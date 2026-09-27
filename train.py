"""
Train a ~1M-parameter GPT-style decoder-only transformer chatbot on DailyDialog.
Exports weights for a pure-JS inference engine (single float32 .bin + config.json).
"""
import json, math, os, random, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- config -----------------------------
BLOCK = 256          # context window
D = 192              # embedding dim
N_LAYER = 2
N_HEAD = 6           # head dim 32
FFN = 800
LR = 1e-3
WARMUP = 150
TOTAL_STEPS = 3000
BATCH = 16
DATA_DIR = "./data/train"
OUT_DIR = "./run"
LOG_EVERY = 25
CKPT_EVERY = 500
SPECIALS = {"<bos>": "\u0002", "<eos>": "\u0003"}

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(1337)
random.seed(1337)

# ----------------------------- data -----------------------------
def load_dialogues(path):
    convs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            utts = [u.strip() for u in line.split("__eou__") if u.strip()]
            if len(utts) >= 2:
                convs.append(utts)
    return convs

convs = load_dialogues(os.path.join(DATA_DIR, "dialogues_train.txt"))
random.shuffle(convs)
val_convs = convs[-400:]
train_convs = convs[:-400]

def render(convs_list):
    """Each dialogue becomes: <bos>User: ...\nBot: ...\n...<eos>"""
    out = []
    for utts in convs_list:
        parts = [SPECIALS["<bos>"]]
        for i, u in enumerate(utts):
            speaker = "User" if i % 2 == 0 else "Bot"
            parts.append(f"{speaker}: {u}\n")
        parts.append(SPECIALS["<eos>"])
        out.append("".join(parts))
    return "".join(out)

train_text = render(train_convs)
val_text = render(val_convs)

chars = sorted(set(train_text))
stoi = {c: i for i, c in enumerate(chars)}
V = len(chars)
print(f"dialogues: {len(train_convs)} train / {len(val_convs)} val | chars: {V} | train corpus: {len(train_text):,} chars", flush=True)

def encode(s):
    return [stoi[c] for c in s]

train_ids = torch.tensor(encode(train_text), dtype=torch.long)
val_ids = torch.tensor(encode(val_text), dtype=torch.long)

def get_batch(data):
    ix = torch.randint(0, len(data) - BLOCK - 1, (BATCH,))
    x = torch.stack([data[i:i+BLOCK] for i in ix])
    y = torch.stack([data[i+1:i+1+BLOCK] for i in ix])
    return x, y

# ----------------------------- model -----------------------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = nn.Linear(D, 3 * D, bias=True)     # qkv
        self.proj = nn.Linear(D, D, bias=True)
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, FFN, bias=True)
        self.fc2 = nn.Linear(FFN, D, bias=True)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.attn(h)
        q, k, v = qkv.split(D, dim=2)
        q = q.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        k = k.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        h = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(V, D)
        self.wpe = nn.Embedding(BLOCK, D)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.T  # tied output head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss

model = TinyGPT()
nn.init.normal_(model.wte.weight, mean=0.0, std=0.02)
nn.init.normal_(model.wpe.weight, mean=0.0, std=0.02)
nparams = sum(p.numel() for p in model.parameters())
print(f"model parameters: {nparams:,} ({nparams/1e6:.3f}M)", flush=True)

def lr_at(step):
    if step < WARMUP:
        return LR * step / WARMUP
    p = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
    return 1e-4 + 0.5 * (LR - 1e-4) * (1 + math.cos(math.pi * p))

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)

@torch.no_grad()
def sample(prompt, n=150, temp=0.8, topk=40):
    model.eval()
    ctx = torch.tensor([encode(SPECIALS["<bos>"] + prompt)], dtype=torch.long)
    out_chars = []
    for _ in range(n):
        idx = ctx[:, -BLOCK:]
        logits, _ = model(idx)
        logits = logits[0, -1] / temp
        if topk:
            kth = torch.topk(logits, topk).values[-1]
            logits[logits < kth] = -1e9
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        if nxt.item() == stoi[SPECIALS["<eos>"]]:
            break
        out_chars.append(chars[nxt.item()])
        ctx = torch.cat([ctx, nxt.view(1, 1)], dim=1)
    model.train()
    return "".join(out_chars)

log_f = open(os.path.join(OUT_DIR, "train.log"), "a")
def log(msg):
    print(msg, flush=True)
    log_f.write(msg + "\n")
    log_f.flush()

best_val = float("inf")
t0 = time.time()
for step in range(1, TOTAL_STEPS + 1):
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = get_batch(train_ids)
    logits, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % LOG_EVERY == 0:
        with torch.no_grad():
            vx, vy = get_batch(val_ids)
            _, vloss = model(vx, vy)
        dt = time.time() - t0
        log(f"step {step}/{TOTAL_STEPS} | train {loss.item():.3f} | val {vloss.item():.3f} | lr {lr:.2e} | {dt/LOG_EVERY:.2f}s/step")
        t0 = time.time()

    if step % 500 == 0:
        log("--- sample after step " + str(step) + " ---")
        log("Bot says: " + sample("User: hello, how are you today?\nBot:").replace("\n", " / "))

    if step % CKPT_EVERY == 0:
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "ckpt.pt"))

    if step % 500 == 0:
        with torch.no_grad():
            vx, vy = get_batch(val_ids)
            _, vloss = model(vx, vy)
        if vloss.item() < best_val:
            best_val = vloss.item()
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))

torch.save(model.state_dict(), os.path.join(OUT_DIR, "ckpt.pt"))

# ----------------------------- export for JS -----------------------------
sd = model.state_dict()
tensors = [("wte", model.wte.weight), ("wpe", model.wpe.weight)]
for b in model.blocks:
    tensors += [
        ("ln1_g", b.ln1.weight), ("ln1_b", b.ln1.bias),
        ("attn_w", b.attn.weight), ("attn_b", b.attn.bias),
        ("proj_w", b.proj.weight), ("proj_b", b.proj.bias),
        ("ln2_g", b.ln2.weight), ("ln2_b", b.ln2.bias),
        ("fc1_w", b.fc1.weight), ("fc1_b", b.fc1.bias),
        ("fc2_w", b.fc2.weight), ("fc2_b", b.fc2.bias),
    ]
tensors += [("lnf_g", model.ln_f.weight), ("lnf_b", model.ln_f.bias)]
flat = torch.cat([t.reshape(-1).float().detach() for _, t in tensors]).numpy().tobytes()
with open(os.path.join(OUT_DIR, "model.bin"), "wb") as f:
    f.write(flat)

meta = {
    "vocab": V, "d": D, "n_layers": N_LAYER, "n_heads": N_HEAD, "ffn": FFN,
    "block": BLOCK, "nparams": nparams,
    "itos": chars,
}
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(meta, f)

log(f"DONE. params={nparams} best_val={best_val:.3f}")
