"""
TinyGPT-100M: resumable, chunked training of a ~100M-param decoder-only
transformer chatbot on DailyDialog + UltraChat. Designed to run in ~4h
chunks on GitHub Actions (CPU), with checkpoint resume between chunks.

Env knobs (used for local testing; CI uses defaults):
  TDIM (832) TLAYERS (12) THEADS (13) TFFN (3328) TTOTAL_STEPS (8000)
  STEPS_PER_RUN (1000)  RESUME (path to ckpt.pt from previous chunk)
"""
import json, math, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- config -----------------------------
BLOCK = 256
D = int(os.environ.get("TDIM", 832))
N_LAYER = int(os.environ.get("TLAYERS", 12))
N_HEAD = int(os.environ.get("THEADS", 13))      # head dim 64
FFN = int(os.environ.get("TFFN", 3328))
TOTAL_STEPS = int(os.environ.get("TTOTAL_STEPS", 8000))
STEPS_PER_RUN = int(os.environ.get("STEPS_PER_RUN", 1000))
RESUME = os.environ.get("RESUME", "")

LR = 6e-4
LR_MIN = 5e-5
WARMUP = 300
BATCH = 16
DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUT_DIR = os.environ.get("OUT_DIR", "./run")
LOG_EVERY = 25
CKPT_EVERY = 200          # intra-chunk checkpoint (overwritten)
SPECIALS = {"<bos>": "\u0002", "<eos>": "\u0003"}

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(1337)
random.seed(1337)

# ----------------------------- data -----------------------------
dialogues = []
with open(os.path.join(DATA_DIR, "corpus.txt"), encoding="utf-8") as f:
    for line in f:
        turns = [t.strip() for t in line.rstrip("\n").split("\t") if t.strip()]
        if len(turns) >= 2:
            dialogues.append(turns)
random.shuffle(dialogues)
val_dialogues = dialogues[-400:]
train_dialogues = dialogues[:-400]

def render(dialogue_list):
    out = []
    for turns in dialogue_list:
        parts = [SPECIALS["<bos>"]]
        for i, t in enumerate(turns):
            speaker = "User" if i % 2 == 0 else "Bot"
            parts.append(f"{speaker}: {t}\n")
        parts.append(SPECIALS["<eos>"])
        out.append("".join(parts))
    return "".join(out)

train_text = render(train_dialogues)
chars = sorted(set(train_text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
train_ids = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)

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
        self.attn = nn.Linear(D, 3 * D, bias=True)
        self.proj = nn.Linear(D, D, bias=True)
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, FFN, bias=True)
        self.fc2 = nn.Linear(FFN, D, bias=True)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.attn(h)
        q, k, v = qkv.split(C, dim=2)
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
        x = self.wte(idx) + self.wpe(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.T
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss

model = TinyGPT()
if not (RESUME and os.path.exists(RESUME)):
    nn.init.normal_(model.wte.weight, mean=0.0, std=0.02)
    nn.init.normal_(model.wpe.weight, mean=0.0, std=0.02)
nparams = sum(p.numel() for p in model.parameters())
print(f"model parameters: {nparams:,} ({nparams/1e6:.3f}M) | vocab {V} | corpus {len(train_text):,} chars", flush=True)

def lr_at(step):
    if step < WARMUP:
        return LR * step / WARMUP
    p = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * p))

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)

start_step = 1
if RESUME and os.path.exists(RESUME):
    ck = torch.load(RESUME, map_location="cpu")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    start_step = ck["step"] + 1
    print(f"resumed from step {ck['step']} (chunk artifact)", flush=True)

end_step = min(start_step + STEPS_PER_RUN - 1, TOTAL_STEPS)
print(f"this chunk: steps {start_step}..{end_step} of {TOTAL_STEPS}", flush=True)

@torch.no_grad()
def sample(prompt, n=80, temp=0.8, topk=40):
    model.eval()
    ctx = torch.tensor([[stoi[c] for c in prompt if c in stoi]], dtype=torch.long)
    out = []
    for _ in range(n):
        logits, _ = model(ctx[:, -BLOCK:])
        logits = logits[0, -1] / temp
        kth = torch.topk(logits, topk).values[-1]
        logits[logits < kth] = -1e9
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        if nxt.item() == stoi[SPECIALS["<eos>"]] or chars[nxt.item()] == "\n":
            break
        out.append(chars[nxt.item()])
        ctx = torch.cat([ctx, nxt.view(1, 1)], dim=1)
    model.train()
    return "".join(out)

t0 = time.time()
for step in range(start_step, end_step + 1):
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = get_batch(train_ids)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % LOG_EVERY == 0:
        dt = time.time() - t0
        print(f"step {step}/{TOTAL_STEPS} | train {loss.item():.3f} | lr {lr:.2e} | {dt/LOG_EVERY:.2f}s/step", flush=True)
        t0 = time.time()

    if step % 1000 == 0:
        print("--- sample after step " + str(step) + " ---")
        print("Bot says: " + sample(SPECIALS["<bos>"] + "User: hello, how are you today?\nBot:"), flush=True)

    if step % CKPT_EVERY == 0 or step == end_step:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
                   os.path.join(OUT_DIR, "ckpt.pt"))

# ----------------------------- final export -----------------------------
if end_step >= TOTAL_STEPS:
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
        "bos": SPECIALS["<bos>"], "eos": SPECIALS["<eos>"],
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(meta, f)
    open(os.path.join(OUT_DIR, "DONE"), "w").write("ok\n")
    print(f"TRAINING COMPLETE. Exported model.bin ({len(flat):,} bytes) + config.json", flush=True)
else:
    print(f"chunk done at step {end_step}; next chunk should resume with RESUME=run/ckpt.pt", flush=True)
