"""Karma Chat — online server.

The model runs HERE (on the server); clients chat over the internet via POST /chat.
Weights are read from the same flat `model.bin` the browser engine uses
(state_dict order == module definition order == export order).
"""
import json
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model import TinyGPT
import tools

app = FastAPI(title="Karma Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

def _model_files():
    """Model weights come from a PRIVATE Hugging Face repo when HF_REPO is set
    (token via HF_TOKEN) — so the model file itself never has to be public.
    Otherwise fall back to the local config.json / model.bin next to app.py."""
    repo = os.getenv("HF_REPO")
    if repo:
        from huggingface_hub import hf_hub_download
        kw = {"repo_id": repo, "repo_type": "model"}
        if os.getenv("HF_TOKEN"):
            kw["token"] = os.getenv("HF_TOKEN")
        print(f"loading model from private HF repo: {repo}", flush=True)
        return (hf_hub_download(filename="config.json", **kw),
                hf_hub_download(filename="model.bin", **kw))
    return "config.json", "model.bin"

CFG_PATH, BIN_PATH = _model_files()
cfg = json.load(open(CFG_PATH, encoding="utf-8"))
itos = cfg["itos"]
stoi = {c: i for i, c in enumerate(itos)}
BOS = cfg.get("bos", "\u0002")
EOS = cfg.get("eos", "\u0003")
BLOCK = cfg["block"]

model = TinyGPT(cfg["vocab"], cfg["d"], cfg["n_layers"], cfg["n_heads"], cfg["ffn"], cfg["block"])

# --- load flat model.bin into the state dict ---
raw = open(BIN_PATH, "rb").read()
flat = torch.frombuffer(bytearray(raw), dtype=torch.float32)
sd = model.state_dict()
off = 0
with torch.no_grad():
    for k, v in sd.items():
        n = v.numel()
        sd[k] = flat[off:off + n].view(v.shape).clone()
        off += n
if off != flat.numel():
    raise RuntimeError(f"weight size mismatch: {off} loaded vs {flat.numel()} available")
model.load_state_dict(sd)
model.eval()
print(f"Karma server ready — {cfg['nparams']:,} parameters loaded", flush=True)


class Msg(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    messages: list[Msg]
    temp: float = 0.75
    topk: int = 40
    max_new: int = 160


@torch.no_grad()
def generate(messages, temp=0.75, topk=40, max_new=160):
    prompt = BOS
    for m in messages:
        prompt += ("User: " if m.role == "user" else "Bot: ") + m.text.strip() + "\n"
    prompt += "Bot:"
    ids = [stoi[c] for c in prompt if c in stoi] or [0]
    budget = max(16, BLOCK - max_new - 2)
    if len(ids) > budget:
        ids = ids[-budget:]
    ctx = torch.tensor([ids], dtype=torch.long)
    out = []
    eos_id = stoi.get(EOS)
    for _ in range(max_new):
        logits = model(ctx[:, -BLOCK:])[0, -1]
        if temp > 0.01:
            logits = logits / temp
            if 0 < topk < logits.numel():
                kth = torch.topk(logits, topk).values[-1]
                logits[logits < kth] = -1e9
            probs = F.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        else:
            nxt = int(logits.argmax())
        if nxt == eos_id:
            break
        ch = itos[nxt]
        if ch == "\n":
            break
        out.append(ch)
        if "".join(out).endswith("User:"):
            out = out[:-6]
            break
        ctx = torch.cat([ctx, torch.tensor([[nxt]])], dim=1)
    return "".join(out).strip()


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"ok": True, "params": cfg["nparams"],
            "tools": ["calculator", "clock", "dice", "wikipedia", "duckduckgo"]}


@app.post("/chat")
def chat(req: ChatRequest):
    # tool layer first: real, live answers for factual queries
    last_user = next((m.text for m in reversed(req.messages) if m.role == "user"), "")
    tool_name, answer = tools.route(last_user)
    if tool_name:
        return {"reply": answer, "source": tool_name}
    reply = generate(req.messages, req.temp, req.topk, req.max_new)
    if not reply:  # model produced nothing useful -> web fallback
        answer = tools.ddg_tool(last_user)
        if answer:
            return {"reply": answer, "source": "duckduckgo"}
    return {"reply": reply, "source": "model"}
