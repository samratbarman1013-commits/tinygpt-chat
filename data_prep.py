"""Build data/corpus.txt for TinyGPT-100M training.

One dialogue per line; turns separated by a single TAB character.
Sources (fixed URLs, deterministic result):
  - DailyDialog train/validation/test (HF: roskoN/dailydialog)
  - UltraChat-200k train_sft shard 0000 (HF parquet convert branch)

Files expected under DATA_DIR (default ./data):
  train/dialogues_train.txt, validation/dialogues_validation.txt,
  test/dialogues_test.txt, uc0.parquet
"""
import os
import re

DATA = os.environ.get("DATA_DIR", "./data")
OUT = os.path.join(DATA, "corpus.txt")
MAX_TURN = 220        # chars per turn (UltraChat only)
MAX_DIALOGUE = 900    # chars per dialogue (UltraChat only)
MAX_TURNS = 8         # turns per dialogue (UltraChat only)


def clean(t, maxlen=None):
    t = t.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if maxlen and len(t) > maxlen:
        t = t[:maxlen].rsplit(" ", 1)[0].rstrip(" ,;:")
    return t


def ascii_ok(t):
    if len(t) < 2:
        return False
    bad = sum(1 for c in t if ord(c) > 126)
    return bad / len(t) < 0.05


dialogues = []

# ---------- 1. DailyDialog (natural short human chats; no truncation) ----------
def load_dd(path):
    convs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            utts = [u.strip() for u in line.split("__eou__") if u.strip()]
            utts = [u for u in utts if ascii_ok(clean(u))]
            if len(utts) >= 2:
                convs.append([clean(u) for u in utts])
    return convs


for split in ["train", "validation", "test"]:
    p = os.path.join(DATA, split, "dialogues_%s.txt" % split)
    if os.path.exists(p):
        d = load_dd(p)
        dialogues += d
        print(f"dailydialog {split}: {len(d)} dialogues", flush=True)
    else:
        raise FileNotFoundError(p)

# ---------- 2. UltraChat (multi-turn, first shard) ----------
import pyarrow.parquet as pq

pf = pq.ParquetFile(os.path.join(DATA, "uc0.parquet"))
n_groups = pf.metadata.num_row_groups
uc = 0
for gi in range(n_groups):
    tb = pf.read_row_group(gi)
    for r in tb.to_pylist():
        turns = []
        want_user = True
        total = 0
        for m in r.get("messages", []):
            if len(turns) >= MAX_TURNS:
                break
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            is_user = role == "user"
            if is_user != want_user:
                continue
            t = clean(m.get("content", ""), MAX_TURN)
            if not ascii_ok(t):
                continue
            if total + len(t) > MAX_DIALOGUE:
                break
            turns.append(t)
            total += len(t)
            want_user = not want_user
        if len(turns) >= 2:
            uc += 1
            dialogues.append(turns)
print(f"ultrachat: {uc} dialogues", flush=True)

# ---------- write ----------
with open(OUT, "w", encoding="utf-8") as f:
    for d in dialogues:
        f.write("\t".join(d) + "\n")

nchars = sum(len("\t".join(d)) for d in dialogues)
print(f"TOTAL: {len(dialogues)} dialogues, {nchars:,} chars -> {OUT}", flush=True)
