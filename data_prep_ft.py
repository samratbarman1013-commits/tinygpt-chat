"""Build the fine-tune corpus: regular dialogues + tool-calling dialogues.

Mix: ~25% of the regular corpus + tool dialogues repeated 3x, so tool-use
patterns make up roughly 30% of the characters the fine-tune sees.
Overwrites data/corpus.txt (train_100m.py reads that path).
"""
import os
import random

random.seed(4242)
DATA = os.environ.get("DATA_DIR", "./data")

regular = [l.rstrip("\n") for l in open(f"{DATA}/corpus.txt", encoding="utf-8") if l.strip()]
tool = [l.rstrip("\n") for l in open(f"{DATA}/tool_dialogues.txt", encoding="utf-8") if l.strip()]

random.shuffle(regular)
keep = regular[: max(1, len(regular) // 4)]
mix = keep + tool * 3
random.shuffle(mix)

with open(f"{DATA}/corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(mix) + "\n")

chars = sum(len(l) for l in mix)
print(f"fine-tune corpus: {len(keep)} regular + {len(tool)} tool x3 = {len(mix)} dialogues, {chars:,} chars")
