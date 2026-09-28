"""Build the fine-tune corpus: FULL regular corpus + tool-calling dialogues x10.

IMPORTANT: the regular corpus is kept COMPLETE — subsampling it would drop
rare characters (emoji etc.), shrink the vocab below 648 and break checkpoint
loading (learned that the hard way on run 1, 2026-09-29).
Mix: full regular corpus (~62M chars) + tool dialogues x10 (~21M chars),
so tool-use patterns are ~25% of what the fine-tune samples.
Overwrites data/corpus.txt (train_100m.py reads that path).
"""
import os
import random

random.seed(4242)
DATA = os.environ.get("DATA_DIR", "./data")

regular = [l.rstrip("\n") for l in open(f"{DATA}/corpus.txt", encoding="utf-8") if l.strip()]
tool = [l.rstrip("\n") for l in open(f"{DATA}/tool_dialogues.txt", encoding="utf-8") if l.strip()]

mix = regular + tool * 10
random.shuffle(mix)

with open(f"{DATA}/corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(mix) + "\n")

chars = sum(len(l) for l in mix)
print(f"fine-tune corpus: {len(regular)} regular (FULL) + {len(tool)} tool x10 = {len(mix)} dialogues, {chars:,} chars")
