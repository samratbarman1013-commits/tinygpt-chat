"""Build the fine-tune corpus: FULL regular corpus + tool-calling dialogues x10.

IMPORTANT vocab guard: the model checkpoint was trained with vocab = 648,
the charset of the ORIGINAL train split (seed-1337 shuffle, last 400 = val).
Re-shuffling the bigger mix can move a val-only rare-char dialogue into
train and grow the vocab (run 1: 334 after bad subsampling, run 2: 649),
which crashes checkpoint loading. So every dialogue is filtered against
the exact 648 charset (derived from corpus.txt BEFORE mixing, via
tool_data.charset_from_corpus) and anything outside is dropped.
"""
import os
import random

import tool_data  # same directory; reads corpus.txt before we overwrite it

random.seed(4242)
DATA = os.environ.get("DATA_DIR", "./data")

BASE = set(tool_data.charset_from_corpus())
print(f"base charset: {len(BASE)} chars")


def ok(line):
    # check turn text only (\t is just the file separator, never rendered)
    return all(c in BASE for t in line.split("\t") for c in t)


regular = [l.rstrip("\n") for l in open(f"{DATA}/corpus.txt", encoding="utf-8") if l.strip()]
tool = [l.rstrip("\n") for l in open(f"{DATA}/tool_dialogues.txt", encoding="utf-8") if l.strip()]

keep_reg = [l for l in regular if ok(l)]
keep_tool = [l for l in tool if ok(l)]
print(f"dropped {len(regular) - len(keep_reg)} regular / {len(tool) - len(keep_tool)} tool dialogues (rare chars)")

mix = keep_reg + keep_tool * 10
random.shuffle(mix)

with open(f"{DATA}/corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(mix) + "\n")

chars = sum(len(l) for l in mix)
print(f"fine-tune corpus: {len(keep_reg)} regular (FULL, filtered) + {len(keep_tool)} tool x10 = {len(mix)} dialogues, {chars:,} chars")

# final assertion: rendered charset can never exceed the checkpoint vocab
allchars = set()
for l in mix:
    for t in l.split("\t"):
        allchars.update(t)
assert allchars | BASE == BASE, f"charset grew by: {allchars - BASE}"
print(f"final safety check passed: {len(allchars)} chars used, subset of {len(BASE)}")
