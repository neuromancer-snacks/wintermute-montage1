#!/usr/bin/env python3
"""collapse_check.py — how many distinct values did the forced channel emit?
Reads existing traces. Writes nothing. Martorell & Bianchi 2026, greedy test."""

import glob, json, sys
from collections import Counter

paths = sys.argv[1:] or sorted(glob.glob("*.jsonl"))
if not paths:
    sys.exit("no traces found")

fields = {}
n_turns = 0

for p in paths:
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") != "turn":
                continue
            n_turns += 1
            for dim in ("D1_valence", "D2_load", "D3_fit"):
                d = row.get("CH2", {}).get(dim, {})
                for field in ("word", "intensity", "confidence"):
                    fields.setdefault((dim, field), Counter())[d.get(field)] += 1

print(f"\nturns: {n_turns}   files: {len(paths)}\n")
for (dim, field), c in fields.items():
    total = sum(c.values())
    distinct = len([v for v in c if v is not None])
    top = c.most_common(3)
    modal_share = top[0][1] / total if total else 0
    print(f"{dim:12s} {field:11s}  distinct={distinct:2d}  "
          f"modal={top[0][0]} at {modal_share:.0%}")
    print(f"{'':26s}{top}")
print()