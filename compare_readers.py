"""Reproduce the retrospective V2/V3 fixed-choice comparison using saved rows."""
import json
from pathlib import Path

from qimag.items_v2 import ITEMS_V2
from qimag.items_v3 import ITEMS_V3


def compare(records, items, encode, arm):
    eligible = [it for it in items if all(len(encode(" " + w)) == 1 for w in it[1:])]
    choices = {w for it in eligible for w in it[1:]}
    rows = [r for r in records if r["arm"] == arm]
    totals = dict(items=len(rows), strongest=0, qwen=0, eligible_reader=0, abstained=0)
    for r in rows:
        if eligible[r["item"]][1:] != (r["target"], r["cue"]):
            raise ValueError("Candidate list differs from original item indices")
        top = next((w for w in r["held"] if w in choices), None)
        totals["strongest"] += r["held"][0] == r["target"]
        totals["qwen"] += r["correct"]
        totals["eligible_reader"] += top == r["target"]
        totals["abstained"] += top is None
    return totals


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    root = Path("results/Qwen_Qwen2.5-1.5B-Instruct")
    for folder, items, arm in (("confirm_v2", ITEMS_V2, "primary"), ("pick_v3", ITEMS_V3, "A_usual")):
        records = (json.loads(line) for line in (root/folder/"rows.jsonl").read_text().splitlines())
        print(folder, compare(records, items, lambda s: tok.encode(s, add_special_tokens=False), arm))
