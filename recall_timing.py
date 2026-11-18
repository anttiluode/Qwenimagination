"""One-item answer-word timing diagnostic. See TIMING.md."""
import argparse
import json
import os
import random
from pathlib import Path
from statistics import mean

import torch

from diagnose import ASK2, PREFIX2, ids_and_ask_start
from qimag import experiment as E, items as I
from qimag.items_v2 import ITEMS_V2
from qimag.lens import Lens
from qimag.model import load
from recall_slots import MODEL, ROOT, L_READ, L_INJ


def timing_prompt(lm, filler):
    marker = "Note:"
    user = marker + "\n\n" + filler + "\n\n" + marker + ASK2
    chat = lm.chat_text([("user", user)], PREFIX2)
    before, found, _ = chat.partition(user)
    if not found:
        raise ValueError("Chat template changed the prompt")
    tokens = lm.tok(chat, add_special_tokens=False, return_offsets_mapping=True)
    ids, spans = tokens["input_ids"], tokens["offset_mapping"]

    def locate(pos):
        return next(i for i, (a, b) in enumerate(spans) if a <= len(before) + pos < b)

    early = locate(len(marker)-1)
    late = locate(user.rindex(marker)+len(marker)-1)
    question = locate(user.rindex(ASK2)+len(ASK2)-len(ASK2.lstrip()))
    if not early < late < question < len(ids)-1 or ids[early] != ids[late]:
        raise ValueError("Expected identical marker tokens before and after the filler")
    return ids, {"none": [], "repeated": list(range(question, len(ids))),
                 "answer": [len(ids)-1], "before_question": [late], "before_gap": [early]}


def trial(setup, item_index, ids, positions, vector):
    delta = None
    if positions:
        delta = vector.new_zeros((len(ids), vector.numel()))
        delta[positions] = vector
    lp = E.run_open(setup, ids, L_INJ, delta)
    return E.score(setup, item_index, lp[-1])


def paired_interval(values):
    rng = random.Random(20261118)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(3000))
    return [mean(values), draws[74], draws[2924]]


def summarize(rows):
    arms = rows[0]["scores"]
    comparisons = [("repeated", "none"), ("answer", "repeated"),
                   ("before_question", "answer"), ("before_gap", "before_question"),
                   ("before_gap", "none")]
    return {
        "items": len(rows),
        "v2_top_choices_reproduced": sum(r["scores"]["v2_replay"]["said"] == r["v2_said"] for r in rows),
        "accuracy": {arm: mean(r["scores"][arm]["correct"] for r in rows) for arm in arms},
        "paired_accuracy_differences_mean_ci95": {
            f"{a}_minus_{b}": paired_interval([r["scores"][a]["correct"] - r["scores"][b]["correct"]
                                               for r in rows])
            for a, b in comparisons},
        "mean_target_logprob_gain_vs_matched_none": {
            a: mean(r["scores"][a]["lp"] - r["scores"]["none"]["lp"] for r in rows)
            for a in arms if a != "v2_replay"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-items", type=int, choices=range(1, 61))
    args = parser.parse_args()
    path = args.results / "delayed_v5" / "v2_lens.pt"
    if not path.exists():
        parser.error(f"Missing {path}. This diagnostic reuses the V5 calibration lens.")
    history = [json.loads(line) for line in (args.results / "confirm_v2" / "rows.jsonl").read_text().splitlines()]
    oracle_rows = [r for r in history if r["arm"] == "oracle"]
    if len(oracle_rows) != 60 or len({r["target"] for r in oracle_rows}) != 60:
        parser.error("Expected the 60 published V2 oracle items")
    lm = load(MODEL, args.device)
    lens = Lens.load(path, lm.device)
    setup = E.Setup(lm, lens)
    usual = torch.stack([E.hold_read(setup, I.HOLD_PROMPT.format(desc=d))[0][L_READ]
                         for d, _, _ in I.ITEMS]).mean(0)
    setup.items = [it for it in ITEMS_V2 if all(lm.single_token(w) is not None for w in it[1:])]
    setup.finalize()
    subset = oracle_rows[:args.max_items]
    rows = []
    for record in subset:
        i = record["item"]
        desc, target, cue = setup.items[i]
        if (target, cue) != (record["target"], record["cue"]):
            raise ValueError("Current tokenizer item list differs from V2")
        resid = E.hold_read(setup, I.HOLD_PROMPT.format(desc=desc))[0][L_READ] - usual
        vector = lens.write(L_INJ, [lens.index[target]], resid.max().reshape(1))
        filler = I.FILLERS[i % len(I.FILLERS)]
        original, start = ids_and_ask_start(lm, [("user", filler + ASK2)], PREFIX2)
        scores = {"v2_replay": trial(setup, i, original, list(range(start, len(original))), vector)}
        ids, conditions = timing_prompt(lm, filler)
        for arm, positions in conditions.items():
            scores[arm] = trial(setup, i, ids, positions, vector)
        rows.append(dict(item=i, target=target, v2_said=record["said"], scores=scores,
                         positions=conditions, prompt_length=len(ids)))
        print(f"{len(rows)}/{len(subset)} {target}", flush=True)
    report = summarize(rows)
    out = args.results / "timing_v6"
    out.mkdir(exist_ok=True)
    dest = out / ("full.json" if len(rows) == 60 else f"smoke-{len(rows)}.json")
    tmp = dest.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(model=MODEL, summary=report, rows=rows), allow_nan=False))
    tmp.replace(dest)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
