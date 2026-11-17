"""Run the whole experiment on one model.

    python run.py --model Qwen/Qwen2.5-1.5B-Instruct
    python run.py --model stub            # offline plumbing test, random weights

Writes results/<tag>/: lens.pt, holds.json, recall.jsonl, summary.json, and
then prints the pre-registered verdict (score.py).
"""
import argparse
import json
import os
import re
import time

import torch

from qimag import items as I
from qimag import experiment as E
from qimag.lens import Lens, compute_jlens, reading_stats
from qimag.model import load


def check_fillers():
    for desc, tgt, cue in I.ITEMS:
        for f in I.FILLERS:
            if re.search(rf"\b{re.escape(tgt)}\b", f, re.I):
                raise SystemExit(f"filler contains target word {tgt!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lens-batch", type=int, default=8)
    ap.add_argument("--max-items", type=int, default=None, help="for quick tests only")
    a = ap.parse_args()
    check_fillers()
    tag = a.out or "results/" + a.model.replace("/", "_")
    os.makedirs(tag, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)

    lm = load(a.model, a.device)
    log(f"model {lm.name}: {lm.n_layers} layers, d={lm.d}, device {a.device}")

    # concept vocabulary: answers, cues and decoys that are single tokens
    words = []
    for w in [t for _, t, _ in I.ITEMS] + [c for _, _, c in I.ITEMS] + I.DECOYS:
        if w not in words and lm.single_token(w) is not None:
            words.append(w)
    lens_path = os.path.join(tag, "lens.pt")
    if os.path.exists(lens_path):
        lens = Lens.load(lens_path, a.device)
        log(f"lens loaded ({len(lens.words)} concepts)")
    else:
        log(f"computing J-lens for {len(words)} concepts")
        J = compute_jlens(lm, [lm.single_token(w) for w in words], I.CALIBRATION,
                          batch=a.lens_batch, log=log)
        mu, sd, hmu = reading_stats(lm, J, I.CALIBRATION)
        lens = Lens(J, mu, sd, hmu, words)
        lens.save(lens_path)
        log("lens saved")

    S = E.Setup(lm, lens)
    if a.max_items:
        S.items = S.items[:a.max_items]
        S.finalize()
    log(f"{len(S.items)} items usable, {len(S.dropped)} dropped (multi-token): "
        + ", ".join(t for _, t, _ in S.dropped))

    holds, null = E.hold_all(S)
    rank, spec = E.g0_table(S, holds)
    l_read, l_inj = E.choose_layers(S, rank)
    log(f"layers chosen on dev: read {l_read}, write {l_inj}")
    null_rank = []
    for _, t, _ in S.items:
        z = null[0][l_read]
        null_rank.append(int((z > z[lens.index[t]]).sum()))
    holds_out = []
    for i, (d, t, c) in enumerate(S.items):
        zr = holds[i][0][l_read]
        top = [lens.words[k] for k in zr.topk(10).indices.tolist()]
        holds_out.append({"item": i, "split": S.split(i), "desc": d, "target": t, "cue": c,
                          "rank_by_layer": rank[i].tolist(), "spec_by_layer": spec[i].tolist(),
                          "rank_read": int(rank[i, l_read]), "spec_read": float(spec[i, l_read]),
                          "null_rank_read": null_rank[i], "top10_read": top})
    json.dump({"model": lm.name, "n_layers": lm.n_layers, "l_read": l_read, "l_inj": l_inj,
               "concepts": len(lens.words), "n_choice": len(S.choice_words), "dropped": [t for _, t, _ in S.dropped],
               "items": holds_out}, open(os.path.join(tag, "holds.json"), "w"), indent=1)

    with open(os.path.join(tag, "recall.jsonl"), "w") as f:
        for i in range(len(S.items)):
            rows, held = E.recall_item(S, i, holds, l_read, l_inj)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            best = {c: max(r["correct"] for r in rows if r["cond"] == c)
                    for c in ("none", "in_context", "cue", "jstate_open", "jstate_closed")}
            log(f"item {i:2d} {S.items[i][1]:>9s}  held={held[:4]}  any-strength correct: {best}")

    from score import score_dir
    score_dir(tag)


if __name__ == "__main__":
    main()
