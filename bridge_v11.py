"""V11: read the thought where it exists (layer 26), install it where it can be used (4-12).
See BRIDGE_V11.md.

V10 read the hold at layers 4-12, where the thought is not yet readable (the item's word was in
the top 8 for 2-6 of 40 items). The v2 diagnostic puts the thought at layers 26-27 (37/38 in the
top 10). V9 showed that a concept swapped in across layers 4-12 steers a later question.
V11 bridges them: the concepts and weights come from the layer-26 read; the swap uses those
concepts' own lens directions at layers 4-12. No selection stage: alpha = 1 (V9), K = 1 primary.
"""
import argparse
import json
import os
import random
from statistics import mean

import torch

from category_transfer import PAIRS, TRAIN, V2_CLUES, bag, classifier
from category_v8 import MODEL, ROOT
from confirm import extend_lens
from qimag.lens import Lens
from qimag.model import load
from thought_swap_v10 import BAND, Lab, cases, summarize

L_READ = 26
ALPHA = 1.0
K_PRIMARY, K_SECONDARY = 1, 8
SEED = 20261118


# The act of holding, not its content (v1: these led 44-53 of 58 holds). Removed so the top
# concept is the thought, not "picture".
MODE_WORDS = {"thought", "thoughts", "imagine", "mind", "think", "brain", "picture", "pictures",
              "photographs", "memory"}


def late_bag(lab, clue, decoy, l_read):
    z = lab.hold(clue)[l_read]
    mode = {i for i, w in enumerate(lab.lens.words) if w.strip().casefold() in MODE_WORDS}
    blocked = sorted(lab.base_block | mode | {lab.lens.index[decoy]})
    return bag(z, blocked)                         # idx, values, sparse


def bridged(lab, idx, values, decoy):
    """Early-band directions for concepts chosen (and weighted) by the late read.

    The swap exchanges coordinates, so it is only fair between directions of equal length
    (the paper swaps equal-magnitude projections). Each direction is rescaled to the length
    of the decoy's own lens vector at that layer. The first V11 run omitted this: the bag
    direction was ~100x shorter than the word direction, so the swap erased the decoy and
    installed almost nothing (every swapped answer fell to about 0)."""
    out = {}
    for l in lab.band:
        d = lab.lens.write(l, idx, values).float()
        a = lab.lens.J[l][lab.lens.index[decoy]].float()
        out[l] = d * (a.norm() / d.norm().clamp_min(1e-12))
    return out


def run(lm, lens, band=BAND, l_read=L_READ, alpha=ALPHA, train=TRAIN, pairs=PAIRS, log=print):
    lab = Lab(lm, lens, band)
    rep = {"model": lm.name, "band": list(band), "read_layer": l_read, "alpha": alpha,
           "K_primary": K_PRIMARY, "K_secondary": K_SECONDARY}
    if lab.probes is None:
        rep["verdict"] = "STOP: yes/no tokens are not single tokens"
        return rep
    te = [p for p in pairs if p[0] in lens.index and p[2] in lens.index]
    rep["test_pairs"] = len(te)
    # Reader centroids from training clues at the same read layer.
    feats = []
    for w, _, c in train:
        z = lab.hold(V2_CLUES[w])[l_read]
        feats.append((bag(z, sorted(lab.base_block))[2], c))
    eligible = None
    rng = random.Random(SEED)
    rows = []
    for c in cases(te):
        c = dict(c)
        idx, values, sparse = late_bag(lab, c["clue"], c["decoy"], l_read)
        c["late_bag"] = [lens.words[i] for i in idx.tolist()]
        c["source_in_bag"] = c["source"] in c["late_bag"]
        c["clean"], c["thought"] = lab.evidence(c["decoy"], bridged(lab, idx[:K_PRIMARY], values[:K_PRIMARY], c["decoy"]), alpha)
        _, c["thought_k8"] = lab.evidence(c["decoy"], bridged(lab, idx[:K_SECONDARY], values[:K_SECONDARY], c["decoy"]), alpha)
        if c["kind"] == "cross":
            _, c["word"] = lab.evidence(c["decoy"], lab.word_dirs(c["source"]), alpha)
            if eligible is None:
                eligible = [i for i in range(len(lens.words)) if i not in lab.base_block]
            ridx = torch.tensor(rng.sample([i for i in eligible if i != lens.index[c["decoy"]]],
                                           K_PRIMARY), device=idx.device)
            _, c["random"] = lab.evidence(c["decoy"], bridged(lab, ridx, values[:K_PRIMARY], c["decoy"]), alpha)
            c["reader_correct"] = float((classifier(feats, sparse) > 0) == c["living"])
        rows.append(c)
        log(f"{len(rows)}/{4 * len(te)} {c['kind']} {c['source']} -> {c['decoy']}: "
            f"top concept {c['late_bag'][0]}")
    t = summarize(rows, ["thought", "thought_k8", "word", "random"])
    k8 = summarize([dict(r, thought=r["thought_k8"]) for r in rows], ["thought", "random"])
    t["thought_k8_cross_minus_within"] = k8["thought_cross_minus_within"]
    t["thought_k8_minus_random_cross"] = k8["thought_minus_random_cross"]
    t["reader_accuracy"] = mean(r["reader_correct"] for r in rows if "reader_correct" in r)
    t["source_in_late_bag"] = mean(r["source_in_bag"] for r in rows if r["kind"] == "cross")
    t["gates"] = {
        "T1_word_swap_still_works": t.get("word_cross_flip", 0) > 0.25,
        "G1_thought_cross_beats_within": t["thought_cross_minus_within"][1] > 0,
        "G2_thought_beats_random_direction": t["thought_minus_random_cross"][1] > 0}
    rep["test"], rep["rows"] = t, rows
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true", help="3 test pairs")
    args = ap.parse_args()
    root = type(ROOT)(args.results)
    lm = load(MODEL, args.device)
    cached = root / "swap_v9" / "v9_lens.pt"
    if cached.is_file():
        lens = Lens.load(cached, lm.device)
    else:
        base = root / "delayed_v5" / "v2_lens.pt"
        if not base.is_file():
            ap.error(f"Missing calibration lens: {base}")
        words = [w for w, _, _ in TRAIN] + [w for p in PAIRS for w in (p[0], p[2])]
        lens = extend_lens(lm, Lens.load(base, lm.device), list(dict.fromkeys(words)), print)
        cached.parent.mkdir(exist_ok=True)
        lens.save(cached)
    rep = run(lm, lens, pairs=PAIRS[:3] if args.quick else PAIRS)
    out = root / "bridge_v11b"
    out.mkdir(exist_ok=True)
    dest = out / ("quick.json" if args.quick else "full.json")
    tmp = dest.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rep, allow_nan=False))
    tmp.replace(dest)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
