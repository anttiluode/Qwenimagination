"""V10: Qwen's own silent thought, read inside the workspace band, swapped into a loaded slot.
See THOUGHT_V10.md.

Hold: "Silently think of <clue>" (the answer is never written). Read the lens at every layer
of the band (4-12, frozen from V9) minus the usual holding state; keep the top K concepts.
Install: on "Item: <decoy>" (a word of the OTHER category), exchange the decoy's lens
coordinate with the thought's direction, clamped at every position across the band (V9's
swap; the thought's direction at layer l is the lens's minimum-norm write of its bag).
Ask: "Is the item above a living thing?" / "made by people?". Does the answer follow the
thought's category instead of the decoy's?
"""
import argparse
import json
import os
import random
from statistics import mean

import torch

from category_transfer import PAIRS, TRAIN, V2_CLUES, bag, bootstrap, classifier
from category_v8 import ANSWER_WORDS, FORMATS, MODEL, ROOT, prompt, resolve
from confirm import extend_lens
from qimag import experiment as E, items as I
from qimag.lens import Lens, _hs, _replace, tap_layers
from qimag.model import load

BAND = (4, 12)          # frozen from V9
KS = (1, 8)
ALPHAS = (1.0, 2.0)
SEED = 20261118


class SpanSwap:
    """Exchange coordinates along two directions per layer (a: decoy, b: thought),
    clamped at every position of every band layer to the clean pass's exchanged values."""

    def __init__(self, lm, band, A, B, ridge=1e-6):
        self.lm, self.band = lm, list(range(band[0], band[1] + 1))
        self.V = {l: torch.stack([A[l], B[l]]).float() for l in self.band}
        self.Gi = {}
        for l, V in self.V.items():
            G = V @ V.T
            self.Gi[l] = torch.linalg.inv(G + ridge * G.diagonal().mean() * torch.eye(2, device=G.device))
        self.want, self.hooks = None, []

    def coords(self, l, h):
        return (h.float() @ self.V[l].T) @ self.Gi[l]

    @torch.no_grad()
    def run(self, ids, alpha):
        x = torch.tensor([ids], device=self.lm.device)
        with tap_layers(self.lm, self.band) as st:
            clean = self.lm.model(input_ids=x).logits[0, -1].float()
            c = {l: self.coords(l, st[l][0]) for l in self.band}
        self.want = {l: v + alpha * (v[:, [1, 0]] - v) for l, v in c.items()}
        for l in self.band:
            self.hooks.append(self.lm.layers[l].register_forward_hook(self._hook(l)))
        try:
            swapped = self.lm.model(input_ids=x).logits[0, -1].float()
        finally:
            for h in self.hooks:
                h.remove()
            self.hooks = []
        return clean, swapped

    def _hook(self, l):
        def hook(mod, inp, out):
            h = _hs(out)
            d = (self.want[l] - self.coords(l, h[0])) @ self.V[l]
            return _replace(out, h + d.to(h.dtype)[None])
        return hook


class Lab:
    def __init__(self, lm, lens, band=BAND):
        self.lm, self.lens, self.band = lm, lens, list(range(band[0], band[1] + 1))
        self.S = E.Setup(lm, lens)
        self.usual = torch.stack([E.hold_read(self.S, I.HOLD_PROMPT.format(desc=d))[0]
                                  for d, _, _ in I.ITEMS]).mean(0)          # [L, C]
        self.probes = resolve(lm, FORMATS["yesno"])
        self.base_block = {i for i, w in enumerate(lens.words)
                           if w.strip().casefold() in ANSWER_WORDS | {"item"}}
        self._holds = {}

    def hold(self, clue):
        if clue not in self._holds:
            self._holds[clue] = E.hold_read(self.S, I.HOLD_PROMPT.format(desc=clue))[0] - self.usual
        return self._holds[clue]

    def thought(self, clue, K, decoy, rng=None):
        """Per band layer: (concept indices, values) and the thought direction."""
        z = self.hold(clue)
        blocked = sorted(self.base_block | {self.lens.index[decoy]})
        dirs, bags = {}, {}
        eligible = None
        for l in self.band:
            idx, values, _ = bag(z[l], blocked)
            idx, values = idx[:K], values[:K]
            if rng is not None:                     # random-direction control, same weights
                if eligible is None:
                    eligible = [i for i in range(len(self.lens.words)) if i not in set(blocked)]
                idx = torch.tensor(rng.sample(eligible, K), device=idx.device)
            dirs[l] = self.lens.write(l, idx, values)
            bags[l] = [self.lens.words[i] for i in idx.tolist()]
        return dirs, bags

    def word_dirs(self, word):
        return {l: self.lens.J[l][self.lens.index[word]].float() for l in self.band}

    def evidence(self, decoy, B, alpha):
        A = self.word_dirs(decoy)
        sw = SpanSwap(self.lm, (self.band[0], self.band[-1]), A, B)
        clean = swapped = 0.0
        for q, prefix, pos, neg, sign in self.probes:
            ids, _ = prompt(self.lm, decoy, q, prefix)
            c, s = sw.run(ids, alpha)
            f = lambda lp: sign * float(torch.logsumexp(lp[pos], 0) - torch.logsumexp(lp[neg], 0))
            clean, swapped = clean + f(c), swapped + f(s)
        return clean, swapped

    def band_bag(self, clue, decoy=None):
        """Thought averaged over the band, for the external reader."""
        z = self.hold(clue)[self.band].mean(0)
        blocked = sorted(self.base_block | ({self.lens.index[decoy]} if decoy else set()))
        return bag(z, blocked)


def cases(items):
    """items: list of (living_word, living_clue, made_word, made_clue).
    cross: thought from one category into a decoy of the other.
    within: thought into a decoy of its own category (next pair's item)."""
    out = []
    n = len(items)
    for p, (lw, lc, mw, mc) in enumerate(items):
        nlw, nlc, nmw, nmc = items[(p + 1) % n]
        out += [dict(pair=p, kind="cross", source=lw, clue=lc, living=True, decoy=mw),
                dict(pair=p, kind="cross", source=mw, clue=mc, living=False, decoy=lw)]
        if n > 1:
            out += [dict(pair=p, kind="within", source=nlw, clue=nlc, living=True, decoy=lw),
                    dict(pair=p, kind="within", source=nmw, clue=nmc, living=False, decoy=mw)]
    return out


def leaves_decoy(r, key):
    """1 if the answer no longer names the decoy's category."""
    e = r[key]
    decoy_living = r["living"] != (r["kind"] == "cross")
    return 0.5 if e == 0 else float((e > 0) != decoy_living)


def summarize(rows, arms):
    out, per = {}, {}
    for arm in arms:
        for kind in ("cross", "within"):
            g = {}
            for r in rows:
                if r["kind"] == kind and arm in r:
                    g.setdefault(r["pair"], []).append(leaves_decoy(r, arm))
            if g:
                per[(arm, kind)] = {p: mean(v) for p, v in g.items()}
                out[f"{arm}_{kind}_flip"] = mean(per[(arm, kind)].values())
    out["clean_correct"] = mean(1 - leaves_decoy(r, "clean") for r in rows)

    def diff(a, b):
        common = sorted(set(per[a]) & set(per[b]))
        return bootstrap([per[a][p] - per[b][p] for p in common])
    if ("thought", "cross") in per and ("thought", "within") in per:
        out["thought_cross_minus_within"] = diff(("thought", "cross"), ("thought", "within"))
    if ("thought", "cross") in per and ("random", "cross") in per:
        out["thought_minus_random_cross"] = diff(("thought", "cross"), ("random", "cross"))
    if ("word", "cross") in per and ("thought", "cross") in per:
        out["word_minus_thought_cross"] = diff(("word", "cross"), ("thought", "cross"))
    return out


def run(lm, lens, band=BAND, ks=KS, alphas=ALPHAS, train=TRAIN, pairs=PAIRS, log=print):
    lab = Lab(lm, lens, band)
    rep = {"model": lm.name, "band": list(band)}
    if lab.probes is None:
        rep["verdict"] = "STOP: yes/no tokens are not single tokens"
        return rep
    living = [(w, V2_CLUES[w]) for w, _, c in train if c == 0]
    made = [(w, V2_CLUES[w]) for w, _, c in train if c == 1]
    tr = [(a, ac, b, bc) for (a, ac), (b, bc) in zip(living, made) if a in lens.index and b in lens.index]
    te = [p for p in pairs if p[0] in lens.index and p[2] in lens.index]
    rep["train_pairs"], rep["test_pairs"] = len(tr), len(te)

    # ---- stage 1: K and alpha on training pairs
    rep["stage1"] = {}
    best = None
    for K in ks:
        for alpha in alphas:
            rows = []
            for c in cases(tr):
                B, _ = lab.thought(c["clue"], K, c["decoy"])
                c = dict(c)
                c["clean"], c["thought"] = lab.evidence(c["decoy"], B, alpha)
                rows.append(c)
            s = summarize(rows, ["thought"])
            gap = s.get("thought_cross_flip", 0) - s.get("thought_within_flip", 0)
            key = f"K{K}_a{alpha:g}"
            rep["stage1"][key] = s
            log(f"stage 1 {key}: cross {s.get('thought_cross_flip')}, within {s.get('thought_within_flip')}")
            if best is None or gap > best[0]:
                best = (gap, K, alpha, key)
    _, K, alpha, key = best
    rep["chosen"] = {"K": K, "alpha": alpha, "key": key}

    # ---- stage 2: held-out pairs, frozen K and alpha
    tr_feats = []
    for a, ac, b, bc in tr:
        tr_feats += [(lab.band_bag(ac)[2], 0), (lab.band_bag(bc)[2], 1)]
    rng = random.Random(SEED)
    rows = []
    for c in cases(te):
        c = dict(c)
        B, bags = lab.thought(c["clue"], K, c["decoy"])
        c["clean"], c["thought"] = lab.evidence(c["decoy"], B, alpha)
        c["bag_by_layer"] = {str(l): v for l, v in bags.items()}
        if c["kind"] == "cross":
            _, c["word"] = lab.evidence(c["decoy"], lab.word_dirs(c["source"]), alpha)
            Br, _ = lab.thought(c["clue"], K, c["decoy"], rng=rng)
            _, c["random"] = lab.evidence(c["decoy"], Br, alpha)
            m = classifier(tr_feats, lab.band_bag(c["clue"])[2])
            c["reader_correct"] = float((m > 0) == c["living"])
        rows.append(c)
        log(f"test {len(rows)}/{4 * len(te)} {c['kind']} {c['source']} -> {c['decoy']}")
    t = summarize(rows, ["thought", "word", "random"])
    t["reader_accuracy"] = mean(r["reader_correct"] for r in rows if "reader_correct" in r)
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
    ap.add_argument("--quick", action="store_true", help="K=8, alpha=1, 3 test pairs")
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
    kw = dict(ks=(8,), alphas=(1.0,), pairs=PAIRS[:3]) if args.quick else {}
    rep = run(lm, lens, **kw)
    out = root / "thought_v10"
    out.mkdir(exist_ok=True)
    dest = out / ("quick.json" if args.quick else "full.json")
    tmp = dest.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rep, allow_nan=False))
    tmp.replace(dest)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("rows", "stage1")}, indent=2))


if __name__ == "__main__":
    main()
