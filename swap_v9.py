"""V9: the paper's own intervention on Qwen. See SWAP_V9.md.

Stage A  locate Qwen's workspace band with the paper's signatures (§4.1): lens kurtosis,
         persistence of the top concept across tokens, and agreement with the output.
Stage B  pick a band and swap strength on training pairs: "Item: wolf", swap wolf <-> chair
         lens coordinates, clamped at every position across the band (§3.3, §3.4).
Stage C  test on held-out pairs. Cross-category swaps should flip "is it alive?";
         same-category swaps should not.
"""
import argparse
import json
import os
import random
from statistics import mean

import torch

from category_transfer import PAIRS, TRAIN, bootstrap
from category_v8 import FORMATS, ROOT, MODEL, prompt, resolve
from confirm import extend_lens
from qimag import items as I
from qimag.lens import Lens, _hs, _replace, tap_layers
from qimag.model import load

# Candidate bands (inclusive) for a 28-layer model; the paper's band starts about a
# third of the way in and ends shortly before the output.
BANDS = [(4, 12), (8, 16), (12, 20), (16, 24), (20, 27), (8, 20), (10, 24), (4, 24)]
ALPHAS = (1.0, 2.0)
SEED = 20261118


# ---------------------------------------------------------------- stage A
@torch.no_grad()
def signatures(lm, lens, texts):
    cids = torch.tensor([lm.single_token(w) or 0 for w in lens.words], device=lm.device)
    rng = random.Random(SEED)
    out = []
    per = {l: dict(kurt=[], same=[], null=[], rho=[]) for l in range(lm.n_layers)}
    for t in texts:
        x = torch.tensor([lm.encode(t)], device=lm.device)
        with tap_layers(lm, range(lm.n_layers)) as st:
            final = lm.model(input_ids=x).logits[0].float()[:, cids]
            hs = {l: st[l][0].float() for l in range(lm.n_layers)}
        fr = final.argsort(-1).argsort(-1).float()
        for l, h in hs.items():
            z = lens.read(h, l)                                   # [T, C]
            zc = z - z.mean(-1, keepdim=True)
            m2, m4 = (zc ** 2).mean(-1), (zc ** 4).mean(-1)
            per[l]["kurt"].append(float((m4 / m2.clamp_min(1e-9) ** 2 - 3).mean()))
            top = z.argmax(-1).tolist()
            per[l]["same"].append(mean(a == b for a, b in zip(top, top[1:])))
            sh = top[:]
            rng.shuffle(sh)
            per[l]["null"].append(mean(a == b for a, b in zip(sh, sh[1:])))
            zr = z.argsort(-1).argsort(-1).float()
            a, b = zr - zr.mean(-1, keepdim=True), fr - fr.mean(-1, keepdim=True)
            per[l]["rho"].append(float(((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))).mean()))
    for l, v in per.items():
        out.append(dict(layer=l, kurtosis=mean(v["kurt"]),
                        persistence=mean(v["same"]) - mean(v["null"]),
                        output_agreement=mean(v["rho"])))
    return out


# ---------------------------------------------------------------- the swap
class ClampSwap:
    """Clamp the 2-D lens coordinates of (source, target) at every position of every
    band layer to the clean pass's values exchanged with strength alpha, leaving
    everything outside their span untouched."""

    def __init__(self, lm, lens, band, s, t):
        self.lm, self.band = lm, list(range(band[0], band[1] + 1))
        self.V = {l: torch.stack([lens.J[l][s], lens.J[l][t]]).float() for l in self.band}
        self.Gi = {l: torch.linalg.inv(V @ V.T) for l, V in self.V.items()}
        self.want, self.hooks = None, []

    def coords(self, l, h):
        return (h.float() @ self.V[l].T) @ self.Gi[l]               # [T, 2]

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
        loading = float(mean(float(c[l][:, 0].abs().mean()) for l in self.band))
        return clean, swapped, loading

    def _hook(self, l):
        def hook(mod, inp, out):
            h = _hs(out)
            d = (self.want[l] - self.coords(l, h[0])) @ self.V[l]
            return _replace(out, h + d.to(h.dtype)[None])
        return hook


def evidence_pair(lm, lens, probes, word, other, band, alpha):
    """Evidence for "living" on "Item: word", clean and with word<->other swapped."""
    sw = ClampSwap(lm, lens, band, lens.index[word], lens.index[other])
    clean = swapped = 0.0
    loads = []
    for q, prefix, pos, neg, sign in probes:
        ids, _ = prompt(lm, word, q, prefix)
        c, s, load_ = sw.run(ids, alpha)
        f = lambda lp: sign * float(torch.logsumexp(lp[pos], 0) - torch.logsumexp(lp[neg], 0))
        clean, swapped = clean + f(c), swapped + f(s)
        loads.append(load_)
    return clean, swapped, mean(loads)


def score(rows):
    """cross: did the answer move to the partner's category? within: did it leave its own?"""
    def flip(r):
        """1 if the answer no longer names the item's own category."""
        e = r["swapped"]
        return 0.5 if e == 0 else float((e > 0) != r["living"])
    per = {}
    for r in rows:
        per.setdefault(r["kind"], {}).setdefault(r["pair"], []).append(flip(r))
    cross = {p: mean(v) for p, v in per.get("cross", {}).items()}
    within = {p: mean(v) for p, v in per.get("within", {}).items()}
    common = sorted(set(cross) & set(within))
    return dict(cross_flip=mean(cross.values()) if cross else None,
                within_flip=mean(within.values()) if within else None,
                cross_minus_within=bootstrap([cross[p] - within[p] for p in common]) if common else None)


def items_with_lens(lens, pairs):
    return [(a, b) for a, b in pairs if a in lens.index and b in lens.index]


def run(lm, lens, bands=BANDS, alphas=ALPHAS, train=TRAIN, pairs=PAIRS, texts=None, log=print):
    report = {"model": lm.name}
    report["signatures"] = signatures(lm, lens, texts if texts is not None else I.CALIBRATION[:8])
    for s in report["signatures"]:
        log("A L{layer:2d}: kurtosis {kurtosis:6.2f}  persistence {persistence:+.3f}  "
            "output agreement {output_agreement:+.3f}".format(**s))
    probes = resolve(lm, FORMATS["yesno"])
    if probes is None:
        report["verdict"] = "STOP: yes/no tokens are not single tokens"
        return report
    bands = [b for b in bands if b[1] < lm.n_layers]
    if not bands:
        report["verdict"] = f"STOP: no band fits a {lm.n_layers}-layer model"
        return report
    living = [w for w, _, c in train if c == 0]
    made = [w for w, _, c in train if c == 1]
    tr_pairs = items_with_lens(lens, list(zip(living, made)))
    report["train_pairs"] = len(tr_pairs)

    def rows_for(cross_pairs, within, band, alpha):
        """cross_pairs: (living, manufactured). within: (word, other, living, pair)."""
        rows = []
        for p, (a, b) in enumerate(cross_pairs):
            for w, o, liv in ((a, b, True), (b, a, False)):
                c, s, load_ = evidence_pair(lm, lens, probes, w, o, band, alpha)
                rows.append(dict(pair=p, kind="cross", item=w, other=o, living=liv,
                                 clean=c, swapped=s, loading=load_))
        for w, o, liv, p in within:
            c, s, load_ = evidence_pair(lm, lens, probes, w, o, band, alpha)
            rows.append(dict(pair=p, kind="within", item=w, other=o, living=liv,
                             clean=c, swapped=s, loading=load_))
        return rows

    def within_of(cross_pairs):
        """Each item swapped with the same-category item of the next pair."""
        n = len(cross_pairs)
        out = []
        for p, (a, b) in enumerate(cross_pairs):
            nxt = cross_pairs[(p + 1) % n]
            if n > 1:
                out += [(a, nxt[0], True, p), (b, nxt[1], False, p)]
        return out

    best = None
    report["stage_b"] = {}
    for band in bands:
        for alpha in alphas:
            r = score(rows_for(tr_pairs, within_of(tr_pairs), band, alpha))
            key = f"L{band[0]}-{band[1]}_a{alpha:g}"
            report["stage_b"][key] = r
            gap = (r["cross_flip"] or 0) - (r["within_flip"] or 0)
            log(f"B {key}: cross flip {r['cross_flip']}, within flip {r['within_flip']}")
            if best is None or gap > best[0]:
                best = (gap, band, alpha, key)
    _, band, alpha, key = best
    report["chosen"] = {"band": band, "alpha": alpha, "key": key}

    test_cross = items_with_lens(lens, [(a, b) for a, _, b, _ in pairs])
    rows = rows_for(test_cross, within_of(test_cross), band, alpha)
    report["test_pairs"] = len(test_cross)
    report["rows"] = rows
    t = score(rows)
    t["clean_accuracy"] = mean(float((r["clean"] > 0) == r["living"]) for r in rows if r["kind"] == "cross")
    ci = t["cross_minus_within"]
    t["gate_S1_swap_moves_category"] = bool(ci and ci[1] > 0)
    report["test"] = t
    log(f"C {key}: clean {t['clean_accuracy']:.2f}, cross flip {t['cross_flip']:.2f}, "
        f"within flip {t['within_flip']:.2f}, gap CI {ci}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true", help="two bands, one alpha, 4 test pairs")
    args = ap.parse_args()
    root = type(ROOT)(args.results)
    path = root / "delayed_v5" / "v2_lens.pt"
    if not path.is_file():
        ap.error(f"Missing calibration lens: {path}")
    lm = load(MODEL, args.device)
    out = root / "swap_v9"
    out.mkdir(exist_ok=True)
    cached = out / "v9_lens.pt"
    if cached.is_file():
        lens = Lens.load(cached, lm.device)
    else:
        words = [w for w, _, _ in TRAIN] + [w for p in PAIRS for w in (p[0], p[2])]
        lens = extend_lens(lm, Lens.load(path, lm.device), list(dict.fromkeys(words)), print)
        lens.save(cached)
    kw = dict(bands=BANDS[1:3], alphas=(1.0,), pairs=PAIRS[:4]) if args.quick else {}
    report = run(lm, lens, **kw)
    dest = out / ("quick.json" if args.quick else "full.json")
    tmp = dest.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(report, allow_nan=False))
    tmp.replace(dest)
    print(json.dumps({k: v for k, v in report.items() if k in ("chosen", "test", "train_pairs",
                                                              "test_pairs", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
