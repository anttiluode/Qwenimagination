"""Rumination: a thought loop that never passes through tokens. See RUMINATE.md.

seed   Qwen silently works out an answer from a clue; read it at layer 26 (V11).
loop   install the current thought into a carrier prompt by the clamped swap across
       layers 4-12 (V9/V11b), let Qwen hold "what it brings to mind", read the change
       at layer 26 against the same carrier without the swap, and take the next thought.
       Fatigue (habituation) on recent thoughts: 0 = rumination, > 0 = wandering.
tag    the strength of the holding-mode words (picture, imagine...) at each step.
check  does a silently installed concept evoke the same associations as the word printed?

Exploratory: nothing here is a pre-registered claim.
"""
import argparse
import json
import os
import random
from statistics import mean

import torch

from bridge_v11 import MODE_WORDS, bridged
from category_transfer import PAIRS
from category_v8 import MODEL, ROOT
from confirm import extend_lens
from qimag import items as I
from qimag.lens import Lens, tap_layers
from qimag.model import load
from thought_swap_v10 import BAND, Lab, SpanSwap

L_READ = 26
PLACEHOLDER = "stone"
CARRIER = ("Item: {word}\n\nSilently think of what the item above brings to mind. "
           "Do not say it or write it down.")
SEEDS = ["the animal whose wool is sheared to make sweaters",
         "the capital city of France",
         "what spins a web and has eight legs",
         "the thing you write with that uses ink"]


class Mind:
    def __init__(self, lab, l_read=L_READ, placeholder=PLACEHOLDER):
        self.lab, self.lm, self.lens, self.l_read = lab, lab.lm, lab.lens, l_read
        self.place = placeholder
        words = [w.strip().casefold() for w in self.lens.words]
        self.mode = [i for i, w in enumerate(words) if w in MODE_WORDS]
        self.blocked = sorted(set(lab.base_block) | set(self.mode) | {self.lens.index[placeholder]})
        self.ids, self.n0 = self._carrier(placeholder)
        self.baseline = self._read(self.ids, self.n0)

    def _carrier(self, word):
        p = CARRIER.format(word=word)
        ids = self.lm.chat([("user", p)], prefix=I.HOLD_REPLY)
        return ids, len(self.lm.chat([("user", p)], prefix=""))

    @torch.no_grad()
    def _read(self, ids, n0, swap=None):
        """Mean layer-l_read lens reads over the hold reply, optionally under a swap."""
        x = torch.tensor([ids], device=self.lm.device)
        hooks = []
        if swap is not None:
            with tap_layers(self.lm, swap.band) as st:
                self.lm.model(input_ids=x)
                c = {l: swap.coords(l, st[l][0]) for l in swap.band}
            swap.want = {l: v[:, [1, 0]] for l, v in c.items()}       # alpha = 1
            hooks = [self.lm.layers[l].register_forward_hook(swap._hook(l)) for l in swap.band]
        try:
            with tap_layers(self.lm, [self.l_read]) as st:
                self.lm.model(input_ids=x)
                h = st[self.l_read][0, n0:].float()
        finally:
            for k in hooks:
                k.remove()
        return self.lens.read(h, self.l_read).mean(0)

    def masked(self, z):
        z = z.clone()
        z[self.blocked] = -torch.inf
        return z

    def seed(self, clue):
        return self.lab.hold(clue)[self.l_read]

    def install(self, concept):
        """Next-thought evidence from installing one concept silently (swap)."""
        idx = torch.tensor([concept], device=self.lm.device)
        B = bridged(self.lab, idx, torch.ones(1, device=self.lm.device), self.place)
        sw = SpanSwap(self.lm, (self.lab.band[0], self.lab.band[-1]),
                      self.lab.word_dirs(self.place), B)
        return self._read(self.ids, self.n0, sw) - self.baseline

    def printed(self, concept):
        """The same, with the word printed in the carrier instead of installed."""
        ids, n0 = self._carrier(self.lens.words[concept].strip())
        return self._read(ids, n0) - self.baseline

    def tag(self, z):
        return float(z[self.mode].mean()) if self.mode else 0.0

    def run(self, clue, steps=10, fatigue=0.0, keep=0.3, top=5):
        state = self.seed(clue)
        tired = torch.zeros_like(state)
        trace = [dict(step=0, thought=None, top=self.top(state, top), tag=self.tag(state))]
        for t in range(1, steps + 1):
            eff = self.masked(state - fatigue * tired)
            c = int(eff.argmax())
            r = self.install(c)
            tired = 0.7 * tired
            tired[c] += float(state[c]) if torch.isfinite(state[c]) else 1.0
            state = keep * state + r
            trace.append(dict(step=t, thought=self.lens.words[c], top=self.top(r, top), tag=self.tag(r)))
        return trace

    def top(self, z, k):
        z = self.masked(z)
        v, i = z.topk(k)
        return [[self.lens.words[j], round(float(x), 2)] for j, x in zip(i.tolist(), v.tolist())]


def overlap(a, b, k=8):
    return len(set(a.topk(k).indices.tolist()) & set(b.topk(k).indices.tolist())) / k


def check(mind, words, seed=0):
    """Does an installed concept evoke what the printed word evokes, more than a random one?"""
    rng = random.Random(seed)
    idx = [mind.lens.index[w] for w in words if w in mind.lens.index and
           mind.lens.index[w] not in set(mind.blocked)]
    rows = []
    for c in idx:
        other = rng.choice([j for j in idx if j != c])
        text = mind.masked(mind.printed(c))
        rows.append(dict(word=mind.lens.words[c],
                         same=overlap(mind.masked(mind.install(c)), text),
                         other=overlap(mind.masked(mind.install(other)), text)))
    return dict(rows=rows, same=mean(r["same"] for r in rows), other=mean(r["other"] for r in rows),
                same_beats_other=sum(r["same"] > r["other"] for r in rows),
                ties=sum(r["same"] == r["other"] for r in rows), n=len(rows))


def show(trace):
    out = []
    for s in trace:
        head = "seed" if s["thought"] is None else f"{s['step']:2d} {s['thought']:>12s} ->"
        out.append(f"  {head} {', '.join(w for w, _ in s['top'])}   (imagined tag {s['tag']:+.2f})")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clue", action="append", help="seed clue(s); default: four examples")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--fatigue", type=float, nargs="*", default=[0.0, 1.0])
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    root = type(ROOT)(args.results)
    lm = load(MODEL, args.device)
    cached = root / "swap_v9" / "v9_lens.pt"
    if not cached.is_file():
        ap.error(f"Missing {cached}; run swap_v9.py once first")
    lens = Lens.load(cached, lm.device)
    if PLACEHOLDER not in lens.index:
        lens = extend_lens(lm, lens, [PLACEHOLDER], print)
    lab = Lab(lm, lens, BAND)
    mind = Mind(lab)
    report = {"model": MODEL, "band": list(BAND), "read_layer": L_READ, "chains": []}
    if not args.no_check:
        words = [w for p in PAIRS for w in (p[0], p[2])]
        report["check"] = check(mind, words)
        c = report["check"]
        print(f"check: installed concept vs printed word, top-8 overlap {c['same']:.2f}; "
              f"a different installed concept {c['other']:.2f}; same > other in "
              f"{c['same_beats_other']}/{c['n']} (ties {c['ties']})")
    for clue in args.clue or SEEDS:
        for f in args.fatigue:
            trace = mind.run(clue, steps=args.steps, fatigue=f)
            report["chains"].append(dict(clue=clue, fatigue=f, trace=trace))
            print(f"\n{clue}  [fatigue {f:g}: {'rumination' if f == 0 else 'wandering'}]")
            print(show(trace))
    out = root / "ruminate"
    out.mkdir(exist_ok=True)
    dest = out / "chains.json"
    tmp = dest.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(report, allow_nan=False))
    tmp.replace(dest)


if __name__ == "__main__":
    main()
