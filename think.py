"""Goal-directed thinking without tokens. See THINK.md.

Start from a silently derived thought. At each step:
  1. install the current thought (swap across layers 4-12) and read what it brings to mind
     at layer 26: these associations are the candidate next thoughts;
  2. install each candidate silently into a goal question ("Can the item above fly?")
     and take Qwen's yes-minus-no;
  3. move to the candidate Qwen answers most "yes" to (never revisiting a thought).
Stop when Qwen answers yes. Verify the found thought with the word printed instead.
Controls take the same steps but choose by strongest association, or at random.

Writes results/.../think/think.json and think.html (a page showing every step).
Exploratory.
"""
import argparse
import html
import json
import os
import random

import torch

from category_v8 import MODEL, ROOT, prompt
from confirm import extend_lens
from qimag.lens import Lens
from qimag.model import load
from ruminate import Mind, PLACEHOLDER
from thought_swap_v10 import BAND, Lab, SpanSwap
from bridge_v11 import bridged

GOALS = [("the animal whose wool is sheared to make sweaters", "Can the item above fly?"),
         ("the thing you write with that uses ink", "Does the item above live in water?"),
         ("the capital city of France", "Is the item above something you can eat?"),
         ("what spins a web and has eight legs", "Is the item above bigger than a car?")]
YES, NO = ["Yes", "yes"], ["No", "no"]
SUFFIX = " Answer yes or no."


class Thinker:
    def __init__(self, mind):
        self.m, self.lm, self.lens = mind, mind.lm, mind.lens
        ids = lambda ws: sorted({self.lm.single_token(w) for w in ws} - {None})
        self.yes, self.no = ids(YES), ids(NO)

    def _yn(self, logits):
        return float(torch.logsumexp(logits[self.yes], 0) - torch.logsumexp(logits[self.no], 0))

    @torch.no_grad()
    def judge(self, concept, question):
        """Qwen's yes-minus-no with the concept installed silently in the slot."""
        ids, _ = prompt(self.lm, self.m.place, question + SUFFIX, "Answer:")
        idx = torch.tensor([concept], device=self.lm.device)
        B = bridged(self.m.lab, idx, torch.ones(1, device=self.lm.device), self.m.place)
        sw = SpanSwap(self.lm, (self.m.lab.band[0], self.m.lab.band[-1]),
                      self.m.lab.word_dirs(self.m.place), B)
        _, swapped = sw.run(ids, 1.0)
        return self._yn(swapped)

    @torch.no_grad()
    def judge_printed(self, word, question):
        ids, _ = prompt(self.lm, word, question + SUFFIX, "Answer:")
        return self._yn(self.lm.model(input_ids=torch.tensor([ids], device=self.lm.device)).logits[0, -1].float())

    def search(self, clue, question, policy, steps=8, k=6, seed=0):
        rng = random.Random(seed)
        start = int(self.m.masked(self.m.seed(clue)).argmax())
        visited, current, trace = {start}, start, []
        first = self.judge(start, question)
        trace.append(dict(step=0, thought=self.lens.words[start], verdict=first, candidates=[]))
        if first > 0:
            return self._finish(trace, question)
        for t in range(1, steps + 1):
            assoc = self.m.masked(self.m.install(current))
            order = assoc.argsort(descending=True).tolist()
            cands = [c for c in order if c not in visited and torch.isfinite(assoc[c])][:k]
            if not cands:
                break
            scored = [(c, self.judge(c, question), float(assoc[c])) for c in cands]
            if policy == "goal":
                nxt = max(scored, key=lambda x: x[1])
            elif policy == "association":
                nxt = scored[0]
            else:
                nxt = rng.choice(scored)
            visited.add(nxt[0])
            current = nxt[0]
            trace.append(dict(step=t, thought=self.lens.words[current], verdict=nxt[1],
                              candidates=[dict(word=self.lens.words[c], verdict=v, association=a,
                                               chosen=c == current) for c, v, a in scored]))
            if nxt[1] > 0:
                break
        return self._finish(trace, question)

    def _finish(self, trace, question):
        last = trace[-1]
        found = last["verdict"] > 0
        out = dict(found=found, steps=last["step"], final=last["thought"], trace=trace)
        if found:
            out["printed_verdict"] = self.judge_printed(last["thought"].strip(), question)
        return out


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen thinking</title><style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#77756f;--card:#fff;--line:#e4e1da;--yes:#2f7d4f;--no:#b4533a;--pick:#1d1d1b}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--mut:#9a978f;--card:#1e1e1c;--line:#2e2d2a;--yes:#6cc28e;--no:#e08a70;--pick:#ecebe6}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1000px;margin:auto}
h1{font-size:22px;margin:0 0 4px}p.m{color:var(--mut);margin:0 0 24px}
.run{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 20px}
.q{font-weight:600}.clue{color:var(--mut);font-size:13px}.pol{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin-top:10px}
.chain{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:8px 0}
.t{padding:3px 9px;border-radius:999px;border:1px solid var(--line)}.t.y{border-color:var(--yes);color:var(--yes);font-weight:600}
.a{color:var(--mut)}.step{margin:6px 0 0 0;font-size:13px}.step b{display:inline-block;min-width:90px}
.c{display:inline-flex;align-items:center;gap:4px;margin:2px 8px 2px 0}.bar{height:8px;border-radius:2px}
.c.on{font-weight:700}.res{font-size:13px;margin-top:6px}
</style></head><body><h1>Qwen thinking without tokens</h1>
<p class="m">Each thought is installed silently (layers 4–12) and never written. Candidates are what it brings to mind (layer 26); bars are Qwen's yes-minus-no to the goal with that candidate installed (green = yes). Bold = chosen.</p>
%s</body></html>"""


def render(runs):
    blocks = []
    for r in runs:
        parts = [f'<div class="run"><div class="q">{html.escape(r["question"])}</div>'
                 f'<div class="clue">seed: {html.escape(r["clue"])}</div>']
        for pol, res in r["policies"].items():
            chain = ' <span class="a">→</span> '.join(
                f'<span class="t{" y" if s["verdict"] > 0 else ""}">{html.escape(s["thought"])}</span>'
                for s in res["trace"])
            parts.append(f'<div class="pol">{html.escape(pol)}</div><div class="chain">{chain}</div>')
            if pol == "goal":
                for s in res["trace"][1:]:
                    cs = "".join(
                        f'<span class="c{" on" if c["chosen"] else ""}">{html.escape(c["word"])}'
                        f'<span class="bar" style="width:{min(60, 4 * abs(c["verdict"])) + 2:.0f}px;'
                        f'background:var(--{"yes" if c["verdict"] > 0 else "no"})"></span></span>'
                        for c in s["candidates"])
                    parts.append(f'<div class="step"><b>step {s["step"]}</b>{cs}</div>')
            res_txt = (f'found <b>{html.escape(res["final"])}</b> in {res["steps"]} steps; '
                       f'printed word check: {"yes" if res.get("printed_verdict", -1) > 0 else "no"}'
                       if res["found"] else f'not found in {res["steps"]} steps')
            parts.append(f'<div class="res">{res_txt}</div>')
        parts.append("</div>")
        blocks.append("".join(parts))
    return PAGE % "\n".join(blocks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clue", help="seed clue (with --question)")
    ap.add_argument("--question", help='goal, e.g. "Can the item above fly?"')
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()
    root = type(ROOT)(args.results)
    lm = load(MODEL, args.device)
    cached = root / "swap_v9" / "v9_lens.pt"
    if not cached.is_file():
        ap.error(f"Missing {cached}; run swap_v9.py once first")
    lens = Lens.load(cached, lm.device)
    if PLACEHOLDER not in lens.index:
        lens = extend_lens(lm, lens, [PLACEHOLDER], print)
    th = Thinker(Mind(Lab(lm, lens, BAND)))
    goals = [(args.clue, args.question)] if args.clue and args.question else GOALS
    runs = []
    for clue, q in goals:
        r = dict(clue=clue, question=q, policies={})
        print(f"\n{q}   (seed: {clue})")
        for pol in ("goal", "association", "random"):
            res = th.search(clue, q, pol, steps=args.steps)
            r["policies"][pol] = res
            chain = " -> ".join(f'{s["thought"]}{"(yes)" if s["verdict"] > 0 else ""}' for s in res["trace"])
            tail = (f'FOUND {res["final"]} in {res["steps"]}, printed check '
                    f'{"yes" if res.get("printed_verdict", -1) > 0 else "no"}') if res["found"] else "not found"
            print(f"  {pol:11s} {chain}   [{tail}]")
        runs.append(r)
    out = root / "think"
    out.mkdir(exist_ok=True)
    for name, text in (("think.json", json.dumps(runs, allow_nan=False)), ("think.html", render(runs))):
        tmp = out / f"{name}.{os.getpid()}.tmp"
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(out / name)
    print(f"\nopen {out / 'think.html'} to watch it think")


if __name__ == "__main__":
    main()
