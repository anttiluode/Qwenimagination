"""Diagnostic v2: post hoc, on items already seen. It cannot confirm anything.
Its only job is to find out which of three design faults of the
pre-registered run (v1) mattered, so a fresh pre-registered run on NEW items
can be designed. Reuses the lens of the v1 run.

    python diagnose.py --model Qwen/Qwen2.5-1.5B-Instruct

The three v1 faults it tests:

  A  The question.  v1 asked "what were you thinking of?" with the prefix
     "I was thinking of". Even with the hold in context the model then named
     the subject ("Japan"), not the answer ("Tokyo"): in-context ceiling 0.13.
     v2 asks for the answer: "...the answer you were silently holding?" with
     the prefix "The answer was".
  B  The selection.  v1 held the 8 highest concepts. They were mostly
     "thought, imagine, mind, think, brain, picture" — the same in every hold.
     v2 also holds the 8 concepts that rose most ABOVE THE USUAL HOLD: the
     hold's reading minus the mean reading of all other holds (a slow
     baseline of "what holding a thought usually looks like"), and minus the
     "think of nothing" hold.
  C  The release.  v1 wrote the state at every position. Lens directions are
     "poised to say" directions, so writing them through the paragraph makes
     the model want to say the thought all the time (oracle needed KL 2.4).
     v2 also tries holding the state outside and releasing it only from the
     question onward.

The layers are re-chosen on dev items from an extended band [L/3, L-2],
since in v1 the answer's rank kept improving above the old band's top.
"""
import argparse
import json
import os
import statistics

import torch

from qimag import experiment as E
from qimag import items as I
from qimag.lens import Lens
from qimag.model import load

ASK2 = "\n\nNow: what was the answer you were silently holding earlier? Answer with one word."
PREFIX2 = "The answer was"
GRID = [0.5, 1.0, 2.0, 4.0, 8.0]
CLOSED = [0.1, 0.4]


def ids_and_ask_start(lm, msgs, prefix):
    """Token ids, and the index of the first token of the question."""
    text = lm.chat_text(msgs, prefix)
    cut = text.rfind(ASK2) if ASK2 in text else text.rfind(prefix)
    return lm.encode(text), len(lm.encode(text[:cut]))


def kl_from(lp_ref, lp, start):
    """Mean KL over positions [start, last) and over [0, start)."""
    k = (lp_ref[:-1].exp() * (lp_ref[:-1] - lp[:-1])).sum(-1)
    return k[start:].mean().item(), (k[:start].mean().item() if start > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results", default=None)
    ap.add_argument("--max-items", type=int, default=None)
    a = ap.parse_args()
    tag = a.results or "results/" + a.model.replace("/", "_")
    out_dir = os.path.join(tag, "diagnostic_v2")
    os.makedirs(out_dir, exist_ok=True)
    lm = load(a.model, a.device)
    lens = Lens.load(os.path.join(tag, "lens.pt"), a.device)
    S = E.Setup(lm, lens)
    if a.max_items:
        S.items = S.items[:a.max_items]
        S.finalize()
    N, L = len(S.items), lm.n_layers
    dev = [i for i in range(N) if S.split(i) == "dev"]
    print(f"{N} items, {len(dev)} dev")

    # ---- holds and three ways of choosing what is held ----------------
    holds, null = E.hold_all(S)
    Z = torch.stack([z for z, _ in holds])                       # [N, L, C]
    loo = (Z.sum(0, keepdim=True) - Z) / (N - 1)                  # usual hold, leaving this one out
    sel = {"absolute": Z, "minus_usual": Z - loo, "minus_null": Z - null[0][None]}
    tidx = torch.tensor([lens.index[t] for _, t, _ in S.items])

    def ranks(V):
        own = V[torch.arange(N), :, tidx]                         # [N, L]
        return (V > own[:, :, None]).sum(-1)                      # [N, L]

    R = {k: ranks(v).cpu() for k, v in sel.items()}
    band = range(L // 3, L - 1)
    l_read = max(band, key=lambda l: (sum(int(R["minus_usual"][i, l] < 10) for i in dev),
                                      -statistics.median(R["minus_usual"][dev, l].tolist())))
    l_inj = max(1, l_read - E.INJECT_OFFSET)
    print(f"layers re-chosen on dev (minus_usual): read {l_read}, write {l_inj}")
    sel_table = {k: [sum(int(R[k][i, l] < 10) for i in range(N) if S.split(i) == "test")
                     for l in range(L)] for k in R}

    rows = []
    for i, (desc, tgt, cue) in enumerate(S.items):
        filler = I.FILLERS[i % len(I.FILLERS)]

        def add(cond, g, release, lp, lp_ref, start, energy=0.0):
            r = E.score(S, i, lp[-1])
            ka, kb = kl_from(lp_ref, lp, start) if lp_ref is not None else (float("nan"),) * 2
            r.update(item=i, split=S.split(i), cond=cond, strength=g, release=release,
                     damage_ask=ka, damage_paragraph=kb, energy=energy)
            rows.append(r)

        # A: ceilings with the new question
        d_ids = lm.chat([("user", f"What is {desc}? Answer with one word.")], prefix=PREFIX2.replace("was", "is"))
        add("direct_question", 0, "-", E.run_open(S, d_ids, l_inj, None), None, 0)
        ic_ids, _ = ids_and_ask_start(lm, [("user", I.HOLD_PROMPT.format(desc=desc)),
                                           ("assistant", I.HOLD_REPLY), ("user", filler + ASK2)], PREFIX2)
        add("in_context_v2", 0, "-", E.run_open(S, ic_ids, l_inj, None), None, 0)
        ids, start = ids_and_ask_start(lm, [("user", filler + ASK2)], PREFIX2)
        lp_none = E.run_open(S, ids, l_inj, None)
        add("none_v2", 0, "-", lp_none, lp_none, start)

        # B x C: what is held x when it is released
        states = {"oracle": (torch.tensor([lens.index[tgt]], device=lm.device), None)}
        for k in ("absolute", "minus_usual"):
            v = sel[k][i, l_read]
            K = v.topk(E.K_STATE).indices
            states[k] = (K, v[K])
        ref = states["absolute"][1].max()
        for name, (K, m) in states.items():
            u = ref[None] if m is None else m
            for g in GRID:
                d = lens.write(l_inj, K, g * u)
                for release in ("throughout", "at_question"):
                    delta = d.repeat(len(ids), 1)
                    if release == "at_question":
                        delta[:start] = 0
                    add(name, g, release, E.run_open(S, ids, l_inj, delta), lp_none, start,
                        d.norm().item())
        K, _ = states["minus_usual"]
        m_abs = Z[i, l_read, K].clamp_min(0.1)     # the loop compares absolute reads
        for g in CLOSED:
            lp, e = E.run_closed(S, ids, l_inj, l_read, K, m_abs, g)
            add("minus_usual_closed", g, "throughout", lp, lp_none, start, e)
        held = [lens.words[k] for k in states["minus_usual"][0].tolist()]
        print(f"item {i:2d} {tgt:>9s}  held(minus_usual)={held[:5]}", flush=True)
        for r in rows:
            if r["item"] == i:
                r["held_minus_usual"] = held

    with open(os.path.join(out_dir, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---- summary: test items only, every strength shown (no budget choice)
    test = lambda r: r["split"] == "test"
    nt = sum(1 for i in range(N) if S.split(i) == "test")
    acc = lambda c, **kw: sum(r["correct"] for r in rows if test(r) and r["cond"] == c and
                              all(r[k] == v for k, v in kw.items()))
    lines = [f"Diagnostic v2 on {lm.name}: read {l_read}, write {l_inj}, {nt} test items "
             f"(post hoc, items already seen; nothing here is confirmed)", ""]
    lines.append(f"A  direct question {acc('direct_question')}/{nt}   in-context, new question "
                 f"{acc('in_context_v2')}/{nt}   none, new question {acc('none_v2')}/{nt}")
    lines.append("")
    lines.append("B  answer in the top 10 of the hold reading, test items, by layer:")
    for k, v in sel_table.items():
        lines.append(f"   {k:>12s}: " + " ".join(f"{x:2d}" for x in v))
    lines.append("")
    lines.append("C  correct/{} on test, and mean KL on the question / on the paragraph:".format(nt))
    for name in ("oracle", "absolute", "minus_usual"):
        for release in ("throughout", "at_question"):
            cells = []
            for g in GRID:
                rs = [r for r in rows if test(r) and r["cond"] == name and r["strength"] == g
                      and r["release"] == release]
                cells.append(f"{g:>4}: {sum(r['correct'] for r in rs):2d} "
                             f"({statistics.mean(r['damage_ask'] for r in rs):.2f}/"
                             f"{statistics.mean(r['damage_paragraph'] for r in rs):.2f})")
            lines.append(f"   {name:>12s} {release:>11s}  " + "  ".join(cells))
    for g in CLOSED:
        rs = [r for r in rows if test(r) and r["cond"] == "minus_usual_closed" and r["strength"] == g]
        lines.append(f"   minus_usual closed loop gain {g}: {sum(r['correct'] for r in rs)} "
                     f"({statistics.mean(r['damage_ask'] for r in rs):.2f}/"
                     f"{statistics.mean(r['damage_paragraph'] for r in rs):.2f})")
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(out_dir, "summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
