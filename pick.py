"""v3: picking the thought out of the bag. One run, two stages, all rules
fixed in PICK_PROTOCOL.md before it runs.

    python pick.py --model Qwen/Qwen2.5-1.5B-Instruct

Stage 1 (development, items already seen in v1 and v2): four ways of
scoring which held concepts are "the thought" are compared by recall. The
best one is chosen by a fixed rule.
Stage 2 (confirmation, fresh items in qimag/items_v3.py): the chosen rule is
tested against the v2 rule and the controls.

The four rules, all read at layer 26 and written at layer 22:

  A  usual       hold - usual hold                       (v2: what rose above habit)
  C  inner       (hold - mention) - (usual hold - usual mention)
                 what the mind added beyond what the words in front of it
                 already evoke, against the habits of both
  M  tuft        relu(A) * sigmoid(C)
                 the content of A multiplied by a gate that asks "is this
                 mine, or did the input put it there?"
  D  late        A at layer 26 - A at layer 22: what is still being built late

"mention" is the same description read without being asked to think of it:
  user: Read this phrase: "<description>".   assistant: Okay, I have read it.
"""
import argparse
import json
import os
import random
import re

import torch

from qimag import experiment as E
from qimag import items as I
from qimag.items_v2 import ITEMS_V2
from qimag.items_v3 import ITEMS_V3
from qimag.lens import Lens, tap_layers
from qimag.model import load
from confirm import extend_lens
from diagnose import ASK2, PREFIX2, ids_and_ask_start, kl_from

L_READ, L_INJ, L_EARLY = 26, 22, 22
K, STRENGTH = 8, 1.0
RULES = ["A_usual", "C_inner", "M_tuft", "D_late"]
MENTION_PROMPT = 'Read this phrase: "{desc}".'
MENTION_REPLY = "Okay, I have read it."


@torch.no_grad()
def read_reply(S, prompt, reply, layers):
    lm = S.lm
    ids = lm.chat([("user", prompt)], prefix=reply)
    n0 = len(lm.chat([("user", prompt)], prefix=""))
    with tap_layers(lm, layers) as st:
        lm.final_hidden(torch.tensor([ids], device=lm.device))
        return {l: S.lens.read(st[l][0, n0:].float(), l).mean(0) for l in layers}


def readings(S, items, layers):
    H = [read_reply(S, I.HOLD_PROMPT.format(desc=d), I.HOLD_REPLY, layers) for d, _, _ in items]
    Mn = [read_reply(S, MENTION_PROMPT.format(desc=d), MENTION_REPLY, layers) for d, _, _ in items]
    return H, Mn


def scores(h, m, U, Um):
    """Each rule's score over all concepts, for one item."""
    A = h[L_READ] - U[L_READ]
    C = (h[L_READ] - m[L_READ]) - (U[L_READ] - Um[L_READ])
    Ae = h[L_EARLY] - U[L_EARLY]
    return {"A_usual": A, "C_inner": C, "M_tuft": A.clamp_min(0) * torch.sigmoid(C), "D_late": A - Ae}


def state(sc, A, k=K):
    """Top-k concepts of a rule; amounts scaled so the largest equals A's largest."""
    idx = sc.topk(k).indices
    u = sc[idx].clamp_min(0)
    u = u / u.max().clamp_min(1e-6) * A.max()
    return idx, u


def known_filter(S, li):
    out = []
    for i, (desc, tgt, cue) in enumerate(S.items):
        d_ids = S.lm.chat([("user", f"What is {desc}? Answer with one word.")], prefix="The answer is")
        ok = E.score(S, i, E.run_open(S, d_ids, li, None)[-1])["correct"]
        out.append(bool(ok) or S.lm.name.startswith("stub"))
    return out


def setup_items(S, items):
    S.items = [it for it in items if all(S.lm.single_token(w) is not None for w in it[1:])]
    S.split = lambda i: "test"
    S.finalize()


def release(S, ids, start, idx, u, s=STRENGTH):
    d = S.lens.write(L_INJ, idx, s * u).repeat(len(ids), 1)
    d[:start] = 0
    return E.run_open(S, ids, L_INJ, d)


def recall_ids(S, i, in_context=False):
    desc = S.items[i][0]
    filler = I.FILLERS[i % len(I.FILLERS)]
    msgs = [("user", filler + ASK2)]
    if in_context:
        msgs = [("user", I.HOLD_PROMPT.format(desc=desc)), ("assistant", I.HOLD_REPLY)] + msgs
    return ids_and_ask_start(S.lm, msgs, PREFIX2)


def first_rank(S, i, sc):
    t = S.lens.index[S.items[i][1]]
    return int((sc > sc[t]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results", default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--layers", default=None, help="read,write,early (plumbing tests only)")
    a = ap.parse_args()
    global L_READ, L_INJ, L_EARLY
    if a.layers:
        L_READ, L_INJ, L_EARLY = map(int, a.layers.split(","))
    tag = a.results or "results/" + a.model.replace("/", "_")
    out = os.path.join(tag, "pick_v3")
    os.makedirs(out, exist_ok=True)
    log = lambda s: print(s, flush=True)

    old = {w.lower() for it in I.ITEMS + ITEMS_V2 for w in it[1:]}
    for d, t, c in ITEMS_V3:
        if t.lower() in old or c.lower() in old:
            raise SystemExit(f"v3 item overlaps v1/v2: {t}/{c}")
        if any(re.search(rf"\b{re.escape(t)}\b", f, re.I) for f in I.FILLERS):
            raise SystemExit(f"paragraph contains answer {t!r}")

    lm = load(a.model, a.device)
    lens = Lens.load(os.path.join(tag, "lens.pt"), a.device)
    lens = extend_lens(lm, lens, [w for it in ITEMS_V2 + ITEMS_V3 for w in it[1:]], log)
    S = E.Setup(lm, lens)
    layers = [L_READ, L_EARLY]

    # ---------------- stage 1: development on v1 + v2 items ----------------
    setup_items(S, I.ITEMS + ITEMS_V2)
    if a.max_items:
        S.items = S.items[:a.max_items]
        S.finalize()
    dev_items = list(S.items)
    known = known_filter(S, L_INJ)
    Hd, Md = readings(S, dev_items, layers)
    n = len(dev_items)
    sumU = {l: sum(h[l] for h in Hd) for l in layers}
    sumUm = {l: sum(m[l] for m in Md) for l in layers}
    dev_rows = []
    for i in range(n):
        if not known[i]:
            continue
        U = {l: (sumU[l] - Hd[i][l]) / (n - 1) for l in layers}       # leave this item out
        Um = {l: (sumUm[l] - Md[i][l]) / (n - 1) for l in layers}
        sc = scores(Hd[i], Md[i], U, Um)
        ids, start = recall_ids(S, i)
        for r in RULES:
            idx, u = state(sc[r], sc["A_usual"])
            res = E.score(S, i, release(S, ids, start, idx, u)[-1])
            dev_rows.append(dict(res, item=i, target=dev_items[i][1], rule=r, first=first_rank(S, i, sc[r])))
    nd = sum(known)
    dev = {r: (sum(x["correct"] for x in dev_rows if x["rule"] == r) / nd,
               sum(x["first"] < 3 for x in dev_rows if x["rule"] == r) / nd) for r in RULES}
    chosen = max(RULES, key=lambda r: (round(dev[r][0], 6), round(dev[r][1], 6), -RULES.index(r)))
    log(f"\nstage 1, {nd} development items (seen before):")
    for r in RULES:
        log(f"  {r:>8s}  recall {dev[r][0]:.2f}   answer in its top 3 {dev[r][1]:.2f}")
    log(f"  chosen: {chosen}")

    # ---------------- stage 2: confirmation on fresh v3 items ----------------
    U = {l: sumU[l] / n for l in layers}
    Um = {l: sumUm[l] / n for l in layers}
    setup_items(S, ITEMS_V3)
    if a.max_items:
        S.items = S.items[:a.max_items]
        S.finalize()
    known = known_filter(S, L_INJ)
    log(f"\nstage 2: {len(S.items)} single-token fresh items, {sum(known)} known "
        f"(unknown: {[t for (d, t, c), k in zip(S.items, known) if not k]})")
    Hv, Mv = readings(S, S.items, layers)
    valid = [i for i in range(len(S.items)) if known[i]]
    rows = []
    for nn, i in enumerate(valid):
        desc, tgt, cue = S.items[i]
        sc = scores(Hv[i], Mv[i], U, Um)
        ids, start = recall_ids(S, i)
        lp_none = E.run_open(S, ids, L_INJ, None)

        def add(arm, lp, extra=None):
            r = E.score(S, i, lp[-1])
            same = lp.shape == lp_none.shape
            r.update(item=i, target=tgt, cue=cue, arm=arm,
                     kl_question=kl_from(lp_none, lp, start)[0] if same else float("nan"))
            r.update(extra or {})
            rows.append(r)

        add("none", lp_none)
        ic, _ = recall_ids(S, i, in_context=True)
        add("in_context", E.run_open(S, ic, L_INJ, None))
        for r in RULES:
            idx, u = state(sc[r], sc["A_usual"])
            add(r, release(S, ids, start, idx, u),
                {"held": [lens.words[k] for k in idx.tolist()], "first": first_rank(S, i, sc[r])})
        idx, u = state(sc[chosen], sc["A_usual"], k=1)
        add("chosen_K1", release(S, ids, start, idx, u))
        top = sc["A_usual"].max()[None]
        add("cue", release(S, ids, start, torch.tensor([lens.index[cue]], device=lm.device), top))
        add("oracle", release(S, ids, start, torch.tensor([lens.index[tgt]], device=lm.device), top))
        j = valid[(nn + 1) % len(valid)]
        scj = scores(Hv[j], Mv[j], U, Um)
        idx, u = state(scj[chosen], scj["A_usual"])
        add("shuffled", release(S, ids, start, idx, u), {"state_from": S.items[j][1]})
        pr = {r: next(x for x in reversed(rows) if x["arm"] == r) for r in (chosen, "A_usual")}
        log(f"{tgt:>11s}  {chosen}: {pr[chosen]['held'][:4]} said {pr[chosen]['said']!r}   "
            f"A said {pr['A_usual']['said']!r}")

    with open(os.path.join(out, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"dev": dev, "chosen": chosen, "n_dev": nd}, open(os.path.join(out, "dev.json"), "w"), indent=1)
    score(rows, chosen, out, lm.name)


def boot(rows, a, b, n=5000):
    A = {r["item"]: r["correct"] for r in rows if r["arm"] == a}
    B = {r["item"]: r["correct"] for r in rows if r["arm"] == b}
    it = sorted(A)
    rng = random.Random(0)
    ds = sorted(sum(A[i] - B[i] for i in (rng.choice(it) for _ in it)) / len(it) for _ in range(n))
    return sum(A[i] - B[i] for i in it) / len(it), ds[int(.025 * n)], ds[int(.975 * n)]


def score(rows, chosen, out, name):
    items = sorted({r["item"] for r in rows})
    n = len(items)
    acc = lambda arm: sum(r["correct"] for r in rows if r["arm"] == arm) / n
    first3 = lambda arm: sum(r["first"] < 3 for r in rows if r["arm"] == arm) / n
    first1 = lambda arm: sum(r["first"] == 0 for r in rows if r["arm"] == arm) / n
    lines = [f"=== v3 selection, {name}: chosen rule {chosen}, {n} fresh items ===", ""]
    for arm in ["none", "in_context"] + RULES + ["chosen_K1", "cue", "shuffled", "oracle"]:
        extra = f"   answer first {first1(arm):.2f}  in top 3 {first3(arm):.2f}" if arm in RULES else ""
        lines.append(f"  {arm:>11s}  recall {acc(arm):.2f}{extra}")
    lines.append("")
    d, lo, hi = boot(rows, chosen, "A_usual")
    gates = [
        ("S1 the chosen rule picks better than v2's",
         f"{chosen} {acc(chosen):.2f} vs A_usual {acc('A_usual'):.2f}  (diff {d:+.2f}, CI {lo:+.2f}..{hi:+.2f})",
         chosen != "A_usual" and acc(chosen) >= acc("A_usual") + 0.10),
        ("S2 it still carries this item's thought",
         f"vs none {acc('none'):.2f}, cue {acc('cue'):.2f}, shuffled {acc('shuffled'):.2f}",
         acc(chosen) >= acc("none") + 0.20 and acc(chosen) >= acc("cue") + 0.10
         and acc(chosen) >= acc("shuffled") + 0.20),
        ("S3 it puts the answer first more often",
         f"answer first {first1(chosen):.2f} vs A_usual {first1('A_usual'):.2f}",
         first1(chosen) >= first1("A_usual") + 0.15),
    ]
    for k, v, ok in gates:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {k}: {v}")
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
