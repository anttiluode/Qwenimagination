"""Confirmation v2 on fresh items (qimag/items_v2.py). Everything is frozen
by CONFIRM_PROTOCOL.md before the first run.

    python confirm.py --model Qwen/Qwen2.5-1.5B-Instruct

Needs results/<model>/lens.pt from run.py. Lens rows for the new words are
computed the same way and appended; old rows are reused as they are.
"""
import argparse
import json
import os
import re

import torch

from qimag import experiment as E
from qimag import items as I
from qimag.items_v2 import ITEMS_V2
from qimag.lens import Lens, compute_jlens, reading_stats
from qimag.model import load
from diagnose import ASK2, PREFIX2, ids_and_ask_start, kl_from

L_READ, L_INJ = 26, 22        # chosen on v1 dev items by diagnose.py, frozen
K_MAIN, S_MAIN = 8, 1.0       # primary arm: 8 concepts, strength 1.0, released at the question
CLOSED_GAIN = 0.1


def extend_lens(lm, lens, words, log):
    new = [w for w in words if w not in lens.index and lm.single_token(w) is not None]
    if not new:
        return lens
    log(f"lens: adding {len(new)} concepts")
    Jn = compute_jlens(lm, [lm.single_token(w) for w in new], I.CALIBRATION, log=log)
    mu, sd, _ = reading_stats(lm, Jn, I.CALIBRATION)
    return Lens(torch.cat([lens.J, Jn], 1), torch.cat([lens.mu, mu], 1),
                torch.cat([lens.sd, sd], 1), lens.hmu, lens.words + new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results", default=None)
    ap.add_argument("--l-read", type=int, default=L_READ)
    ap.add_argument("--l-inj", type=int, default=L_INJ)
    ap.add_argument("--max-items", type=int, default=None)
    a = ap.parse_args()
    tag = a.results or "results/" + a.model.replace("/", "_")
    out = os.path.join(tag, "confirm_v2")
    os.makedirs(out, exist_ok=True)
    log = lambda s: print(s, flush=True)
    for _, t, _ in ITEMS_V2:
        for f in I.FILLERS:
            if re.search(rf"\b{re.escape(t)}\b", f, re.I):
                raise SystemExit(f"filler contains answer {t!r}")

    lm = load(a.model, a.device)
    lens = Lens.load(os.path.join(tag, "lens.pt"), a.device)
    lens = extend_lens(lm, lens, [w for _, t, c in ITEMS_V2 for w in (t, c)], log)
    lr, li = a.l_read, a.l_inj

    # the usual hold: a slow baseline learned from the v1 holds (past experience)
    S = E.Setup(lm, lens)
    usual = torch.stack([E.hold_read(S, I.HOLD_PROMPT.format(desc=d))[0][lr]
                         for d, _, _ in I.ITEMS]).mean(0)

    items = [it for it in ITEMS_V2 if all(lm.single_token(w) is not None for w in it[1:])]
    if a.max_items:
        items = items[:a.max_items]
    S.items, S.split = items, (lambda i: "test")
    S.finalize()
    # pre-registered filter: the model must know the answer when asked directly
    known = []
    for i, (desc, tgt, cue) in enumerate(S.items):
        d_ids = lm.chat([("user", f"What is {desc}? Answer with one word.")], prefix="The answer is")
        ok = E.score(S, i, E.run_open(S, d_ids, li, None)[-1])["correct"]
        known.append(ok or lm.name.startswith("stub"))   # random weights know nothing: plumbing only
    log(f"{len(items)} single-token items, {sum(known)} known by the model "
        f"(dropped as unknown: {[t for (d, t, c), k in zip(items, known) if not k]})")

    holds = [E.hold_read(S, I.HOLD_PROMPT.format(desc=d))[0][lr] for d, _, _ in S.items]
    N = len(S.items)
    valid = [i for i in range(N) if known[i]]
    rows = []
    for n, i in enumerate(valid):
        desc, tgt, cue = S.items[i]
        filler = I.FILLERS[i % len(I.FILLERS)]
        ids, start = ids_and_ask_start(lm, [("user", filler + ASK2)], PREFIX2)
        lp_none = E.run_open(S, ids, li, None)

        def add(arm, lp, extra=None):
            r = E.score(S, i, lp[-1])
            same = lp.shape == lp_none.shape     # in_context is a different sequence
            qa, pa = kl_from(lp_none, lp, start) if same else (float("nan"), float("nan"))
            r.update(item=i, target=tgt, cue=cue, arm=arm, kl_question=qa, kl_paragraph=pa)
            r.update(extra or {})
            rows.append(r)

        def release(K, u, s):
            d = lens.write(li, K, s * u).repeat(len(ids), 1)
            d[:start] = 0
            return E.run_open(S, ids, li, d)

        add("none", lp_none)
        ic_ids, _ = ids_and_ask_start(lm, [("user", I.HOLD_PROMPT.format(desc=desc)),
                                           ("assistant", I.HOLD_REPLY), ("user", filler + ASK2)], PREFIX2)
        add("in_context", E.run_open(S, ic_ids, li, None))

        resid = holds[i] - usual
        held = resid.topk(K_MAIN).indices
        add("primary", release(held, resid[held], S_MAIN),
            {"held": [lens.words[k] for k in held.tolist()]})
        for s in (0.5, 2.0):
            add(f"residue_K8_s{s}", release(held, resid[held], s))
        for k in (1, 3):
            Kk = resid.topk(k).indices
            add(f"residue_K{k}", release(Kk, resid[Kk], S_MAIN))
        Ka = holds[i].topk(K_MAIN).indices
        add("absolute", release(Ka, holds[i][Ka], S_MAIN))
        u1 = resid[held].max()[None]
        add("cue", release(torch.tensor([lens.index[cue]], device=lm.device), u1, S_MAIN))
        add("oracle", release(torch.tensor([lens.index[tgt]], device=lm.device), u1, S_MAIN))
        j = valid[(n + 1) % len(valid)]
        rj = holds[j] - usual
        Kj = rj.topk(K_MAIN).indices
        add("shuffled", release(Kj, rj[Kj], S_MAIN), {"state_from": S.items[j][1]})
        lp, _ = E.run_closed(S, ids, li, lr, held, holds[i][held].clamp_min(0.1), CLOSED_GAIN)
        add("closed_loop_throughout", lp)
        prim = next(r for r in reversed(rows) if r["arm"] == "primary")
        log(f"{tgt:>10s}  held={prim['held'][:5]}  primary said {prim['said']!r}")

    with open(os.path.join(out, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    score(rows, len(valid), out, lm.name, lr, li)


def score(rows, n, out, name, lr, li):
    acc = lambda arm: sum(r["correct"] for r in rows if r["arm"] == arm) / n
    own_cue = lambda arm: sum(r["said"] == r["cue"] for r in rows if r["arm"] == arm)
    in_held = sum(r["said"] in r["held"] for r in rows if r["arm"] == "primary")
    p, no, ab, cu, sh, ic = (acc(x) for x in ("primary", "none", "absolute", "cue", "shuffled", "in_context"))
    gates = [
        ("C1 a derived thought carried outside the tokens", f"primary {p:.2f} vs none {no:.2f}", p >= no + 0.20),
        ("C2 the residue, not the loud state", f"primary {p:.2f} vs absolute {ab:.2f}", p >= ab + 0.10),
        ("C3 the thought, not the cue", f"primary {p:.2f} vs cue {cu:.2f}", p >= cu + 0.10),
        ("C4 it is this item's thought", f"shuffled {sh:.2f} (none {no:.2f}), primary-shuffled {p - sh:+.2f}",
         sh <= no + 0.05 and p - sh >= 0.20),
        ("C5 carried beats re-deriving in context", f"primary {p:.2f} vs in-context {ic:.2f}", p >= ic),
    ]
    arms = ["none", "in_context", "primary", "residue_K8_s0.5", "residue_K8_s2.0", "residue_K1",
            "residue_K3", "absolute", "cue", "oracle", "shuffled", "closed_loop_throughout"]
    lines = [f"=== Confirmation v2: {name}, read {lr}, write {li}, {n} fresh items known to the model ===", ""]
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm and r["kl_question"] == r["kl_question"]]
        rs = rs or [{"kl_question": float("nan"), "kl_paragraph": float("nan")}]
        lines.append(f"  {arm:>24s}  recall {acc(arm):.2f}   said own cue {own_cue(arm):2d}   "
                     f"KL question {sum(r['kl_question'] for r in rs) / len(rs):.2f}   "
                     f"KL paragraph {sum(r['kl_paragraph'] for r in rs) / len(rs):.2f}")
    lines.append(f"\n  primary: said a word from its held bag in {in_held}/{n}")
    lines.append("")
    for k, v, ok in gates:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {k}: {v}")
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")
    json.dump({"gates": [{"gate": k, "value": v, "pass": ok} for k, v, ok in gates]},
              open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
