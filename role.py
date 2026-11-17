"""V4: two derived thoughts in ONE role-bound external state.

See ROLE_PROTOCOL.md. No real-model result exists until role.py is run.
The role binder is engineered, not learned. --max-pairs <30 is smoke only.
"""
import argparse
import json
import math
import os
import random
import re

import torch

from qimag import experiment as E
from qimag import items as I
from qimag.items_v2 import ITEMS_V2
from qimag.items_v3 import ITEMS_V3
from qimag.lens import Lens
from qimag.model import load
from qimag.role_memory import RoleBinder, topk
from confirm import extend_lens
from diagnose import kl_from

L_READ, L_INJ, K = 26, 22, 8
N_PAIRS, SEED = 30, 20261118
PREFIX = "The answer was"
QUESTION = ("\n\nEarlier there were two separate silent answers, one first and one "
            "second. What was the answer held in the {role} slot? "
            "Give the answer itself as one word.")
ARMS = ("none", "bound", "unbound", "wrong_role", "single",
        "cue", "oracle", "in_context")


def recall_prompt(lm, filler, role, source=None):
    ask = QUESTION.format(role=role)
    messages = [("user", filler + ask)]
    if source is not None:
        a, b = source
        messages = [
            ("user", "First silent answer: " + I.HOLD_PROMPT.format(desc=a[0])),
            ("assistant", I.HOLD_REPLY),
            ("user", "Second silent answer: " + I.HOLD_PROMPT.format(desc=b[0])),
            ("assistant", I.HOLD_REPLY),
        ] + messages
    text = lm.chat_text(messages, PREFIX)
    pos = text.rfind(ask)
    if pos < 0:
        raise RuntimeError("no question boundary found")
    return lm.encode(text), len(lm.encode(text[:pos]))


def clean_source(it):
    desc, target, _ = it
    pat = rf"\b{re.escape(target)}\b"
    return (not re.search(pat, desc, re.I) and
            all(not re.search(pat, f, re.I) for f in I.FILLERS))


def known_answer(S, i):
    desc = S.items[i][0]
    ids = S.lm.chat(
        [("user", f"What is {desc}? Answer with one word.")],
        prefix="The answer is")
    return bool(E.score(S, i, E.run_open(S, ids, L_INJ, None)[-1])["correct"])


def choose_pairs(items, count):
    indices = list(range(len(items)))
    random.Random(SEED).shuffle(indices)
    pairs = []
    for a, b in zip(indices[::2], indices[1::2]):
        first, second = items[a], items[b]
        if first[1].lower() == second[1].lower():
            continue
        if re.search(rf"\b{re.escape(first[1])}\b", second[0], re.I):
            continue
        if re.search(rf"\b{re.escape(second[1])}\b", first[0], re.I):
            continue
        pairs.append((a, b))
    return pairs[:count]


def hold_bag(S, usual, item):
    z = E.hold_read(S, I.HOLD_PROMPT.format(desc=item[0]))[0][L_READ]
    residual = (z - usual).clamp_min(0)
    ids = residual.topk(K).indices
    out = torch.zeros(len(S.lens.words), device=S.lm.device)
    out[ids] = residual[ids]
    return out.cpu().tolist()


def normalized_entries(S, values, norm):
    pairs = topk(values, K)
    if not pairs:
        return None
    idx = torch.tensor([i for i,_ in pairs], device=S.lm.device, dtype=torch.long)
    strength = torch.tensor([v for _,v in pairs], device=S.lm.device, dtype=torch.float32)
    return idx, strength * (norm / strength.norm().clamp_min(1e-8))


def release(S, ids, start, selected):
    if selected is None:
        return E.run_open(S, ids, L_INJ, None)
    idx, amount = selected
    delta = S.lens.write(L_INJ, idx, amount).repeat(len(ids), 1)
    delta[:start] = 0
    return E.run_open(S, ids, L_INJ, delta)


def paired_boot(rows, arm, control, n=3000, seed=123):
    """Bootstrap by pair, not by the two correlated role queries."""
    by_pair = {}
    for r in rows:
        if r["arm"] in (arm, control):
            by_pair.setdefault(r["pair"], {}).setdefault(r["arm"], []).append(r["correct"])
    keys = sorted(i for i,v in by_pair.items() if arm in v and control in v)
    d = [sum(by_pair[i][arm])/len(by_pair[i][arm]) -
         sum(by_pair[i][control])/len(by_pair[i][control]) for i in keys]
    if not d:
        return [None, None, None]
    rng = random.Random(seed)
    draws = sorted(sum(d[rng.randrange(len(d))] for _ in d)/len(d) for _ in range(n))
    return [sum(d)/len(d), draws[int(.025*n)], draws[int(.975*n)]]


def summarize(rows, out, model, n_pairs, smoke):
    acc = lambda arm: (sum(r["correct"] for r in rows if r["arm"] == arm) /
                       max(1, sum(r["arm"] == arm for r in rows)))
    other_hits = sum(r["said"] == r["other_target"]
                     for r in rows if r["arm"] == "wrong_role")
    gates = {
        "R1_bound_vs_none": acc("bound") >= acc("none") + .15,
        "R2_bound_vs_unbound": acc("bound") >= acc("unbound") + .08,
        "R3_bound_vs_wrong_role": acc("bound") >= acc("wrong_role") + .10,
        "R4_bound_near_single": acc("bound") >= acc("single") - .10,
        "R5_wrong_role_returns_other": other_hits >= .15*(2*n_pairs),
    }
    summary = {
        "model": model,
        "n_pairs": n_pairs,
        "n_queries": 2*n_pairs,
        "smoke_only": smoke,
        "accuracy": {arm: acc(arm) for arm in ARMS},
        "wrong_role_returned_other_target": other_hits,
        "paired_bootstrap_95pct": {
            f"{a}_vs_{b}": paired_boot(rows,a,b)
            for a,b in (("bound","none"),("bound","unbound"),
                        ("bound","wrong_role"),("bound","single"))
        },
        "gates": gates,
        "all_pass": bool(not smoke and n_pairs >= 20 and all(gates.values())),
        "evidence_limit": ("Fixed engineered role keys; old individual items "
                           "in new pairings; NOT learned relations or generative imagination."),
    }
    with open(os.path.join(out,"summary.json"),"w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results",default=None)
    ap.add_argument("--max-pairs",type=int,default=N_PAIRS,
                    help="1..29 for smoke; 30 for full exploratory test")
    args = ap.parse_args()
    if not 1 <= args.max_pairs <= N_PAIRS:
        raise SystemExit("--max-pairs must be 1..30")
    tag = args.results or "results/" + args.model.replace("/","_")
    out = os.path.join(tag,"roles_v4")
    os.makedirs(out,exist_ok=True)
    lm = load(args.model,args.device)
    if lm.n_layers <= L_READ:
        raise SystemExit("V4 fixed layer 26 requires Qwen2.5-1.5B; not stub")
    base = Lens.load(os.path.join(tag,"lens.pt"),args.device)
    pool = [it for it in ITEMS_V2 + ITEMS_V3
            if clean_source(it) and
            all(lm.single_token(w) is not None for w in it[1:])]
    cache = os.path.join(out,"lens_extended.pt")
    if os.path.exists(cache):
        lens = Lens.load(cache,args.device)
        if not all(w in lens.index for it in pool for w in it[1:]):
            raise SystemExit("Cached lens missing words: delete roles_v4/lens_extended.pt")
    else:
        lens = extend_lens(lm,base,[w for it in pool for w in it[1:]],print)
        lens.save(cache)

    S = E.Setup(lm,lens)
    S.items = pool
    S.finalize()
    known = [it for i,it in enumerate(pool) if known_answer(S,i)]
    S.items = known
    S.finalize()
    pairs = choose_pairs(known,args.max_pairs)
    print(f"{len(pool)} clean token items; {len(known)} known; "
          f"{len(pairs)} disjoint pairs",flush=True)
    if not pairs:
        raise SystemExit("No eligible pairs")

    # Baseline from the original v1 holdings, not from this paired test.
    historical = E.Setup(lm,lens)
    usual = torch.stack([
        E.hold_read(historical,I.HOLD_PROMPT.format(desc=d))[0][L_READ]
        for d,_,_ in I.ITEMS
    ]).mean(0)
    selected = sorted(set(i for pair in pairs for i in pair))
    bags = {i: hold_bag(S,usual,S.items[i]) for i in selected}
    binder = RoleBinder(len(lens.words),seed=SEED)
    rows = []
    for pair_id,(ai,bi) in enumerate(pairs):
        it_a,it_b = S.items[ai], S.items[bi]
        A,B = bags[ai],bags[bi]
        packed = binder.pack(A,B)
        union = [a+b for a,b in zip(A,B)]
        for role,idx,other in (("first",ai,bi),("second",bi,ai)):
            item = S.items[idx]
            target,cue = item[1],item[2]
            other_target = S.items[other][1]
            filler = I.FILLERS[pair_id % len(I.FILLERS)]
            ids,start = recall_prompt(lm,filler,role)
            lp_none = E.run_open(S,ids,L_INJ,None)
            own = A if role == "first" else B
            norm = max(math.sqrt(sum(v*v for v in own)),1e-6)
            sources = {
                "bound": binder.read(packed,role),
                "unbound": union,
                "wrong_role": binder.read(
                    packed,"second" if role == "first" else "first"),
                "single": own,
                "cue": [norm if j==lens.index[cue] else 0.0
                        for j in range(len(lens.words))],
                "oracle": [norm if j==lens.index[target] else 0.0
                           for j in range(len(lens.words))],
            }
            for arm in ARMS:
                held = None
                if arm == "in_context":
                    ic_ids,_ = recall_prompt(
                        lm,filler,role,source=(it_a,it_b))
                    lp = E.run_open(S,ic_ids,L_INJ,None)
                    damage = None
                elif arm == "none":
                    lp,damage = lp_none,0.0
                else:
                    entries = normalized_entries(S,sources[arm],norm)
                    lp = release(S,ids,start,entries)
                    damage = kl_from(lp_none,lp,start)[0]
                    if arm in ("bound","wrong_role","single"):
                        held = [lens.words[i] for i,_ in topk(sources[arm],K)]
                result = E.score(S,idx,lp[-1])
                result.update(
                    pair=pair_id,role=role,item=idx,target=target,cue=cue,
                    other_target=other_target,arm=arm,kl_question=damage,
                    provenance=packed.tags[0 if role=="first" else 1])
                if held is not None:
                    result["held"] = held
                rows.append(result)
            b = next(x for x in reversed(rows) if x["arm"]=="bound")
            print(f"pair {pair_id:02d} {role:6s} {target:12s} "
                  f"read={b['held'][:4]} said={b['said']}",flush=True)

    with open(os.path.join(out,"rows.jsonl"),"w",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row)+"\n")
    summarize(rows,out,lm.name,len(pairs),
              smoke=args.max_pairs<N_PAIRS or len(pairs)<20)


if __name__ == "__main__":
    main()
