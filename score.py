"""Apply the pre-registered gates of PROTOCOL.md to a results directory.

    python score.py results/Qwen_Qwen2.5-1.5B-Instruct
"""
import json
import math
import os
import random
import sys
from collections import defaultdict

BUDGETS = [0.02, 0.05, 0.1, 0.2, 0.5]   # mean KL per position, nats
MAIN_BUDGET = 0.2
METHODS = ["oracle", "cue", "snapshot", "jstate_open", "jstate_closed",
           "jstate_closed_surprise"]


def load(tag):
    holds = json.load(open(os.path.join(tag, "holds.json")))
    rows = [json.loads(l) for l in open(os.path.join(tag, "recall.jsonl"))]
    return holds, rows


def table(rows):
    """t[cond][strength][split] -> list of rows"""
    t = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        t[r["cond"]][r["strength"]][r["split"]].append(r)
    return t


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def at_budget(t, method, budget):
    """Strength chosen on dev (best dev recall with dev damage <= budget,
    ties to the weaker strength); returns (strength, test rows) or (None, none rows)."""
    best = None
    for g in sorted(t[method]):
        dev = t[method][g]["dev"]
        if not dev or mean([r["damage"] for r in dev]) > budget:
            continue
        acc = mean([r["correct"] for r in dev])
        if best is None or acc > best[0]:
            best = (acc, g)
    if best is None:
        return None, t["none"][0.0]["test"]
    return best[1], t[method][best[1]]["test"]


def boot_diff(a_rows, b_rows, n=2000, seed=0):
    """95% bootstrap CI of mean(a.correct) - mean(b.correct), paired by item."""
    a = {r["item"]: r["correct"] for r in a_rows}
    b = {r["item"]: r["correct"] for r in b_rows}
    items = sorted(set(a) & set(b))
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        s = [rng.choice(items) for _ in items]
        ds.append(mean([a[i] - b[i] for i in s]))
    ds.sort()
    return ds[int(0.025 * n)], ds[int(0.975 * n)]


def score_dir(tag):
    holds, rows = load(tag)
    t = table(rows)
    test_items = [h for h in holds["items"] if h["split"] == "test"]
    out = {"model": holds["model"], "l_read": holds["l_read"], "l_inj": holds["l_inj"],
           "n_test": len(test_items), "n_dev": len(holds["items"]) - len(test_items)}

    hit = mean([h["rank_read"] < 10 for h in test_items])
    spec = mean([h["spec_read"] > 0 for h in test_items])
    null_hit = mean([h["null_rank_read"] < 10 for h in test_items])
    none = mean([r["correct"] for r in t["none"][0.0]["test"]])
    ic = mean([r["correct"] for r in t["in_context"][0.0]["test"]])

    recall = {}
    for m in METHODS:
        recall[m] = {}
        for b in BUDGETS:
            g, rs = at_budget(t, m, b)
            recall[m][b] = {"strength": g, "recall": mean([r["correct"] for r in rs]),
                            "damage": mean([r["damage"] for r in rs]) if g is not None else 0.0}
    rb = lambda m, b=MAIN_BUDGET: recall[m][b]["recall"]
    j_best = "jstate_closed" if rb("jstate_closed") >= rb("jstate_open") else "jstate_open"

    gates = {}
    gates["G0a workspace holds the derived thought"] = {
        "value": f"top-10 hit {hit:.2f} (null hold {null_hit:.2f}), specific {spec:.2f}",
        "rule": "hit >= 0.40 and specific >= 0.80", "pass": hit >= 0.40 and spec >= 0.80}
    gates["G0b the write channel works"] = {
        "value": f"oracle recall {rb('oracle', 0.5):.2f} at budget 0.5",
        "rule": ">= 0.50", "pass": rb("oracle", 0.5) >= 0.50}
    _, a_rows = at_budget(t, j_best, MAIN_BUDGET)
    n_rows = t["none"][0.0]["test"]
    lo, hi = boot_diff(a_rows, n_rows)
    gates["G1 a thought carried outside the tokens"] = {
        "value": f"{j_best} {rb(j_best):.2f} vs none {none:.2f}  (diff CI {lo:+.2f}..{hi:+.2f})",
        "rule": "J-state >= none + 0.20 at budget 0.2", "pass": rb(j_best) >= none + 0.20}
    _, c_rows = at_budget(t, "cue", MAIN_BUDGET)
    lo, hi = boot_diff(a_rows, c_rows)
    gates["G2 the thought, not the cue"] = {
        "value": f"{j_best} {rb(j_best):.2f} vs cue {rb('cue'):.2f}  (diff CI {lo:+.2f}..{hi:+.2f})",
        "rule": "J-state >= cue + 0.10 at budget 0.2", "pass": rb(j_best) >= rb("cue") + 0.10}
    diffs = [rb("jstate_closed", b) - rb("jstate_open", b) for b in BUDGETS]
    gates["G3 the closed loop earns its place"] = {
        "value": "closed - open by budget: " + ", ".join(f"{b}:{d:+.2f}" for b, d in zip(BUDGETS, diffs)),
        "rule": "closed >= open - 0.02 at every budget, and >= open + 0.10 at one",
        "pass": min(diffs) >= -0.02 and max(diffs) >= 0.10}
    g0 = gates["G0a workspace holds the derived thought"]["pass"] and \
        gates["G0b the write channel works"]["pass"]

    out.update(gates=gates, recall_by_budget=recall, none=none, in_context=ic,
               chance=1.0 / holds["n_choice"],
               g0_passed=g0)
    json.dump(out, open(os.path.join(tag, "summary.json"), "w"), indent=1)

    print(f"\n=== {holds['model']}  read layer {holds['l_read']}, write layer {holds['l_inj']}, "
          f"{out['n_test']} test items ===")
    print(f"chance {1.0 / holds['n_choice']:.3f}   none {none:.2f}   in-context ceiling {ic:.2f}")
    print("recall on test, strength chosen on dev, by damage budget (nats/position):")
    print("  " + "".join(f"{b:>9}" for b in BUDGETS) + "   method")
    for m in METHODS:
        print("  " + "".join(f"{recall[m][b]['recall']:9.2f}" for b in BUDGETS) + f"   {m}")
    print()
    for k, g in gates.items():
        print(f"[{'PASS' if g['pass'] else 'FAIL'}] {k}: {g['value']}   ({g['rule']})")
    if not g0:
        print("\nG0 failed: the model's workspace or the write channel is not good enough "
              "for G1-G3 to mean anything. Reported, not interpreted.")
    return out


if __name__ == "__main__":
    score_dir(sys.argv[1])
