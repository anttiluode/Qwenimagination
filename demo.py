"""See it in words: hold a thought, open a fresh context, and let the model
answer freely (greedy) under each route. Uses the lens and layers of a
finished run.

    python demo.py results/Qwen_Qwen2.5-1.5B-Instruct --model Qwen/Qwen2.5-1.5B-Instruct
    python demo.py results/... --model ... --desc "the colour of a ripe banana"
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from qimag import experiment as E
from qimag import items as I
from qimag.lens import Lens, Injector, tap_layers
from qimag.model import load


@torch.no_grad()
def generate(S, ids, l_inj, l_read, n_new=12, delta=None, K=None, m=None, gamma=None):
    """Greedy continuation. Open loop if `delta` is given, closed loop if `gamma` is."""
    lm, lens = S.lm, S.lens
    inj = Injector(lm, l_inj)
    inj.delta = delta
    a = None if m is None else torch.zeros_like(m)
    cache, out, seq = None, [], list(ids)
    try:
        with tap_layers(lm, [l_read]) as st:
            q = 0
            while q < len(seq) and len(out) < n_new:
                x = torch.tensor([[seq[q]]], device=lm.device)
                o = lm.model(input_ids=x, past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                if gamma is not None:
                    z = lens.read(st[l_read][0, -1], l_read, K)
                    a = torch.minimum(torch.clamp(a + gamma * (m - z), min=0.0), E.A_MAX * m)
                    inj.delta = lens.write(l_inj, K, a)
                if q == len(seq) - 1:
                    nxt = o.logits[0, -1].argmax().item()
                    out.append(nxt)
                    seq.append(nxt)
                    if nxt in (getattr(lm.tok, "eos_token_id", None), lm.encode("<|im_end|>")[0]):
                        break
                q += 1
    finally:
        inj.remove()
    return lm.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--desc", default=None, help="your own thing to think of")
    ap.add_argument("--n", type=int, default=6, help="number of test items to show")
    a = ap.parse_args()
    lm = load(a.model, a.device)
    lens = Lens.load(os.path.join(a.results, "lens.pt"), a.device)
    summ = json.load(open(os.path.join(a.results, "summary.json")))
    l_read, l_inj = summ["l_read"], summ["l_inj"]
    rb = summ["recall_by_budget"]
    pick = lambda m: rb[m][str(E_BUDGET)]["strength"]
    S = E.Setup(lm, lens)
    descs = [a.desc] if a.desc else [S.items[i][0] for i in range(len(S.items)) if S.split(i) == "test"][:a.n]
    for desc in descs:
        Z, H = E.hold_read(S, I.HOLD_PROMPT.format(desc=desc))
        zr = Z[l_read]
        K = zr.topk(E.K_STATE).indices
        m = zr[K]
        print(f"\n### think of: {desc}")
        print("held J-state:", ", ".join(f"{lens.words[k]}({v:.1f})" for k, v in zip(K.tolist(), m.tolist())))
        ids = lm.chat([("user", I.FILLERS[0] + I.ASK)], prefix=I.ANSWER_PREFIX)
        print(f"  none          : {I.ANSWER_PREFIX}{generate(S, ids, l_inj, l_read)}")
        g = pick("jstate_open")
        if g is not None:
            d = lens.write(l_inj, K, g * m)
            print(f"  jstate_open   : {I.ANSWER_PREFIX}{generate(S, ids, l_inj, l_read, delta=d)}   (strength {g})")
        g = pick("jstate_closed")
        if g is not None:
            print(f"  jstate_closed : {I.ANSWER_PREFIX}{generate(S, ids, l_inj, l_read, K=K, m=m, gamma=g)}   (gain {g})")


E_BUDGET = 0.2

if __name__ == "__main__":
    main()
