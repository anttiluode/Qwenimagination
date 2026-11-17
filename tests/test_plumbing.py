"""Plumbing tests on a random Qwen-shaped model. They check the machinery is
exact, not that anything is learned.

    python -m pytest -q tests
"""
import torch
import torch.nn.functional as F

from qimag import experiment as E
from qimag import items as I
from qimag.lens import Lens, compute_jlens, reading_stats, tap_layers
from qimag.model import load_stub

torch.manual_seed(0)
LM = load_stub("cpu")
WORDS = [w for w in dict.fromkeys([t for _, t, _ in I.ITEMS] + [c for _, _, c in I.ITEMS]
                                  + I.DECOYS[:30]) if LM.single_token(w) is not None]
CIDS = [LM.single_token(w) for w in WORDS]
J = compute_jlens(LM, CIDS, I.CALIBRATION[:3], chunk=48, batch=16, log=lambda s: None)
LENS = Lens(J, *reading_stats(LM, J, I.CALIBRATION[:3]), WORDS)
S = E.Setup(LM, LENS)


def test_lens_matches_finite_difference():
    """j[l,t] must equal the average gradient, checked by brute force on one text."""
    ids = LM.encode(I.CALIBRATION[0])[:20]
    Jone = compute_jlens(LM, CIDS[:3], [LM.decode(ids)], chunk=64, batch=3, log=lambda s: None)
    x = torch.tensor([LM.encode(LM.decode(ids))])
    l, t, eps = 3, 1, 1e-3
    W = LM.unembed[CIDS[t]].double()

    def total(delta_at):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            h = h.clone()
            q, v = delta_at
            h[0, q] += v
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        hh = LM.layers[l].register_forward_hook(hook)
        with torch.no_grad():
            z = (LM.final_hidden(x)[0].double() @ W).sum()
        hh.remove()
        return z

    T = x.shape[1]
    g = torch.zeros(LM.d, dtype=torch.float64)
    for q in range(T):
        for k in range(LM.d):
            e = torch.zeros(LM.d)
            e[k] = eps
            g[k] += (total((q, e)) - total((q, -e))) / (2 * eps)
    g /= T
    assert torch.allclose(Jone[l, t].double(), g, rtol=2e-2, atol=2e-4)


def test_write_moves_read_exactly():
    l, K = 4, torch.tensor([0, 3, 5])
    h = torch.randn(LM.d)
    u = torch.tensor([1.0, -2.0, 0.5])
    z0 = LENS.read(h, l, K)
    z1 = LENS.read(h + LENS.write(l, K, u), l, K)
    assert torch.allclose(z1 - z0, u, atol=1e-3)


def test_closed_loop_with_zero_gain_equals_no_injection():
    ids = E.recall_ids(S, 0)
    lp_none = E.run_open(S, ids, 2, None)
    K = torch.tensor([0, 1])
    lp, energy = E.run_closed(S, ids, 2, 6, K, torch.tensor([1.0, 1.0]), 0.0)
    assert energy == 0.0
    assert torch.allclose(lp, lp_none, atol=1e-4)
    assert E.kl(lp_none, lp) < 1e-5


def test_constant_drive_equals_open_loop():
    """A closed loop whose drive is pinned at a_max from the first token on
    must equal the open loop writing the same vector at every later position."""
    ids = E.recall_ids(S, 1)
    K = torch.tensor([2, 4])
    m = torch.tensor([100.0, 200.0])   # far above any reading, so the deficit is positive
    old = E.A_MAX
    E.A_MAX = 1.0
    try:
        lp_c, _ = E.run_closed(S, ids, 2, 6, K, m, 1e6)   # saturates at a = m after token 0
    finally:
        E.A_MAX = old
    d = LENS.write(2, K, m)
    delta = d.repeat(len(ids), 1)
    delta[0] = 0
    lp_o = E.run_open(S, ids, 2, delta)
    assert torch.allclose(lp_c, lp_o, atol=1e-4)


def test_recall_context_has_no_hold():
    for i, (desc, tgt, cue) in enumerate(S.items[:6]):
        text = LM.decode(E.recall_ids(S, i))
        assert desc not in text and f" {tgt} " not in text
        assert desc in LM.decode(E.recall_ids(S, i, in_context=True))


def test_full_item_runs():
    holds, null = E.hold_all(S)
    rank, spec = E.g0_table(S, holds)
    l_read, l_inj = E.choose_layers(S, rank)
    rows, held = E.recall_item(S, 0, holds, l_read, l_inj)
    conds = {r["cond"] for r in rows}
    assert conds == {"none", "in_context", "oracle", "cue", "jstate_open", "snapshot",
                     "jstate_closed", "jstate_closed_surprise"}
    assert len(held) == E.K_STATE
    assert all(r["damage"] >= -1e-6 for r in rows if r["cond"] != "in_context")
