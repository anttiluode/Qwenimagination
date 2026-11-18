"""V9 plumbing on a random Qwen-shaped model."""
import torch

import category_v8 as V8
import swap_v9 as V
from category_transfer import PAIRS, TRAIN
from qimag import items as I
from qimag.lens import Lens, compute_jlens, reading_stats
from qimag.model import load_stub

EXTRA = V8.QUESTION_TEXTS + ["Yes No"] + [t for p in PAIRS for t in p] + [t for r in TRAIN for t in r[:2]]
torch.manual_seed(0)
LM = load_stub("cpu", extra=EXTRA)
WORDS = list(dict.fromkeys([w for w, _, _ in TRAIN] + [w for p in PAIRS[:4] for w in (p[0], p[2])]))
WORDS = [w for w in WORDS if LM.single_token(w) is not None]
J = compute_jlens(LM, [LM.single_token(w) for w in WORDS], I.CALIBRATION[:2],
                  chunk=48, batch=16, log=lambda s: None)
LENS = Lens(J, *reading_stats(LM, J, I.CALIBRATION[:2]), WORDS)


def test_clamp_exchanges_span_and_leaves_the_rest():
    s, t = LENS.index["sheep"], LENS.index["hammer"]
    sw = V.ClampSwap(LM, LENS, (3, 3), s, t)
    h = torch.randn(1, 5, LM.d)
    c = sw.coords(3, h[0])
    sw.want = {3: c[:, [1, 0]]}
    new = sw._hook(3)(None, None, (h,))[0]
    assert torch.allclose(sw.coords(3, new[0]), c[:, [1, 0]], atol=1e-4)
    V3 = sw.V[3]
    P = V3.T @ sw.Gi[3] @ V3                       # projector onto the span
    assert torch.allclose((new[0] - h[0]) @ P, new[0] - h[0], atol=1e-4)


def test_alpha_zero_is_no_intervention():
    probes = V8.resolve(LM, V8.FORMATS["yesno"])
    c, s, _ = V.evidence_pair(LM, LENS, probes, "sheep", "hammer", (2, 5), 0.0)
    assert abs(c - s) < 1e-4
    c, s, _ = V.evidence_pair(LM, LENS, probes, "sheep", "hammer", (2, 5), 1.0)
    assert abs(c - s) > 1e-6


def test_signatures_cover_every_layer():
    sig = V.signatures(LM, LENS, I.CALIBRATION[:1])
    assert [s["layer"] for s in sig] == list(range(LM.n_layers))


def test_end_to_end_and_scoring():
    r = V.run(LM, LENS, bands=[(2, 4), (3, 6)], alphas=(1.0,), pairs=PAIRS[:3],
              texts=I.CALIBRATION[:1], log=lambda s: None)
    assert r["train_pairs"] == 10 and len(r["stage_b"]) == 2
    kinds = [x["kind"] for x in r["rows"]]
    assert kinds.count("cross") == kinds.count("within") == 2 * r["test_pairs"]
    rows = [dict(pair=0, kind="cross", living=True, swapped=-1.0),
            dict(pair=0, kind="cross", living=False, swapped=1.0),
            dict(pair=0, kind="within", living=True, swapped=1.0),
            dict(pair=0, kind="within", living=False, swapped=-1.0)]
    s = V.score(rows)
    assert s["cross_flip"] == 1.0 and s["within_flip"] == 0.0
