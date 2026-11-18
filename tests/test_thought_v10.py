"""V10 plumbing on a random Qwen-shaped model."""
import torch

import category_v8 as V8
import thought_swap_v10 as V
from category_transfer import PAIRS, TRAIN
from qimag import items as I
from qimag.lens import Lens, compute_jlens, reading_stats
from qimag.model import load_stub

EXTRA = V8.QUESTION_TEXTS + ["Yes No"] + [t for p in PAIRS for t in p] + [t for r in TRAIN for t in r[:2]]
torch.manual_seed(0)
LM = load_stub("cpu", extra=EXTRA)
WORDS = list(dict.fromkeys([w for w, _, _ in TRAIN] + [w for p in PAIRS[:3] for w in (p[0], p[2])]
                           + ["dog", "wool", "metal", "tool", "living", "item"]))
WORDS = [w for w in WORDS if LM.single_token(w) is not None]
J = compute_jlens(LM, [LM.single_token(w) for w in WORDS], I.CALIBRATION[:2],
                  chunk=48, batch=16, log=lambda s: None)
LENS = Lens(J, *reading_stats(LM, J, I.CALIBRATION[:2]), WORDS)


def test_span_swap_exchanges_coordinates():
    A = {3: torch.randn(LM.d)}
    B = {3: torch.randn(LM.d)}
    sw = V.SpanSwap(LM, (3, 3), A, B)
    h = torch.randn(1, 6, LM.d)
    c = sw.coords(3, h[0])
    sw.want = {3: c[:, [1, 0]]}
    new = sw._hook(3)(None, None, (h,))[0]
    assert torch.allclose(sw.coords(3, new[0]), c[:, [1, 0]], atol=1e-3)


def test_thought_blocks_answer_words_and_decoy_and_k1_is_a_word_direction():
    lab = V.Lab(LM, LENS, band=(2, 4))
    clue = PAIRS[0][1]
    dirs, bags = lab.thought(clue, 8, "chair")
    for words in bags.values():
        assert len(words) == 8
        assert not {w.casefold() for w in words} & (V8.ANSWER_WORDS | {"item", "chair"})
    d1, b1 = lab.thought(clue, 1, "chair")
    for l in d1:
        j = LENS.J[l][LENS.index[b1[l][0]]]
        assert abs(torch.nn.functional.cosine_similarity(d1[l], j, 0)) > 0.999


def test_alpha_zero_is_no_change():
    lab = V.Lab(LM, LENS, band=(2, 4))
    B, _ = lab.thought(PAIRS[0][1], 8, "chair")
    c, s = lab.evidence("chair", B, 0.0)
    assert abs(c - s) < 1e-4


def test_cases_and_scoring():
    cs = V.cases([("wolf", "w", "chair", "c"), ("lion", "l", "bicycle", "b")])
    assert [(c["kind"], c["source"], c["decoy"]) for c in cs[:4]] == [
        ("cross", "wolf", "chair"), ("cross", "chair", "wolf"),
        ("within", "lion", "wolf"), ("within", "bicycle", "chair")]
    # thought from wolf into chair, answer now "living": left the decoy's category
    assert V.leaves_decoy(dict(kind="cross", living=True, thought=2.0), "thought") == 1.0
    # thought from lion into wolf, answer still living: did not leave
    assert V.leaves_decoy(dict(kind="within", living=True, thought=2.0), "thought") == 0.0


def test_end_to_end():
    r = V.run(LM, LENS, band=(2, 4), ks=(1, 8), alphas=(1.0,), pairs=PAIRS[:3], log=lambda s: None)
    assert r["test_pairs"] == 3 and set(r["stage1"]) == {"K1_a1", "K8_a1"}
    t = r["test"]
    for key in ("thought_cross_flip", "thought_within_flip", "word_cross_flip",
                "random_cross_flip", "reader_accuracy", "gates"):
        assert key in t
    assert len(r["rows"]) == 12
