"""V11 plumbing on a random Qwen-shaped model."""
import torch

import bridge_v11 as V
from tests.test_thought_v10 import LENS, LM
from category_transfer import PAIRS
from thought_swap_v10 import Lab


def test_bridge_uses_late_concepts_with_early_directions():
    lab = Lab(LM, LENS, band=(2, 4))
    idx, values, _ = V.late_bag(lab, PAIRS[0][1], "chair", 6)
    assert LENS.index["chair"] not in idx.tolist()
    dirs = V.bridged(lab, idx[:1], values[:1])
    for l, d in dirs.items():
        j = LENS.J[l][idx[0]]
        assert abs(torch.nn.functional.cosine_similarity(d, j, 0)) > 0.999


def test_end_to_end():
    r = V.run(LM, LENS, band=(2, 4), l_read=6, pairs=PAIRS[:3], log=lambda s: None)
    t = r["test"]
    assert r["test_pairs"] == 3 and len(r["rows"]) == 12
    for key in ("thought_cross_flip", "thought_k8_cross_flip", "word_cross_flip", "random_cross_flip",
                "thought_k8_cross_minus_within", "reader_accuracy", "source_in_late_bag", "gates"):
        assert key in t
