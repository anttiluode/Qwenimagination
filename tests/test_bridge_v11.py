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
    dirs = V.bridged(lab, idx[:1], values[:1], "chair")
    for l, d in dirs.items():
        j = LENS.J[l][idx[0]]
        assert abs(torch.nn.functional.cosine_similarity(d, j, 0)) > 0.999
        assert abs(d.norm() - LENS.J[l][LENS.index["chair"]].norm()) < 1e-3


def test_one_concept_thought_swap_equals_the_word_swap():
    """The bug in the first V11 run: same concept, different result."""
    lab = Lab(LM, LENS, band=(2, 4))
    c = LENS.index["sheep"]
    idx, values = torch.tensor([c]), torch.tensor([0.7])
    _, via_thought = lab.evidence("chair", V.bridged(lab, idx, values, "chair"), 1.0)
    _, via_word = lab.evidence("chair", lab.word_dirs("sheep"), 1.0)
    assert abs(via_thought - via_word) < 1e-3


def test_mode_words_are_blocked():
    lab = Lab(LM, LENS, band=(2, 4))
    idx, _, _ = V.late_bag(lab, PAIRS[0][1], "chair", 6)
    assert not {LENS.words[i].casefold() for i in idx.tolist()} & V.MODE_WORDS


def test_end_to_end():
    r = V.run(LM, LENS, band=(2, 4), l_read=6, pairs=PAIRS[:3], log=lambda s: None)
    t = r["test"]
    assert r["test_pairs"] == 3 and len(r["rows"]) == 12
    for key in ("thought_cross_flip", "thought_k8_cross_flip", "word_cross_flip", "random_cross_flip",
                "thought_k8_cross_minus_within", "reader_accuracy", "source_in_late_bag", "gates"):
        assert key in t
