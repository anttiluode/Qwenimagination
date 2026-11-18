"""Check leakage guards and scoring when the required response code flips."""
import torch

from category_transfer import PAIRS, TRAIN, bag, summarize


def test_disjoint_fit_and_category_word_mask():
    held_out = [word for left, _, right, _ in PAIRS for word in (left, right)]
    assert len(set(held_out)) == 40
    assert not set(held_out).intersection(word for word, _, _ in TRAIN)
    assert [sum(c == label for _, _, c in TRAIN) for label in (0, 1)] == [10, 10]
    # Mask the four highest coordinates, including an arbitrarily large score.
    state = torch.arange(12, dtype=torch.float32)
    idx, values, sparse = bag(state, [8, 9, 10, 11])
    assert idx.tolist() == list(range(7, -1, -1))
    assert torch.equal(sparse[idx], values)
    assert torch.count_nonzero(sparse[8:]) == 0


def test_reader_and_two_question_arms_have_equal_item_weight():
    rows = []
    for item in range(2):
        rows.append(dict(pair=0, arm="reader", margin=(-1)**item))
        for order in range(2):
            for arm, margin in (("derived", 1), ("none", (-1)**(item+order)), ("swapped", -1)):
                rows.append(dict(pair=0, arm=arm, mapping=order, margin=margin))
    result = summarize(rows)
    assert result["accuracy"] == {"derived": 1, "none": .5, "reader": .5, "swapped": 0}
    assert result["accuracy_by_letter_mapping"]["none"] == [.5, .5]
