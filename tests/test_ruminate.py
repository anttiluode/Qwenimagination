"""Rumination loop plumbing on a random Qwen-shaped model."""
import math

import torch

import ruminate as R
from category_transfer import PAIRS
from tests.test_thought_v10 import LENS, LM
from thought_swap_v10 import Lab

LAB = Lab(LM, LENS, band=(2, 4))
MIND = R.Mind(LAB, l_read=6, placeholder="sheep")


def test_installing_the_placeholder_into_itself_changes_nothing():
    self_ = float(MIND.install(LENS.index["sheep"]).abs().max())
    other = float(MIND.install(LENS.index["hammer"]).abs().max())
    assert self_ < 0.01 * other


def test_loop_never_thinks_a_blocked_word_and_records_tags():
    trace = MIND.run(PAIRS[0][1], steps=3, fatigue=1.0)
    assert len(trace) == 4 and trace[0]["thought"] is None
    for s in trace[1:]:
        assert LENS.index[s["thought"]] not in set(MIND.blocked)
        assert math.isfinite(s["tag"])


def test_full_fatigue_never_repeats_immediately():
    trace = MIND.run(PAIRS[1][1], steps=4, fatigue=1e6)
    thoughts = [s["thought"] for s in trace[1:]]
    assert all(a != b for a, b in zip(thoughts, thoughts[1:]))


def test_check_runs():
    c = R.check(MIND, ["hammer", "pen", "dove", "puppy"])
    assert c["n"] >= 2 and 0 <= c["same"] <= 1
