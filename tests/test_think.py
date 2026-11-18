"""Goal-directed thinking plumbing on a random Qwen-shaped model."""
import think as T
from category_transfer import PAIRS
from tests.test_ruminate import MIND

TH = T.Thinker(MIND)
Q = "Is the item above a living thing?"


def test_goal_policy_picks_qwens_most_yes_candidate_and_never_revisits():
    res = TH.search(PAIRS[0][1], Q, "goal", steps=3, k=4)
    seen = [s["thought"] for s in res["trace"]]
    assert len(seen) == len(set(seen))
    for s in res["trace"][1:]:
        best = max(s["candidates"], key=lambda c: c["verdict"])
        assert best["chosen"] and best["word"] == s["thought"]


def test_association_policy_follows_strongest_association():
    res = TH.search(PAIRS[0][1], Q, "association", steps=3, k=4)
    for s in res["trace"][1:]:
        assert s["candidates"][0]["chosen"]


def test_page_renders():
    runs = [dict(clue="c", question=Q, policies={p: TH.search(PAIRS[1][1], Q, p, steps=2, k=3)
                                                for p in ("goal", "association", "random")})]
    page = T.render(runs)
    assert page.startswith("<!doctype html>") and "Qwen thinking" in page
