"""V8 plumbing on a random Qwen-shaped model. Checks the machinery, not learning."""
import torch

import category_v8 as V
from category_transfer import PAIRS, TRAIN
from qimag import experiment as E, items as I
from qimag.lens import Lens, compute_jlens, reading_stats
from qimag.model import load_stub

EXTRA = V.QUESTION_TEXTS + ["Yes No Living Manufactured Natural Artificial"] + \
        [t for p in PAIRS for t in p] + [t for row in TRAIN for t in row[:2]]
torch.manual_seed(0)
LM = load_stub("cpu", extra=EXTRA)
WORDS = list(dict.fromkeys([w for w, _, _ in TRAIN] + [p[0] for p in PAIRS[:2]] +
                           ["living", "yes", "A", "dog", "wool", "metal", "tree", "tool"]))
WORDS = [w for w in WORDS if LM.single_token(w) is not None]
J = compute_jlens(LM, [LM.single_token(w) for w in WORDS], I.CALIBRATION[:2],
                  chunk=48, batch=16, log=lambda s: None)
LENS = Lens(J, *reading_stats(LM, J, I.CALIBRATION[:2]), WORDS)
S = E.Setup(LM, LENS)
USUAL = torch.stack([E.hold_read(S, I.HOLD_PROMPT.format(desc=d))[0][6]
                     for d, _, _ in I.ITEMS[:4]]).mean(0)


def test_all_formats_resolve_and_answer_words_are_blocked():
    for name, probes in V.FORMATS.items():
        assert V.resolve(LM, probes) is not None, name
    assert {"living", "manufactured", "yes", "no", "a", "b"} <= V.ANSWER_WORDS


def test_neutral_prompt_calibrates_to_zero_and_zero_write_is_no_write():
    probe = V.Probe(S)
    probes = V.resolve(LM, V.FORMATS["yesno"])
    assert probe.calibrated(probes, V.NEUTRAL) == 0
    zero = torch.zeros(LM.d)
    for where in V.WHERE:
        assert abs(probe.calibrated(probes, V.NEUTRAL, (3, where, zero))) < 1e-5


def test_answer_write_touches_only_the_last_position():
    probes = V.resolve(LM, V.FORMATS["yesno"])
    q, prefix, *_ = probes[0]
    ids, q0 = V.prompt(LM, V.NEUTRAL, q, prefix)
    assert 0 < q0 < len(ids) - 1
    vec = torch.randn(LM.d)
    delta = vec.new_zeros((len(ids), LM.d))
    delta[len(ids) - 1] = vec
    a = E.run_open(S, ids, 3, None)
    b = E.run_open(S, ids, 3, delta)
    assert torch.allclose(a[:-1], b[:-1], atol=1e-5)
    assert not torch.allclose(a[-1], b[-1])


def test_stops_when_printed_word_fails(monkeypatch):
    monkeypatch.setattr(V, "PASS_TEXT", 1.01)
    r = V.run(LM, LENS, USUAL, layers=(3,), gains=(1.0,), l_read=6, pairs=PAIRS[:1], log=lambda s: None)
    assert r["verdict"].startswith("STOP at gate T0") and not r["stage1"]


def test_full_path_runs_end_to_end(monkeypatch):
    monkeypatch.setattr(V, "PASS_TEXT", 0.0)
    r = V.run(LM, LENS, USUAL, layers=(2, 4), gains=(1.0,), l_read=6, pairs=PAIRS[:2], log=lambda s: None)
    assert len(r["stage1"]) == 4 and r["write"]["layer"] in (2, 4)
    t = r["test"]
    assert t["accuracy"]["none"] == 0.5
    assert {"text", "derived", "random", "reader"} <= set(t["accuracy"])
    assert len(r["rows"]) == 4
    for row in r["rows"]:
        assert not V.ANSWER_WORDS & {w.strip().casefold() for w in row["bag"]}


def test_summary_gates():
    rows = []
    for p in range(6):
        for living in (True, False):
            s = 1 if living else -1
            rows.append(dict(pair=p, living=living,
                             evidence={"none": 0.0, "text": s, "derived": s, "random": -s, "reader": s}))
    t = V.summarize(rows)
    assert t["accuracy"] == {"derived": 1, "none": .5, "random": 0, "reader": 1, "text": 1}
    assert all(t["gates"].values())
