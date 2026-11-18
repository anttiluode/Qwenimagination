"""V8: can Qwen use a carried thought to answer a different question? See CATEGORY_V8.md.

Stage 0  pick a question format on the 20 training items, using the PRINTED word.
         Stop unless the best format reaches PASS_TEXT. (V7 failed exactly here.)
Stage 1  pick the write (layer, strength, positions) on the same training items.
Stage 2  test the frozen choice on the 40 held-out V7 items, against controls.

Every decision is calibrated against the neutral "Item: item" prompt with no write,
so a fixed lean toward "Yes" or "A" cannot pass. Positive evidence means "living".
"""
import argparse
import json
import os
import random
from pathlib import Path
from statistics import mean

import torch

from category_transfer import PAIRS, TRAIN, bag, bootstrap, classifier
from qimag import experiment as E, items as I
from qimag.lens import Lens
from qimag.model import load

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path("results/Qwen_Qwen2.5-1.5B-Instruct")
L_READ = 26
LAYERS = (10, 14, 18, 22)
GAINS = (1.0, 2.0, 4.0)
WHERE = ("answer", "question")
PASS_TEXT = 0.85
NEUTRAL = "item"
SEED = 20261118


def _letters():
    q = "Classify the item above. Use A for {} and B for {}. Answer with the letter only."
    return [(q.format("living", "manufactured"), "The letter is", ["A"], ["B"], +1),
            (q.format("manufactured", "living"), "The letter is", ["A"], ["B"], -1)]


def _words(a, b):
    q = "Is the item above {} or {}? Answer with one word."
    pos, neg = [a, a.capitalize()], [b, b.capitalize()]
    return [(q.format(a, b), "The item is", pos, neg, +1),
            (q.format(b, a), "The item is", pos, neg, +1)]


YES, NO = ["Yes", "yes"], ["No", "no"]
# Each probe: (question, assistant prefix, words meaning +, words meaning -, sign).
# Evidence for "living" = sum over probes of sign * (logsumexp(+) - logsumexp(-)).
FORMATS = {
    "yesno": [("Is the item above a living thing? Answer yes or no.", "Answer:", YES, NO, +1),
              ("Is the item above made by people? Answer yes or no.", "Answer:", YES, NO, -1)],
    "living_manufactured": _words("living", "manufactured"),
    "natural_artificial": _words("natural", "artificial"),
    "letters_v7": _letters(),
}
QUESTION_TEXTS = [q for probes in FORMATS.values() for q, prefix, *_ in probes] + \
                 [prefix for probes in FORMATS.values() for _, prefix, *_ in probes] + \
                 ["Item: " + NEUTRAL]
# Words a derived bag must never carry: every answer word of every format.
ANSWER_WORDS = {w.casefold() for probes in FORMATS.values()
                for *_, pos, neg, _ in probes for w in pos + neg}


def resolve(lm, probes):
    """Token ids per probe, or None if an answer word is not one token."""
    out = []
    for q, prefix, pos, neg, sign in probes:
        p = sorted({lm.single_token(w) for w in pos} - {None})
        n = sorted({lm.single_token(w) for w in neg} - {None})
        if not p or not n or set(p) & set(n):
            return None
        out.append((q, prefix, p, n, sign))
    return out


def prompt(lm, word, question, prefix):
    user = f"Item: {word}\n\n{question}"
    chat = lm.chat_text([("user", user)], prefix)
    start = chat.index(user) + len(user) - len(question)
    enc = lm.tok(chat, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    q0 = next(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a <= start < b)
    return ids, q0


class Probe:
    def __init__(self, setup):
        self.S = setup
        self.cache = {}

    def evidence(self, probes, word, write=None):
        """write = (layer, where, vector) or None."""
        lm, total = self.S.lm, 0.0
        for q, prefix, pos, neg, sign in probes:
            ids, q0 = prompt(lm, word, q, prefix)
            layer, delta = 0, None
            if write is not None:
                layer, where, vec = write
                delta = vec.new_zeros((len(ids), vec.numel()))
                delta[[len(ids) - 1] if where == "answer" else list(range(q0, len(ids)))] = vec
            lp = E.run_open(self.S, ids, layer, delta)[-1]
            total += sign * float(torch.logsumexp(lp[pos], 0) - torch.logsumexp(lp[neg], 0))
        return total

    def calibrated(self, probes, word, write=None):
        key = id(probes)
        if key not in self.cache:
            self.cache[key] = self.evidence(probes, NEUTRAL)
        return self.evidence(probes, word, write) - self.cache[key]


def correct(e, living):
    if e == 0:
        return 0.5
    return float((e > 0) == living)


def run(lm, lens, usual, layers=LAYERS, gains=GAINS, l_read=L_READ, pairs=PAIRS,
        train=TRAIN, log=print):
    S = E.Setup(lm, lens)
    probe = Probe(S)
    blocked = [i for i, w in enumerate(lens.words) if w.strip().casefold() in ANSWER_WORDS]
    blocked_set = set(blocked)
    eligible = [i for i in range(len(lens.words)) if i not in blocked_set]

    def extract(desc):
        state = E.hold_read(S, I.HOLD_PROMPT.format(desc=desc))[0][l_read] - usual
        return bag(state, blocked)

    report = {"model": lm.name, "stage0": {}, "stage1": {}, "test": {}}

    # ---- stage 0: format, printed word, training items only
    formats = {name: resolve(lm, p) for name, p in FORMATS.items()}
    for name, probes in formats.items():
        if probes is None:
            report["stage0"][name] = "unavailable (answer word not one token)"
            continue
        acc = mean(correct(probe.calibrated(probes, w), c == 0) for w, _, c in train)
        report["stage0"][name] = acc
        log(f"stage 0 {name}: printed-word accuracy {acc:.2f}")
    scored = [(v, -i, n) for i, (n, v) in enumerate(report["stage0"].items()) if isinstance(v, float)]
    if not scored:
        report["verdict"] = "STOP: no format usable with this tokenizer"
        return report
    best, _, name = max(scored)
    report["format"] = name
    if best < PASS_TEXT:
        report["verdict"] = (f"STOP at gate T0: best printed-word accuracy {best:.2f} < {PASS_TEXT}. "
                             "The question format is the problem; no write was tested.")
        return report
    probes = formats[name]

    # ---- stage 1: the write, training items only
    bags = {w: extract(desc) for w, desc, _ in train}
    configs = []
    for layer in layers:
        for gain in gains:
            for where in WHERE:
                es = [probe.calibrated(probes, NEUTRAL,
                                       (layer, where, lens.write(layer, bags[w][0], gain * bags[w][1])))
                      for w, _, _ in train]
                acc = mean(correct(e, c == 0) for e, (_, _, c) in zip(es, train))
                signed = mean(e if c == 0 else -e for e, (_, _, c) in zip(es, train))
                key = f"L{layer}_g{gain:g}_{where}"
                report["stage1"][key] = {"accuracy": acc, "signed_evidence": signed}
                configs.append((acc, signed, layer, gain, where, key))
                log(f"stage 1 {key}: accuracy {acc:.2f}, signed evidence {signed:+.3f}")
    acc, _, layer, gain, where, key = max(configs)
    report["write"] = {"layer": layer, "gain": gain, "where": where, "key": key, "train_accuracy": acc}

    # ---- stage 2: frozen choice on held-out items
    train_feats = [(bags[w][2], c) for w, _, c in train]
    rng = random.Random(SEED)
    rows = []
    for p, (left, clue_l, right, clue_r) in enumerate(pairs):
        for word, clue, living in ((left, clue_l, True), (right, clue_r, False)):
            idx, values, sparse = extract(clue)
            ridx = torch.tensor(rng.sample(eligible, len(idx)), device=idx.device)
            e = {"none": 0.0,
                 "text": probe.calibrated(probes, word),
                 "derived": probe.calibrated(probes, NEUTRAL,
                                             (layer, where, lens.write(layer, idx, gain * values))),
                 "random": probe.calibrated(probes, NEUTRAL,
                                            (layer, where, lens.write(layer, ridx, gain * values))),
                 "reader": classifier(train_feats, sparse)}
            if word in lens.index:
                vec = lens.write(layer, [lens.index[word]], (gain * values.max()).reshape(1))
                e["word"] = probe.calibrated(probes, NEUTRAL, (layer, where, vec))
            rows.append(dict(pair=p, item=word, living=living, evidence=e,
                             bag=[lens.words[i] for i in idx.tolist()]))
        log(f"test {p + 1}/{len(pairs)}")
    report["rows"] = rows
    report["test"] = summarize(rows)
    return report


def summarize(rows):
    arms = sorted({a for r in rows for a in r["evidence"]})
    per_pair = {}
    for arm in arms:
        groups = {}
        for r in rows:
            if arm in r["evidence"]:
                groups.setdefault(r["pair"], []).append(correct(r["evidence"][arm], r["living"]))
        per_pair[arm] = {p: mean(v) for p, v in groups.items()}
    out = {"accuracy": {a: mean(v.values()) for a, v in per_pair.items()},
           "signed_evidence": {a: mean(r["evidence"][a] * (1 if r["living"] else -1)
                                       for r in rows if a in r["evidence"]) for a in arms}}
    diffs = {}
    for a, b in (("derived", "none"), ("derived", "random"), ("derived", "reader"),
                 ("text", "none"), ("word", "none")):
        if a in per_pair and b in per_pair:
            common = sorted(set(per_pair[a]) & set(per_pair[b]))
            diffs[f"{a}_minus_{b}"] = bootstrap([per_pair[a][p] - per_pair[b][p] for p in common])
    out["paired_accuracy_differences_mean_ci95"] = diffs
    g1 = diffs.get("derived_minus_none", [0, 0, 0])[1] > 0
    g2 = diffs.get("derived_minus_random", [0, 0, 0])[1] > 0
    out["gates"] = {"T1_text_works_on_test": out["accuracy"].get("text", 0) >= PASS_TEXT,
                    "G1_derived_beats_none": g1, "G2_derived_beats_random_write": g2}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, choices=range(1, len(PAIRS) + 1), default=len(PAIRS))
    ap.add_argument("--results", type=Path, default=ROOT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    path = args.results / "delayed_v5" / "v2_lens.pt"
    if not path.is_file():
        ap.error(f"Missing calibration lens: {path}")
    lm = load(MODEL, args.device)
    lens = Lens.load(path, lm.device)
    S = E.Setup(lm, lens)
    usual = torch.stack([E.hold_read(S, I.HOLD_PROMPT.format(desc=d))[0][L_READ]
                         for d, _, _ in I.ITEMS]).mean(0)
    report = run(lm, lens, usual, pairs=PAIRS[:args.max_pairs])
    out = args.results / "category_v8"
    out.mkdir(exist_ok=True)
    dest = out / ("full.json" if args.max_pairs == len(PAIRS) else f"smoke-{args.max_pairs}.json")
    tmp = dest.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(report, allow_nan=False))
    tmp.replace(dest)
    short = {k: v for k, v in report.items() if k not in ("rows", "stage1")}
    print(json.dumps(short, indent=2))


if __name__ == "__main__":
    main()
