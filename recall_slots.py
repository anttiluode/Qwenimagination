"""V5 delayed retrieval. See DELAYED_RECALL.md for the fixed experiment."""
import argparse
import copy
import json
import os
import random
import re
from pathlib import Path
from statistics import mean

import torch

from confirm import extend_lens
from qimag import experiment as E, items as I
from qimag.items_v2 import ITEMS_V2
from qimag.lens import Injector, Lens
from qimag.model import load
from role import L_INJ, L_READ, K

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path("results/Qwen_Qwen2.5-1.5B-Instruct")
LABELS = ("Alpha", "Beta")
SEED = 20261118


def prompt_parts(lm, order, filler, words=None):
    """Locate writes and the first question word in the fully tokenized chat."""
    prefix = "Remember the word stored under each label."
    marker_ends = []
    for label in order:
        prefix += "\n" + label + ":"
        if words is None:
            marker_ends.append(len(prefix))
        else:
            prefix += " " + words[label]
    prefix += "\n\n" + filler
    base = None
    questions = {}
    for label in LABELS:
        user = prefix + "\n\nWhich word was stored under " + label + "? Answer with one word."
        rendered = lm.chat_text([("user", user)], "The answer was")
        before, found, _ = rendered.partition(user)
        if not found:
            raise ValueError("Chat template changed the user text")
        tokens = lm.tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        offsets = tokens["offset_mapping"]
        full = tokens["input_ids"]

        def token_at(char):
            for index, (start, end) in enumerate(offsets):
                if start <= char < end:
                    return index
            raise ValueError(f"No token covers prompt character {char}")

        # Punctuation may share a token with following newlines. Cache all
        # complete tokens before "Which" and write on the token containing ":".
        split = token_at(len(before) + len(prefix) + len("\n\n"))
        positions = [token_at(len(before) + end - 1) for end in marker_ends]
        if base is None:
            base, markers = full[:split], positions
        elif base != full[:split] or markers != positions:
            raise ValueError("Questions produced different pre-question tokens or markers")
        questions[label] = full[split:]
    return base, markers, questions


@torch.no_grad()
def prefix_cache(lm, ids, positions=(), writes=(), layer=L_INJ):
    hook = Injector(lm, layer)
    try:
        if len(positions) != len(writes):
            raise ValueError("One vector per marker is required")
        if len(writes):
            hook.delta = writes[0].new_zeros((len(ids), lm.d))
            hook.delta[list(positions)] = torch.stack(writes)
        output = lm.model.model(input_ids=torch.tensor([ids], device=lm.device), use_cache=True)
        return output.past_key_values
    finally:
        hook.remove()


@torch.no_grad()
def query(lm, cache, suffix):
    hidden = lm.final_hidden(torch.tensor([suffix], device=lm.device),
                             past_key_values=copy.deepcopy(cache), use_cache=True)
    return lm.model.get_output_embeddings()(hidden[0, -1]).float()


def eligible(lm, root):
    # Only the pre-existing knowledge filter supplies item identity.
    # No accuracy, ranking or held-concept fields enter selection.
    path = root / "confirm_v2" / "rows.jsonl"
    known = {row["target"] for line in path.read_text().splitlines()
             if (row := json.loads(line))["arm"] == "none"}
    items = [it for it in ITEMS_V2 if it[1] in known]
    random.Random(SEED).shuffle(items)
    pairs = []
    for a, b in zip(items[::2], items[1::2]):
        filler = I.FILLERS[len(pairs) % len(I.FILLERS)]
        base, _, tails = prompt_parts(lm, LABELS, filler)
        public = " ".join(lm.decode(base + tail) for tail in tails.values())
        if a[1] != b[1] and all(not re.search(r"\b" + re.escape(it[1]) + r"\b", public, re.I)
                               for it in (a, b)):
            pairs.append((a, b))
    return pairs


def run_pair(lm, setup, pair_id, pair, vectors, oracle):
    a, b = pair
    targets = (a[1], b[1])
    ids = [lm.single_token(w) for w in targets]
    filler = I.FILLERS[pair_id % len(I.FILLERS)]
    rows = []
    for order in (LABELS, LABELS[::-1]):
        base, marks, tails = prompt_parts(lm, order, filler)
        for arm in ("derived", "oracle", "text", "none"):
            for assigned in (targets, targets[::-1]):
                mapping = dict(zip(LABELS, assigned))
                questions = tails
                if arm == "text":
                    tokens, _, questions = prompt_parts(lm, order, filler, mapping)
                    cache = prefix_cache(lm, tokens)
                elif arm == "none":
                    cache = prefix_cache(lm, base)
                else:
                    bank = vectors if arm == "derived" else oracle
                    writes = [bank[mapping[label]] for label in order]
                    cache = prefix_cache(lm, base, marks, writes)
                for label in LABELS:
                    logits = query(lm, cache, questions[label])
                    asked = mapping[label]
                    sign = 1 if asked == targets[0] else -1
                    margin = sign * float(logits[ids[0]] - logits[ids[1]])
                    word = setup.choice_words[int(logits[setup.choice_ids].argmax())]
                    rows.append(dict(pair=pair_id, arm=arm, order=list(order), assigned=list(assigned),
                                     ask=label, target=asked, said=word, margin=margin,
                                     correct=word == asked))
    return rows


def summary(rows, smoke):
    report = {"pairs": len(set(r["pair"] for r in rows)), "smoke": smoke, "arms": {}}
    for arm in ("derived", "oracle", "text", "none"):
        group = [r for r in rows if r["arm"] == arm]
        margins = [mean(r["margin"] for r in group if r["pair"] == p)
                   for p in sorted(set(r["pair"] for r in group))]
        rng = random.Random(SEED)
        draws = sorted(mean(rng.choices(margins, k=len(margins))) for _ in range(5000))
        report["arms"][arm] = dict(mean_margin=mean(margins),
                                   margin_ci95=[draws[124], draws[4874]],
                                   pairwise_accuracy=mean((r["margin"] > 0) + .5*(r["margin"] == 0)
                                                          for r in group),
                                   exact_accuracy=mean(r["correct"] for r in group))
    arms = report["arms"]
    report["role_signal"] = (not smoke and report["pairs"] >= 20 and
                             abs(arms["none"]["mean_margin"]) < 1e-4 and
                             all(arms[a]["margin_ci95"][0] > 0 for a in ("derived", "text")) and
                             arms["derived"]["pairwise_accuracy"] >= .60)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, choices=range(1, 31), default=30)
    ap.add_argument("--results", type=Path, default=ROOT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    root = args.results
    if not (root / "lens.pt").is_file():
        ap.error(f"Expected the original Qwen lens at {root / 'lens.pt'} (see run.py)")
    lm = load(MODEL, args.device)
    pairs = eligible(lm, root)[:args.max_pairs]
    if not pairs:
        raise ValueError("No eligible pairs in confirm_v2")
    out = root / "delayed_v5"
    out.mkdir(exist_ok=True)
    lensfile = out / "v2_lens.pt"
    if lensfile.exists():
        lens = Lens.load(lensfile, lm.device)
    else:
        base = Lens.load(root / "lens.pt", lm.device)
        lens = extend_lens(lm, base, list(dict.fromkeys(w for it in ITEMS_V2 for w in it[1:])), print)
        tmp = out / f"lens-{os.getpid()}.tmp"
        lens.save(tmp)
        tmp.replace(lensfile)
    setup = E.Setup(lm, lens)
    usual = torch.stack([E.hold_read(setup, I.HOLD_PROMPT.format(desc=d))[0][L_READ]
                         for d, _, _ in I.ITEMS]).mean(0)
    setup.items = [it for it in ITEMS_V2 if all(lm.single_token(w) is not None for w in it[1:])]
    setup.finalize()
    vectors, oracle, rows = {}, {}, []
    for i, pair in enumerate(pairs):
        for desc, target, _ in pair:
            state = E.hold_read(setup, I.HOLD_PROMPT.format(desc=desc))[0][L_READ] - usual
            values, idx = state.topk(K)
            vectors[target] = lens.write(L_INJ, idx, values)
            oracle[target] = lens.write(L_INJ, [lens.index[target]], values[:1])
        rows.extend(run_pair(lm, setup, i, pair, vectors, oracle))
        print(f"{i+1}/{len(pairs)} pairs", flush=True)
    report = summary(rows, args.max_pairs < 30)
    path = out / ("full.json" if args.max_pairs == 30 else f"smoke-{args.max_pairs}.json")
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"model": MODEL, "summary": report, "rows": rows}, allow_nan=False))
    tmp.replace(path)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
