"""Can an upstream write influence a later, remapped category decision? See CATEGORY.md."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
from statistics import mean

import torch
import torch.nn.functional as F

from qimag import experiment as E, items as I
from qimag.items_v2 import ITEMS_V2
from qimag.lens import Lens, tap_layers
from qimag.model import load
from recall_slots import MODEL, ROOT, L_READ, prefix_cache, query

LAYER = 10
LABELS = ("living", "manufactured")
CODES = ("A", "B")
NEUTRAL = "item"
V2_CLUES = {word: desc for desc, word, _ in ITEMS_V2}
TRAIN = [(word, V2_CLUES[word], label)
         for label, words in enumerate(("sheep elephant tiger fox bat puppy dove pilot teacher baker",
                                        "hammer pen mirror clock socks book camera knife fork broom"))
         for word in words.split()]
# Each row has one member of each class. Pairing fixes the swap and bootstrap units.
PAIRS = [
    ("wolf", "the wild canine that howls in packs", "chair", "what you pull out from a desk to sit on"),
    ("lion", "the big cat whose males grow a mane", "bicycle", "what has two wheels and pedals for riding"),
    ("horse", "what a jockey rides in a race", "spoon", "what you use to scoop soup at dinner"),
    ("cow", "what moos and gives us milk", "bottle", "what has a narrow neck and holds a drink"),
    ("whale", "the huge sea mammal that breathes through a blowhole", "radio", "what you tune to a station to hear a broadcast"),
    ("goat", "the horned farm creature famous for climbing and bleating", "guitar", "the six-string instrument you strum"),
    ("duck", "what quacks and swims on ponds", "piano", "the keyboard instrument with hammers hitting strings"),
    ("rabbit", "what hops with long ears and a short fluffy tail", "drum", "the instrument whose stretched skin you strike"),
    ("shark", "the finned ocean predator with rows of replaceable teeth", "bell", "what rings when its metal cup is struck"),
    ("bee", "what visits flowers and makes honey", "helmet", "what a cyclist straps over the head for protection"),
    ("frog", "the amphibian that starts life as a tadpole", "basket", "what you weave from wicker to carry things"),
    ("deer", "the woodland grazer whose males commonly grow antlers", "lamp", "what you switch on beside your bed for light"),
    ("snake", "the legless reptile that sheds its skin", "key", "what you insert and turn to unlock a door"),
    ("monkey", "the primate often shown swinging by its arms with a long tail", "pencil", "the wooden writing implement with a graphite core"),
    ("ant", "the tiny six-legged colony dweller that forms trails to food", "bowl", "the deep round dish used for serving soup"),
    ("spider", "what spins a web and has eight legs", "truck", "the heavy road vehicle used to haul freight"),
    ("owl", "the nocturnal bird known for hooting", "shoe", "what you lace onto a foot for walking outside"),
    ("crow", "the common black bird that caws", "shirt", "the sleeved garment worn on the upper body"),
    ("oak", "the tree that produces acorns", "wheel", "the round part that turns on a vehicle axle"),
    ("rose", "the flowering shrub with thorny stems associated with Valentine's Day", "jacket", "the outer garment with sleeves that you zip over your clothes"),
]


def bag(state, blocked):
    state = state.clone()
    state[blocked] = -torch.inf
    values, indices = state.topk(8)
    if not torch.isfinite(values).all():
        raise ValueError("Insufficient eligible concept directions")
    sparse = torch.zeros_like(state)
    sparse[indices] = values
    return indices, values, sparse


def parts(lm, word):
    """Keep only the item slot in the prefix. Both questions follow the same cache."""
    prefix = "Item: " + word
    cuts = []
    suffixes = []
    for a, b in (LABELS, LABELS[::-1]):
        ask = f"\n\nClassify the item above. Use A for {a} and B for {b}. Answer with the letter only."
        chat = lm.chat_text([("user", prefix + ask)], "The letter is")
        start = chat.index(prefix + ask)
        enc = lm.tok(chat, add_special_tokens=False, return_offsets_mapping=True)
        pos = next(i for i, (s, e) in enumerate(enc["offset_mapping"])
                   if s <= start + len(prefix) - 1 < e)
        ids = enc["input_ids"]
        if lm.single_token(word) != ids[pos]:
            raise ValueError(f"Expected a single source token for {word!r}")
        cuts.append(ids[:pos+1])
        suffixes.append(ids[pos+1:])
    if cuts[0] != cuts[1]:
        raise ValueError("The later question changed the stored prefix")
    return cuts[0], suffixes


@torch.no_grad()
def last_state(lm, ids):
    with tap_layers(lm, [LAYER]) as layers:
        lm.final_hidden(torch.tensor([ids], device=lm.device))
        return layers[LAYER][0,-1].float().clone()


def classifier(train, test):
    # Supervised nearest-centroid reader on the very same eight weighted coordinates.
    centers = [F.normalize(torch.stack([F.normalize(v, dim=0) for v,c in train if c==label]).mean(0),
                           dim=0) for label in (0,1)]
    return float(F.normalize(test, dim=0) @ (centers[0]-centers[1]))


def bootstrap(values):
    rng = random.Random(20261118)
    draws = sorted(mean(rng.choices(values,k=len(values))) for _ in range(5000))
    return [mean(values), draws[124], draws[4874]]


def summarize(rows):
    arms = sorted({r["arm"] for r in rows})
    scores = {}
    report = {}
    for arm in arms:
        per_pair = defaultdict(list)
        for r in rows:
            if r["arm"] == arm:
                per_pair[r["pair"]].append((r["margin"] > 0) + .5*(r["margin"]==0))
        scores[arm] = [mean(group) for _,group in sorted(per_pair.items())]
        report[arm] = mean(scores[arm])
    differences = {a:bootstrap([x-y for x,y in zip(scores["derived"],scores[a])])
                   for a in ("none","swapped","reader")}
    by_mapping = {}
    for arm in arms:
        if arm == "reader":
            continue
        by_mapping[arm] = [mean((r["margin"] > 0) + .5*(r["margin"] == 0)
                                for r in rows if r["arm"] == arm and r["mapping"] == order)
                           for order in (0,1)]
    return {"accuracy":report,"accuracy_by_letter_mapping":by_mapping,
            "derived_minus_control_mean_ci95":differences}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-pairs",type=int,choices=range(1,len(PAIRS)+1),default=len(PAIRS))
    p.add_argument("--results",type=Path,default=ROOT)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    path = args.results / "delayed_v5" / "v2_lens.pt"
    if not path.is_file():
        p.error(f"Missing calibration lens: {path}")
    lm = load(MODEL,args.device)
    lens = Lens.load(path,lm.device)
    setup = E.Setup(lm,lens)
    usual = torch.stack([E.hold_read(setup,I.HOLD_PROMPT.format(desc=d))[0][L_READ]
                         for d,_,_ in I.ITEMS]).mean(0)
    forbidden = {w.casefold() for w in LABELS+CODES}
    blocked = [i for i,w in enumerate(lens.words) if w.strip().casefold() in forbidden]

    def extract(desc):
        state = E.hold_read(setup,I.HOLD_PROMPT.format(desc=desc))[0][L_READ] - usual
        return bag(state,blocked)

    train = [(extract(desc)[2],c) for _,desc,c in TRAIN]
    base, questions = parts(lm,NEUTRAL)
    baseline_state = last_state(lm,base)
    no_cache = prefix_cache(lm,base,layer=LAYER)
    choices = [lm.single_token(c) for c in CODES]
    if None in choices or choices[0] == choices[1]:
        raise ValueError("Expected two distinct one-token response letters")

    def read(cache, suffixes):
        margins = []
        for suffix in suffixes:
            logits = query(lm,cache,suffix)
            margins.append(float(logits[choices[0]]-logits[choices[1]]))
        return margins

    none = read(no_cache,questions)
    rows, bags = [], {}
    for pair_no,(left,clue_l,right,clue_r) in enumerate(PAIRS[:args.max_pairs]):
        features = [extract(clue) for clue in (clue_l,clue_r)]
        writes = [lens.write(LAYER,idx,values) for idx,values,_ in features]
        for label,word in enumerate((left,right)):
            idx,values,feat = features[label]
            bags[word] = list(zip([lens.words[i] for i in idx.tolist()],values.tolist()))
            printed,printed_q = parts(lm,word)
            if printed[:-1]!=base[:-1] or printed_q!=questions:
                raise ValueError(f"Token context for {word} differs outside the slot")
            native = last_state(lm,printed)-baseline_state
            margins = {"none":none,"text":read(prefix_cache(lm,printed,layer=LAYER),questions)}
            for arm,delta in (("derived",writes[label]),("swapped",writes[1-label]),("token_patch",native)):
                cache = prefix_cache(lm,base,[len(base)-1],[delta],layer=LAYER)
                margins[arm] = read(cache,questions)
            for arm,raw in margins.items():
                for order,value in enumerate(raw):
                    rows.append(dict(pair=pair_no,item=word,arm=arm,mapping=order,
                                     margin=value if label==order else -value, A_minus_B=value))
            m = classifier(train,feat)
            rows.append(dict(pair=pair_no,item=word,arm="reader",margin=m if label==0 else -m))
        print(f"{pair_no+1}/{args.max_pairs} pairs",flush=True)
    report = summarize(rows)
    out = args.results / "category_v7"
    out.mkdir(exist_ok=True)
    dest = out / ("full.json" if args.max_pairs==len(PAIRS) else f"smoke-{args.max_pairs}.json")
    tmp = dest.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(model=MODEL,layer=LAYER,pairs=args.max_pairs,
                                  summary=report,bags=bags,rows=rows),allow_nan=False))
    tmp.replace(dest)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
