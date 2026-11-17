# v3 protocol: picking the thought out of the bag

Committed before `pick.py` has run on a real model.

## The problem

In v2 the carried state held the answer in 57 of 60 bags, but the model
named it only 30% of the time. It often said a neighbour or the cue (*pen*
→ *ink*, *lungs* → *chest*, *December* → *January*). Writing the answer
alone gets 0.82, so the loss is in choosing which held concept is *the*
thought.

## The idea

A bag mixes two sources: concepts the **words in front of the model**
already evoke (outer), and concepts the model **added by thinking** (inner).
The cue and its associations are mostly outer, and the derived answer
should be mostly inner. So the model also reads the same description
*without* being asked to think of it (the "mention"):

> user: Read this phrase: "the capital city of France".
> assistant: Okay, I have read it.

Four rules score the concepts (read layer 26, write layer 22, as in v2):

| rule | score | meaning |
|---|---|---|
| A_usual | hold − usual hold | v2: what rose above the usual holding |
| C_inner | (hold − mention) − (usual hold − usual mention) | what the mind added beyond the words, against habit |
| M_tuft | relu(A) · sigmoid(C) | content multiplied by a "was this mine?" gate |
| D_late | A at layer 26 − A at layer 22 | what is still being built late |

The usual hold and usual mention are means over past items: leave one out
in development, all v1 + v2 items in confirmation. Each rule holds its top
8 concepts. Amounts are scaled so the largest equals A's largest, so rules
differ in *what* they hold, not in how hard they push. Release only at the
question, strength 1.0, the same question and scoring as v2.

## Two stages, one run

1. **Development** on v1 + v2 items, already seen, keeping only items the
   model answers when asked directly. The rule with the highest recall is
   chosen. Ties go to the higher top-3 rate, then the earlier rule in the
   order A, C, M, D.
2. **Confirmation** on **66 fresh items** (`qimag/items_v3.py`; no answer or
   cue shared with v1 or v2, checked in code). Arms: none, in-context, all
   four rules, the chosen rule with K = 1, cue, shuffled (the next item's
   state under the chosen rule), oracle.

## Gates (fresh items)

| Gate | Rule |
|---|---|
| S1 the chosen rule picks better than v2's | chosen ≠ A, and chosen recall ≥ A recall + 0.10 |
| S2 it still carries this item's thought | chosen ≥ none + 0.20, ≥ cue + 0.10, ≥ shuffled + 0.20 |
| S3 it puts the answer first more often | answer ranked first by the chosen rule ≥ A + 0.15 |

If development chooses A, S1 fails by definition. That would mean none of
the three new rules beat v2.

## Predictions, before the run

- **Chosen rule:** C or M, about 60%. D is the least likely.
- **The main risk:** just *reading* "the capital city of France" may already
  evoke Paris. If so, the mention contains the answer, C subtracts it, and
  A wins development. Of the new rules, M is the least exposed to this,
  because it keeps A's content and only gates it.
- **S1: about 40%.** Many v2 errors were neighbours of the same kind
  (*December*/*January*). The words don't explain those either, so
  "inner minus outer" may not separate them.
- **S2: pass, about 80%.** This repeats v2's controls on new items.
- **S3: about 45%.**
