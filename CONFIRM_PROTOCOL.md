# Confirmation v2 protocol

Committed before `confirm.py` has run on a real model.

## Why a v2

The pre-registered v1 run failed all five gates (see README). The
diagnostic (`diagnose.py`, items already seen) found three design faults:

- the question made the model name the subject;
- the held state was dominated by a constant "I'm thinking" signal;
- writing "about to say" directions over the whole paragraph was costly.

It also found a fix for each. Those fixes were chosen while looking at v1
items, so they can't be confirmed on them. This run uses **73 fresh items**
(`qimag/items_v2.py`). None of their answers or cues was an answer or cue
in v1.

## Frozen design

- **Model** Qwen2.5-1.5B-Instruct, fp32. **Read layer 26, write layer 22**
  (chosen on v1 dev items by the diagnostic).
- **Lens**: v1's lens. Rows for new words are computed by the same method
  and appended.
- **Usual hold** (the slow baseline): the mean hold reading over all v1
  items at the read layer. It is learned from past holds only and never
  sees a v2 item.
- **Held state** (primary): the 8 concepts whose reading rose most above
  the usual hold, with those rises as amounts.
- **Release**: nothing is written during the paragraph. From the first
  token of the question on, the state is written at strength 1.0.
- **Question**: "...what was the answer you were silently holding
  earlier? Answer with one word." Prefix: "The answer was".
- **Score**: forced choice over all v2 answers and cues. Naming the cue is
  wrong.
- **Filter**: only items whose answer and cue are single tokens, and which
  the model answers correctly when asked directly ("What is {desc}?").
  Unknown items are listed, not scored.
- **Arms**: none; in-context (hold in the conversation, same question);
  primary; strength 0.5 and 2.0; K = 1 and 3; absolute top 8 (v1's
  selection); cue lens vector; answer lens vector (oracle); shuffled
  (another item's state, the next one in order); closed loop (v1 style,
  written throughout). Cue and oracle use the size of the state's largest
  amount.

## Gates (all scored items)

| Gate | Rule |
|---|---|
| C1 a derived thought carried outside the tokens | primary ≥ none + 0.20 |
| C2 the residue, not the loud state | primary ≥ absolute + 0.10 |
| C3 the thought, not the cue | primary ≥ cue + 0.10 |
| C4 it is this item's thought | shuffled ≤ none + 0.05, and primary − shuffled ≥ 0.20 |
| C5 carried beats re-deriving in context | primary ≥ in-context |

## Predictions, before the run

- **C1 pass.** On v1 items the same design got about 0.34 against 0.00.
- **C2 pass.** The absolute selection carried almost nothing (0.08–0.11).
- **C3 probably pass, about 65%.** The cue route is new. If the model can
  take the cue at question time and derive the answer then, the cue could
  match the carried state. That would mean the carried state works as a
  prompt, not as a held answer.
- **C4 pass.** A shuffled state should make the model name the other
  item's thought, not this one's.
- **C5 a coin flip, about 55%.** On v1 items it was 13 against 8 out of 38,
  too small a gap to trust.
- **Not gated:** K = 1 should beat K = 8. In v1 the answer was nearly
  always in the bag (top 3 in 34/38) but only first in 18/38, and the
  model often picked a neighbour from the bag. The closed loop should stay
  near none.
