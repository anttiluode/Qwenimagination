# V8: can Qwen use a carried thought to answer a different question?

Written before any real run.

**Why V8.** V7 asked the right question but never tested it. Qwen 1.5B answered "A" in all 400 decisions, in every arm, including with the word printed in the prompt. With the letter bias removed, the printed word reached only about 0.60. V7's own protocol required the printed-word control to work first, and it failed. V8 makes that check a stage that runs, and stops the run, before any write is tested.

**Question.** Qwen silently works out an item from a clue (the wool clue → sheep). Its extracted thought, eight concept weights with every answer word removed, is written into a fresh prompt that names no item ("Item: item"). Qwen is then asked a question whose answer was never written: is it living or manufactured? If the answer follows the carried thought, Qwen used it for a second inference.

**Items.** V7's sets, unchanged: 20 training items (10 living, 10 manufactured) and 40 held-out test items in 20 living/manufactured pairs. Stages 0 and 1 see only the training items.

**Calibration.** Every arm's evidence is measured against the same neutral prompt with no write. A fixed lean toward one answer gives zero evidence, so it cannot score. Positive evidence means "living". Ties count half.

**Stage 0: the question format (printed word, training items).** Four formats, each asked two ways so a fixed lean cancels:

| Format | The two questions | Scored tokens |
|---|---|---|
| yesno | "a living thing?" and "made by people?" | Yes/yes vs No/no |
| living_manufactured | "living or manufactured?" in both orders | the two words |
| natural_artificial | "natural or artificial?" in both orders | the two words |
| letters_v7 | V7's question, both letter mappings | A vs B |

A format whose answer words are not single tokens is skipped. The best format by printed-word accuracy is used, ties going to the earlier row. **Gate T0: if it is below 0.85, the run stops and reports the format as the problem.**

**Stage 1: the write (training items).** Layers 10, 14, 18, 22 × strengths 1, 2, 4 × positions (final answer token only, or every token from the question onward). V6 showed writes act at the answer position. The configuration with the best training accuracy is frozen, ties broken by mean signed evidence. 24 configurations on 20 items can overfit, which is why stage 2 uses new items.

**Stage 2: test (40 held-out items), frozen choice.**

| Arm | What is written |
|---|---|
| none | nothing (0.50 by construction) |
| text | the item printed, no write (format check on new items) |
| derived | the item's own carried thought |
| random | the same weights on random concept directions |
| word | the item word's own lens direction (where the lens has it) |
| reader | no Qwen: nearest-centroid classifier on the same bag (V7: 0.91) |

V7's swapped arm is dropped: with a neutral prompt it is the derived arm of the partner, so it adds no independent trial.

**Gates (95% bootstrap over the 20 pairs):**

- **T1**: text ≥ 0.85 on the test items, or the result is inconclusive.
- **G1**: derived − none, lower bound > 0. Qwen used the carried thought for a question it was never given the answer to.
- **G2**: derived − random, lower bound > 0. It was the thought, not any disturbance.

derived − reader is reported, not gated. If Qwen falls below the reader, the extracted bag already holds the category and Qwen is only a weaker reader of it. Matching or beating it would be the interesting outcome.

**Predictions (Claude, before the run).** T0 passes with yesno (≈75%). G1 passes (≈45%). Derived falls below the reader (≈80%). The word arm beats the derived arm.

**Run.**

    python3.13 category_v8.py --max-pairs 2
    python3.13 category_v8.py

Output: results/Qwen_Qwen2.5-1.5B-Instruct/category_v8/full.json. It reuses the V5 lens; no lens building.
