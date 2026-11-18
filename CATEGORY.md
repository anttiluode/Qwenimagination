# V7: use a carried item to classify it

**Question.** If a clue evokes sheep, can a single pre-question write make a later answer depend both on the item's semantic category and on the question's instructions? This is a binary category test. Success could use coarse animal or object associations without recovering the specific word, or carrying out a multi-step inference.

**Items.** The 20 fixed test pairs in category_transfer.py each contain a living organism and a manufactured object. Every arm sees all 40 items. No exclusions based on model performance. These are held out from classifier fitting. Some concepts may be familiar from calibration or prior experiments. The 20 fixed training items are disjoint by target word and balanced by class. The categories and clues are hand supplied; results apply to this item set.

**Writes.** Reuse the V5 extended V2 lens and V2 hold prompt. At layer 26 subtract the mean V1 hold and select the eight highest remaining concept coordinates after excluding both category names and both response letters (case insensitive). The signed values are used for both the gain-1 layer-10 write and the direct reader. We chose this earlier layer before the run. The recipient token is always the same neutral word in “Item: item”. A single write is applied to its post-layer residual. Cache the prefix, remove the write hook, and fork that same cache for the two later questions. One uses A for living, B for manufactured; the other reverses that code. Score A versus B at the next token. Category names occur in the question; neither the clue nor the identity of the item occurs in the neutral prefix.

| Arm | Input to the decision |
|---|---|
| Derived | One write from the clue's eight weighted coordinates |
| Swapped | The opposite-category partner's derived vector on the same neutral token |
| None | The neutral prefix with no write |
| Text | The true item printed at the same slot |
| Token patch | Replace the neutral token's post-layer-10 state with the state of the printed item at that position. Lower-layer cache entries remain neutral. |
| Reader | Nearest class centroid on the same sparse weighted bags, with cosine similarity. Class centroids use only the 20 training bags. It predicts a semantic class directly. |

The swap is deliberate: it tests directional reversal and is algebraically related to the derived arm on a balanced pair. It supplies no additional independent trials. The letter reversal tests whether a fixed push toward one response can explain accuracy. The native patch uses the true word, making it a diagnostic control for the transfer path. It receives more information than the derived arm.

**Readout.** Report forced-choice accuracy (ties count half) averaging the two code mappings, and intervals for paired accuracy differences by resampling 20 whole pairs. The direct reader contributes one semantic decision per item. Scores and bags are saved for inspection. Interpret the full run in this order:

1. Check text can do the task and none averages 50%. A failed text control makes the downstream test inconclusive.
2. A positive lower 95% interval for derived minus none, with the swapped condition reversing the advantage, supports item-specific downstream category use on this prompt. Inspect accuracy under each mapping before claiming code following.
3. The reader measures how much of that decision a simple classifier can already make from the same information. A positive lower paired interval against the reader would support added performance over this particular classifier. A strong reader and weak Qwen would favor extraction as the working component.
4. If the native patch works and derived fails, the tested write format is a plausible limitation. Failure of both leaves the chosen layer, slot and task unresolved.

**Run.** From the repository root, after the V5 lens is available:

    python3.13 category_transfer.py --max-pairs 2
    python3.13 category_transfer.py

The full file is results/Qwen_Qwen2.5-1.5B-Instruct/category_v7/full.json. Smoke runs use separate filenames and the same fixed training set. No fine-tuning, lens rebuilding, layer search, or remote compute is required with the existing local files.
