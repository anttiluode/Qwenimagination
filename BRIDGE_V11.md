# V11: read the thought in the mouth, install it in the room

Written before any real run.

**Why.** Two measurements on Qwen2.5-1.5B don't overlap:

- **Where a concept can steer a later question (V9):** a lens-coordinate swap clamped across layers 4–12. Bands starting at layer 12 or later did nothing.
- **Where a silent thought is readable (v2 diagnostic):** the answer is in the top 10 of the hold reading (minus the usual hold) for 4–8 of 38 items at layers 1–16, rising from layer 17, 37 of 38 at layers 26–27.

V10 read the thought at layers 4–12 and carried nothing (the item's word was in the top 8 for 2–6 of 40 items). V11 bridges the two: the **concepts and their weights come from the layer-26 read**, and the swap uses **those concepts' own lens directions at layers 4–12**.

**No selection stage.** Two of this morning's runs were steered by a choice made on 9 training pairs (V3's rule pick, V10's α = 2). V11 fixes everything in advance:

- band 4–12, α = 1 (V9);
- read layer 26, with the v2 usual-hold subtraction;
- **K = 1 is the primary**: the late read's top concept, swapped in by its early directions. This is V9's swap with the word chosen by Qwen's own silent thought.
- K = 8 (the whole bag, weighted, as one direction per layer) is reported alongside, without a gate.

Answer words of every V8 format, "item", and the decoy word are excluded from the bag.

**Cases, arms, gates.** As in V10, on V7/V8's 20 held-out pairs: the thought swapped into a slot of the other category (cross) and into a slot of its own category (within); the true word (V9 control); a random concept's early directions with the same weight; and the external reader on the same layer-26 bag (trained on the 20 training clues).

- **T1**: word swap flips more than 0.25 (V9 reproduced).
- **G1**: thought cross − thought within, 95% bootstrap lower bound > 0.
- **G2**: thought cross − random cross, lower bound > 0.

Also reported: how often the item's own word is in the late bag, and flips split by direction. V9 and V10 both found manufactured slots very hard to flip even with the true word (0/20), so most of the signal will come from living slots.

**Predictions (Claude, before the run).** The item's word is in the late bag for about two thirds of items. T1 passes (≈85%). G1 passes for K = 1 (≈55%): the top concept is often a neighbour (*rabbit* for the wool clue), but a neighbour usually has the right category, which is all this question needs. G2 passes (≈50%). The thought flips fewer answers than the true word (≈80%).

**What each outcome means.**

- **G1 and G2 pass:** Qwen's own unspoken thought, carried outside the tokens and put back where reasoning happens, decides the answer to a question it was never given the answer to. That is the milestone, on a 1.5B model.
- **They fail while T1 passes:** the late-read concepts aren't the right ones, or one concept isn't enough. The next step is a larger model, where the paper finds the workspace.

**Run.**

    python3.13 bridge_v11.py

Reuses V9's cached lens. Output: `results/Qwen_Qwen2.5-1.5B-Instruct/bridge_v11/full.json`.
