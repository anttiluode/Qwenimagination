# V10: Qwen's own silent thought, used for a second inference

Written before any real run.

**What V9 found.** Swapping a word's lens coordinates (wolf ↔ chair), clamped at every position across layers 4–12, makes Qwen answer a *different* question ("is it a living thing?") about the swapped-in word. Bands starting at layer 12 or later did nothing. Gate passed on held-out pairs: cross flips 0.35, same-category flips 0.10, CI +0.15..+0.35.

**The question now.** Replace the swapped-in *word* with Qwen's own *thought*. Qwen is asked to silently think of something from a clue ("the wild canine that howls in packs"). The answer is never written. We read its lens state inside the band, carry it to a fresh conversation, swap it into a slot holding a word of the other category ("Item: chair"), and ask whether the item is living. If the answer follows the thought, Qwen used its own unspoken thought for a second inference.

**Frozen from earlier runs.** Band 4–12 (V9). Yes/no format (V8: 0.95 training, 1.00 test with the printed word). V7/V8's 10 training pairs and 20 held-out pairs. V2's hold prompt and "usual hold" subtraction, now taken at every band layer instead of layer 26.

**The thought.** At each band layer: the hold's lens reads minus the usual hold, with every answer word of any V8 format, "item", and the decoy word removed; keep the top K concepts. Its direction at that layer is the lens's minimum-norm write of that bag. For K = 1 this is exactly the direction of a single word, so K = 1 is V9 run on Qwen's own top concept.

**The swap.** V9's: in the span of the decoy's lens direction and the thought's direction, the clean pass's two coordinates are exchanged (strength α), clamped at every position of every band layer. Everything outside the span is untouched.

**Cases (per pair).** Cross: the living item's thought into the manufactured word's slot, and the reverse. Within: a thought into a slot of its own category (the next pair's item), which should not move the answer.

**Stage 1 (training pairs).** K ∈ {1, 8} × α ∈ {1, 2}; freeze the one with the largest cross-flip minus within-flip.

**Stage 2 (held-out pairs).** A flip is an answer that no longer names the decoy's category.

| Arm | What gets swapped in |
|---|---|
| thought, cross | Qwen's own thought, other category |
| thought, within | Qwen's own thought, same category as the decoy |
| word, cross | the true item word (V9's control; the ceiling) |
| random, cross | the thought's weights on random concept directions |
| reader | no Qwen: nearest-centroid classifier on the band-averaged bag |

**Gates (95% bootstrap over pairs).**

- **T1**: word swap still flips more than 0.25 (V9 reproduced on these decoys).
- **G1**: thought cross − thought within, lower bound > 0. *The milestone.*
- **G2**: thought cross − random cross, lower bound > 0. It's the thought, not any direction.

**Predictions (Claude, before the run).** T1 passes (≈85%). G1 passes (≈45%): the hold reads in early layers are noisier than the word itself, and the top concept is sometimes a neighbour (*dog* for wolf), which still has the right category. The thought flips fewer answers than the word (≈85%). K = 1 is chosen over K = 8 (≈55%).

**Run.**

    python3.13 thought_swap_v10.py --quick
    python3.13 thought_swap_v10.py

Reuses V9's cached lens. Output: `results/Qwen_Qwen2.5-1.5B-Instruct/thought_v10/full.json`.
