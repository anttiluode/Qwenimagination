# V9: the paper's own intervention, on a 1.5B Qwen

Written before any real run.

**Why.** Rereading Gurnee et al. (2026) against V2–V8 showed that we never ran the intervention the paper validated:

| | The paper | V2–V8 |
|---|---|---|
| Layers | the workspace band, from about a third of the way in to shortly before the output; the last layers are "motor" (§4.1) | read at 26 of 28, writes mostly at 22 |
| Intervention | a **swap** of two lens coordinates, **clamped** at every position across the whole band (§3.3, §3.4) | an **addition** at one layer |
| Starting state | the source concept is already present; its "workspace loading" predicts success (§3.4) | an empty slot ("Item: item") |
| Across tokens | broadcast heads, concentrated in the first half of the band (§4.3.2) | single writes at layer 22, after them |

The paper's two-hop swaps (spider → ant changes "8 legs" to "6") work 54–70% of the time on Claude models. Whether small models have a usable workspace is one of its open questions (§9.1). V9 asks it of Qwen2.5-1.5B.

**Stage A: find the band.** On calibration text, at every layer, three of the paper's signatures, computed over our lens vocabulary:

- **kurtosis** of the lens reads (high = a few concepts stand out);
- **persistence**: how often the top concept stays the same at the next token, minus a shuffled baseline;
- **output agreement**: rank correlation between the lens reads and the model's final logits (rises sharply in the motor layers).

Printed and saved; they are for reading, not used by a rule.

**Stage B: choose band and strength on training pairs.** The yes/no format from V8 (printed-word accuracy 0.95 on training, 1.00 on test). Prompt: "Item: wolf", then "Is the item above a living thing?" and "…made by people?". A cross swap exchanges the lens coordinates of the item and a partner from the other category (wolf ↔ chair); a within swap uses a partner from the same category (wolf ↔ lion). Coordinates are those in the two-dimensional span of the two lens vectors; everything outside the span is untouched. They are clamped to the clean pass's exchanged values (strength α: α = 2 doubles the exchange, as in the paper's Fig. 19) at every position of every band layer.

Bands: 4–12, 8–16, 12–20, 16–24, 20–27, 8–20, 10–24, 4–24. α ∈ {1, 2}. The pair with the largest cross-flip minus within-flip on the 10 training pairs is frozen.

**Stage C: test on V7/V8's 20 held-out pairs.**

- **cross flip**: after wolf ↔ chair, does "Item: wolf" get answered as manufactured?
- **within flip**: after wolf ↔ lion, does it stop being answered as living? This catches a swap that merely breaks the answer.
- **Gate S1**: cross flip − within flip, 95% bootstrap over pairs, lower bound > 0.

Also saved per item: clean evidence, swapped evidence, and the source's loading, to compare with the paper's loading result.

Words without lens rows get them computed once (as V5 did) and cached in `swap_v9/v9_lens.pt`, which is git-ignored.

**Reading the outcome.**

- **S1 passes:** Qwen 1.5B has a workspace band where a concept, swapped in, is *used* to answer a different question. Then the earlier failures were about where and how we wrote. The next step is writing Qwen's own derived thought there, read inside the band rather than at layer 26.
- **S1 fails with the swaps having little effect:** a 1.5B model has no workspace that steers reasoning this way, at least not one our lens can reach. The next step is a larger model (a 3B fits a 12 GB card).
- **Both flips high:** the swap breaks answers rather than steering them.

**Predictions (Claude, before the run).** Stage A shows a band starting around layer 8–10 with output agreement climbing steeply after about 22 (≈65%). S1 passes (≈55%). The chosen band lies inside 8–24, not 20–27 (≈75%).

**Run.**

    python3.13 swap_v9.py --quick
    python3.13 swap_v9.py

The first run adds lens rows for missing words (a few minutes), then reuses them. Output: `results/Qwen_Qwen2.5-1.5B-Instruct/swap_v9/full.json`.
