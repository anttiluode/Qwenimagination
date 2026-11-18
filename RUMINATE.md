# Rumination: a thought loop that never passes through tokens

Exploratory. Nothing here is a pre-registered claim; it is the first loop built from the pieces that held up.

**The pieces.**

- **Read** (v2, V11): while Qwen silently holds a derived answer, its lens reading at layer 26, minus the usual holding state, contains it (the item's word was in the top 8 for 37 of 40 items in V11b).
- **Install** (V9, V11b): exchanging a concept's lens coordinate with a word already in the prompt, clamped at every position across layers 4–12, with equal magnitudes, changes how Qwen answers a different question about it. In V11b, Qwen's own carried thought kept a living slot "living" in 14/20 cases, against 6/20 for random directions.

**The loop.**

1. **Seed:** Qwen silently works out an answer from a clue ("the animal whose wool is sheared…"); read layer 26.
2. **Install** the current top thought into a carrier, "Item: stone. Silently think of what the item above brings to mind." *stone* is exchanged with the thought across layers 4–12; the thought is never written.
3. **Read** what Qwen now holds at layer 26, minus the same carrier with no swap. That difference is the next thought's evidence.
4. **State:** `state = 0.3 · state + reading`. The next thought is the strongest concept in the state, minus **fatigue** on recently used thoughts (a trace that decays by 0.7 per step).
5. Repeat.

Fatigue 0 should get stuck on one thought (rumination). Fatigue 1 should move on (wandering). These are the bistable and excitable rungs of the TuftSheet ladder.

**The imagined tag.** The holding-mode words (*picture, imagine, photographs, thought…*) are blocked from being thoughts, and their mean reading is printed at every step. v1 found them loud in almost every hold. They are a native "I am holding something" signal, the tag SyntheticQualia said our machines lacked.

**The check.** For each of the 40 V7/V8 test words: install it silently, print it in the carrier instead, and compare the top 8 associations. The installed word should overlap its printed version more than a *different* installed word does. If not, the loop is wandering on noise, not on thoughts.

**Run.**

    python3.13 ruminate.py
    python3.13 ruminate.py --clue "the colour of a ripe banana" --steps 15 --fatigue 0 0.5 1

Output: `results/Qwen_Qwen2.5-1.5B-Instruct/ruminate/chains.json`. Reuses V9's cached lens; adds a lens row for *stone* if missing.
