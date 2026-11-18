# Thinking toward a goal, without tokens

Exploratory. Builds on the rumination loop (RUMINATE.md), where a silently installed word evoked nearly the same associations as the printed word (top-8 overlap 0.41 against 0.07 for a different word, 34/40).

**What changes.** In the rumination loop our rule picked the next thought (the strongest association minus fatigue). Here **Qwen's own answers** pick it:

1. **Start** from a silently derived thought (the hold read at layer 26).
2. **Associate:** install the current thought across layers 4–12 and read what it brings to mind at layer 26. The top 6 not yet visited are the candidates.
3. **Judge:** install each candidate silently into a goal question ("Item: stone. Can the item above fly? Answer yes or no.") and take Qwen's yes-minus-no.
4. **Move** to the candidate Qwen answers most "yes" to. Stop when it answers yes (at most 8 steps).
5. **Verify** the found thought by asking the same question with the word printed.

No thought or candidate is ever written as a token until the final check.

**Controls,** from the same start with the same candidates each step:

- **association**: move to the strongest association (the wandering loop, without fatigue's numbers, never revisiting);
- **random**: move to a random candidate.

If goal-directed search finds a "yes" thought in fewer steps, and more often, than the controls, the loop is using Qwen's judgments to steer where its thinking goes.

**Default goals.**

| Seed clue | Goal |
|---|---|
| the animal whose wool is sheared to make sweaters | Can the item above fly? |
| the thing you write with that uses ink | Does the item above live in water? |
| the capital city of France | Is the item above something you can eat? |
| what spins a web and has eight legs | Is the item above bigger than a car? |

**Run.**

    python3.13 think.py
    python3.13 think.py --clue "the colour of a ripe banana" --question "Is the item above a kind of weather?"

Writes `results/Qwen_Qwen2.5-1.5B-Instruct/think/think.json` and **`think.html`**, a page showing each run's chain of thoughts and, for every step, each candidate with Qwen's verdict as a bar.
