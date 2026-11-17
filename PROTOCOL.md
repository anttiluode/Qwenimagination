# Protocol: holding a thought outside the tokens

Committed before any run on a real model. Nothing in this repository has
been tuned on a real model: the workspace that built it could not reach
Hugging Face, so every constant below was set in advance and only checked
for plumbing on a random-weight model (`tests/`).

**Pre-registered run:** `python run.py --model Qwen/Qwen2.5-1.5B-Instruct`,
fp32, on the first execution. Runs on other models are reported as extra
results under the same rules, never in place of this one.

## Question

Can a frozen language model hold a thought it *derived*, one that was
never written as a token, across a context that does not contain it, and
report it at the end? The thought must be carried by a small state that
lives outside the token stream, in the model's J-space coordinates.

## Method (fixed)

- **Lens:** averaged Jacobian of each concept's logit, summed over the
  current and later positions, with respect to the residual after each
  layer, averaged over `items.CALIBRATION`. Reads are z-scored on the same
  text. Concepts: every single-token answer, cue and decoy in `items.py`.
- **Items:** `items.ITEMS`. Items whose answer or cue is not one token are
  dropped. Item *i* is dev if *i* mod 3 = 0, test otherwise.
- **Hold:** the model is told to silently think of the description; the
  J-space is read at each layer over its fixed reply and averaged.
- **Layers:** the read layer is the one in [L/3, 5L/6] with the highest
  dev top-10 hit rate (ties: lower median rank). The write layer is the
  read layer − 4.
- **Held state:** the 8 concepts with the highest z at the read layer and
  their z values m. No filtering: cue words and words like "mind" stay in
  if the model put them there.
- **Recall context:** a fresh conversation with a neutral paragraph and
  the question "what were you thinking of earlier?", ending in the
  assistant prefix "I was thinking of". The hold is not in it.
- **Routes:** none; in-context (ceiling); oracle (answer word's lens
  vector); cue (cue word's lens vector); snapshot (raw hidden state at the
  write layer); jstate_open (held state written at constant strength);
  jstate_closed (residue loop: after every token the read layer is read,
  the deficit m − z is integrated into a non-negative drive capped at 16·m,
  and the drive is written at the next token); jstate_closed_surprise
  (as closed, write scaled by exp(−0.5·max(0, s − median s)), where s is the
  model's next-token surprise). Grids: open routes 0.5–16, snapshot
  0.25–4, closed gain 0.05–0.8, as in `experiment.py`.
- **Recall score:** forced choice over all answers **and all cue words**.
  Correct only if the item's answer is ranked first, so naming the cue
  ("France" for the capital of France) counts as wrong.
- **Damage:** mean KL(none ‖ route) of next-token distributions over the
  recall context, answer slot excluded.
- **Budgets:** for each route and damage budget B ∈ {0.02, 0.05, 0.1, 0.2,
  0.5}, the strength is chosen on **dev** (best dev recall with dev damage
  ≤ B); the gate uses the **test** recall at that strength. If no strength
  fits, the route counts as "none".
- **J-state** in G1/G2 means the better of jstate_open and jstate_closed at
  budget 0.2.

## Gates (test items)

| Gate | Rule |
|---|---|
| **G0a** the workspace holds the derived thought | answer in the top 10 of the hold reading at the read layer in ≥ 40% of items, and its z higher in its own hold than on average in other items' holds in ≥ 80% |
| **G0b** the write channel works | oracle recall ≥ 0.50 at budget 0.5 |
| **G1** a thought carried outside the tokens | J-state recall ≥ none + 0.20 at budget 0.2 |
| **G2** the thought, not the cue | J-state recall ≥ cue recall + 0.10 at budget 0.2 |
| **G3** the closed loop earns its place | jstate_closed ≥ jstate_open − 0.02 at every budget, and ≥ jstate_open + 0.10 at one or more |

If G0a or G0b fails, G1–G3 are reported but not interpreted: either a
1.5B model does not hold the derived concept where the lens can see it, or
the write channel cannot deliver even the answer itself.

Report only, no rule: the in-context ceiling, snapshot, the surprise-gated
loop, the null-hold baseline, every budget column, and bootstrap CIs.

## My predictions, before the run

- **G0a: uncertain, slightly more likely to pass.** The paper's results are
  for large production models. Whether a 1.5B model puts "Paris" into its
  J-space when told to think of the capital of France, without saying it,
  is the real unknown of this repo.
- **G0b: pass.** Writing an answer's own lens vector is close to steering.
- **G1: pass if G0 passes.**
- **G2: pass, with a smaller margin than G1.** The held state will contain
  the cue too, and under strict scoring a cue in the state can win.
- **G3: fail.** I expect the closed loop to act as a self-calibrating open
  loop. In a feedforward pass the read four layers above the write mostly
  sees the write itself, so the loop says less about the model's own
  holding than the name suggests. This is the same kind of prediction as
  "the scalar copy will win" in the earlier repos.
