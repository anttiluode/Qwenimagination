# Qwenimagination

**Can a frozen language model hold a thought it never wrote down, carry it
through a conversation that doesn't contain it, and say it at the end?**

Anthropic's J-space paper (Gurnee, Sofroniew, … Lindsey, *Verbalizable
Representations Form a Global Workspace in Language Models*, arXiv
2607.15495, 2026) found that a language model's "workspace" is a small set
of directions it is poised to put into words. The model can hold a concept
there on request, reason through it, and report it. The same paper names
the limit (§9.3): this workspace is rebuilt at every token. The only way a
transformer can hold a thought longer than one forward pass is to write it
out as tokens and read it back, *"punctuated at regularly spaced intervals
by a significant bandwidth constraint."* A brain holds a thought in
recurrent activity, without speaking it.

This repo tests the smallest version of the missing piece: a **J-space state
that lives outside the token stream**, carried by a small loop, the way the
WaveComputer / ApicalRuleReversalMachine / WaveCortex / TuftSheet line
carried a belief between events.

## The experiment

```
 HOLD (context 1)                          RECALL (context 2, a fresh one)
 "Silently think of the capital city of     a neutral paragraph about painting
  France. Don't say it."                    a fence ... "what were you thinking
 "Okay. I am holding it in my mind."        of earlier?"   "I was thinking of ___"
        │                                            ▲
        └── read the J-space ── held state ──────────┘  written into the residual
            (the 8 concepts the model                   stream at every token
             itself put there)
```

- **The answer is derived, never written.** "Paris" is not in either
  context. If the model says it, the thought came through the state.
- **The hold is not in the recall context,** so attention can't look it up.
- **Strict scoring.** Forced choice over every answer *and every cue word*.
  Saying "France" for the capital of France counts as wrong.

The routes compared, each swept over a fixed strength grid:

| route | what is carried |
|---|---|
| none | nothing |
| in_context | the hold conversation itself (ceiling: ordinary attention) |
| oracle | the answer word's lens vector (cheating; measures the write channel) |
| cue | the cue word's lens vector ("France"): is the cue enough? |
| snapshot | the raw hidden state at hold time |
| **jstate_open** | the model's own held concepts, written at constant strength |
| **jstate_closed** | the same concepts held by a residue loop: read the J-space a few layers up after every token, integrate the deficit, write only what's missing |
| jstate_closed_surprise | as closed, but quieter when the model's next-token surprise is high (the world is talking) |

The closed loop is this line's two-form unit moved into an LLM: content
(which concepts) is kept apart from drive (how hard to push). Drive is set
by the local residue, not by a fixed strength.

Because pushing harder always helps recall but also disturbs the model,
every route is compared **at matched damage**: the KL between the model's
next-token distributions with and without the injection.

## Status: protocol committed, real run pending

The workspace that built this couldn't reach Hugging Face, so **nothing has
been run on a real Qwen yet**. The protocol ([PROTOCOL.md](PROTOCOL.md)) was
pushed before any real-model run, with every constant fixed in advance and
predictions written down, including that **G3, the closed loop beating the
open one, will probably fail**.

What *has* been checked, on a random-weight Qwen-shaped model (`tests/`):

- the lens equals a brute-force finite-difference gradient;
- a write moves exactly the chosen reads by exactly the asked amount;
- a closed loop with zero gain is bit-identical to no injection;
- a saturated closed loop equals the open loop;
- the recall context never contains the hold;
- the full pipeline runs, and on random weights it stops at G0, as it
  should: random weights can't derive "Paris", but writing "Paris" directly
  still works (oracle 1.00).

## Run it

On a 12 GB GPU (an RTX 3060 is enough):

```
pip install -r requirements.txt
python run.py --model Qwen/Qwen2.5-1.5B-Instruct
python demo.py results/Qwen_Qwen2.5-1.5B-Instruct --model Qwen/Qwen2.5-1.5B-Instruct
python demo.py results/Qwen_Qwen2.5-1.5B-Instruct --model Qwen/Qwen2.5-1.5B-Instruct --desc "the colour of a ripe banana"
```

`run.py` computes the lens (a few minutes), reads all holds, chooses the
layers on dev items, runs every route on every item (estimated half an hour on a 3060; not yet timed),
and prints the pre-registered verdict. `demo.py` shows it in words: the
held state, then the model's free answer in a fresh context under each
route. `python -m pytest -q tests` reruns the plumbing checks offline.

## Gates

| | Gate | Rule (test items) |
|---|---|---|
| G0a | the workspace holds the derived thought | answer in the top 10 of the hold reading in ≥ 40%, specific in ≥ 80% |
| G0b | the write channel works | oracle recall ≥ 0.50 |
| G1 | a thought carried outside the tokens | J-state ≥ none + 0.20 |
| G2 | the thought, not the cue | J-state ≥ cue + 0.10 |
| G3 | the closed loop earns its place | closed ≥ open at every damage budget, + 0.10 at one |

If G0 fails, the rest isn't interpreted. That would mean a 1.5B model
doesn't hold the derived concept where the lens can see it, which is a
result in its own right.

## Honest edges

- **The thought is supplied as an instruction.** This is held imagination,
  not spontaneous imagination.
- **One-token concepts only,** and a hand-written concept list of about 300
  words, not the whole vocabulary.
- **The state is a fixed set of concepts.** The loop sets how hard to push,
  not what to hold. A state that revises *what* it holds from surprise, and
  carries an "imagined" tag so the model can tell it from perception, is
  the next step, not this one.
- **A small model.** The paper's results are for large production models.

## Where this sits

QualiaClaude → WaveComputer → ApicalRuleReversalMachine → WaveCortex →
SyntheticQualia → TuftSheet built a small machine that holds a belief
between events, carries its confidence, and revises it by surprise. This
repo asks whether the same move gives a real language model what the
J-space paper says it lacks: a thought held over time without speaking it.
