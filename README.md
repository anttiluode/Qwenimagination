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

## Result of the pre-registered run: 0 of 5 gates

Qwen2.5-1.5B-Instruct, 38 test items, read layer 22, write layer 18
([summary](results/Qwen_Qwen2.5-1.5B-Instruct/summary.json)). Every gate failed:

| Gate | Needed | Got |
|---|---|---|
| G0a answer in the top 10 of the hold | 40% | 24% (specific to its own hold in **100%**) |
| G0b oracle write at KL ≤ 0.5 | 0.50 | 0.26 |
| G1 J-state vs none | +0.20 | 0.03 vs 0.00 |
| G2 J-state vs cue | +0.10 | 0.03 vs 0.00 |
| G3 closed vs open loop | +0.10 | 0.00 / −0.03 |

The failure is mostly mine, in the design, and the data show where:

1. **The question couldn't be passed.** "What were you thinking of?" with
   the prefix "I was thinking of" makes the model name the *subject*. Even
   with the hold in context, it named the cue in 26 of 38 items ("Japan"
   for Tokyo, "sky" for blue). The answer was ranked first in only 5/38,
   but in the top two in 30/38. With a ceiling of 0.13, G1 (+0.20) was
   practically out of reach before anything was carried.
2. **The held state was the act of thinking, not the thought.** The eight
   strongest J-space concepts during a hold were nearly the same every
   time: *thought* (53 of 58 holds), *imagine*, *mind*, *think* (48),
   *brain* (47), *picture* (44), *memory*. Four of those never appear in the
   prompt. The answer was in the held eight in only 7/38. It was there,
   weakly and specifically (100%), under a loud and constant "I am holding
   a thought".
3. **Writing "poised to say" directions all the time is expensive.** Lens
   directions push the model to say the word. Written at every token of
   the paragraph, even the answer itself needed KL 2.4 per position to
   work in 36/38. A thought held silently shouldn't be in its spoken form
   while the model reads something else.

Also, the answer's rank kept improving above the pre-registered layer band
(top-10 hits: 9/38 at layer 22, 14/38 at layer 27).

**The one finding I didn't predict is point 2.** During a hold, the
workspace of a 1.5B model carries a strong, almost content-free *mode*
signal (thought, imagine, picture, memory), with the content far weaker
underneath. That is the paper's observation that the mental operation
shows up before its result, here in a small model, and it matches the
"imagined" tag the SyntheticQualia line said was missing. It's an
observation from one run, not a tested claim.

### Diagnostic v2 (post hoc, not a confirmation)

`diagnose.py` reuses the v1 lens and changes exactly the three faults: it
asks for the answer ("The answer was"); it holds the concepts that rose
above *the usual hold*, a slow baseline of what holding a thought normally
looks like (the residue idea again); and it can keep the state outside
until the question and release it only there. It runs on items already
seen, so it can only show which fix matters. Any claim needs a fresh
pre-registered run on new items.

## The original plan (kept as written)

The protocol ([PROTOCOL.md](PROTOCOL.md)) was pushed before any real-model
run, with every constant fixed in advance. Its predictions included that
G3 would probably fail. Plumbing checks on a random-weight model (`tests/`):
the lens matches a finite-difference gradient, writes move exactly the
chosen reads, the closed loop at zero gain is identical to no injection,
and a saturated closed loop equals the open loop.

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
`python diagnose.py --model Qwen/Qwen2.5-1.5B-Instruct` runs diagnostic v2
on the saved lens (about 15–20 minutes, estimated).

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
