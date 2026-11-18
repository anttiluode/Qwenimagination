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

## Result: confirmed on fresh items (v2, 5 of 5 gates)

The first pre-registered run (v1) failed every gate, mostly through my
design. A diagnostic found the three faults. A second protocol
([CONFIRM_PROTOCOL.md](CONFIRM_PROTOCOL.md)) froze the fixes and was pushed
before running on **60 fresh items** the model knows, none seen while
designing. Qwen2.5-1.5B-Instruct, read layer 26, write layer 22
([rows](results/Qwen_Qwen2.5-1.5B-Instruct/confirm_v2/rows.jsonl)):

| route (released only at the question) | recall | KL on the paragraph |
|---|---|---|
| none | 0.02 | 0 |
| hold kept in the conversation (in-context) | 0.18 | – |
| **carried residue (primary)** | **0.30** | **0** |
| loudest 8 concepts (v1's selection) | 0.05 | 0 |
| the cue word ("France") | 0.00 (said the cue 44/60) | 0 |
| another item's residue (shuffled) | 0.00 | 0 |
| the answer word itself (oracle) | 0.82 | 0 |
| closed loop, written throughout | 0.05 | 0.08 |

| gate | difference (95% bootstrap CI) | |
|---|---|---|
| C1 carried vs none | +0.28 (+0.17..+0.40) | pass |
| C2 residue vs loudest concepts | +0.25 (+0.15..+0.37) | pass |
| C3 thought vs cue | +0.30 (+0.18..+0.42) | pass |
| C4 own vs another item's state | +0.30 (+0.18..+0.42) | pass |
| C5 carried vs in-context | +0.12 (−0.03..+0.27) | pass by its rule, but the CI includes 0 |

**What this shows.** A thought the model *derived* and never wrote ("the
capital city of …" → the city) can be lifted out of its workspace, kept
outside the tokens through an unrelated paragraph, and released when it's
asked for. The paragraph is untouched. The model then names it about a
third of the time, against 2% without the state and 0% with the bare cue.
The thought is the **residue of the workspace against its usual holding
state**: the J-space concepts that rose above what holding a thought
normally looks like. The loudest concepts are the act of holding itself
(*thought, imagine, picture, mind*), and they carry almost nothing.

**The state carries this item's content, not a general "remember" push.**
Another item's state never produced this item's answer, and in 20 of 60
cases it made the model name *the other item's* answer.

**Where it stops.** The answer is in the held bag 57/60 times, first in
18, top three in 31. The model says a word from the bag, but often a
neighbour or the cue: *cheese* comes back as *cheese*, but *pen* comes back
as *ink*, *lungs* as *chest*, *December* as *January*. Writing the answer
alone gets 0.82, so the gap between 0.30 and 0.82 is choosing the thought
from its neighbourhood. That's the bag-of-concepts limit of the J-space
paper, now measured. Holding fewer concepts (K = 1: 0.32) didn't fix it;
I predicted it would.

**Not earned:** the closed loop (0.05). What works is a still residue
released at the right moment, not a loop that keeps it alive.

**An observation, not tested:** 21 of 60 held bags contain *picture*,
*photographs* or *imagine* after the usual hold is subtracted. All 21 are
things you can picture (elephant, beach, rainbow, lava, queen, desert).
None of the abstract items (late, weak, south, January, gravity, lose)
have them.

Scope: one 1.5B model, one-token concepts, a supplied instruction to think
of something, a late layer (26 of 28, nearer "about to say" than the
paper's mid-network workspace).

## v3: picking the thought out of the bag (failed, 0 of 3 gates)

[PICK_PROTOCOL.md](PICK_PROTOCOL.md), committed before the run. Four rules
for choosing which held concepts are *the* thought were compared on 112
seen items, and the winner was tested on **50 fresh items**:

| rule | development (seen) | fresh items |
|---|---|---|
| A, v2: above the usual hold | 0.30 | **0.22** |
| C, inner: minus what merely reading the words evokes | 0.25 | 0.12 |
| M, tuft: A gated by "was this mine?" | 0.32 (chosen) | 0.18 |
| D, late: still growing in the last layers | 0.29 | 0.18 |
| none / cue / another item's state / in-context | | 0.02 / 0.00 / 0.02 / 0.08 |
| the answer itself (oracle) | | 0.68 |

- **No new rule beat v2.** The tuft rule won development by 0.02, which is
  noise. The protocol should have required a margin before replacing A.
  That's my error. On fresh items it was 0.04 worse (CI −0.10..+0.00).
- **"Inner minus outer" made things worse:** −0.10 (CI −0.20..−0.02). The
  risk named in the protocol happened. Just *reading* "the capital city of
  France" already evokes Paris, so subtracting the reading subtracts the
  answer. In this model, being asked to think adds the *mode* (picture,
  imagine) more than the content: the content arrives on reading.
- **v2's carried state replicated on a third item set:** A beat none by
  +0.20 (CI +0.08..+0.32), the cue by +0.22 and another item's state by
  +0.20, all with CIs clear of zero. It's lower than v2's 0.30. This item
  set has strong default answers: with no state the model said *never*,
  *always* or *above* in 50 of 50 cases, words that are answers or cues
  of the opposite-word items.
- **The bag problem is still there.** Under A the model said a word from
  the held bag in 31 of 50 cases, but the right one in 11.

**What this suggests:** a bag can't say which of its members is the answer.
*Paris* and *France* are both in it, and only the relation "capital of"
says which is the thought. A bag of concepts has no relations. Picking
the thought needs something that binds a concept to its role, the
multiplicative (c₃) term the earlier machines kept running into, and not
another way of ranking the bag.

## V5: write once, retrieve later (Astra; negative)

[DELAYED_RECALL.md](DELAYED_RECALL.md), [full rows](results/Qwen_Qwen2.5-1.5B-Instruct/delayed_v5/full.json).
Two items are written once, each on the colon after its label (`Alpha:`,
`Beta:`, layer 22). The model then reads a paragraph, the state is cached,
and the writer is off before either question exists. Assignments and label
order are crossed, so recency or item strength can't score. 29 pairs, 232
questions per arm:

| before the question | requested beats partner | requested ranked first |
|---|---|---|
| both words printed beside their labels | 214/232 | 176/232 |
| two derived thought writes | 117/232 | 0/232 |
| two answer-word writes (oracle) | 116/232 | 0/232 |
| nothing written | 116/232 | 0/232 |

The task is easy for Qwen when the words are in the text. Single-token
writes at layer 22 are not retrieved. Even the clean answer word fails,
so the loss is in the write-and-retrieve path, not in the derived thought.
The writes do reach later computation: target-vs-partner scores move by
about 0.025 logits per query on average. But the first word was the same
as with no write in all 232 queries. V2 still stands: there the state was
released *during* the answer, not stored for later retrieval.

**Next (one step at a time, one item, the answer word only):** reproduce
V2's write during the question; reduce it to one write at the answer
prefix; move that write to just before the question; then put the
paragraph between write and question. The first step that breaks shows
where retrieval fails. Only once a single item is recoverable does a
second item test binding.

## V4: role-bound working space (built, not run)

*Status note (Nov 18):* not run on Qwen, and as built it couldn't test
binding. The script picks the role key for the asked question and unbinds
the bag **before** injecting, so Qwen only ever receives one bag. The
binder also barely changes the bag (in a simulation at this vocabulary
size, the decoded top 8 overlap the original by 0.97). Kept as an
engineering baseline. V5 below is the version where Qwen has to do the
selection itself.

**Question:** Can the small neural state that carried *one* derived answer
also carry **two distinct answers with recoverable roles**?

V3 showed that a bag of concepts is not enough to identify what each
concept *means* in a relationship. The v4 experiment implements a **single
non-text memory vector** with two role addresses, *first* and *second*.
It stores two separately derived J-space residues, then asks Qwen for one
of the two after a fresh, unrelated context.

- **Bound**: recover the requested role from one packed vector.
- **Unbound**: simply add both bags, losing which is which.
- **Wrong role**: deliberately retrieve the other thought.
- **Single state, no state, cue, oracle, in-context**: honest controls.
- **Fork**: an isolated, externally supplied hypothetical update with an
  explicitly marked *imagined* source (unit-tested plumbing, **not**
  learned imagination).

These role transformations are engineered by us (orthogonal
signed-permutation/Hadamard maps). They are *not* Qwen learning relations,
and no real Qwen v4 outcome has been measured yet. All v1-v3 results
below are preserved; the v4 gate thresholds are fixed in
[ROLE_PROTOCOL.md](ROLE_PROTOCOL.md).

**Run on the RTX 3060 (after a quick smoke check):**

    python3.13 -m unittest discover -s tests -p "test_role_memory.py" -v
    python3.13 role.py --model Qwen/Qwen2.5-1.5B-Instruct --max-pairs 2
    python3.13 role.py --model Qwen/Qwen2.5-1.5B-Instruct

The first run extends and caches the lens; full results go to
`results/Qwen_Qwen2.5-1.5B-Instruct/roles_v4/`.
A smoke run is **not** confirmation.

The larger program is mapped in
[INNER_WORLD_ROADMAP.md](INNER_WORLD_ROADMAP.md): learned relational roles,
one world-state shared by perception and simulation, source attribution,
counterfactual action, and eventually self-directed internal inquiry.

## The experiment (v2, as confirmed)

```
 HOLD (context 1)                          RECALL (context 2, a fresh one)
 "Silently think of the capital city of     a neutral paragraph about painting
  France. Don't say it."                    a fence ...  │  "what was the answer you
 "Okay. I am holding it in my mind."                     │   were silently holding?"
        │                                   nothing      │  "The answer was ___"
        │                                   written here ▼
        └─ read J-space, layer 26                  release the held state
           minus the usual hold  ─── held ───►  into layer 22, from the
           = Paris, France, London, ...            question onward
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

## History: the first pre-registered run (v1) failed 0 of 5 gates

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
   *brain* (47), *picture* (44), *memory*. Five of those never appear in the
   prompt (thought, imagine, brain, picture, memory). The answer was in the held eight in only 7/38. It was there,
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

### Diagnostic result, and the confirmation run

On v1 items (so nothing confirmed): subtracting the usual hold put the
answer in the top 10 of the hold reading in **37/38** at layer 26 (14/38
without it). The answer is in the held bag almost always (top 3 in
34/38). The model says a word from the bag, but often a neighbour
("Tokyo" for Cairo) or the cue. Recall went from 0.08–0.11 to 0.32–0.37.
Released only at the question, it costs the paragraph nothing (KL 0.00).
With the hold in context and the new question, the model scored 0.21: the
state carried from outside beat the model re-deriving it in context.

`confirm.py` tests this on 72 fresh items with every choice frozen in
[CONFIRM_PROTOCOL.md](CONFIRM_PROTOCOL.md), pushed before it ran.

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
`python confirm.py --model Qwen/Qwen2.5-1.5B-Instruct` runs the v2
confirmation on fresh items (a few minutes of lens, then the items).
`python pick.py --model Qwen/Qwen2.5-1.5B-Instruct` runs v3, picking the
thought out of the bag ([PICK_PROTOCOL.md](PICK_PROTOCOL.md)).

## v1 gates (for the record)

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
