# Qwenimagination

**A frozen Qwen model produces a short list of concepts that carries information about the answer to a clue.** An external script keeps that list across contexts. On two fixed-choice datasets, reading the list directly recovered more answers than writing it back into Qwen. A clean answer-word write was effective at the final answer position. Writing the same direction before the question produced a small score change and no change to any top choice in the 60-item timing test.

The project uses the J-space lens from Gurnee, Sofroniew, Lindsey et al., *Verbalizable Representations Form a Global Workspace in Language Models* (2026). The WaveCortex and TuftSheet projects motivated the longer-term question of whether a model can use a silently supplied state later. The experiments below distinguish external storage, direct readout, and retrieval by Qwen.

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

**What V2 establishes.** The script reads eight concept directions that rose above Qwen's usual holding state and stores their weights externally. Injecting them during the later question raised forced-choice accuracy from 1/60 to 18/60. The hold text itself was absent from the recall context. The paragraph received no intervention by design.

**The state carries this item's content, not a general "remember" push.**
Another item's state never produced this item's answer, and in 20 of 60
cases it made the model name *the other item's* answer.

**Where it stops.** The answer is in the held bag 57/60 times, first in
18, top three in 31. The model says a word from the bag, but often a
neighbour or the cue: *cheese* comes back as *cheese*, but *pen* comes back
as *ink*, *lungs* as *chest*, *December* as *January*. Writing the answer
alone gets 49/60. Selection and readback can both contribute to this gap.
Using one held concept (K = 1) gave 0.32.

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

## Direct reader comparison (retrospective)

The reader takes the first word in a saved ranked bag that belongs to the original forced-choice vocabulary (all eligible answer **and cue** words for that dataset). It never sees the clue or the correct label. An empty intersection counts as wrong. The evaluation uses the same items as the model readback. The V3 comparison uses the V2-style A_usual arm.

| Readout | V2 (60 items) | V3 (50 fresh items) |
|---|---:|---:|
| First word in the bag, unrestricted | 18/60 | 10/50 |
| Qwen after injecting the bag during the question | 18/60 | 11/50 |
| First eligible word in the bag | **29/60** | **25/50** |

For the wool clue, the list began *rabbit, sheep*. Sheep was the first eligible choice. After injection Qwen ranked *brown* first. The simple reader benefits from the known candidate list, just as Qwen's scorer does. This is retrospective evidence for item information in the extracted bag and a limitation of the readback method. It leaves open the role of binding and performance on open-ended questions.

[Recompute the comparison](compare_readers.py) · [V2 rows](results/Qwen_Qwen2.5-1.5B-Instruct/confirm_v2/rows.jsonl) · [V3 rows](results/Qwen_Qwen2.5-1.5B-Instruct/pick_v3/rows.jsonl).

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

## V6: timing a clean answer-word write

The replay reproduced **all 60 original V2 oracle choices**, including the 11 errors. The subsequent conditions used an otherwise identical marked prompt. These were clean answer-word directions supplied by the experimenter. V2's derived-bag result is a separate measurement.

| Layer-22 write schedule | Exact choices | Mean target log-probability gain over marked no-write |
|---|---:|---:|
| Original V2 oracle replay | 49/60 | — (original prompt) |
| Repeated, question through final answer-prefix token | 48/60 | +7.819 |
| Final answer-prefix token only | 46/60 | +7.650 |
| Marker immediately before question | 2/60 | +0.062 |
| Marker before paragraph | 2/60 | +0.026 |
| No write | 2/60 | 0 |

Moving the single write 25 tokens upstream cost 44 correct choices (paired difference −73.3 percentage points, 95% bootstrap interval −85.0 to −61.7). For both upstream positions the target score improved on all 60 items, yet every top choice matched the no-write condition. Position, surrounding tokens and distance change together. The result applies to this layer, direction, and task.

[Timing protocol](TIMING.md) · [Full rows](results/Qwen_Qwen2.5-1.5B-Instruct/timing_v6/full.json).

## V7: can a later question use the carried content? (inconclusive)

Write once before the question, then ask for the item's category. The meanings of answer letters A and B reverse between questions asked from copies of the same stored state. The derived bag, swapped bag, empty slot, printed word, native-token patch, and an external classifier are compared on the same 40 evaluation items. The classifier is trained on 20 other items.

**Result: every Qwen arm scored exactly 0.50, and the test never got started.** Qwen answered "A" in all 400 decisions, in every arm, by 1 to 6 logits, including when the word was printed in the prompt. Reversing the letter meaning turns "always A" into exactly 50%. With the letter bias removed (each pair compared under both mappings), the printed word reaches only about 0.60. The protocol's first check, that the printed word can do the task, failed, so the activation arms say nothing either way. The external reader on the same bags got **0.91**.

| Arm | Accuracy | Mean A−B lean | Pairs in the right direction (bias removed) |
|---|---|---|---|
| Printed word | 0.50 | 2.74 | 11/20 |
| Real-token patch | 0.50 | 3.27 | 11/20 |
| Derived write | 0.50 | 4.08 | 15/20 (95% CI touches 0) |
| No write | 0.50 | 4.14 | – |
| Reader on the bag | **0.91** | – | – |

[Protocol](CATEGORY.md) · [Full rows](results/Qwen_Qwen2.5-1.5B-Instruct/category_v7/full.json).

## V8: the same question, with the format checked first (negative)

V8 picks the question format on the 20 training items using the printed word, and stops unless it reaches 0.85. Only then does it choose the write (layer, strength, answer position or whole question) on the same training items, and test the frozen choice on V7's 40 held-out items. Every answer is calibrated against the neutral prompt with no write, so a fixed lean can't score.

**The format check passed, so this is a real result.** Yes/no ("a living thing?" / "made by people?") reached 0.95 with the printed word on training items and 1.00 on the test items. No write passed: 23 of 24 write settings scored 0.50 on training, and the chosen one (layer 14, strength 1, over the question) scored 0.50 on test.

| Arm (40 held-out items) | Accuracy | Living minus manufactured, within pair |
|---|---|---|
| Printed word | **1.00** | +11.2 (20/20 pairs right) |
| Reader on the bag | **0.91** | – |
| Derived thought written in | 0.50 | +0.15 (8/20, CI −0.39..+0.76) |
| Same weights, random directions | 0.50 | −0.16 (9/20) |
| The item word's own direction | 0.61 | −0.38 (4/8) |

Every write pushed every item toward "living" by 1–7 units, whatever it carried: a generic disturbance, not content. Even writing the clean direction for *chair* moved Qwen the same way as *wolf*.

**What this means.** J-lens directions are "about to say this word" directions. Written at the answer they make Qwen say the word (V2, V6), and that is all they do: they don't make Qwen *understand* the word well enough to answer a different question about it. The read holds (a classifier gets the category from the bag 91% of the time); the lens write is a speech lever, not a thought.

[Protocol and predictions](CATEGORY_V8.md) · [Full rows](results/Qwen_Qwen2.5-1.5B-Instruct/category_v8/full.json).

## Rereading the paper: what V2–V8 actually tested

Gurnee et al. validate a different intervention from the one these experiments used. They **swap** two lens coordinates, **clamped** at every position across the **workspace band** (from about a third of the way in to shortly before the output), on a concept that is **already present**. Their two-hop swaps (spider → ant turns "8 legs" into "6") work 54–70% of the time. The last few layers are "motor": lens content there drives the imminent output and isn't available for further processing (§4.1). Content travels between tokens through broadcast heads in the first half of the band (§4.3.2).

Our read was at layer 26 of 28 and our writes were single additions, mostly at layer 22, into an empty slot. Seen that way:

- **V2** reproduces the paper's injected-thought result: late "about to say" content, written at the answer, gets said.
- **V5/V6**: a single write at layer 22 sits after the broadcast heads, so later tokens don't fetch it.
- **V8**: the paper's two-hop test, done with a weaker method. Even the clean word direction failed.

None of V2–V8 tested the paper's method on Qwen. V9 does.

## V9: the paper's swap on Qwen 1.5B (gate passed)

Swap two lens coordinates, clamped at every position across a band of layers, on a concept already in the prompt: "Item: wolf", then "is it a living thing?" / "made by people?". Cross swap: *wolf* ↔ *chair*. Control: *wolf* ↔ *lion*, same category. Band and strength chosen on 9 training pairs; tested on V8's 20 held-out pairs.

**Result: swapping the concept in layers 4–12 changes how Qwen answers a different question about it.** Gate S1 passed: cross flips 0.35, same-category flips 0.10, difference +0.25 (95% CI +0.15..+0.35).

| On held-out items | Living → manufactured partner | Manufactured → living partner |
|---|---|---|
| Answer flipped (of items answered right before the swap) | **9/15** (wolf → chair: +2.6 → −9.4) | 0/20 |
| Moved toward the partner's category | 18/20 | 18/20 |
| Same-category control, moved the same way | 11/20 (mean −0.2) | 14/20 (mean +0.5) |

Manufactured items start far more certain (about −10 against +1..+6 for living), so they move toward "living" (bicycle −8.5 → −2.1) but don't cross. Items that flipped had higher workspace loading (28.6 against 20.2), as the paper reports for its swaps.

**Where it works.** Every band that includes layers 4–12 moved answers. Bands starting at 12 or later (12–20, 16–24, 20–27) did nothing at all: by then the category has already been computed from "wolf". The layer signatures agree: concepts start persisting across tokens around layer 9, and from about layer 21 the lens increasingly just predicts the next token (output agreement 0.35 at layer 20, 0.74 at layer 26).

**What this changes.** Qwen 1.5B does have a usable workspace route, early in the network. Our read (layer 26) and writes (mostly 22) sat in its "mouth"; this run shows the "room" is around layers 4–12, and the paper's swap, not a single added write, is what steers it.

[Protocol](SWAP_V9.md) · [Full rows](results/Qwen_Qwen2.5-1.5B-Instruct/swap_v9/full.json).

## V10: Qwen's own silent thought, swapped in (negative, and it didn't carry the thought)

V9's swap, with Qwen's thought read in layers 4–12 of the hold in place of a word. Strength and bag size chosen on 9 training pairs (K = 8, α = 2), tested on 20 held-out pairs. **Both gates failed.**

| Held-out, swapped into a slot of the other category | Answers flipped | Mean shift away from the slot's category |
|---|---|---|
| Qwen's thought | 0.55 | +6.2 |
| The true word (V9's control) | 0.48 | +4.3 |
| Same weights, random directions | 0.48 | +5.5 |
| Qwen's thought into a same-category slot | 0.38 | +3.7 |

Thought − same-category: +0.18 (CI −0.03..+0.35). Thought − random: +0.08 (CI −0.13..+0.28). At α = 2 any direction breaks the answer, so random directions flip as often as the real word. α = 2 won stage 1 on 9 pairs by noise.

**The thought was not in what we carried.** The bags read at layers 4–12 are the same words for every clue (*tail, road, truck, pizza, box, small*); the item's own word was in the top 8 for only 2–6 of 40 items at any of those layers. The v2 diagnostic had already measured where the thought appears: the answer is in the top 10 of the hold reading (minus the usual hold) for 4–8 of 38 items at layers 1–16, rising from layer 17, and 37 of 38 at layers 26–27.

**The picture.** In this 1.5B model the place where a concept can be swapped in and used (V9: layers 4–12, at the word itself) comes before the place where a silently derived thought becomes readable (layers 17–27). The thought arrives after the room has closed.

[Protocol](THOUGHT_V10.md) · [Full rows](results/Qwen_Qwen2.5-1.5B-Instruct/thought_v10/full.json).

## V11: read the thought in the mouth, install it in the room (ready to run)

The thought is readable late (layer 26); a concept can steer a later question only early (layers 4–12). V11 takes the concepts and weights from the layer-26 read and swaps them in by their own layer-4–12 directions. Nothing is selected on training pairs: α = 1 and the top concept (K = 1) are fixed in advance; the whole bag (K = 8) is reported alongside.

    python3.13 bridge_v11.py

[Protocol and predictions](BRIDGE_V11.md).

## V4: role-bound working space (built, not run)

*Status note (Nov 18):* not run on Qwen, and as built it couldn't test
binding. The script picks the role key for the asked question and unbinds
the bag **before** injecting, so Qwen only ever receives one bag. The
binder also barely changes the bag (in a simulation at this vocabulary
size, the decoded top 8 overlap the original by 0.97). Kept as an
engineering baseline. V5 above is the version where Qwen has to do the
selection itself.

**Question:** Can the small neural state that carried *one* derived answer
also carry **two distinct answers with recoverable roles**?

V3 motivated an exploratory slot-binding design. The v4 experiment implements a **single
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

- **The answer is omitted from the hold text.** "Paris" is not in either
  context. The no-write and shuffled controls measure how much the item-specific state helps.
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

## History: the first pre-registered run (v1) passed 0 of 5 gates

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

- **The instruction asks the model to think of a specified item.** The measurement is a concept read during the fixed reply.
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
