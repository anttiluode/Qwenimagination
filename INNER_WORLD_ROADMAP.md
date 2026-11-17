# InnerWorld — the research target beyond carrying a concept

The purpose is **not** a visualizer that makes a model *look* thoughtful.

The target is an agent that maintains an internal causal scene, uses the same
state for perception and imagination, can test alternative actions without
touching the outside world, and knows which of its beliefs came from observation,
inference, or its own simulation.

It must earn every step against matched simpler alternatives. No experiment
here establishes phenomenal consciousness.

## 0. What Qwenimagination actually established

- V1 was negative (0/5 gates).
- V2: 30% derived-concept recall vs 2% none on 60 new items, fixed instructions,
  carried outside the token stream. Specific shuffling changes *which* answer
  the model gives, not just whether it remembers. The nominal in-context win
  has a CI overlapping zero.
- V3: the v2 transfer replicated at 22% on 50 further items; the proposed
  multiplicative selector did **not** improve it (18%). No working closed loop.
- The held state is a sparse **concept cloud**, not a learned graph of
  relations. The first obstacle is selecting the correct role of a concept.

## 1. V4 — role-addressable workspace (this branch)

Two independently derived answer clouds, stored in a single compressed
external vector. Query role first or second. Controls: no state, unbound
mixture, wrong role, single corresponding bag, cue, oracle, in-context.
The role maps are externally engineered, and a test success would establish
**a usable interface**, not a learned neural role-binding mechanism.

## 2. V5 — learned roles, not keys supplied by us

Replace fixed random orthogonal role codes with a small learned read/write
interface trained to answer queries such as:

    first[object=ball, position=left]
    second[object=box, position=right]

Supply training only on **some** combinations. Test newly recombined
objects, roles, sources, and contexts. The exact bindings cannot be looked
up as memorized pairs.

**Falsifiers:** exchange role labels, randomly rotate role vectors, damage
half the connections, compare with a structured dictionary and a text
scratchpad at matched capacity. If a trivial dictionary wins, report it.

## 3. V6 — persistent world model

Generate a tiny offline world with two moving objects, one occluder,
camera movements, and occasional acoustic contact events.

The model must estimate latent positions and velocities from sparse
observations, survive the objects passing behind an occluder, and predict
what it would see or hear later. Give separate results for:

- no sensory input (pure imagination rollout)
- missing only one modality (cross-sense update)
- a genuinely changed outside world (prediction correction)
- a corrupted sensor (source reliability)
- a counterfactual camera/action intervention

Evaluate pixel/position trajectory error, event calibration, hidden object
identity, and ability to generalize to longer occlusions. Compare against
scalar Kalman filtering, recurrent GRU, explicit object-slot state, and
a small transformer with textual history.

**A crucial requirement:** perception and imagination must share the SAME
learned latent state; two separate lookup tables don't count.

## 4. V7 — belief provenance / internal vs external

Every workspace claim gets at least a *functional* provenance variable:
observed, inferred, imagined, or uncertain channel. This is more than an
application flag. Have the agent **infer** source reliability by comparing
independent channels and temporal consistency.

For a controlled test, corrupt vision but not sound. A good model should
infer that vision is unreliable without rewriting the stable outside world
to match the corrupted signal.

Compare: no provenance, explicit provenance supplied by us, provenance
learned from cross-modal evidence. Measure false-world-updating rates.

The migraine analogy motivated this; none of this is evidence of an
actual aura mechanism.

## 5. V8 — self-directed internal inquiry

Let a bounded workspace ask its own next question: what would distinguish
two currently plausible latent worlds? A good agent chooses actions or
simulations that reduce uncertainty and improve future predictions.

The essential control is random internal simulation at matched compute.
A second control is prompt-chain reasoning with an explicit text scratchpad.

**Breakthrough criterion:** the agent chooses useful new hypothetical tests,
produces predictions that distinguish competing causes, and wins on new
worlds it has never seen, without getting its next question from a human.

## Architectural sketch (proposed, not demonstrated)

    sensors + actions -> evidence -> distributed latent world W(t)
                                          |
                          perception -- selective read -- workspace Q(t)
                                          |                 |
                                   provenance            choice
                                          |                 |
                            imagined future W'(t+1) <- action proposal
                                          |
                        compare branches, return only needed concepts

Qwen can serve as the linguistic interface to inspect or query a world
model, but should not be allowed to solve the benchmark by reading the
original answer from a natural-language memory store.

## Compute discipline

Keep a CPU numerical pilot and paired controls for every architecture.
Use the RTX 3060 only for tests whose code, gates, and seeds are pinned.
Retain failures; do not turn exploratory repairs into preregistration.
Before running an expensive confirmation, require the direct/oracle
readout to be possible in that exact response format.

## Important limits of the J-space analogy

Anthropic's J-space is a measured subset of verbalizable, broadcast
transformer representations; it is not itself a recurrent world simulator.
A continuous state can be an *external companion* to the token-based
workspace, but the role-selective interface and any claimed cognitive
advantage require causal and baseline tests.

**Large ambition, narrow claims at each stage.**
