# V4 — Relational working space: two simultaneous silent thoughts

**Status: code + protocol only. No real-model results yet.** This is an
exploratory, fixed-parameter extension of the earlier v2/v3 results, not
another claimed confirmation. The original v1/v2/v3 data and negative
results remain intact.

## Why

V2 transferred some specific concept information through a fresh context:
30% recall (60 fresh items), replicated at 22% (50 further fresh items).
V3's attempt to choose the correct element from the bag by gating *failed*.
The missing piece could be a relation: this concept plays *this role*.

The first falsifiable step is **not** a full world simulator: ask whether
two different Qwen-derived concept clouds can coexist in one compact
external non-text state and be separately recovered by querying their role.

## Frozen setup

- **Model:** Qwen/Qwen2.5-1.5B-Instruct, float32, frozen; read layer **26**,
  write layer **22**; J-lens extended using the original calibration corpus.
- **Items:** all eligible items from previously used ITEMS_V2 + ITEMS_V3,
  excluding target words present in their own prompt or the distractor
  paragraphs, non-single-token target/cue words, and items failing direct
  answer knowledge on this exact model. These are **old items**.
- **Pairs:** deterministic shuffle with seed **20261118**, disjoint pairs
  of eligible known items; exclude cross-description target leakage.
  Evaluate up to **30 pairs × two roles = 60 role-specific queries**.
  These are previously untested *combinations*, not new concepts.
- **Hold:** individually query each description with the original silent
  hold prompt and fixed hold reply; no target answer is generated as text.
  A v2-style 8-concept positive residue is the hold reading (layer 26)
  minus the mean historical hold over v1 items.
- **Memory:** two sparse held residues, packed into **ONE numeric vector**.
  Role A ("first") and B ("second") use fixed independent signed-permutation
  orthogonal Walsh-Hadamard operators, with seed **20261118**. Unbinding
  incurs cross-talk. The decoded top 8 positive concepts are injected
  at layer 22 **only from the role-specific question onward**.
- **Comparison:** same one-word answer prefix across arms; concept-vector
  magnitudes normalized to the queried single-state bag. Measure mean
  KL disturbance at the question and exact forced-choice answer, including
  distractor cue words. This is not full free-form semantic accuracy.
- **Baselines:** none; single corresponding bag (capacity upper bound);
  two bags simply added without role keys (unbound); wrong role decoded;
  cue-only; answer-word oracle; both original holds in the token context.
- **No learned parameters** in the binding layer, no fine-tuning, no
  selection of winners on evaluation outcomes.
- **Negative controls:** (a) wrong role and (b) unbound mixture. Require
  specific restoration of target, not generic topic or "remember" nudge.

## Gates: pre-specified exploratory criteria

These gates only apply if at least **20 pairs** survive knowledge and
leakage filters. A **smoke run** with --max-pairs below 30 is not confirmation.

| Gate | Criterion over role-specific queries |
|---|---|
| R1 Carrying | bound >= none + **0.15** |
| R2 Binding | bound >= unbound + **0.08** |
| R3 Role discrimination | bound >= wrong-role + **0.10** |
| R4 Capacity | bound >= single-slot - **0.10** |
| R5 Role reversal control | wrong-role answer is the *other* item's target in >= **15%** of queries |

Also report 95% bootstrap CIs by **pair**, not by treating two queries
from one pair as independent. A procedural gate passing is not the same
as a statistically clear effect if its interval includes zero.

### Meaning of a success

- Qwen's *previously derived*, task-specific latent content can be carried
  through a packed, externally engineered role-dependent memory interface
  and later influence a fresh-context answer appropriately.
- The system would have a usable **role-specific access interface**, not
  merely a list of simultaneous concepts.

### Meaning of a failure

- Crosstalk destroys or obscures weak concept signals, or Qwen cannot
  consistently use a retrieved concept bag when asked for a specific role.
- The single-slot result may remain real; failure would not refute v2.

## Crucial limits

This is not general imagination. The two roles and the binder are
hand-coded, not discovered by Qwen; the slots contain **bag-like concept
signals**, not learned causal relationships (e.g., "Paris IS-CAPITAL-OF
France"). The imagined branch in qimag/role_memory.py is supplied as an
external delta—it does not come from a learned world transition. A genuine
dynamic workspace must eventually *generate, revise, and verify* such
transitions, maintain source attribution (observed/inferred/imagined),
and beat scalar/text and fixed-memory baselines.

The original paper's J-space is verbalizable activation directions. A
successful v4 would not prove subjective experience, consciousness or a
neural substrate of physical imagination.

## Running

On the Windows machine with Qwen weights already downloaded:

    python3.13 -m unittest tests.test_role_memory -v
    python3.13 role.py --model Qwen/Qwen2.5-1.5B-Instruct --max-pairs 2
    python3.13 role.py --model Qwen/Qwen2.5-1.5B-Instruct

The first role run computes and caches an **extended lens** under
results/Qwen_Qwen2.5-1.5B-Instruct/roles_v4/lens_extended.pt. The full
run writes rows.jsonl and summary.json alongside that file. Do not
interpret the smoke run or tune after inspecting full-run data and call it
the original confirmation.

## The next real ambition (not claimed here)

1. A **learned** relation/role/filler representation that must generalize
   to recombinations never shown during training.
2. A persistent **world state**, not just a concept cloud, supporting
   multiple objects under occlusion and audio/visual common cause.
3. An **offline transition operator**: choose actions, advance that same
   representation, compare branches, and causally test predictions.
4. A **source-of-belief mechanism** distinguishing observed, inferred,
   imagined and channel-fault states; wrong-source and swapped-role
   interventions must produce distinctive failures.
5. A **selective workspace interface**: read different relations and
   explain choices without exposing the entire scene as text.

The night-shift contribution tests step 1's **interface prerequisite**,
not step 1's learned relation itself.
