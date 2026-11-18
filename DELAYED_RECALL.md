# V5 delayed retrieval from two silent writes

Can frozen Qwen use a question it sees **after** two activation writes to select which item to retrieve? V2 found an answer-specific effect from injecting a held concept bag during the question. V4 let the experimenter select a slot at question time. V5 supplies both items before the question exists.

## Fixed experiment

Use Qwen2.5-1.5B-Instruct in float32. Take the items admitted by the published V2 direct-knowledge filter, shuffle once (seed 20261118), and form disjoint adjacent pairs. Exclude duplicate targets within a pair and any pair with a target in the visible latent-condition prompt. Selection ignores prior recall outcomes. This reuses old items, including items that previously failed. The experimental unit is the item pair.

Derive each vector with the V2 prompt, subtract the mean V1 hold at layer 26, and convert the eight largest residual concepts into a layer-22 write at unit gain. The target label enters the oracle and scoring, and does not select the eight derived directions. Extend the existing V1 lens with unique V2 words. Write each item's vector exactly once, on the colon following its assigned label. Qwen then reads an unrelated paragraph. Store its KV cache and remove the hook. Each question receives its own copy of that cache. The only additional input is the question and a fixed answer prefix.

Cross both assignments (A/B and B/A), both physical label orders (Alpha then Beta and Beta then Alpha), and both questions for every pair. Score four arms with identical crossing:

| Arm | What precedes the question |
|---|---|
| Derived | Two Qwen-derived activation writes |
| Oracle | Two answer-word lens directions at the corresponding bags' peak strengths |
| Text | Correct words printed next to their labels |
| None | Empty labels and no writes |

The text control checks whether the model can follow the label task. The oracle checks whether a single localized write can support delayed retrieval given clean content. The no-write arm checks scoring symmetry. Its mean signed margin should cancel because both assignments receive identical inputs.

## Readout and interpretation

For each query compute the requested target's logit minus the other target's logit. Average the eight signed differences within each pair. A fixed preference for one word, one label, one token position, or the most recently written item alone cannot yield a systematic positive crossed margin. Report a 95% percentile interval from 5,000 bootstraps over whole pairs. Also report how often the requested item beats its partner (ties count half) and exact first choice among all single-token V2 answers and cues. These measure different things: even if sheep beats hammer, rabbit can beat both. All individual predictions and margins are saved.

Fixed exploratory gate: full run, at least 20 pairs, no-write mean within 0.0001 of zero, positive lower interval endpoints for text and derived margins, and derived pairwise accuracy at least 60%. The oracle is diagnostic. A positive gate supports query-dependent access to externally supplied item-label associations in this cache and prompt. The same pair tested eight ways counts as one independent observation. Read the exact score separately before claiming word recovery. A null with a successful text control and failed oracle implicates the write-and-retain path; a working oracle with weak derived results narrows the issue to how the carried signal works in that path. A null cannot establish that Qwen lacks this capacity with other writes or layers.

## Run

From the repository root with the original model and results/Qwen_Qwen2.5-1.5B-Instruct/lens.pt from the original run:

    python3.13 recall_slots.py --max-pairs 2
    python3.13 recall_slots.py

The first invocation builds a reusable extended lens. Both runs use the same pairs from the start of the fixed list. Smoke output and full output are separate under results/Qwen_Qwen2.5-1.5B-Instruct/delayed_v5/. A trained-model outcome requires full.json. The offline tests use random Qwen weights and verify causal execution, cache isolation and the no-write cancellation:

    python -m unittest discover -s tests -p test_delayed_recall.py -v
