# When does a one-item write stop helping?

This diagnostic uses the 60 known V2 items, its original candidate set, the V5 cached V2 lens, and V2's answer-word direction at layer 22. Each item receives its own V2 oracle strength at every selected position. The V2 oracle achieved 49/60 exact forced choices. Check the replay against the published individual choices first. This measures the write path with a supplied answer. Derived content and two-slot selection are later questions.

    python3.13 recall_timing.py

Use --max-items 2 for a smoke run. The full output goes to results/Qwen_Qwen2.5-1.5B-Instruct/timing_v6/full.json.

First replay the original V2 prompt and release schedule. Then place identical Note: markers at each end of the original filler. Hold this marked prompt fixed for five conditions.

| Condition | Positions receiving the same answer-word vector |
|---|---|
| none | No positions |
| repeated | Every token from the first question word through the final answer-prefix token |
| answer | Final answer-prefix token only |
| before_question | Colon token after the paragraph |
| before_gap | Identical colon token before the paragraph |

The marked repeated arm bridges the old and new prompts. Compare it with the replay and with matched none before interpreting the ladder. Compare answer with repeated to measure the effect of removing all earlier writes while keeping the final one. Compare before_question with answer to test moving one write upstream into a different token context. Compare before_gap with before_question to test the additional intervening paragraph on matching marker tokens. Delay, absolute position and surrounding context still covary. The repeated arm also delivers more total intervention. These are timing and placement comparisons within this setup.

Report exact accuracy, candidate-set target log probability, and item-paired accuracy differences with bootstrap intervals. Read a gain in score without an accuracy gain as weak influence. If the positive control fails to reproduce, investigate that before assigning a memory mechanism. A durable single-item effect merits confirmation on fresh items before adding roles. This is an exploratory localization on previously used items.
