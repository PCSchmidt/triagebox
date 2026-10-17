# Prompt tuning for the 1.5B models, and error bars (milestone 0.7b)

All runs: dev split (100 answerable + 100 unanswerable), BM25 top 4, temperature 0, seed 0, accuracy only (no timing). Test split untouched.

## Why the 1.5B models scored so badly

The ladder ([ladder.md](ladder.md)) showed 1.5B Q8_0 answering 49% of answerable questions correctly and 1.5B Q4_K_M only 28%, with 32% and 61% of answerable questions ending as refusals at system level. Reading the replies showed the "refusals" were mostly not refusals. The 1.5B models copied the prompt's example id (`doc1#3`) or put `doc` in front of a real id (`doc19p5#0` for `a19p5#0`), so the citation policy rejected correct answers: for 1.5B Q8_0, **25 of the 100 answerable answers were rejected for an invented citation**, and only 4 were true refusals. The v2 example was a trap for small models.

## Variants tried

- **v3:** no example id; a placeholder `[ID]` and the instruction to copy the id exactly, with nothing added.
- **v4:** the example id is the id of the first passage shown. (Worse; see below.)

## Results (system level: what the citation policy accepts)

Accuracy with bootstrap 95% intervals over questions; "not answered" is the share of answerable questions that ended as a refusal or a rejected answer.

| Model | v2 accuracy | v3 accuracy | v2 not answered | v3 not answered | Paired change v3 vs v2 (better / worse questions) |
| --- | --- | --- | --- | --- | --- |
| 1.5B FP16 | 49% [40-58] | 56% [46-66] | 31% | 22% | 14 / 7 |
| 1.5B Q8_0 | 49% [40-58] | 58% [48-68] | 32% | 21% | 16 / 7 |
| 1.5B Q4_K_M | 28% [20-37] | 33% [24-42] | 61% | 55% | 16 / 11 |
| 7B Q8_0 | 74% [66-82] | 69% [60-78] | 15% | 22% | 2 / 7 |
| 7B Q4_K_M | 74% [65-83] | 69% [60-78] | 15% | 18% | 1 / 6 |

v4 was worse than v3 for every 1.5B model (system accuracy 41%, 24%, 41% for Q8_0, Q4_K_M, FP16): with a real id in the example the models often dropped the citation altogether.

## What can and cannot be concluded

- **Solid (mechanical, not statistical):** for 1.5B Q8_0, rejections for invented citations fell from 25 to 1. Rejections for a missing citation rose from 2 to 12, so the prompt moved the failure, it did not remove it.
- **Suggestive only:** v3 raised 1.5B accuracy by 5 to 9 points, but no paired sign test reaches p < 0.05 (smallest p = 0.09, for Q8_0), and the intervals overlap. Do not read "v3 is better" as established.
- **v3 does not help the 7B models:** accuracy 74% to 69%, also not significant (p = 0.13 and 0.18). Prompt v2 stays the choice for 7B. A prompt tuned on the small model and checked on the large one is not free.
- **Quantisation:** 7B Q8_0 and Q4_K_M are indistinguishable on every metric, with identical intervals (74% [66-82] vs 74% [65-83]). The 1.5B Q4_K_M model is clearly worse than Q8_0 and FP16 under both prompts; its intervals do not overlap Q8_0's under v2 ([20-37] vs [40-58]).
- **The 1.5B models still cannot refuse.** At model level (before the citation policy) the 1.5B Q8_0 model answers 87% of the unanswerable questions under v3. The lower "hallucination" numbers at system level mostly reflect answers thrown away for citation faults, not good judgment. Only 1.5B Q4_K_M refuses more (45% at model level), and it does so by also failing answerable questions.

## Why bootstrap intervals and not repeated runs

At temperature 0 with a fixed seed a repeat gives the same answers (it moved 1 of 200 answers once). So "repeat the run for error bars" would show only numerical noise. The uncertainty that matters is the sample of questions, so the intervals above resample questions with replacement (2,000 resamples). They are in `results/bootstrap_ci_dev.json`, produced by `scripts/bootstrap_ci.py`. `scripts/prompt_sweep.py` produced the runs.

## Limits

One dataset, one run per configuration, prompt chosen on the dev questions it was evaluated on. Intervals are for question sampling only; they do not cover model or retrieval choices.
