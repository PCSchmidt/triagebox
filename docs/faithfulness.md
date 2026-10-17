# Faithfulness: does the cited passage support the answer?

Milestone 0.6. The citation policy only checks that a cited passage was *retrieved*. [results.md](results.md) showed that this does nothing against hallucination (every citation was valid, 31% hallucination). This page asks the next question: does the cited passage actually say the answer?

Run judged: Qwen2.5-7B Q4_K_M, prompt v2, dev split, 116 accepted answers (85 to answerable questions, 31 to unanswerable ones). Judge: Qwen2.5-7B-Instruct Q8_0 via llama.cpp, shown the question, the answer and only the cited passages. Single run, dev split only.

## What the judge says

| | Answers | Judge: supported | Judge: not supported |
| --- | --- | --- | --- |
| To answerable questions | 85 | 85.9% | 14.1% |
| To unanswerable questions | 31 | 64.5% | 35.5% |

## The judge is not reliable enough to trust alone

It was checked two ways.

- **Against my hand labels** (43 items: all 31 answered-unanswerable items and the 12 answerable items the judge rejected). Agreement 72% (31 of 43). The judge was too strict 8 times (for example rejecting "four divisions of graduate research" for a passage saying "four divisions of graduate research", and rejecting 1413 because the passage said "Warsaw" where the question said the old name "Warszowa") and too lenient 4 times (for example accepting "medicine" for "what field is respiration used in", when the passage is about oxygen therapy).
- **Against a deterministic check on answerable questions**: is the gold answer a substring of the cited passage? Agreement 89% (76 of 85). The judge accepted 2 answers whose gold string is absent from the cited passage and rejected 7 whose gold string is present.

The hand labels are one labeller (me), not a random sample, and four items are genuinely ambiguous (entity swaps such as Warsaw versus Poland). The labels are in `results/hand_labels_dev.json` so they can be disputed. The judge is also the same model family as the model being judged, so shared blind spots are possible. Treat the judge as a noisy screen, not a measurement.

## What the checks found, independent of the judge

1. **About half of the "hallucinations" are not hallucinations.** Of the 31 answers to questions the dataset labels unanswerable, I judged **17 supported** by the cited passage and 14 not. The dataset's near-miss questions often contain a typo, a swapped detail the model reasonably ignored, or a question the passage does answer ("Duct PTT Telecom", "the third of France", "Of Poland's inhabitants in 1901"). By this reading, the unsupported-answer rate is about **14 of 100** unanswerable questions, not the 31% reported in [results.md](results.md). Both numbers are real: 31% is "answered a question the benchmark says is unanswerable"; 14% is "gave an answer its passage does not support". The second is the one that matters operationally, and the first overstates it.
2. **Citations can be valid and wrong.** In **7 of 85** answers to answerable questions (8%) the answer matches the gold answer's wording but the gold string is *absent from the cited passage*. The model got the answer right (from the other retrieved passages, or from its own memory: it cited a passage about Warsaw for the Baltic "Singing Revolution") and cited the wrong chunk. The citation policy accepts these. A reader following the citation would not find the answer.

## What this changes

- The headline hallucination figure should be reported alongside the supported-by-passage figure, with the caveat that the second rests on hand labels from one labeller.
- Citation validity (100%) means "cited something retrieved", not "cited something relevant". A grounding check (the deterministic one above, or the judge) is a better acceptance test, and the deterministic one needs no model.
- The next experiment is to run the deterministic check inside the pipeline as a policy (reject answers whose cited passages do not contain the answer string) and measure what it costs in false refusals. It only applies to short extractive answers, so it is not a general solution.

## Reproducing

```sh
python scripts/judge.py --run qwen2.5-7b-q4_k_m-p2_dev --judge q8_0        # writes results/judge_*.json
python scripts/judge_review.py --run ... --kind unanswerable             # print items to label
python scripts/judge_agreement.py --run qwen2.5-7b-q4_k_m-p2_dev         # agreement with hand labels
```

Per-item verdicts are under `data/judge/` (untracked, they quote the benchmark).

## Not done

Hand labels for the 73 answerable items the judge accepted (so false "supported" on answerable questions is only covered by the deterministic check); a second labeller; a judge from a different model family; the test split.
