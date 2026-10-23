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

## A grounding policy, tested offline

`triage(..., min_support=t)` now optionally rejects an answer unless at least a fraction `t` of its content words appear in the passages it cites (`lexical_support`; no model involved). `scripts/grounding_policy.py` re-scores every saved dev run as if that policy had been on, so no new model runs were needed. The threshold was swept on the dev split only; results are in `results/grounding_policy_dev.json`.

Qwen2.5-7B Q4_K_M, prompt v2, dev split (100 + 100 questions):

| Policy | Accuracy | False refusal | Hallucination (dataset labels) | Hand-labelled unsupported answers removed (of 14) | Hand-labelled supported answers lost (of 17) |
| --- | --- | --- | --- | --- | --- |
| none | 74% | 14% | 31% | 0 | 0 |
| support >= 0.5 | 69% | 23% | 28% | 2 | 1 |
| support >= 0.7 | 67% | 25% | 28% | 2 | 1 |

Qwen2.5-7B Q8_0 behaves the same (74% to 68%, hallucination 30% to 28%). The 1.5B models lose more.

Reading it:

- **It does what it was built for, and only that.** At 0.5, all 5 correct answers it rejects are the ones whose cited passage lacks the answer (Donald Davies, the Edict of Nantes, the Singing Revolution, the Presbyterian Church, photosynthesis). Those answers are right but unfaithful, so the "accuracy loss" is partly the policy doing its job: a reader following those citations would not find the answer. At 0.7 two more correct answers are lost (a paraphrased answer and a short phrase); those are false alarms.
- **It barely touches the larger problem.** Of the 14 answers to unanswerable questions that the cited passage does not support, it removes 2. The other 12 reuse words from a passage that is about the right topic ("15 kilometres" for a question about a different distance), so word overlap cannot tell them apart. Detecting that needs the question and the passage compared, which is what the judge tries to do and does only 72% reliably.
- **So it is not worth turning on as the default.** It costs 5 to 7 points of accuracy to remove 2 of 14 unsupported answers. The more useful use of the check is as a flag on the 8% of answers with wrong citations, not as a gate.

Limits: one model family, one dev subset, thresholds picked after seeing the dev results (not confirmed on the test split), and the hand labels are from one labeller.

## What this changes

- The headline hallucination figure should be reported alongside the supported-by-passage figure, with the caveat that the second rests on hand labels from one labeller.
- Citation validity (100%) means "cited something retrieved", not "cited something relevant". A grounding check (the deterministic one above, or the judge) is a better acceptance test, and the deterministic one needs no model.
- A lexical grounding check was tried as a pipeline policy (next section): it catches the wrong-citation answers but almost none of the unsupported ones, and costs accuracy. Detecting an unsupported-but-on-topic answer needs a comparison of question and passage, not word overlap.

## Reproducing

```sh
python scripts/judge.py --run qwen2.5-7b-q4_k_m-p2_dev --judge q8_0        # writes results/judge_*.json
python scripts/judge_review.py --run ... --kind unanswerable             # print items to label
python scripts/judge_agreement.py --run qwen2.5-7b-q4_k_m-p2_dev         # agreement with hand labels
```

Per-item verdicts are under `data/judge/` (untracked, they quote the benchmark).

## A judge from a different model family

Added 2026-10-23. The judge above is Qwen2.5-7B, the same family as the answerers, so it can share their blind spots. As a check I ran the same judging prompt with **Phi-3-mini-4k-instruct** (3.8B, 4-bit, MIT licence, Microsoft: a different family, trained on different data) over the same 116 accepted answers of the 7B Q4_K_M dev run and compared it with my hand labels (the same 43 labelled items) and with the deterministic gold-in-cited-passage check (85 answerable questions). It is a second *model* labeller, not a second person: there is still only one human labeller.

| | Qwen2.5-7B Q8_0 judge | Phi-3-mini judge |
| --- | --- | --- |
| Agreement with my hand labels (43 items) | 72.1% | 69.8% |
| Answers I labelled unsupported that it caught (of 19) | 15 | 6 |
| Answers I labelled supported that it wrongly rejected (of 24) | 8 | 0 |
| Agreement with "gold answer is in the cited passage" (85 answerable) | 89.4% | 97.6% |
| Share of the 31 answered unanswerable questions it calls supported | 64.5% | 96.8% |
| Share of the 85 answered answerable questions it calls supported | 85.9% | 94.1% |

Same agreement, opposite errors: the Qwen judge is strict (it catches most unsupported answers and wrongly rejects a third of supported ones), the Phi-3 judge is lenient (it almost never rejects, so it misses 13 of 19 unsupported answers but also never wrongly rejects a supported one). Neither is a reliable screen on its own; the verbatim quote check ([detector.md](detector.md)) removes more unsupported answers (8 of 14) than either. A combination (reject only if both judges reject) would keep Phi-3's zero false rejections but could catch at most the 6 it catches; I did not test combinations. The 43 items are not a random sample (they are all answered unanswerable questions plus the judge's rejections of answerable ones), so the percentages describe this set, not a general rate.

## Not done

Hand labels for the 73 answerable items the judge accepted (so false "supported" on answerable questions is only covered by the deterministic check); a second *human* labeller; combinations of judges; the test split. (A judge from a different model family is in the previous section.)
