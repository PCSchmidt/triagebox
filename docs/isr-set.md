# The ISR-flavoured set (milestone 0.2b)

A small, hand-written evaluation set in the style of surveillance reports, with a locked test split. Files: `eval/isr/` (tracked), built by `scripts/build_isr_set.py`.

## What it is, and what it is not

- **Fictional reports I wrote for this project.** 12 pairs of near-identical short reports (ship sightings, air tracks, patrol summaries, emitter notes, imagery notes, sensor logs) with invented places and callsigns, plus 40 templated filler reports per split. Public intelligence documents are hard to license and hard to label cleanly, and writing the text gives exact ground truth.
- **Not real data.** It tests reading dense, abbreviated, number-heavy text and refusing correctly. It does not show how a system performs on real reporting. The same person (the assistant) wrote the reports and the questions, so the questions share the reports' phrasing; that probably makes it easier than a set written by someone else.
- **Small.** Each split has 42 answerable and 18 or 19 unanswerable questions. A 90% accuracy has a 95% interval of roughly ±9 points, an 11% hallucination rate (2 of 18) roughly ±14. Treat differences of under about 10 points as noise.

## Question kinds (the id prefix)

| Kind | Meaning | Per split |
| --- | --- | --- |
| A | answerable from one report | 36 |
| M | answerable only by comparing the two reports of a pair | 6 |
| U1 | unanswerable: stated in no report | 6-7 |
| U2 | unanswerable: stated only in the *sibling* report (the bait case) | 6 |
| U3 | unanswerable: false premise (assumes something the report contradicts) | 6 |

Pairs go whole into dev or test, so no pair straddles the split. `build_isr_set.py` fails if an answerable question's gold answer is not in its report(s) or a U1 question uses a word that appears in its reports. `tests/test_isr_set.py` also checks that dev and test share nothing, and that the test files still match the SHA-256 hashes in `eval/isr/LOCK.txt`, so an edit after locking fails the tests.

## How it was used

- Choices (prompt version, BM25 settings) were made earlier on SQuAD dev, not on this set. Dev here was used to characterise behaviour and find problems. The test split was run **once** per configuration, after dev, with no change in between. Retrieval was compared on test once.
- This was the first use of the test split; no later change to the pipeline should be tuned on these numbers.

## Results on the locked test split (system level: what the citation policy accepts)

| Configuration | Accuracy (42 answerable) | False refusal | Hallucination (18 unanswerable) | Model-level hallucination |
| --- | --- | --- | --- | --- |
| 7B Q4_K_M, prompt v2 | 90.5% | 4.8% | 11.1% | 11.1% |
| 7B Q8_0, prompt v2 | 90.5% | 7.1% | 11.1% | 11.1% |
| 1.5B FP16, prompt v3 | 76.2% | 11.9% | 50.0% | 66.7% |
| 1.5B Q8_0, prompt v3 | 73.8% | 14.3% | 55.6% | 61.1% |
| 1.5B Q4_K_M, prompt v3 | 50.0% | 45.2% | 27.8% | 27.8% |

Per kind on test (count correct, or count wrongly answered):

| Configuration | A /36 | M /6 | U1 /6 | U2 /6 | U3 /6 |
| --- | --- | --- | --- | --- | --- |
| 7B Q4_K_M | 35 | 3 | 1 | 0 | 1 |
| 7B Q8_0 | 34 | 4 | 1 | 0 | 1 |
| 1.5B FP16 | 30 | 2 | 3 | 3 | 3 |
| 1.5B Q8_0 | 29 | 2 | 3 | 3 | 4 |
| 1.5B Q4_K_M | 21 | 0 | 1 | 2 | 2 |

(Dev showed the same pattern: the 7B models got all 36 single-report questions right, 2 of 6 comparisons, and answered 1 of 19 unanswerable questions.)

## What it shows

- **The 7B models handle this set well.** 90% accuracy, 11% hallucination, against 74% and 31% on SQuAD. Q8_0 and Q4_K_M are indistinguishable again.
- **The bait did not catch the 7B models.** On U2 (the fact exists only in the sibling report) they refused 6 of 6 on test and on dev. This set's unanswerables are much easier than SQuAD's near-miss ones, which were written by people trying to fool a reader. So the low 7B hallucination here says the set is easy on that dimension, not that the problem is solved.
- **The 1.5B models take the bait.** FP16 and Q8_0 answered half of the unanswerable questions, including 3 of 6 U2 questions, so they borrow a fact from the sibling report. 1.5B Q4_K_M refuses more but also fails more answerable questions (21 of 36 single-report ones correct).
- **Comparison questions fail mostly because retrieval does.** On dev, for the four comparison questions the 7B model refused, the retrieved passages did not include both reports. The questions ("which of the two tracks flew higher?") name no entity, so BM25 has nothing to match. A model that sees both reports could answer. On test the 7B models got 3 and 4 of 6.

## Retrieval on this set (recall@k; every gold document must be in the top k)

| Retriever | Dev @4 | Test @4 | Test @8 |
| --- | --- | --- | --- |
| BM25 | 90.5% | 92.9% | 92.9% |
| BGE-small dense | 95.2% | 88.1% | 97.6% |
| Hybrid (RRF) | 95.2% | 92.9% | 92.9% |

Intervals are about ±10 points. Dense is better than BM25 on dev, worse at k = 4 on test, and ahead only at k = 8: no retriever is clearly ahead. The corpus has 52 chunks, so the top 8 is 15% of it. The earlier SQuAD finding (BM25 at least as good as dense) is neither confirmed nor reversed.

## Limits

Fictional, single author, small, and easy on unanswerables for large models. A better set would have a second writer, reports from real public sources, more comparison and multi-step questions, and unanswerable questions written adversarially. The faithfulness judge and the unsupported-answer detector have not been run on this set.
