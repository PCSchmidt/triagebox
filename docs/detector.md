# Detecting unsupported answers (quote then verify)

[faithfulness.md](faithfulness.md) found that the citation policy only checks that a cited id was retrieved, and that a word-overlap check ([lexical grounding](faithfulness.md#a-grounding-policy-tested-offline)) removed 2 of 14 answers that their cited passage does not support. This page tries to do better.

## The method

For each accepted answer, a second model call (Qwen2.5-7B Q8_0) is asked to **copy, word for word, the sentence of the cited passage that states the answer**, or reply NONE. Code then checks that the copy occurs in a cited passage (`triagebox.verify.quote_match`: longest verbatim run divided by quote length, 1.0 = verbatim). The answer is kept only if the match is at least 0.9; otherwise it is turned into a refusal. The model only extracts, which small models do reliably; it cannot talk its way past the check, because the verdict comes from string matching. `scripts/quote_policy.py` re-scores saved runs under the policy; `scripts/detector_eval.py` measures how well the signal separates supported from unsupported answers.

The 0.9 threshold and the prompt were fixed before any result was seen. A second, more extractive prompt (version 2) was tried on dev only; see below.

## A bug in my own first analysis

My first evaluation reported that "question entity coverage" (the names and numbers in the question being present in the cited passage) separated supported from unsupported answers with AUROC 0.93 on the ISR dev set and 0.74 on the SQuAD set, with no model involved. That was wrong. The function returned zeros for every answer where the model replied NONE, so the "model-free" signal secretly contained the quote result. After fixing it (a test now checks that entity coverage does not depend on the quote) the real values are **0.61 (ISR dev) and 0.52 (SQuAD)**: entity coverage alone is useless. The earlier table was never published, and I found the error because the end-to-end policy removed none of the answers the AUROC said it should.

## How well the signal separates (dev; AUROC, 95% bootstrap interval over answers)

Labelled answers: on ISR, a correct answer to an answerable question is supported, and any answer to an unanswerable question is unsupported (true by construction). On SQuAD, my hand labels plus a gold-in-cited-passage check (105 answers from the 7B Q4_K_M run; only 19 are unsupported).

| Signal | ISR dev (196 supported, 34 unsupported) | SQuAD (86 supported, 19 unsupported) |
| --- | --- | --- |
| Lexical overlap (old) | 0.64 [0.55-0.73] | 0.67 [0.56-0.78] |
| Entity coverage (fixed) | 0.61 [0.54-0.68] | 0.52 [0.46-0.60] |
| **Quote found (verbatim >= 0.9)** | **0.89 [0.85-0.92]** | **0.74 [0.63-0.85]** |
| Quote found x min(answer, entity coverage) | 0.82 [0.79-0.86] | 0.73 [0.60-0.84] |

Combining the quote with coverage measures did not help, so the policy is the quote check alone. On SQuAD the interval is wide (19 unsupported answers) and overlaps the lexical baseline's; on ISR the gain is clear.

## What the policy does end to end

Accuracy / false refusal / hallucination, no policy vs quote policy (prompt version 1, system level).

SQuAD dev (100 + 100 questions):

| Model | No policy | Quote policy |
| --- | --- | --- |
| 7B Q4_K_M | 74 / 14 / 31% | 56 / 35 / 19% |
| 7B Q8_0 | 74 / 15 / 30% | 56 / 37 / 18% |
| 1.5B Q8_0 | 58 / 19 / 64% | 32 / 58 / 19% |
| 1.5B Q4_K_M | 33 / 48 / 31% | 19 / 66 / 10% |

On the hand-labelled 7B run the policy removed **8 of the 14** answers that the passage does not support, and wrongly removed 4 of the 17 that it does. For comparison the lexical policy removed 2 of 14 and cost 5 to 7 accuracy points.

ISR **locked test split**, run once (42 answerable, 18 unanswerable):

| Model | No policy | Quote policy |
| --- | --- | --- |
| 7B Q4_K_M | 90.5 / 4.8 / 11.1% | 76.2 / 21.4 / 11.1% |
| 7B Q8_0 | 90.5 / 7.1 / 11.1% | 76.2 / 21.4 / 11.1% |
| 1.5B FP16 | 76.2 / 11.9 / 50.0% | 52.4 / 45.2 / 11.1% |
| 1.5B Q8_0 | 73.8 / 14.3 / 55.6% | 52.4 / 45.2 / 11.1% |
| 1.5B Q4_K_M | 50.0 / 38.1 / 27.8% | 42.9 / 50.0 / 11.1% |

## What it shows

- **It works as a filter and it is expensive.** The quote check catches most unsupported answers, but about one correct answer in five also fails it. Almost all of those are the extractor saying NONE for an answer that is plainly in the passage ("260 kilometres (160 mi)", "Tarn Field"). The model is too strict about what "states" an answer. That is a limit of this 7B extractor, not of the idea.
- **For the 1.5B models it is a bargain, in the sense that matters for safety.** On the locked test split it cut the share of unanswerable questions answered from 50-56% to 11%, at a cost of about 24 points of accuracy.
- **For the 7B models on this set it only costs.** Hallucination did not move (the 2 of 18 answers that remain pass the quote check, so they are verbatim-supported on a plain reading) and accuracy fell 14 points. On the harder SQuAD dev set it cut hallucination by about 12 points for an 18-point accuracy cost.
- **A more extractive prompt (version 2, dev only)** gave back some accuracy (SQuAD 7B Q4_K_M 59% vs 56%) but removed fewer unsupported answers (6 of 14 vs 8), and the AUROC did not improve (0.71 vs 0.74 SQuAD, 0.89 vs 0.89 ISR). It is the same trade-off, not a better one, so version 1 stays.
- **Not a free lunch in compute.** The check adds a second, longer model call per answer. Latency and energy of that call were not measured.

## Limits

Labels: the SQuAD "unsupported" set is 19 answers, 14 labelled by me alone; ISR labels are exact but the ISR unanswerable questions are easy for large models. All checks use one extractor (Qwen2.5-7B Q8_0, same family as the answerers). The quote must be near-verbatim, so it cannot verify an answer that needs two sentences or a computation (the comparison questions). The policy is not wired into `triage()`; it is an offline re-scoring until the cost question is settled. The two-model setup would need both models resident on an edge device.
