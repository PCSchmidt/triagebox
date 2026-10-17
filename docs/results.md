# Results

The five-configuration quantisation ladder, with speed and energy, is in [ladder.md](ladder.md). This page covers the first model (Qwen2.5-7B Q4_K_M) and the prompt comparison.

All numbers are for one model, one benchmark split, and a single run of each configuration. The evaluation runs on this page were on a thermally throttled laptop GPU, so **no speed or energy figure is quoted from them**; accuracy figures are not affected by throttling. Properly gated speed and energy measurements are in [ladder.md](ladder.md).

## Setup

- **Model:** Qwen2.5-7B-Instruct, Q4_K_M GGUF (4.68 GB), official Apache-2.0 release from Qwen. Run with llama.cpp build b11700 (CUDA 12.4), all layers on the GPU, context 4096, temperature 0, seed 0.
- **Data:** the SQuAD 2.0 **dev split** of this project ([data.md](data.md)): 100 answerable and 100 unanswerable questions (the first 100 of each kind), interleaved. The test split has not been used.
- **Retrieval:** BM25, top 4 chunks of up to 200 words. In this 100-question subset the gold paragraph is in the top 4 for **91%** of answerable questions, which is the ceiling on accuracy. (Over the whole dev split it is 93.7%.)
- **Two levels of metrics.** *Model level* scores what the model said. *System level* scores what TriageBox accepts: an answer must have text and cite only passages that were retrieved, otherwise it is turned into a refusal.

## Prompt v1 vs. v2

| | Accuracy | Hallucination | False refusal | Citation validity | Malformed replies |
| --- | --- | --- | --- | --- | --- |
| Prompt v1, model level | 60% | 23% | 17% | 85.7% | 12.5% |
| Prompt v1, system level | 52% | 19% | 44% | 100% | 0% |
| **Prompt v2, model level** | **74%** | **31%** | 14% | **100%** | 0.5% |
| **Prompt v2, system level** | **74%** | **31%** | 15% | 100% | 0% |

Definitions are in [eval-protocol.md](eval-protocol.md). Hallucination is the fraction of unanswerable questions that got an answer instead of a refusal.

Prompt v2 differs from v1 only by stating the two reply formats explicitly and giving an example (`ANSWER: 1998 [doc1#3]`). Choosing v2 over v1 was done on the dev split; the test split remains untouched for final numbers.

### What happened with prompt v1

With the v1 prompt, 25 of the 200 replies were **only a citation with no answer text**, for example `ANSWER: [a8p5#2]` (18 on answerable questions, 7 on unanswerable). The first version of the parser counted those as answers, and the citation policy **accepted 22 of the 25**, so the system was accepting empty answers. That was a hole in the policy, found by reading the replies, and it is now closed: an answer needs text besides its citations, otherwise it is malformed. Other v1 failures: the model cited a chunk number that was never shown (right paragraph, invented suffix) or an invented source, and in some cases wrote `ANSWER: INSUFFICIENT_EVIDENCE`, which the parser now reads as a refusal.

Numbers before the parser fix (prompt v1, lenient parser), kept for transparency in `results/qwen2.5-7b-q4_k_m-p1-lenient-parser_dev.json`: model level 60% accuracy, 30% hallucination, 17% false refusal, 86.2% citation validity; system level 52%, 25%, 28%.

### Reading the comparison

- **Prompt v2 fixed the format, not the judgment.** Malformed replies went from 12.5% to 0.5% and citation validity from 86% to 100%. Accuracy rose from 60% to 74% against a 91% ceiling.
- **The hallucination figure got worse, honestly.** Under v1, 23% looked better only because many replies were unusable and so did not count as answers. With well-formed replies, the model answers about 31% of the near-miss unanswerable questions instead of refusing.
- **The citation policy did nothing against hallucination under v2.** Every citation is valid, so nothing is rejected: system-level and model-level hallucination are both 31%. A model that cites a real, retrieved paragraph for an invented answer passes the structural check. This is the main reason a judge model for faithfulness is on the roadmap; [faithfulness.md](faithfulness.md) follows it up and finds that about half of the 31% are answers the cited passage does support.
- **Statistical caution.** With 100 questions per kind, a rate near 30% has a 95% interval of roughly ±9 points. The accuracy gain (60% to 74%) is well outside that; the hallucination change (23% to 31%) is not distinguishable from noise on its own.

## Reproducibility

Repeating the same run changed the answer to about 1 question in 200 (false refusal 16% vs. 17%), even at temperature 0 with a fixed seed. Treat a one-question difference as noise.

## Why these runs report no speed or energy

A probe sending the same request repeatedly (`scripts/throttle_probe.py`) showed the GPU at 89-94 C with the SM clock falling from a 1785 MHz maximum to 210 MHz, with software-thermal, hardware-thermal and hardware-slowdown throttle reasons set. Decode speed fell from about 53 tokens/s early in the evaluation run to about 15 tokens/s later, and GPU power from 114 W to 72 W. Speed and energy measured under those conditions describe the throttle, not the model. `run_eval.py` now logs the GPU clock, temperature and throttle reasons during every run and marks the timing `NOT TRUSTWORTHY` when the median clock is below 90% of maximum or any power or thermal throttle was seen; all runs so far carry that flag.

The ladder therefore uses a different protocol (cool the GPU before each short burst, interleave configurations across rounds, log clock, temperature and throttle reasons beside each number); see [ladder.md](ladder.md). Sustained speed under throttling is also what a fanless edge device would deliver, so it remains worth reporting separately from the cooled-burst speed.

## Limitations

- One model, one quantisation, one retrieval setting, one dataset (Wikipedia text, not intelligence-style reports), single runs.
- Accuracy is a containment test against short gold answers; it can miss correct paraphrases and credit lucky matches.
- Hallucination here means "answered a near-miss unanswerable question", scored by the dataset's labels. Whether an answer is faithful to the cited passage is not yet judged.
- Prompt choice was made on the dev split; any further tuning must stay on dev.
