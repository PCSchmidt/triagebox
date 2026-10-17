# Quantisation ladder (Qwen2.5-Instruct, dev split)

Five configurations of one model family on the same questions, prompt and retrieval, so differences come from model size and precision only. Single run each, on the SQuAD 2.0 **dev** subset (100 answerable + 100 unanswerable questions), prompt v2, BM25 top 4 (gold paragraph retrieved for 91% of the answerable questions, which caps accuracy), temperature 0, llama.cpp b11700, RTX A4500 Laptop GPU. Definitions are in [eval-protocol.md](eval-protocol.md); the single-model background is in [results.md](results.md).

Files (official Apache-2.0 GGUF releases): 1.5B at FP16 3.56 GB, Q8_0 1.89 GB, Q4_K_M 1.12 GB; 7B at Q8_0 8.1 GB and Q4_K_M 4.68 GB. 7B at FP16 (15 GB) was not run because it does not fit a 16 GB card with a KV cache.

## Accuracy and refusal behaviour

**Model level (what the model said)**

| Model | Precision | Accuracy | Hallucination | False refusal | Citation validity | Malformed |
| --- | --- | --- | --- | --- | --- | --- |
| 1.5B | FP16 | 63.0% | 92.0% | 3.0% | 78.2% | 0.5% |
| 1.5B | Q8_0 | 64.0% | 92.0% | 4.0% | 75.7% | 0.5% |
| 1.5B | Q4_K_M | 57.0% | 90.0% | 5.0% | 46.5% | 1.0% |
| 7B | Q8_0 | 74.0% | 30.0% | 15.0% | 100.0% | 0.0% |
| 7B | Q4_K_M | 74.0% | 31.0% | 14.0% | 100.0% | 0.5% |

**System level (after the citation policy turns invalid answers into refusals)**

| Model | Precision | Accuracy | Hallucination | False refusal |
| --- | --- | --- | --- | --- |
| 1.5B | FP16 | 49.0% | 76.0% | 31.0% |
| 1.5B | Q8_0 | 49.0% | 72.0% | 32.0% |
| 1.5B | Q4_K_M | 28.0% | 33.0% | 61.0% |
| 7B | Q8_0 | 74.0% | 30.0% | 15.0% |
| 7B | Q4_K_M | 74.0% | 31.0% | 15.0% |

Hallucination is the share of unanswerable questions that received an answer instead of a refusal.

## What the accuracy results show

1. **Model size dominates precision.** Both 7B configurations answer about 74% of questions correctly and invent an answer to about 30% of the unanswerable ones. The 1.5B models answer 57-64% correctly and invent an answer to **90-92%** of unanswerable questions: they almost never say there is not enough evidence (false refusal 3-5%). A 4.68 GB 7B model at 4-bit is far more trustworthy on this task than a 3.56 GB 1.5B model at 16-bit.
2. **At 7B, going from Q8_0 to Q4_K_M cost nothing measurable.** The two differ by at most one question out of 100 on every metric, which is the size of the run-to-run wobble measured earlier (about 1 in 200). The 4-bit file is 42% smaller.
3. **At 1.5B, only the 4-bit version suffers.** FP16 and Q8_0 are indistinguishable (63% vs 64% accuracy). Q4_K_M loses 6-7 points of accuracy and, more strikingly, citation validity falls from 76-78% to 46.5%: the small model at 4-bit increasingly cites passages it was not shown. This matches the general expectation that smaller models tolerate aggressive quantisation less well, but it is one observation on one model family, not a rule.
4. **The citation policy "fixes" the small models only by refusing almost everything.** At the system level the 1.5B Q4_K_M model's hallucination falls from 90% to 33%, but its false refusal rises to 61% and accuracy falls from 57% to 28%. Most of that apparent caution is the policy rejecting broken citations, not the model judging the evidence. System-level hallucination must always be read beside false refusal.
5. **Even the best configuration invents answers about 30% of the time on near-miss questions.** This is the quantity a faithfulness judge and better prompting or retrieval need to reduce. Smaller files do not help here; a larger model did.

## Caveats

- One model family, one run per configuration. With 100 questions per kind, a rate near 30% has roughly ±9 points of 95% uncertainty and a rate near 90% roughly ±6; differences of a few points between configurations (for example 63% vs 64%, or 30% vs 31%) are not meaningful.
- **Prompt v2 was tuned on the 7B Q4_K_M model.** The 1.5B models may behave differently with their own prompt, so their poor refusal behaviour is partly a statement about this prompt.
- Accuracy is a containment test against short gold answers. Dev split only; the test split has not been used. The source text is Wikipedia, not intelligence-style reporting.
- The GPU was throttled during these runs (median SM clock 20-69% of maximum). That does not affect accuracy; it is why speed and energy are measured separately, below.

## Speed and energy

![Quality versus GPU energy per answer: accuracy and hallucination against joules per answer for five configurations](ladder-quality-vs-energy.svg)

The chart shows the system-level numbers from the tables in this document. Each point is one configuration; circles are 1.5B models and squares are 7B models. A line joins the efficient set on the accuracy panel: the configurations that no other configuration beats on both accuracy and energy.

### How it was measured

The laptop GPU throttles under sustained load (it reached 94-97 C and a 210 MHz clock in long runs), so a long evaluation run cannot be timed. Timing was done separately, in short bursts:

- Before every burst, wait for the GPU to cool to 60 C or below.
- A burst is 8 requests (the same 8 questions for every configuration), after 3 discarded warm-up requests, with a fresh server for each burst.
- 5 configurations x 3 rounds, with the order rotated each round, so no configuration always runs first or on a warmer GPU.
- A burst counts only if the GPU reported no thermal, hardware-slowdown or power-brake throttle reason and stayed at or below 85 C. A software power cap is tolerated and recorded.
- Energy is GPU board power from `nvidia-smi` (100 ms samples), integrated over the whole burst and divided by the number of answers. It is GPU power only, not system power, and it includes the small idle gaps between requests.

**All 15 bursts were accepted.** The only throttle bits seen were the idle bit and the software power cap (mask 0x5), the hottest burst reached 73 C, and decode speed varied by only 1-5% between rounds.

### Results

| Model | Precision | Bursts accepted | Decode tokens/s, median (range) | Round-to-round spread | Time to first token (ms) | GPU energy per answer (J), median (per round) |
| --- | --- | --- | --- | --- | --- | --- |
| 1.5B | FP16 | 3/3 | 103.9 (102.7-107.6) | x1.05 | 99 | 21.1 (21.8, 20.2, 21.1) |
| 1.5B | Q8_0 | 3/3 | 163.1 (158.2-163.4) | x1.03 | 95 | 17.4 (17.1, 18.0, 17.4) |
| 1.5B | Q4_K_M | 3/3 | 191.6 (190.8-197.5) | x1.04 | 104 | 12.2 (12.8, 12.2, 9.5) |
| 7B | Q8_0 | 3/3 | 47.6 (47.4-48.9) | x1.03 | 302 | 63.7 (63.7, 63.7, 63.0) |
| 7B | Q4_K_M | 3/3 | 59.0 (58.9-59.4) | x1.01 | 311 | 56.1 (60.9, 56.1, 54.3) |

### What the speed and energy results show

1. **Smaller files decode faster.** Within each model size, fewer bits per weight means more tokens per second: 7B Q4_K_M is about 24% faster than Q8_0, and 1.5B Q4_K_M is about 84% faster than FP16. That is what memory-bandwidth-bound decoding would predict; it was not profiled.
2. **Time to first token depends on model size, not precision**: about 100 ms for the 1.5B models and about 300 ms for the 7B models, for prompts of roughly a thousand tokens. Prompt processing is compute-bound.
3. **At 7B, Q4_K_M is the better point.** It matched Q8_0 on every quality metric within noise, decodes about 24% faster, and used about 12% less energy per answer (the three rounds do not overlap: 54-61 J against 63-64 J).
4. **The 7B models cost about 3 to 5 times the energy of the 1.5B models** (56 J against 12-21 J) and buy a large quality gain: system-level accuracy 74% against 28-49%, and hallucination 30% against 33-76%, with the caveat that the small models' low system-level hallucination comes from the citation policy rejecting most of their answers.
5. **Efficient set on the accuracy panel:** 1.5B Q4_K_M, 1.5B Q8_0 and 7B Q4_K_M. 1.5B FP16 is beaten by 1.5B Q8_0 (same accuracy, cheaper), and 7B Q8_0 by 7B Q4_K_M. On hallucination, both 7B models are far ahead of every 1.5B model.
6. **Which to choose.** If the job is to answer or refuse reliably over documents, 7B Q4_K_M is the best configuration tested: it equals the best quality at the lowest 7B cost. The 1.5B models are not usable for this task with this prompt (they answer 90%+ of unanswerable questions). A 1.5B Q4_K_M model is the cheapest, but its 61% false-refusal rate at the system level makes it unusable too.

### Limits of the energy numbers

- **Energy follows time on this board.** While generating, the GPU runs near its power limit (about 110-118 W), so energy per answer is roughly power times time and the energy ordering mirrors the latency ordering. A device with a much lower power budget could change the ranking between configurations; that is what the Jetson measurements are for.
- **GPU power only**, sampled every 100 ms. System power, CPU and memory are not included. The burst average of 51-96 W is below the in-request draw because it includes idle gaps between requests, which matter more for the fast 1.5B models.
- **Energy is noisier than speed.** Per-round energy varies by up to about 30% for 1.5B Q4_K_M (9.5-12.8 J) and by about 12% for 7B Q4_K_M. Differences between the 1.5B configurations are small relative to that noise; the 3-5x gap between the model sizes is not.
- 3 rounds, 8 requests per burst, one GPU. Answer lengths differ between models, which is part of the real cost of each configuration.

### How the method changed (and the earlier data)

The timing method was revised twice, and the earlier data are kept in `results/`.

1. **First rule rejected every burst.** The initial acceptance rule required the SM clock to stay at 90% of its maximum and no throttle reason of any kind. All 15 bursts failed it (`results/timing_ladder_first_rule.json`), even though the GPU was cool (58-60 C at the start, at most 73 C). The cause: under load the board runs at its power limit, which sets a software power-cap reason and a clock below the unconstrained maximum. That is normal steady behaviour, unlike the thermal collapse seen in long runs. Speeds were stable across rounds, consistent with a steady state. The rule was revised to tolerate the software power cap and reject thermal, hardware-slowdown and power-brake reasons, before the runs reported here. The reported numbers therefore describe this board at its rated power limit, not unconstrained boost.
2. **Per-request energy was too noisy.** A second pass estimated energy per request from the one or two power samples inside each short request, which gave implausible spreads (for example 9.3 to 17.5 J for the same configuration; `results/timing_ladder_per_request_energy.json`). The reported pass integrates power over the whole burst instead.
3. **Speeds reproduced across all three passes**, for example 7B Q4_K_M at 58.8, 59.2 and 60.8 tokens/s (first pass), 60.4, 59.4 and 59.6 (second) and 59.4, 58.9 and 59.0 (third).
