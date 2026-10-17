# Roadmap

| Milestone | Deliverable | Needs | Status |
| --- | --- | --- | --- |
| 0.1 | Scaffold: retrieval, citation policy, metrics, signed bundles, diode, audit log, integrity check, CI | Laptop | Done |
| 0.2a | Benchmark v0: SQuAD 2.0 subset with disjoint dev and test splits; BM25 retrieval baseline | Laptop | Done |
| 0.2b | ISR-flavoured evaluation set (answerable, comparison, and three kinds of unanswerable), with a locked test split | Laptop | Done (hand-written fictional reports, not public documents; see isr-set.md). Test split used once for the five ladder configurations |
| 0.3 | First model running locally through the `LLM` interface; end-to-end run on the dev set (Qwen2.5-7B Q4_K_M, two prompt versions); GPU throttling detected and timing flagged | Desktop GPU | Done (accuracy only; timing withheld) |
| 0.3b | Thermally controlled timing protocol: cool-down gating, short bursts, interleaved rounds, clock, temperature and throttle reasons logged beside each number | Desktop GPU | Done (rule revised once after the first pass; see ladder.md) |
| 0.4 | Quantisation ladder (1.5B at FP16, Q8_0, Q4_K_M; 7B at Q8_0, Q4_K_M); accuracy, hallucination rate, citation validity; interleaved speed | Desktop GPU | Done (dev split, single run per configuration) |
| 0.5 | Embedding retrieval versus BM25 on the same questions | Desktop GPU | Done on dev (see retrieval.md): dense alone is worse than BM25; hybrid at best slightly better |
| 0.6 | Judge-model faithfulness with hand-checked agreement | Desktop GPU | Done on the 7B Q4_K_M p2 dev run (see faithfulness.md); judge agrees 72% with hand labels, so it is a noisy screen. Lexical grounding policy tested offline: catches wrong citations, misses unsupported answers, costs 5-7 points of accuracy |
| 0.7 | Energy per answer and the quality-versus-joules chart | Laptop GPU; wall-power meter for system energy | Done for GPU board energy on the laptop GPU; wall-power and edge-device measurements remain |
| 0.7b | Prompt tuned for the 1.5B models; error bars on the ladder | Desktop GPU | Done on dev (see prompt-tuning.md): bootstrap intervals instead of repeats; prompt v3 fixes invented citations in 1.5B models, accuracy gain suggestive only |
| 0.8 | Jetson Orin Nano measurements | Jetson | Planned |
| 0.9 | Disconnected-sync demo: signed bundles and diode between two instances | Laptop | Planned |
| Later | LoRA or distilled small model compared against the quantised baseline | Desktop GPU | Planned |

A milestone is done only when its README section leads with measured results.
