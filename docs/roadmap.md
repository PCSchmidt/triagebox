# Roadmap

| Milestone | Deliverable | Needs | Status |
| --- | --- | --- | --- |
| 0.1 | Scaffold: retrieval, citation policy, metrics, signed bundles, diode, audit log, integrity check, CI | Laptop | Done |
| 0.2a | Benchmark v0: SQuAD 2.0 subset with disjoint dev and test splits; BM25 retrieval baseline | Laptop | Done |
| 0.2b | ISR-flavoured evaluation set over public documents (answerable and three kinds of unanswerable), with a locked test split | Laptop | Planned |
| 0.3 | First model running locally through the `LLM` interface; end-to-end run on the dev set; time-to-first-token and decode-rate harness validated | Desktop GPU | Planned |
| 0.4 | Quantisation ladder (FP16 where it fits, INT8, INT4) at two model sizes; accuracy, hallucination rate, citation validity; interleaved latency | Desktop GPU | Planned |
| 0.5 | Embedding retrieval versus BM25 on the same questions | Desktop GPU | Planned |
| 0.6 | Judge-model faithfulness with hand-checked agreement | Desktop GPU | Planned |
| 0.7 | Energy per answer and the quality-versus-joules Pareto chart | Power meter | Planned |
| 0.8 | Jetson Orin Nano measurements | Jetson | Planned |
| 0.9 | Disconnected-sync demo: signed bundles and diode between two instances | Laptop | Planned |
| Later | LoRA or distilled small model compared against the quantised baseline | Desktop GPU | Planned |

A milestone is done only when its README section leads with measured results.
