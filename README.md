# TriageBox

[![CI](https://github.com/PCSchmidt/triagebox/actions/workflows/ci.yml/badge.svg)](https://github.com/PCSchmidt/triagebox/actions/workflows/ci.yml)

An air-gapped box that answers questions about a set of documents using a small, quantised language model, and **refuses to answer without citing the passages it was shown**. The question this project exists to answer is how much quality each step down in model size and precision costs, per joule, on edge hardware.

**Status: scaffold plus an evaluation set and retrieval baseline; no model results yet.** The retrieval, citation policy, evaluation metrics, signed sync bundles, one-way export, audit log and boot-time integrity checks are implemented and unit-tested on a laptop (pure Python plus `cryptography` for signatures). A SQuAD 2.0 benchmark subset (dev and test splits, half unanswerable) is prepared and BM25 retrieves the gold paragraph in the top 4 for 93.7% of dev questions (see [data](docs/data.md)). No language model has been run, so there are no accuracy, hallucination, latency or energy numbers yet. See the [roadmap](docs/roadmap.md).

Part of [physical-ai-portfolio](https://github.com/PCSchmidt/physical-ai-portfolio). It reuses the measurement discipline from [Sentinel Node](https://github.com/PCSchmidt/sentinel-node).

## Problem

Disconnected, power-limited systems cannot call a cloud model, and a small model on them will sometimes make things up. Two things have to be measured honestly: how often each model/precision invents an answer when the documents do not contain one, and what it costs in time and energy to run it. The headline figure will be quality versus joules per answer across model sizes, precisions and runtimes.

## What exists today

| Module | What it does | Tested |
| --- | --- | --- |
| `rag` | Word-window chunking and a pure-Python BM25 retriever | Ranking, overlap, edge cases |
| `prompt` | Citation-forced prompt; reply protocol `ANSWER: ... [id]` or `INSUFFICIENT_EVIDENCE` | Parsing, malformed replies |
| `citations` | Checks every cited id was actually retrieved; lexical-support proxy | Invented, mixed and missing citations |
| `pipeline` | Retrieve → generate → enforce citation policy → audit | Accept/reject for each failure mode |
| `dataset` | JSONL corpus/question files; SQuAD 2.0 converter with disjoint dev/test splits | Counts, determinism, round trip |
| `metrics` | Hallucination rate on unanswerable questions, false refusals, accuracy, citation validity, gold-citation recall | Mixed sets and missing denominators |
| `bench` | Time to first token, decode rate, energy per answer, Pareto front | Fake-clock timing, dominance |
| `bundle` | Ed25519-signed store-and-forward bundles; replay, tamper and conflict handling | 10 rejection and merge cases |
| `diode` | Allowlist-only export of summaries | No field outside the allowlist leaves |
| `audit` | Hash-chained log of every prompt and answer | Edit, reorder and removal detected |
| `integrity` | Boot-time hash check of model and index files; fails closed | Modified and missing files |

## What the checks do and do not show

- A **valid citation** means the model cited a passage it was actually shown. It does not mean that passage supports the claim. `lexical_support` is a cheap overlap proxy, not a faithfulness judge; a judge model plus hand-checked samples are planned.
- The **audit log** detects edits, reordering and removal in the middle. Truncating the end is not detectable from the log alone (documented and tested).
- **Bundles** reject replays only if the receiver persists the highest accepted sequence number. Key distribution and revocation are out of scope.
- No part of this has been security-reviewed.

## Planned measurements (not yet done)

1. An ISR-flavoured evaluation set over public documents, to follow the SQuAD 2.0 benchmark ([protocol](docs/eval-protocol.md)).
2. For each model size, precision (FP16, INT8, INT4) and runtime: answer accuracy, hallucination rate on unanswerable questions, citation validity, tokens per second, time to first token, and joules per answer.
3. One quality-versus-joules Pareto chart, with repeated and interleaved measurements.
4. The same on a Jetson Orin Nano once available.

## Run the tests

```sh
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Scope

Summarising and answering questions about public, unclassified documents. No targeting, weapons or engagement functionality, and no controlled or employer information.

## License

MIT. See [LICENSE](LICENSE).
