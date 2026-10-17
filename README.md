# TriageBox

[![CI](https://github.com/PCSchmidt/triagebox/actions/workflows/ci.yml/badge.svg)](https://github.com/PCSchmidt/triagebox/actions/workflows/ci.yml)

An air-gapped box that answers questions about a set of documents using a small, quantised language model, and **refuses to answer without citing the passages it was shown**. The question this project exists to answer is how much quality each step down in model size and precision costs, per joule, on edge hardware.

**Status: first model result in; no quantisation comparison, speed or energy numbers yet.** The retrieval, citation policy, evaluation metrics, signed sync bundles, one-way export, audit log and boot-time integrity checks are implemented and unit-tested on a laptop (pure Python plus `cryptography` for signatures). A SQuAD 2.0 benchmark subset (dev and test splits, half unanswerable) is prepared (see [data](docs/data.md)), and one model, Qwen2.5-7B-Instruct at Q4_K_M, has been run through the full pipeline on the dev split (below). Nothing has run on an edge device, and the speed and energy figures are withheld because the laptop GPU was thermally throttled. See the [roadmap](docs/roadmap.md).

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
| `llamacpp`, `power` | Streaming client for a local llama-server; GPU power, clock, temperature and throttle sampling with a timing-validity rule | Fake HTTP server; parsing and validity rules |
| `bundle` | Ed25519-signed store-and-forward bundles; replay, tamper and conflict handling | 10 rejection and merge cases |
| `diode` | Allowlist-only export of summaries | No field outside the allowlist leaves |
| `audit` | Hash-chained log of every prompt and answer | Edit, reorder and removal detected |
| `integrity` | Boot-time hash check of model and index files; fails closed | Modified and missing files |

## First result: Qwen2.5-7B-Instruct (Q4_K_M), SQuAD 2.0 dev subset

100 answerable and 100 unanswerable questions, BM25 top-4 retrieval (the gold paragraph is retrieved for 91% of the answerable ones, which caps accuracy), temperature 0, llama.cpp on an RTX A4500 Laptop GPU. "Hallucination" is the share of unanswerable questions that got an answer instead of a refusal. Full write-up, definitions and caveats in [docs/results.md](docs/results.md).

| Prompt | Accuracy | Hallucination | False refusal | Citation validity | Malformed replies |
| --- | --- | --- | --- | --- | --- |
| v1 | 60% | 23% | 17% | 85.7% | 12.5% |
| v2 (explicit format example) | **74%** | **31%** | 14% | **100%** | 0.5% |

What it shows, and what it doesn't:

- **The format failure was the first problem.** With the v1 prompt, 25 of 200 replies were only a citation with no answer text, and the citation policy initially *accepted* 22 of them. That hole is closed (an answer must contain text), and a prompt with an explicit example removed almost all malformed replies.
- **The honest hallucination figure is about 31%.** The v1 figure of 23% looked better only because many replies were unusable. About a third of near-miss unanswerable questions get an invented answer from this model at this precision.
- **The structural citation check does not stop that.** With well-formed replies every citation is valid, so the policy rejects nothing; a real, retrieved paragraph can be cited for an invented answer. Judging whether the cited passage supports the answer is the next measurement.
- **Caveats:** one model, one quantisation, one run; the prompt was chosen on the dev split and the test split is untouched; with 100 questions per kind a rate near 30% has roughly ±9 points of uncertainty, so the 23%-to-31% change is not distinguishable from noise while the accuracy gain is not. Repeating a run changed about 1 question in 200.
- **No speed or energy numbers on purpose.** The GPU hit 89-97 C and its clock fell from 1785 MHz to as low as 210 MHz during sustained generation, so any speed or power figure from these runs would describe the throttle, not the model. The runner now records clock, temperature and throttle reasons and flags such runs as untrustworthy.

## What the checks do and do not show

- A **valid citation** means the model cited a passage it was actually shown. It does not mean that passage supports the claim. `lexical_support` is a cheap overlap proxy, not a faithfulness judge; a judge model plus hand-checked samples are planned.
- The **audit log** detects edits, reordering and removal in the middle. Truncating the end is not detectable from the log alone (documented and tested).
- **Bundles** reject replays only if the receiver persists the highest accepted sequence number. Key distribution and revocation are out of scope.
- No part of this has been security-reviewed.

## Planned measurements (not yet done)

1. A thermally controlled timing protocol (cool-down gating, short bursts, interleaved rounds), then for each model size and precision: tokens per second, time to first token and joules per answer.
2. The quantisation ladder: 1.5B at FP16, Q8_0 and Q4_K_M; 7B at Q8_0 and Q4_K_M (7B FP16 does not fit a 16 GB card with a KV cache), each scored on accuracy, hallucination and citation validity.
3. A judge model for citation faithfulness, with a hand-checked sample and reported agreement.
4. An ISR-flavoured evaluation set over public documents, to follow the SQuAD 2.0 benchmark ([protocol](docs/eval-protocol.md)), and the final numbers on the locked test split.
5. One quality-versus-joules Pareto chart, and the same on a Jetson Orin Nano once available.

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
