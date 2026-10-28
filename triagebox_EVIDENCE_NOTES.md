# triagebox: Evidence Notes

Companion to triagebox.md. Labels: Confirmed, Ran it, Per the docs, My reading, Ask the author, Author says. Paths are repository-relative. Probes and tests ran in a scratch copy of the tracked files (git archive of HEAD 70f99ca) with a throwaway virtual environment outside the clone; nothing was written in the clone. No .env file exists in the tree; none was read. No model was run and no paid call was made. Builder-tool naming follows the standing instruction: AI coding agents only.

## E1 Fact sheet

| Fact | Value | Label |
| --- | --- | --- |
| Name, address, licence | triagebox; github.com/PCSchmidt/triagebox; MIT | Confirmed |
| Kind | Measurement study plus citation-forced RAG (retrieval-augmented generation) harness for small local models | Confirmed |
| Status | README: ladder measured on a laptop GPU; edge device and benchmark test half not yet used; version 0.1.0 | Confirmed |
| Size | about 1,370 lines in src/triagebox (19 modules), 22 scripts, 12 test files, 11 docs | Confirmed |
| Tests | 112 collected and passing (pytest, Python in a scratch venv with pytest and cryptography) | Ran it |
| Commits | 13, one author name, 2026-10-16 to 2026-10-23 | Confirmed |
| Repo metadata | public, 0 stars, 0 forks, last push 2026-10-23 (GitHub API metadata) | Confirmed |
| CI | One workflow (pytest on Python 3.10 and 3.12). Last five-plus runs: the four newest failed, the earlier four succeeded (run list) | Confirmed |
| Needs | Python 3.10+, optional cryptography; real runs need llama.cpp server, GGUF files (1.1 to 8.1 GB each), an NVIDIA GPU, internet for downloads | Per the docs |
| Deployed | No | Confirmed (nothing in repo suggests otherwise) |
| Related | physical-ai-portfolio (index), sentinel-node (measurement habits) | Per the docs |
| Builder credits | All 13 commits carry the same co-author credit naming an AI tool; the guide uses "AI coding agents" only. No agent instruction file in the tree | Confirmed |

## E2 Reading and running log

Read: README, all 11 docs, pyproject, workflow, .gitignore, src/triagebox/{pipeline,prompt,citations,rag,verify,audit,bundle,diode,integrity,metrics (head),judge,llamacpp (head),embed (head)}, tests/test_pipeline.py (head) and test titles for the rest, eval/isr/LOCK.txt, head of the ISR corpus and question files, scripts/download_models.sh (head), a selection of results JSON files (see E3).

Not read: scripts other than names and the download script head; power.py, bench.py, serverproc.py bodies; most of the tests; the data files of the benchmark (not committed); GitHub history beyond commit dates and the run list.

Ran (scratch copy): `pytest -q -p no:cacheprovider` (112 passed); probe script (E4); sha256 of the locked test corpus under LF and CRLF line endings (E8); gh api for repo metadata and run list; `gh run view --log-failed` for the newest run. Final state: `git status --short` in the clone prints nothing.

## E3 Claims check

Level L1 = the doc matches the committed result file; none was re-run (needs GPU, models, downloads).

| ID | Claim (source) | Check | Verdict |
| --- | --- | --- | --- |
| CL-01 | Ladder system-level accuracy / hallucination / false refusal for 5 configs (README, ladder.md) | results/qwen2.5-*-p2_dev.json system_level and model_level read: 49/76/31, 49/72/32, 28/33/61, 74/30/15, 74/31/15; model-level hallucination 0.92, 0.92, 0.90, 0.30, 0.31 | Supported (L1) |
| CL-02 | ISR locked test, 7B Q4_K_M: 90.5 / 4.8 / 11.1 (isr-set.md) | results/isr-qwen2.5-7b-q4_k_m-p2_test.json: accuracy 0.9048, false refusal 0.0476, hallucination 0.1111, n 42 and 18 | Supported (L1) |
| CL-03 | Quote policy on ISR test (detector.md): 1.5B 50-56% to 11%; 7B 11.1% unchanged, accuracy 90.5 to 76.2 | results/quote_policy_isr_test.json | Supported (L1) |
| CL-04 | Timing/energy table (ladder.md) | Values quoted from README/ladder.md; timing_ladder.json not recomputed | Per the docs |
| CL-05 | Hand-label results (17 supported of 31, 7 of 85 wrong citation, judge agreement 72% and 69.8%) | faithfulness.md only; hand_labels_dev.json not recomputed | Per the docs |
| CL-06 | Retrieval recall table (retrieval.md) | Per the docs; JSON not recomputed | Per the docs |
| CL-07 | Bootstrap intervals, prompt v3 results (prompt-tuning.md) | Per the docs | Per the docs |
| CL-08 | Locked test files unchanged (LOCK.txt; test_isr_set.py) | Fingerprint in LOCK.txt equals the CRLF form of the committed file, not the committed LF blob (see E8) | Partly: content consistent, check fails on Linux |
| CL-09 | Protocol written before results (eval-protocol.md) | File added in the first commit (2026-10-16 20:00 -0400); first results commit 2026-10-16/17 | Supported by dates (My reading: no outside timestamp) |
| CL-10 | "The policy is not wired into triage()" (detector.md Limits; roadmap 0.6b) | pipeline.py has a QuoteCheck parameter and tests exist; latest commit message says wired | Contradicted by code (doc drift) |
| CL-11 | README status line says the locked test split "is still to come", then reports the ISR locked test run | Both appear in the same paragraph; the ISR set is the project's own locked split, the benchmark's test half is the unused one | Doc inconsistency (wording) |
| CL-12 | ISR test false refusal for 1.5B Q4_K_M: 45.2% (isr-set.md, prompt v3) vs 38.1% (detector.md no-policy row) | different prompt versions (v3 vs the quote-policy file's base run); not a discrepancy | Reconciled on review |

## E4 Rule attacks (scratch copy, stand-in model returning fixed replies)

Corpus: two one-sentence documents (harbor, air). Question: "Who owned the trawler Nordkap?"

| Probe | Result |
| --- | --- |
| Valid id, answer about flag, not owner | accepted (relevance not checked) |
| Invented id harbor#9 | rejected: not retrieved |
| Real id plus invented id | rejected |
| No citation | rejected |
| Reply "ANSWER: [harbor#0]" only | rejected: malformed |
| INSUFFICIENT_EVIDENCE | rejected as refusal |
| lower-case "answer:" tag; "Sure! ANSWER: ..." | rejected: malformed |
| Empty question, model cites harbor#0 | rejected: not retrieved (no hits) |
| Id with a space "[harbor #0]" | rejected: no citations found |
| Quote check, extractor returns the real flag sentence | accepted (verbatim, does not answer) |
| Quote check, extractor returns invented sentence | rejected, match 0.32 |
| Quote check, extractor returns NONE | rejected |
| Audit log: edit middle, reorder, remove middle | each detected at index 1 |
| Audit log: truncate end | not detected (documented) |
| Bundle: replayed seq, tampered payload, wrong key | each rejected with its own error |
| Equal-version merge conflict | reported, resolved by larger hash |
| Diode: extra field | dropped; strict mode raises |
| Boot check: modified, then missing file | each raises with a reason |
| chunk_text with overlap equal to window | ValueError |

Guide examples: BM25 toy over three chunks gave a single hit (0.90), "flag" not matching "flagged" (no stemming); quote_match of the invented sentence in the guide scores 0.405 against the real sentence. Toy examples, labelled as such.

## E5 Code map

src/triagebox: rag.py (tokenize, chunk_text, Bm25Index: k1 1.5, b 0.75, stable sort), prompt.py (four prompt versions, parse_response), citations.py (check_citations, lexical_support), pipeline.py (triage, QuoteCheck, TriageResult; the entry point), verify.py (quote prompts, quote_match via longest common run, other coverage measures), audit.py (hash chain, verify_chain), bundle.py (Ed25519 manifests, verify_bundle, merge), diode.py (allowlist export), integrity.py (boot_check), metrics.py (hallucination, false refusal, accuracy, citation validity), llamacpp.py (local server client), embed.py (dense and hybrid index), judge.py, power.py (GPU sampling and validity), bench.py, serverproc.py, dataset.py. bundle, diode and integrity are not imported by any other module or script (grep): only by tests. scripts/: run_eval, ladder, quote_cost, judge, detector_eval, retrieval_compare, bootstrap_ci, build_isr_set, prepare_squad, etc.

## E6 Model call sites

| Where | Model | Purpose | Settings | Cost |
| --- | --- | --- | --- | --- |
| llamacpp.LlamaServer.generate, via pipeline.triage and scripts/run_eval.py | Qwen2.5-Instruct 1.5B (FP16, Q8_0, Q4_K_M), 7B (Q8_0, Q4_K_M) GGUF | answerer | temperature 0, seed 0, max_tokens 256, ctx 4096, BM25 top 4 chunks of up to 200 words (code default chunk 120) | local; no money |
| pipeline.QuoteCheck.llm | Qwen2.5-7B Q8_0 | extractor for quote check; one extra call per answer that passes the citation policy | quote prompt v1 (v2 tried on dev), threshold 0.9 | per quote_cost.json: x1.9 to x2.8 time, x1.95 to x3.0 GPU energy |
| scripts/judge.py, judge.py | Qwen2.5-7B Q8_0; Phi-3-mini-4k-instruct 4-bit | faithfulness judge | single judge prompt | local |
| embed.EmbeddingClient | BGE-small-en-v1.5 Q8_0 GGUF | dense retrieval comparison | query prefix | local |

No hosted or paid model is called anywhere (grep for provider names finds only the phrase "OpenAI-compatible" in two docstrings). Model licences are per the docs (Apache-2.0 for Qwen2.5 GGUF releases; MIT for BGE-small and Phi-3-mini); not independently checked. Reason for choosing the Qwen2.5 family: licence and runtime availability only (design.md); not otherwise documented. ISR set text: per isr-set.md the reports and questions were written by an assistant, and the README calls the set "hand-written"; see questions.

## E7 Tests and CI

112 tests over 12 files, all unit level with fakes (scripted model, fake HTTP server, fake clock). No end-to-end test with a real model; no property tests; coverage not measured. Spot checks: test_invented_citation_is_rejected (asserts rejection, reason text and the invalid id), test_edit_in_the_middle_is_detected (audit), test_dev_and_test_share_no_documents_or_questions (ISR set). Written alongside the code by agents.

CI: runs `pytest -q` on Python 3.10 and 3.12 (pip install -e ".[dev]"). Newest run (2026-10-23, commit 70f99ca) failed: 1 failed, 111 passed on 3.12; failing test is test_locked_test_files_have_not_been_edited (test_corpus.jsonl fingerprint mismatch). Runs on 0cc037e, 11175ec, 440c6ce also failed; runs on earlier commits succeeded.

## E8 Plan versus results and history

Eval protocol and design in the scaffold commit 592bc09 (2026-10-16 20:00 -0400), before the first model result (956003e, 2026-10-17). ISR set built and locked in 0cc037e (2026-10-17 12:25); the ISR test result files carry meta date 2026-10-17T16:23Z. The lock hash for test_corpus.jsonl (217279c3...) equals the SHA-256 of the file with CRLF line endings; the blob in git (LF) hashes to 4edd470e..., which is what CI computes. Ran it: sed to add CRs gave 217279c3...; git cat-file blob gave 4edd470e.... Cause of the mismatch is therefore line endings, not content (My reading: lock computed on a Windows working copy with CRLF).

All 13 commits carry the same co-author credit naming an AI tool. Commit sizes were not analysed. Gates: no gates.yaml or approvals in this repo (E9 not applicable).

## E9 Gates and approvals

None in this repository.

## E10 Environment

No lockfile; dependencies [] with optional cryptography>=42; pytest>=8 in dev extra. Model and llama.cpp versions recorded in result meta (llama.cpp build b11700 per docs). Benchmark download fingerprint recorded in the docs. Results depend on model files and the GPU, so a fresh install reproduces the tests but not the measurements without downloads.

## E11 Security and privacy

No secrets found by reading tracked files; .gitignore excludes .env, key files and model files. Outside access: scripts/prepare_squad.py (benchmark download) and scripts/download_models.sh (model downloads); the library code talks only to a local server address. No security review (README). Data is public or fictional. Not scanned: git history beyond commit messages.

## E12 Shortcomings register

| # | Shortcoming | Severity | Disclosed? | Options | Recommendation |
| --- | --- | --- | --- | --- | --- |
| 1 | Citation validity does not mean relevance or support; quote check accepts a real non-answering sentence | High | Yes (detector.md, README) | A: question-to-passage check by a model; B: require key question terms in the quote; C: document | A or B then measure; medium effort, high payoff |
| 2 | CI red since the lock commit (line endings in the lock fingerprint) | Medium | No | A: regenerate LOCK.txt from the LF blob and add a gitattributes rule; B: hash after normalising line endings | A or B; small effort, medium payoff (note: changing the lock needs a statement that content is unchanged) |
| 3 | Detector evidence is thin: 19 unsupported SQuAD answers, one labeller; ISR set easy and agent-written | Medium | Yes | A: second human labeller; B: new adversarial set from public documents | A; medium |
| 4 | Bundles, one-way export and boot check are not wired into any pipeline or script | Medium | Partly (README: "implemented and unit-tested") | A: build the sync demo (roadmap 0.9); B: label clearly as standalone | A |
| 5 | Edge-device (Jetson) numbers absent; laptop GPU energy mirrors latency | Medium | Yes | Run the ladder on the device | small-medium effort, high payoff |
| 6 | Prompt tuned on the 7B dev questions that are also reported; one run per setting | Medium | Yes | Report test-half numbers once; repeat with different seeds | |
| 7 | Stale docs: detector.md Limits and roadmap 0.6b say quote check is not wired; README status line mentions the locked test split as still to come | Low | No | Update | small |
| 8 | Minor number mismatch between isr-set.md and quote_policy_isr_test.json for 1.5B Q4_K_M false refusal (45.2 vs 38.1) | Low | No | Reconcile | small |
| 9 | Audit tail truncation not detectable | Low | Yes | Anchor the head hash in each signed bundle (the module's own suggestion) | |

## E13 Method detail

BM25: idf = ln(1 + (N - df + 0.5)/(df + 0.5)); score = sum over query tokens present of idf * tf*(k1+1)/(tf + k1*(1 - b + b*len/avg_len)), k1 1.5, b 0.75. Tokenizer: lower-case alphanumeric runs, 30 stop words, no stemming. Only query tokens that appear somewhere in the corpus are kept. Toy example in the guide computed with the repo's function (idf of "trawler" 0.981; score 0.90). quote_match: longest common substring of normalised text divided by quote length (difflib SequenceMatcher). Bootstrap: 2,000 resamples of questions (per docs). Recall@k and RRF (k = 60) per retrieval.md.

## E14 Discrepancies

See CL-08, CL-10, CL-11, CL-12. Also: README says 200-word chunks in experiments while the code default is 120 (experiments pass 200); README calls ISR set "hand-written" while isr-set.md says the assistant wrote the reports and questions.

## E15 Production readiness

Placement: Works locally. Clean start: yes for tests (pip install, pytest); measurements need downloads. Checks block a bad change: CI exists but is currently red. Quality measured against baselines: yes (dev and locked splits, intervals). Safe failure: refusal by default; no timeouts beyond a request timeout in the client. Human review: none. Versioned prompts: yes (four numbered versions recorded in results). Logs: hash-chained audit log; GPU sampling logs in runs. Data and secrets: public or fictional data; no secrets found. Missing for deployable: edge run, hosting, sign-in, monitoring, rollback, security review, key management.

## E16 Review dispositions

Accuracy review done 2026-10-28; see triagebox.review.md. On a Windows checkout (autocrlf) the lock test passes and 112 tests pass; on Linux CI 111 pass and 1 fails (line endings).

## E17 Completeness and backlog

Done and tested (unit level): retrieval, prompt protocol, citation policy, metrics, bundles, diode, audit, boot check, dataset converters. Measured on dev: ladder, retrieval comparison, prompt tuning, judge, quote check. Measured once on the ISR test half: five configurations. Not done: Jetson measurements (planned 0.8), disconnected-sync demo (0.9), LoRA or distilled model comparison (later), second human labeller, benchmark test half, wall-power measurement. Order: CI fix, doc drift fix, Jetson run, sync demo. Milestone exit test: five settings measured on the device with the same protocol.

## Diagram sources

Images are rendered PNGs (triagebox-learned-vs-rules, -parts, -results), drawn with matplotlib in a throwaway venv; every number in the results chart equals a number in the README ladder tables. Parts picture shows bundles, export and boot check as grey dashed, matching E5 (not wired).

## Revision log

First draft written, then trimmed from about 5,300 to about 4,850 plain-wc words (including table markup). Images viewed after rendering; no clipped or overlapping text remained after one layout fix.

## Revision log (v2 revision after reader check)

Previous text archived as triagebox.v1draft.md (4,862 words); revised guide is about 4,100 words (plain wc -w). All ten reader-check fixes applied.

1. Opening now says what "small model" and "answer quality" mean before the numbers.
2. Opening numbers cut to 74 of 100 right and 31 of 100 wrongly answered; the "14 of 100" figure is stated with its dependence on one hand labeller. Other figures moved to the results chapter.
3. Trawler example introduced in the problem chapter, before the results table; "wrongly answered" defined there and in the results chapter.
4. FP16, Q8_0, Q4_K_M, Ed25519, GGUF, llama.cpp, dev/test halves, bootstrap and hash defined at first use; SHA-256, CC BY-SA, replay, allowlist, stemming detail and similar dropped or moved here. Glossary extended (hash, bootstrap, CI, dev and test halves).
5. Modules chapter trimmed to the question path; signed bundles, export and boot check become one short aside saying they are not wired into anything; licences left to these notes (E1, E13).
6. Prompt versions v1 to v4 reduced to a short story; judge percentages reduced to "about 70%".
7. Edge cases grouped (refused, three that get through); CI story stated in two sentences: red since the locked-test commit, lock fingerprint matches Windows line endings, stored file has Unix ones.
8. Results reduced to the table plus three takeaways; caveats grouped in one paragraph.
9. Trawler blind spot told once (problem chapter) and called the relevance gap thereafter; "laptop only" stated once in the results caveats.
10. Tagline softened to "refuses any answer that does not cite one of the passages it was shown", with a line in the opening that the rules check form, not relevance.

Also: first-person "I closed the hole" and "I tried" changed to neutral wording (not evidenced as the author's acts). Images unchanged: no label or number changed (the parts picture keeps its Ed25519 label, now defined in the text); both previously viewed, no clipped or overlapping text. No tool named as builder; no local paths or secrets in the guide. Not published.
