# Design

## Data flow

```text
question ─▶ retrieve (BM25 → later embeddings) ─▶ prompt with passages ─▶ LLM (quantised, local)
                                                                              │
         audit log ◀── accept / reject ◀── citation policy ◀── parse reply ◀──┘
              │
              └─▶ summaries ─▶ diode export / signed bundle ─▶ (link, when available)
```

The model never sees anything but the retrieved passages and the question, and its reply must follow a fixed protocol, so refusals and citations are checked by code, not by guessing at phrasing.

## Decisions made

- **Citation-forced, protocol-checked answers.** `ANSWER:` with `[passage-id]` citations or `INSUFFICIENT_EVIDENCE`. An answer with no citation, or citing a passage that was not retrieved for that question, is rejected.
- **Validity is structural.** The policy proves a citation is to something the model was shown, not that it supports the claim. The evaluation adds a lexical proxy and a judge model with hand-checked samples.
- **BM25 first.** A strong, dependency-free baseline; embedding retrieval is added later and compared on the same questions.
- **Signed bundles with Ed25519, sequence numbers and merge-by-version.** Chosen so a disconnected node can accept updates it can fully verify, reject replays, and converge deterministically after conflicting edits.
- **Allowlist diode.** Only id, summary, citations and timestamp can leave; strict mode makes extra fields an error.
- **Fail-closed boot.** Model and index files are hashed against a manifest; any missing or changed file stops the box.
- **Hash-chained audit.** Every prompt and reply is recorded in a chain that exposes edits, reordering and removal.

## Open decisions

- **Model family and sizes.** A permissively licensed family at about 3B and 7-8B parameters. A 16 GB desktop GPU holds 7-8B models at INT8 and INT4 but not FP16; the FP16 point may need a smaller model.
- **Runtimes.** llama.cpp (GGUF quantisations) first because it runs natively on Windows and on Jetson. vLLM does not run natively on Windows. TensorRT-based LLM runtimes on Windows and Jetson need checking; their inclusion depends on what installs.
- **Evaluation corpus.** Which public documents; licences recorded per document.
- **Judge model** for faithfulness, and how its agreement with hand-checked labels is reported.
- **Distillation.** Whether a distilled or LoRA-tuned small model is worth building after the quantisation ladder shows where quality falls off.

## Known limitations

- No language model has been run; the `LLM` protocol has only test doubles.
- Key management and revocation, secure storage of the audit head, and any real security review are out of scope so far.
