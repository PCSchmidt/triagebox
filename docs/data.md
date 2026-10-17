# Data

## Benchmark v0: SQuAD 2.0 (dev file)

The first evaluation set is a subset of SQuAD 2.0, a public benchmark with ground truth, not a hand-made toy set.

- **Source:** Rajpurkar, Jia and Liang, "Know What You Don't Know: Unanswerable Questions for SQuAD" (ACL 2018). Licensed CC BY-SA 4.0. <https://rajpurkar.github.io/SQuAD-explorer/>
- **Why this one:** about half the questions are unanswerable, and they were written by crowdworkers to look answerable from a paragraph that does not contain the answer. They are near-miss unanswerables, which is the case where a small model is most tempted to invent an answer.
- **What it is not:** Wikipedia text, not intelligence-style reporting. It measures refusal and citation behaviour and the cost of quantisation; it says nothing about domain performance. An ISR-flavoured set built from public documents is still planned (see the evaluation protocol).
- **Provenance:** `scripts/prepare_squad.py` downloads the dev file (SHA-256 `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8` at the time of download, recorded in `data/squad_v2/SOURCE.txt`) and builds the splits below. The generated files are not committed; rerun the script to recreate them.

### Splits

The 35 dev articles are shuffled with seed 0. Eighteen form the **development** split and seventeen the **test** split, so no article or paragraph appears in both.

| Split | Documents (paragraphs) | Answerable questions | Unanswerable questions |
| --- | --- | --- | --- |
| dev | 595 | 300 | 300 |
| test | 609 | 300 | 300 |

Prompts, retrieval settings and thresholds are tuned on **dev only**. The test split is used once per final configuration.

### Retrieval baseline (BM25, dev split, answerable questions)

Pure-Python BM25 over 200-word chunks (631 chunks). The gold paragraph counts as retrieved if any top-k chunk comes from it. No language model involved.

| k | Gold paragraph in top k |
| --- | --- |
| 1 | 80.3% (241/300) |
| 2 | 87.3% (262/300) |
| 4 | 93.7% (281/300) |
| 8 | 96.0% (288/300) |

With k = 4 a perfect model could still answer at most about 94% of answerable dev questions from what it is shown, so end-to-end accuracy figures need this ceiling beside them. Retrieval for unanswerable questions is not scored: they have no gold paragraph, and the paragraph they were written against usually *is* retrieved, which is what makes them hard.

### Attribution

This project's evaluation sets are derived from SQuAD 2.0 and carry its CC BY-SA 4.0 licence. Derived question files, if ever published, must be shared under the same terms with attribution.
