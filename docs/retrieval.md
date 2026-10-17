# Retrieval: BM25 vs embeddings vs hybrid (milestone 0.5)

Question: is retrieval the bottleneck? BM25 puts the gold paragraph in the top 4 for 93.7% of answerable dev questions, so it caps accuracy at that level. Does an embedding retriever do better?

Setup: dev split, 631 chunks of up to 200 words, 300 answerable questions. Embeddings are BGE-small-en-v1.5 (Q8_0 GGUF, MIT licence) served by llama.cpp, with the BGE query prefix on questions; cosine similarity. Hybrid is reciprocal rank fusion (k = 60) of the BM25 and dense rankings. Recall@k = the gold paragraph is among the top-k chunks. Bootstrap 95% intervals over questions. No answering model is involved (`scripts/retrieval_compare.py`).

| Retriever | Recall@1 | Recall@2 | Recall@4 | Recall@8 |
| --- | --- | --- | --- | --- |
| BM25 | 80.3% [76-84] | 87.3% [83-91] | 93.7% [91-96] | 96.0% [94-98] |
| BGE-small dense | 72.7% [68-77] | 83.7% [80-88] | 90.3% [87-94] | 94.7% [92-97] |
| Hybrid (RRF) | 78.3% [74-83] | 90.0% [87-93] | 95.0% [92-97] | 97.7% [96-99] |

## Reading it

- **Dense retrieval alone is worse than BM25** here, by 3 to 8 points, and the gap at recall@1 is outside the intervals.
- **The hybrid is at best slightly better** than BM25 at k = 2 to 8 (+1 to +3 points), and the intervals overlap heavily. At k = 1 it is not better. I did not run a paired test, so I would not claim the hybrid wins.
- **Retrieval is not the main bottleneck for the 7B models.** The ceiling at k = 4 is 94 to 95% against 74% accuracy; the loss is in the model's reading and in the citation policy.
- **Caveat that favours BM25:** SQuAD questions were written by people looking at the paragraph, so they share many words with it, which is the case lexical matching does best. Intelligence-style reports with paraphrase and abbreviation could reverse this; this benchmark cannot show it. Also, a 33M-parameter embedding model is small, and the corpus is only 631 chunks, so the retrieval problem is easy.

Not done: a larger embedding model, the test split, an end-to-end run with the hybrid retriever feeding the answering model.
