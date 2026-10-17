"""How often does BM25 put the gold paragraph in the top k? (No language model involved.)

    python scripts/retrieval_check.py --split dev

Recall@k is measured over answerable questions: the gold paragraph counts as
retrieved if any of the top-k chunks comes from it. Run on the dev split only
while building the system; the test split is for final numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from triagebox.dataset import load_corpus, load_items
from triagebox.rag import Bm25Index, chunk_text

DATA = Path(__file__).resolve().parents[1] / "data" / "squad_v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--max-words", type=int, default=200)
    args = ap.parse_args()

    docs = load_corpus(DATA / f"{args.split}_corpus.jsonl")
    items = [i for i in load_items(DATA / f"{args.split}_questions.jsonl") if i.answerable]
    chunks = [c for doc_id, text in docs.items() for c in chunk_text(doc_id, text, max_words=args.max_words, overlap=0)]
    index = Bm25Index(chunks)
    print(f"{args.split}: {len(docs)} documents -> {len(chunks)} chunks; {len(items)} answerable questions")

    ks = (1, 2, 4, 8)
    hits = {k: 0 for k in ks}
    for it in items:
        ranked = [c.doc_id for c, _ in index.search(it.question, k=max(ks))]
        for k in ks:
            if it.gold_docs[0] in ranked[:k]:
                hits[k] += 1
    for k in ks:
        print(f"  recall@{k}: {hits[k] / len(items):.3f}  ({hits[k]}/{len(items)})")


if __name__ == "__main__":
    main()
