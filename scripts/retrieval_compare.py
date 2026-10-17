"""BM25 vs embeddings vs hybrid retrieval on the same questions (no answering model involved).

    python scripts/retrieval_compare.py --split dev

Embeddings: BGE-small-en-v1.5 (Q8_0 GGUF, MIT licence) served by llama.cpp. Recall@k is
over answerable questions: the gold paragraph counts as found if any of the top-k
chunks comes from it. Dev split only while building; per the project rules the test
split is for final numbers. Bootstrap 95% intervals are over questions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from triagebox.dataset import load_corpus, load_items
from triagebox.embed import DenseIndex, EmbeddingClient, HybridIndex
from triagebox.rag import Bm25Index, chunk_text
from triagebox.serverproc import ServerProc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "squad_v2"
EXE = ROOT / "tools" / "llama-b11700" / "llama-server.exe"
MODEL = ROOT / "models" / "bge-small" / "bge-small-en-v1.5-q8_0.gguf"
KS = (1, 2, 4, 8)


def ci(flags: list[int], resamples: int = 2000) -> tuple[float, float, float]:
    rng = random.Random(0)
    n = len(flags)
    s = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return sum(flags) / n, s[int(0.025 * resamples)], s[int(0.975 * resamples) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    docs = load_corpus(DATA / f"{args.split}_corpus.jsonl")
    items = [i for i in load_items(DATA / f"{args.split}_questions.jsonl") if i.answerable]
    chunks = [c for d, t in docs.items() for c in chunk_text(d, t, max_words=200, overlap=0)]
    print(f"{args.split}: {len(chunks)} chunks, {len(items)} answerable questions", flush=True)

    bm25 = Bm25Index(chunks)
    with ServerProc(EXE, MODEL, port=args.port, ctx=4096, log=ROOT / "data" / "logs" / "embed_server.log",
                    extra_args=("--embeddings", "-b", "2048", "-ub", "2048")) as s:
        dense = DenseIndex(chunks, EmbeddingClient(s.url))
        hybrid = HybridIndex(bm25, dense)
        systems = {"BM25": bm25, "BGE-small dense": dense, "Hybrid (RRF)": hybrid}
        out = {}
        for name, index in systems.items():
            hits = {k: [] for k in KS}
            for it in items:
                ranked = [c.doc_id for c, _ in index.search(it.question, k=max(KS))]
                for k in KS:
                    hits[k].append(int(it.gold_docs[0] in ranked[:k]))
            out[name] = {f"recall@{k}": dict(zip(("estimate", "low", "high"), (round(x, 3) for x in ci(hits[k])))) for k in KS}
            print(name, "  ".join(f"@{k} {100 * out[name][f'recall@{k}']['estimate']:.1f}% [{100 * out[name][f'recall@{k}']['low']:.0f}-{100 * out[name][f'recall@{k}']['high']:.0f}]" for k in KS), flush=True)
    (ROOT / "results" / f"retrieval_compare_{args.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
