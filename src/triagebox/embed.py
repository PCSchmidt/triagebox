"""Embedding retrieval and a BM25 + embedding hybrid, with the same ``search`` interface as ``Bm25Index``.

Standard library only: vectors come from a local ``llama-server --embeddings``
(OpenAI-compatible ``/v1/embeddings``) and are compared with a plain dot product on
unit vectors. That is slow for large corpora and fine for the thousands of chunks
used here; a vector store (sqlite-vec) would replace ``DenseIndex`` at scale.
"""

from __future__ import annotations

import json
import math
import urllib.request
from typing import Sequence

from .rag import Bm25Index, Chunk

# BGE models retrieve better when the *query* (not the passages) carries this prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8081", timeout: float = 300.0, batch: int = 16):
        self.base_url, self.timeout, self.batch = base_url.rstrip("/"), timeout, batch

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            body = json.dumps({"input": list(texts[i : i + self.batch])}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/v1/embeddings", data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = sorted(json.loads(resp.read())["data"], key=lambda d: d["index"])
            out.extend(_unit(d["embedding"]) for d in data)
        return out


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class DenseIndex:
    def __init__(self, chunks: list[Chunk], client: EmbeddingClient, query_prefix: str = BGE_QUERY_PREFIX):
        if len({c.id for c in chunks}) != len(chunks):
            raise ValueError("chunk ids must be unique")
        self.chunks, self.client, self.query_prefix = chunks, client, query_prefix
        self._vecs = client.embed([c.text for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = self.client.embed([self.query_prefix + query])[0]
        scored = [(c, sum(a * b for a, b in zip(q, v))) for c, v in zip(self.chunks, self._vecs)]
        scored.sort(key=lambda cs: -cs[1])
        return scored[:k]


class HybridIndex:
    """Reciprocal rank fusion of BM25 and dense rankings: score = sum 1 / (rrf_k + rank)."""

    def __init__(self, bm25: Bm25Index, dense: DenseIndex, rrf_k: int = 60, depth: int = 50):
        self.bm25, self.dense, self.rrf_k, self.depth = bm25, dense, rrf_k, depth

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for ranking in (self.bm25.search(query, k=self.depth), self.dense.search(query, k=self.depth)):
            for rank, (c, _) in enumerate(ranking, 1):
                fused[c.id] = fused.get(c.id, 0.0) + 1.0 / (self.rrf_k + rank)
                chunks[c.id] = c
        order = sorted(fused, key=lambda cid: -fused[cid])
        return [(chunks[cid], fused[cid]) for cid in order[:k]]
