"""Chunking and a pure-Python BM25 retriever.

Lexical retrieval is the baseline. Embedding retrieval (and a vector store such
as sqlite-vec) comes later and will be compared against this on the same
questions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were which with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc_id>#<n>", the string a model must cite
    doc_id: str
    text: str


def chunk_text(doc_id: str, text: str, max_words: int = 120, overlap: int = 20) -> list[Chunk]:
    """Split ``text`` into word windows of ``max_words`` with ``overlap`` words shared."""
    if max_words <= 0 or not 0 <= overlap < max_words:
        raise ValueError("need max_words > 0 and 0 <= overlap < max_words")
    words = text.split()
    if not words:
        return []
    step = max_words - overlap
    chunks = []
    for n, start in enumerate(range(0, len(words), step)):
        window = words[start : start + max_words]
        chunks.append(Chunk(f"{doc_id}#{n}", doc_id, " ".join(window)))
        if start + max_words >= len(words):
            break
    return chunks


class Bm25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if len({c.id for c in chunks}) != len(chunks):
            raise ValueError("chunk ids must be unique")
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(chunks)) if chunks else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Top ``k`` chunks with a positive score, best first (ties broken by order)."""
        q = [t for t in tokenize(query) if t in self._idf]
        scored = []
        for i, tf in enumerate(self._tf):
            norm = 1.0 - self.b + self.b * (self._len[i] / self._avg if self._avg else 0.0)
            s = sum(
                self._idf[t] * tf[t] * (self.k1 + 1.0) / (tf[t] + self.k1 * norm) for t in q if tf[t]
            )
            if s > 0.0:
                scored.append((self.chunks[i], s))
        scored.sort(key=lambda cs: -cs[1])  # stable: equal scores keep corpus order
        return scored[:k]
