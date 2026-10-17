from triagebox.embed import DenseIndex, EmbeddingClient, HybridIndex, _unit
from triagebox.rag import Bm25Index, Chunk

CHUNKS = [Chunk("a#0", "a", "red apples grow on trees"), Chunk("b#0", "b", "blue whales swim in the sea"), Chunk("c#0", "c", "green grass")]


class FakeClient(EmbeddingClient):
    """Embeds text as counts of three marker words, so tests need no server."""

    def __init__(self):
        pass

    def embed(self, texts):
        keys = ("apples", "whales", "grass")
        return [_unit([float(t.lower().count(k)) + 0.01 for k in keys]) for t in texts]


def test_unit_vectors_have_length_one():
    assert abs(sum(x * x for x in _unit([3.0, 4.0])) - 1.0) < 1e-9


def test_dense_search_ranks_by_similarity():
    idx = DenseIndex(CHUNKS, FakeClient(), query_prefix="")
    assert idx.search("where do whales live", k=1)[0][0].id == "b#0"


def test_hybrid_rescues_a_query_bm25_cannot_match():
    bm25 = Bm25Index(CHUNKS)
    assert bm25.search("xyzzy", k=1) == []  # no lexical overlap at all
    hybrid = HybridIndex(bm25, DenseIndex(CHUNKS, FakeClient(), query_prefix=""))
    assert len(hybrid.search("apples xyzzy", k=2)) == 2


def test_hybrid_ranks_a_chunk_found_by_both_first():
    bm25 = Bm25Index(CHUNKS)
    hybrid = HybridIndex(bm25, DenseIndex(CHUNKS, FakeClient(), query_prefix=""))
    assert hybrid.search("apples", k=1)[0][0].id == "a#0"
