import pytest

from triagebox.rag import Bm25Index, Chunk, chunk_text, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Port of Hodeidah is in Yemen.") == ["port", "hodeidah", "yemen"]


def test_chunk_windows_and_overlap():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_text("d", text, max_words=100, overlap=20)
    assert [c.id for c in chunks] == ["d#0", "d#1", "d#2"]
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert len(first) == 100 and first[-20:] == second[:20]
    assert chunks[-1].text.split()[-1] == "w249"  # nothing dropped at the end


def test_short_text_is_one_chunk_and_empty_is_none():
    assert len(chunk_text("d", "just a few words")) == 1
    assert chunk_text("d", "   ") == []


@pytest.mark.parametrize("max_words,overlap", [(0, 0), (10, 10), (10, -1)])
def test_bad_chunk_parameters_rejected(max_words, overlap):
    with pytest.raises(ValueError):
        chunk_text("d", "a b c", max_words=max_words, overlap=overlap)


CORPUS = [
    Chunk("a#0", "a", "The cargo ship docked at the northern port on Tuesday."),
    Chunk("b#0", "b", "Satellite imagery shows three trucks near the airfield."),
    Chunk("c#0", "c", "Weather over the strait was clear with light winds."),
]


def test_bm25_ranks_the_relevant_chunk_first():
    idx = Bm25Index(CORPUS)
    top = idx.search("trucks near the airfield", k=3)
    assert top[0][0].id == "b#0"
    assert all(score > 0 for _, score in top)


def test_bm25_returns_nothing_without_a_match_and_respects_k():
    idx = Bm25Index(CORPUS)
    assert idx.search("submarine torpedo", k=3) == []
    assert len(idx.search("the", k=3)) == 0  # stopword only
    assert len(idx.search("port trucks weather", k=2)) == 2


def test_bm25_empty_corpus_and_duplicate_ids():
    assert Bm25Index([]).search("anything") == []
    with pytest.raises(ValueError):
        Bm25Index([Chunk("x", "d", "one"), Chunk("x", "d", "two")])
