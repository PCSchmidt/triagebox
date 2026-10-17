import pytest

from triagebox.citations import check_citations, lexical_support, split_sentences
from triagebox.prompt import ParsedResponse, build_prompt, extract_citations, parse_response
from triagebox.rag import Chunk


def test_prompt_contains_every_passage_id_and_the_protocol():
    chunks = [Chunk("d1#0", "d1", "alpha text"), Chunk("d2#4", "d2", "beta text")]
    p = build_prompt("what is alpha?", chunks)
    assert "[d1#0]" in p and "[d2#4]" in p and "alpha text" in p
    assert "INSUFFICIENT_EVIDENCE" in p and "ANSWER:" in p and "what is alpha?" in p


def test_prompt_without_context_says_so():
    assert "no context retrieved" in build_prompt("q", [])


def test_parse_answer_with_citations():
    r = parse_response("ANSWER: Three trucks were seen [b#0]. The port was busy [a#0][a#0].")
    assert r.kind == "answer"
    assert r.citations == ("b#0", "a#0", "a#0")


def test_parse_refusal_with_and_without_reason():
    assert parse_response("INSUFFICIENT_EVIDENCE").kind == "refusal"
    r = parse_response("INSUFFICIENT_EVIDENCE: nothing about that")
    assert r.kind == "refusal" and r.text == "nothing about that"


@pytest.mark.parametrize("raw", ["", "Sure! The answer is 42.", "ANSWER:", "answer: lowercase tag"])
def test_everything_else_is_malformed(raw):
    assert parse_response(raw).kind == "malformed"


def test_extract_citations_ignores_bracketed_phrases_with_spaces():
    assert extract_citations("see [d1#2] and [not a citation] and [x]") == ("d1#2", "x")


def test_check_citations_flags_invented_ids():
    chk = check_citations("Trucks seen [b#0]. A convoy left [z#9].", {"a#0", "b#0"})
    assert chk.valid == ("b#0",) and chk.invalid == ("z#9",)
    assert not chk.all_valid
    assert chk.sentences == 2 and chk.uncited_sentences == 0


def test_check_citations_counts_uncited_sentences():
    chk = check_citations("Trucks seen [b#0]. Nobody knows why.", {"b#0"})
    assert chk.sentences == 2 and chk.uncited_sentences == 1 and chk.all_valid


def test_no_citations_is_not_all_valid():
    assert not check_citations("An unsupported claim.", {"a#0"}).all_valid


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_lexical_support_is_a_fraction_of_content_words():
    passage = ["Three trucks were parked near the northern airfield."]
    assert lexical_support("Trucks parked near the airfield [b#0]", passage) == 1.0
    partial = lexical_support("Trucks parked near the hangar", passage)
    assert 0.0 < partial < 1.0
    assert lexical_support("Submarines surfaced", passage) == 0.0
    assert lexical_support("", passage) == 0.0
