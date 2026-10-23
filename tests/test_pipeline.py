from triagebox.audit import AuditLog, verify_chain
from triagebox.pipeline import triage
from triagebox.rag import Bm25Index, Chunk

CORPUS = [
    Chunk("port#0", "port", "The cargo ship docked at the northern port on Tuesday."),
    Chunk("air#0", "air", "Satellite imagery shows three trucks near the airfield."),
]
INDEX = Bm25Index(CORPUS)


class Scripted:
    """Test double that returns a fixed reply and records the prompt it saw."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply


def run(reply, question="Where are the trucks?", **kw):
    return triage(question, INDEX, Scripted(reply), **kw)


def test_a_properly_cited_answer_is_accepted():
    r = run("ANSWER: Three trucks are near the airfield [air#0].")
    assert r.accepted and r.parsed.kind == "answer"
    assert r.retrieved_ids[0] == "air#0"


def test_invented_citation_is_rejected():
    r = run("ANSWER: Three trucks are near the airfield [made-up#7].")
    assert not r.accepted and "not retrieved" in r.reason
    assert r.check.invalid == ("made-up#7",)


def test_one_good_and_one_invented_citation_is_still_rejected():
    r = run("ANSWER: Trucks [air#0] and a convoy [ghost#1].")
    assert not r.accepted


def test_answer_without_citations_is_rejected():
    r = run("ANSWER: Three trucks are near the airfield.")
    assert not r.accepted and r.reason == "answer has no citations"


def test_refusal_and_malformed_are_never_accepted_as_answers():
    assert not run("INSUFFICIENT_EVIDENCE").accepted
    r = run("I think probably three trucks")
    assert not r.accepted and "protocol" in r.reason


def test_citing_a_passage_that_exists_but_was_not_retrieved_is_rejected():
    # port#0 is in the corpus but this question retrieves only the airfield chunk.
    r = run("ANSWER: Trucks [port#0].", question="trucks airfield")
    assert "port#0" not in r.retrieved_ids and not r.accepted


def test_prompt_given_to_the_model_contains_only_retrieved_passages():
    llm = Scripted("INSUFFICIENT_EVIDENCE")
    triage("trucks airfield", INDEX, llm)
    assert "[air#0]" in llm.prompts[0] and "[port#0]" not in llm.prompts[0]


def test_every_call_is_audited_and_the_log_verifies():
    log = AuditLog()
    run("ANSWER: Three trucks [air#0].", audit=log, ts="t0")
    run("INSUFFICIENT_EVIDENCE", question="submarine", audit=log, ts="t1")
    assert len(log.entries) == 2 and verify_chain(log.entries) is None
    assert log.entries[0]["event"]["accepted"] is True
    assert log.entries[1]["event"]["kind"] == "refusal"
    log.entries[0]["event"]["accepted"] = False  # tamper
    assert verify_chain(log.entries) == 0


def test_grounding_policy_rejects_an_answer_the_cited_passage_does_not_contain():
    invented = "ANSWER: Seven tanks are near the airfield [air#0]."
    assert run(invented).accepted  # structurally fine, so accepted without the policy
    r = run(invented, min_support=0.8)
    assert not r.accepted and "not found in the cited passages" in r.reason


def test_grounding_policy_keeps_a_grounded_answer():
    assert run("ANSWER: Three trucks near the airfield [air#0].", min_support=0.8).accepted


def test_grounding_policy_checks_the_cited_passage_not_just_any_retrieved_one():
    # "docked" appears only in port#0 (retrieved), but the answer cites air#0.
    r = run("ANSWER: The ship docked [air#0].", question="Where did the cargo ship dock at the airfield port?", min_support=0.8)
    assert not r.accepted


# ---------------------------------------------------------------------------------------------- quote check
from triagebox.pipeline import QuoteCheck  # noqa: E402

GOOD = "ANSWER: Three trucks are near the airfield [air#0]."


def test_quote_check_accepts_an_answer_the_extractor_can_quote():
    ex = Scripted("Satellite imagery shows three trucks near the airfield.")
    r = run(GOOD, quote_check=QuoteCheck(ex))
    assert r.accepted and r.quote.startswith("Satellite imagery") and r.quote_match == 1.0
    assert len(ex.prompts) == 1 and "Three trucks are near the airfield" in ex.prompts[0] and "[air#0]" in ex.prompts[0]
    assert "[air#0]" not in ex.prompts[0].split("Answer given:")[1]  # the citation tag is stripped from the answer


def test_quote_check_rejects_when_the_extractor_finds_no_supporting_sentence():
    r = run(GOOD, quote_check=QuoteCheck(Scripted("NONE")))
    assert not r.accepted and "copied verbatim" in r.reason and r.quote is None


def test_quote_check_rejects_a_quote_that_is_not_in_the_passage():
    r = run(GOOD, quote_check=QuoteCheck(Scripted("Three tanks were seen near the harbour at dawn.")))
    assert not r.accepted and "not found verbatim" in r.reason and r.quote_match < 0.9


def test_quote_check_is_not_asked_when_the_citation_policy_already_refused():
    ex = Scripted("anything")
    for reply in ("INSUFFICIENT_EVIDENCE", "ANSWER: Trucks [ghost#1].", "ANSWER: no citation here.", "garbled"):
        assert not run(reply, quote_check=QuoteCheck(ex)).accepted
    assert ex.prompts == []


def test_quote_check_threshold_and_audit():
    log = AuditLog()
    ex = Scripted("Satellite imagery shows three trucks near the airfield.")
    run(GOOD, audit=log, ts="t1", quote_check=QuoteCheck(ex))
    assert log.entries[0]["event"]["quote_match"] == 1.0 and log.entries[0]["event"]["quote"].startswith("Satellite")
    # a half-right quote passes a loose threshold and fails the default one
    partial = Scripted("Satellite imagery shows three trucks near the harbour.")
    assert not run(GOOD, quote_check=QuoteCheck(partial)).accepted
    assert run(GOOD, quote_check=QuoteCheck(partial, threshold=0.8)).accepted
    assert verify_chain(log.entries) is None
