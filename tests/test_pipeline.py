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
