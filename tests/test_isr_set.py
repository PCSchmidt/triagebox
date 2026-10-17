import hashlib
import re
from pathlib import Path

from triagebox.dataset import load_corpus, load_items

ISR = Path(__file__).resolve().parents[1] / "eval" / "isr"


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def test_locked_test_files_have_not_been_edited():
    expected = {}
    for line in (ISR / "LOCK.txt").read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line)
        if m:
            expected[m.group(2)] = m.group(1)
    assert set(expected) == {"test_corpus.jsonl", "test_questions.jsonl"}
    for name, digest in expected.items():
        assert hashlib.sha256((ISR / name).read_bytes()).hexdigest() == digest, f"{name} changed after it was locked"


def test_answerable_gold_answers_appear_in_their_gold_documents():
    for split in ("dev", "test"):
        docs = load_corpus(ISR / f"{split}_corpus.jsonl")
        for it in load_items(ISR / f"{split}_questions.jsonl"):
            if it.answerable:
                text = " ".join(norm(docs[d]) for d in it.gold_docs)
                assert any(norm(g) in text for g in (it.gold_answer, *it.alt_answers)), it.id
            else:
                assert not it.gold_docs and it.gold_answer is None, it.id


def test_dev_and_test_share_no_documents_or_questions():
    d, t = load_corpus(ISR / "dev_corpus.jsonl"), load_corpus(ISR / "test_corpus.jsonl")
    assert not set(d) & set(t) and not set(d.values()) & set(t.values())
    qd = {i.question for i in load_items(ISR / "dev_questions.jsonl")}
    qt = {i.question for i in load_items(ISR / "test_questions.jsonl")}
    assert not qd & qt
