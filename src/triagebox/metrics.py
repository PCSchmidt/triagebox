"""Evaluation metrics, defined precisely.

Terms (see ``docs/eval-protocol.md``):

* **hallucination rate**: of the *unanswerable* questions, the fraction the system
  answered instead of refusing.
* **false refusal rate**: of the *answerable* questions, the fraction it refused.
* **accuracy**: of the *answerable* questions, the fraction answered with text
  containing a gold answer string (case- and whitespace-normalised; an item may
  list several acceptable strings). This is a
  crude containment test, suitable for short factual answers only.
* **citation validity**: of all citations in answers, the fraction pointing at a
  passage that was actually retrieved. Structural only; not support.
* **gold citation recall**: of answered answerable questions, the fraction citing
  at least one passage from a gold document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .citations import check_citations
from .prompt import ParsedResponse


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    answerable: bool
    gold_docs: tuple[str, ...] = ()
    gold_answer: str | None = None
    alt_answers: tuple[str, ...] = ()  # other acceptable answer strings


@dataclass(frozen=True)
class Outcome:
    item: EvalItem
    parsed: ParsedResponse
    retrieved_ids: frozenset[str]


@dataclass(frozen=True)
class Scores:
    n_answerable: int
    n_unanswerable: int
    accuracy: float | None
    false_refusal_rate: float | None
    hallucination_rate: float | None
    citation_validity: float | None
    gold_citation_recall: float | None
    malformed_rate: float | None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _contains_gold(item: EvalItem, text: str) -> bool:
    golds = [g for g in (item.gold_answer, *item.alt_answers) if g]
    return any(_norm(g) in _norm(text) for g in golds)


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _doc_of(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0]


def score(outcomes: list[Outcome]) -> Scores:
    answerable = [o for o in outcomes if o.item.answerable]
    unanswerable = [o for o in outcomes if not o.item.answerable]

    correct = sum(
        1
        for o in answerable
        if o.parsed.kind == "answer" and _contains_gold(o.item, o.parsed.text)
    )
    false_refusals = sum(1 for o in answerable if o.parsed.kind == "refusal")
    hallucinated = sum(1 for o in unanswerable if o.parsed.kind == "answer")

    cited = valid = 0
    for o in outcomes:
        if o.parsed.kind == "answer":
            chk = check_citations(o.parsed.text, set(o.retrieved_ids))
            cited += len(chk.cited)
            valid += len(chk.valid)

    answered_answerable = [o for o in answerable if o.parsed.kind == "answer"]
    gold_hit = sum(
        1
        for o in answered_answerable
        if any(_doc_of(c) in set(o.item.gold_docs) for c in o.parsed.citations)
    )
    malformed = sum(1 for o in outcomes if o.parsed.kind == "malformed")

    return Scores(
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        accuracy=_rate(correct, len(answerable)),
        false_refusal_rate=_rate(false_refusals, len(answerable)),
        hallucination_rate=_rate(hallucinated, len(unanswerable)),
        citation_validity=_rate(valid, cited),
        gold_citation_recall=_rate(gold_hit, len(answered_answerable)),
        malformed_rate=_rate(malformed, len(outcomes)),
    )
