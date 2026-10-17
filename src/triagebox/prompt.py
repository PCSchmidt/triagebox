"""Citation-forced prompt and response protocol.

The model must reply in one of two shapes:

* ``ANSWER: <text>`` with every claim followed by a citation such as ``[doc1#3]``
* ``INSUFFICIENT_EVIDENCE`` (optionally followed by a short reason)

Anything else is *malformed*. Fixing the protocol makes refusals and citations
machine-checkable instead of relying on phrase matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .rag import Chunk

ANSWER_TAG = "ANSWER:"
REFUSE_TAG = "INSUFFICIENT_EVIDENCE"
_CITATION = re.compile(r"\[([^\[\]\s]+)\]")


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[{c.id}]\n{c.text}" for c in chunks) if chunks else "(no context retrieved)"
    return (
        "You answer questions using ONLY the context below. Do not use outside knowledge.\n"
        f"If the context does not contain the answer, reply with exactly {REFUSE_TAG}.\n"
        f"Otherwise reply starting with '{ANSWER_TAG}' and follow every claim with the id of the "
        "supporting passage in square brackets, for example [doc1#3].\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n"
    )


@dataclass(frozen=True)
class ParsedResponse:
    kind: str  # "answer" | "refusal" | "malformed"
    text: str = ""
    citations: tuple[str, ...] = field(default_factory=tuple)


def extract_citations(text: str) -> tuple[str, ...]:
    """Citation ids in order of appearance, duplicates kept."""
    return tuple(_CITATION.findall(text))


def parse_response(raw: str) -> ParsedResponse:
    s = raw.strip()
    if s.startswith(REFUSE_TAG):
        return ParsedResponse("refusal", s[len(REFUSE_TAG) :].strip(" :-\n"))
    if s.startswith(ANSWER_TAG):
        body = s[len(ANSWER_TAG) :].strip()
        if not body:
            return ParsedResponse("malformed", s)
        return ParsedResponse("answer", body, extract_citations(body))
    return ParsedResponse("malformed", s)
