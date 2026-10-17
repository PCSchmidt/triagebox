"""Citation-forced prompt and response protocol.

The model must reply in one of two shapes:

* ``ANSWER: <text>`` with every claim followed by a citation such as ``[doc1#3]``;
  a reply that is only citations, with no answer text, is malformed
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

PROMPT_VERSIONS = (1, 2, 3, 4)


def build_prompt(question: str, chunks: list[Chunk], version: int = 1) -> str:
    """Prompt for ``question`` over ``chunks``.

    Version 2 adds an explicit format example. Versions 3 and 4 exist because small models
    copied v2's example id (``doc1#3``) or prefixed ``doc`` onto real ids: v3 uses a
    placeholder instead of an id, v4 uses the id of the first passage shown.
    """
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt version {version}")
    context = "\n\n".join(f"[{c.id}]\n{c.text}" for c in chunks) if chunks else "(no context retrieved)"
    if version == 1:
        head = (
            "You answer questions using ONLY the context below. Do not use outside knowledge.\n"
            f"If the context does not contain the answer, reply with exactly {REFUSE_TAG}.\n"
            f"Otherwise reply starting with '{ANSWER_TAG}' and follow every claim with the id of the "
            "supporting passage in square brackets, for example [doc1#3].\n\n"
        )
    elif version in (3, 4):
        example_id = "ID" if version == 3 else (chunks[0].id if chunks else "ID")
        head = (
            "You answer questions using ONLY the context below. Do not use outside knowledge.\n"
            "Each passage starts with its id in square brackets on its own line.\n"
            "Reply in exactly one of these two formats.\n"
            f"1) If the context answers the question: '{ANSWER_TAG} <the answer, in a few words or one sentence> [{example_id}]'. "
            "Replace the id in brackets with the id of the passage that contains the answer, copied exactly as it appears "
            "in the context, with nothing added to it. The answer text comes first.\n"
            f"2) If the context does not contain the answer: '{REFUSE_TAG}'.\n\n"
        )
    else:
        head = (
            "You answer questions using ONLY the context below. Do not use outside knowledge.\n"
            "Reply in exactly one of these two formats.\n"
            f"1) If the context answers the question: '{ANSWER_TAG} <the answer, in a few words or one sentence> "
            "[passage-id]'. "
            f"For example: '{ANSWER_TAG} 1998 [doc1#3]'. The answer text comes first; the bracketed id only says "
            "which passage supports it. Copy the id exactly as it appears in the context.\n"
            f"2) If the context does not contain the answer: '{REFUSE_TAG}'.\n\n"
        )
    return f"{head}CONTEXT:\n{context}\n\nQUESTION: {question}\n"


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
        if body.startswith(REFUSE_TAG):  # models sometimes write "ANSWER: INSUFFICIENT_EVIDENCE"
            return ParsedResponse("refusal", body[len(REFUSE_TAG) :].strip(" :-\n"))
        if not _CITATION.sub("", body).strip():  # only citations, no answer text
            return ParsedResponse("malformed", s)
        return ParsedResponse("answer", body, extract_citations(body))
    return ParsedResponse("malformed", s)
