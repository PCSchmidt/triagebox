"""Retrieve → generate → enforce citations → audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .audit import AuditLog
from .citations import CitationCheck, check_citations
from .prompt import ParsedResponse, build_prompt, parse_response
from .rag import Bm25Index


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class TriageResult:
    question: str
    parsed: ParsedResponse  # what the model said
    accepted: bool  # True only for an answer that passed the citation policy
    check: CitationCheck | None
    retrieved_ids: tuple[str, ...]
    reason: str  # why it was accepted or rejected


def triage(
    question: str,
    index: Bm25Index,
    llm: LLM,
    k: int = 4,
    audit: AuditLog | None = None,
    ts: str = "",
    prompt_version: int = 1,
) -> TriageResult:
    """Answer ``question`` from the index, rejecting any answer that breaks the citation policy.

    Policy: an answer is accepted only if it cites at least one passage and every
    citation is a passage that was retrieved for this question. Refusals and
    malformed replies are never accepted as answers.
    """
    hits = index.search(question, k=k)
    chunks = [c for c, _ in hits]
    retrieved = tuple(c.id for c in chunks)
    raw = llm.generate(build_prompt(question, chunks, prompt_version))
    parsed = parse_response(raw)

    check = None
    if parsed.kind == "refusal":
        accepted, reason = False, "model reported insufficient evidence"
    elif parsed.kind == "malformed":
        accepted, reason = False, "reply did not follow the ANSWER / INSUFFICIENT_EVIDENCE protocol"
    else:
        check = check_citations(parsed.text, set(retrieved))
        if not check.cited:
            accepted, reason = False, "answer has no citations"
        elif check.invalid:
            accepted, reason = False, f"answer cites passages that were not retrieved: {list(check.invalid)}"
        else:
            accepted, reason = True, "all citations refer to retrieved passages"

    if audit is not None:
        audit.append(
            {
                "question": question,
                "retrieved": list(retrieved),
                "raw_reply": raw,
                "kind": parsed.kind,
                "accepted": accepted,
                "reason": reason,
            },
            ts,
        )
    return TriageResult(question, parsed, accepted, check, retrieved, reason)
