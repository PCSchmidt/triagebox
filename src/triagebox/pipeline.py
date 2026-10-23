"""Retrieve → generate → enforce citations → audit."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .audit import AuditLog
from .citations import CitationCheck, check_citations, lexical_support
from .prompt import ParsedResponse, build_prompt, parse_response
from .rag import Bm25Index
from .verify import QUOTE_PROMPTS, clean_quote, quote_match


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class QuoteCheck:
    """Quote-then-verify support check (docs/detector.md): an extractor model is asked to copy, word for word, the
    sentence of the cited passage(s) that states the answer; the answer is kept only if at least ``threshold`` of
    that copy occurs verbatim in a cited passage. The model's job is extraction; the verdict is deterministic.
    The check costs one more model call per answer that passed the citation policy."""

    llm: LLM  # the extractor; it may be a different, larger model than the answerer
    prompt_version: int = 1
    threshold: float = 0.9


_CITATION_TAG = re.compile(r"\s*\[[^\[\]\s]+\]")


@dataclass(frozen=True)
class TriageResult:
    question: str
    parsed: ParsedResponse  # what the model said
    accepted: bool  # True only for an answer that passed the citation policy
    check: CitationCheck | None
    retrieved_ids: tuple[str, ...]
    reason: str  # why it was accepted or rejected
    quote: str | None = None  # the extractor's copied sentence, when the quote check ran
    quote_match: float | None = None  # how much of it occurs verbatim in a cited passage


def triage(
    question: str,
    index: Bm25Index,
    llm: LLM,
    k: int = 4,
    audit: AuditLog | None = None,
    ts: str = "",
    prompt_version: int = 1,
    min_support: float | None = None,
    quote_check: QuoteCheck | None = None,
) -> TriageResult:
    """Answer ``question`` from the index, rejecting any answer that breaks the citation policy.

    Policy: an answer is accepted only if it cites at least one passage and every
    citation is a passage that was retrieved for this question. Refusals and
    malformed replies are never accepted as answers. With ``min_support`` set, an
    answer is also rejected unless at least that fraction of its content words
    appear in the passages it cites (a lexical grounding check; see docs/faithfulness.md). With ``quote_check``
    set, an answer that passes everything else is also rejected unless an extractor model can copy a sentence
    from the cited passages that contains it (see :class:`QuoteCheck`).
    """
    hits = index.search(question, k=k)
    chunks = [c for c, _ in hits]
    retrieved = tuple(c.id for c in chunks)
    raw = llm.generate(build_prompt(question, chunks, prompt_version))
    parsed = parse_response(raw)

    check = None
    quote, qmatch = None, None
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
        elif min_support is not None and lexical_support(parsed.text, [c.text for c in chunks if c.id in check.valid]) < min_support:
            accepted, reason = False, f"answer words are not found in the cited passages (support < {min_support})"
        else:
            accepted, reason = True, "all citations refer to retrieved passages"
            if quote_check is not None:
                cited = [c for c in chunks if c.id in check.valid]
                answer = _CITATION_TAG.sub("", parsed.text).strip()
                shown = "\n\n".join(f"[{c.id}] {c.text}" for c in cited)
                prompt = QUOTE_PROMPTS[quote_check.prompt_version].format(passages=shown, question=question, answer=answer)
                quote = clean_quote(quote_check.llm.generate(prompt))
                qmatch = quote_match(quote, [c.text for c in cited]) if quote else 0.0
                if qmatch < quote_check.threshold:
                    accepted = False
                    reason = "no sentence of the cited passages, copied verbatim, states the answer" if quote is None else f"the quoted sentence is not found verbatim in the cited passages (match {qmatch:.2f})"

    if audit is not None:
        audit.append(
            {
                "question": question,
                "retrieved": list(retrieved),
                "raw_reply": raw,
                "kind": parsed.kind,
                "accepted": accepted,
                "reason": reason,
                **({"quote": quote, "quote_match": qmatch} if quote_check is not None and parsed.kind == "answer" and check is not None and check.valid else {}),
            },
            ts,
        )
    return TriageResult(question, parsed, accepted, check, retrieved, reason, quote, qmatch)
