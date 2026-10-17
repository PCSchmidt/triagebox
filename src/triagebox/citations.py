"""Structural citation checks and a lexical grounding proxy.

What these checks do and do not establish:

* ``check_citations`` verifies that every cited id is one of the passages that
  was actually retrieved and shown to the model. It does NOT show the cited
  passage supports the claim.
* ``lexical_support`` is a cheap proxy for support: the fraction of a sentence's
  content words that appear in the cited passages. High overlap is necessary
  but not sufficient for a faithful claim, and low overlap flags a sentence for
  review. A judge model and spot-checks by hand are still needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .prompt import extract_citations
from .rag import tokenize

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CITATION_TEXT = re.compile(r"\[[^\[\]\s]+\]")


@dataclass(frozen=True)
class CitationCheck:
    cited: tuple[str, ...]
    valid: tuple[str, ...]  # cited ids that were in the retrieved set
    invalid: tuple[str, ...]  # cited ids that were not (invented or from elsewhere)
    sentences: int
    uncited_sentences: int

    @property
    def all_valid(self) -> bool:
        return bool(self.cited) and not self.invalid


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def check_citations(answer_text: str, retrieved_ids: set[str]) -> CitationCheck:
    cited = extract_citations(answer_text)
    valid = tuple(c for c in cited if c in retrieved_ids)
    invalid = tuple(c for c in cited if c not in retrieved_ids)
    sentences = split_sentences(answer_text)
    uncited = sum(1 for s in sentences if not extract_citations(s))
    return CitationCheck(cited, valid, invalid, len(sentences), uncited)


def lexical_support(sentence: str, passages: list[str]) -> float:
    """Fraction of the sentence's content words found in ``passages`` (0 if none)."""
    words = set(tokenize(_CITATION_TEXT.sub(" ", sentence)))
    if not words:
        return 0.0
    available = set()
    for p in passages:
        available.update(tokenize(p))
    return len(words & available) / len(words)
