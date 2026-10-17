"""Quote-then-verify support check for answers.

A judge model is asked to copy, word for word, the sentence of the cited passage that
states the answer to the question. Code then checks that the copy really occurs in a
cited passage and that it covers the key words of both the question and the answer.
The model's job is extraction, which small models do far more reliably than judging;
the verdict itself is deterministic. See docs/detector.md for how well it works.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from .rag import tokenize

QUOTE_PROMPT = """Passage(s):
{passages}

Question: {question}
Answer given: {answer}

Copy, word for word, the single sentence from the passage(s) above that states this answer to this question.
If no sentence in the passage(s) states it, reply with exactly NONE.
Reply with only the copied sentence, or NONE."""

# Version 2 asks for the sentence that CONTAINS the answer's words, which small models do more willingly
# than judging whether a sentence "states" the answer.
QUOTE_PROMPT_V2 = """Passage(s):
{passages}

Question: {question}
Answer given: {answer}

Find the sentence in the passage(s) above that contains the words or numbers of the answer given, and copy it word for word.
If the answer's words or numbers do not appear in the passage(s), reply with exactly NONE.
Reply with only the copied sentence, or NONE."""

QUOTE_PROMPTS = {1: QUOTE_PROMPT, 2: QUOTE_PROMPT_V2}

# Question words and other tokens that say nothing about the topic asked about.
_QUESTION_WORDS = frozenset(
    "what which who whom whose when where why how many much did does do can could would should will "
    "between during after before near about any there than also both two".split()
)
_BRACKETED_ID = re.compile(r"\[[^\[\]\s]+\]")


def fold(s: str) -> str:
    """Lower-case and strip accents, so 'Czerniaków' matches 'Czerniakow'."""
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"'`]", "", fold(s))).strip(" .")


def clean_quote(reply: str) -> str | None:
    """The quoted sentence, or None if the reply is NONE or empty."""
    q = _BRACKETED_ID.sub("", reply).strip().strip("\"'`").strip()
    if not q or q.upper().rstrip(".") == "NONE":
        return None
    return q


def _stem(w: str) -> str:
    """Crude plural/verb-ending stripper, enough to match 'stand' with 'stands' and 'ship' with 'ships'."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def content_words(text: str, drop_question_words: bool = False) -> set[str]:
    words = set(tokenize(fold(_BRACKETED_ID.sub(" ", text))))
    if drop_question_words:
        words -= _QUESTION_WORDS
    return {_stem(w) for w in words}


def coverage(words: set[str], text: str) -> float:
    """Fraction of ``words`` that occur among the content words of ``text`` (1.0 for an empty set)."""
    if not words:
        return 1.0
    have = content_words(text)
    return len(words & have) / len(words)


def question_entities(question: str) -> set[str]:
    """Stemmed names, capitalised terms (not the first word) and numbers in the question."""
    toks = re.findall(r"[A-Za-z0-9À-ſ][\wÀ-ſ-]*", question)
    ents = set()
    for i, t in enumerate(toks):
        if any(c.isdigit() for c in t) or (i > 0 and t[0].isupper() and t.lower() not in _QUESTION_WORDS):
            ents |= content_words(t)
    return ents


def quote_match(quote: str, passages: list[str]) -> float:
    """How much of the quote occurs verbatim in a passage: longest common run / quote length (1.0 = verbatim)."""
    nq = norm(quote)
    if not nq:
        return 0.0
    best = 0.0
    for p in passages:
        np_ = norm(p)
        m = SequenceMatcher(None, nq, np_, autojunk=False).find_longest_match(0, len(nq), 0, len(np_))
        best = max(best, m.size / len(nq))
    return best


@dataclass(frozen=True)
class Support:
    match: float  # share of the quote found as one verbatim run in a cited passage (0 if there was no quote)
    question_coverage: float  # share of the question's topic words present in the cited passages
    entity_coverage: float  # share of the question's names and numbers present in the cited passages
    answer_coverage: float  # share of the answer's words present in the quote

    @property
    def found(self) -> bool:
        return self.match >= 0.9


def verify(question: str, answer: str, quote: str | None, cited_passages: list[str]) -> Support:
    joined = " ".join(cited_passages)
    return Support(
        quote_match(quote, cited_passages) if quote else 0.0,
        coverage(content_words(question, drop_question_words=True), joined),
        coverage(question_entities(question), joined),
        coverage(content_words(answer), quote) if quote else 0.0,
    )
