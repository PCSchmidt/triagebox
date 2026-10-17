"""Prompt and verdict parser for the faithfulness judge (see scripts/judge.py)."""

from __future__ import annotations

import re

JUDGE_PROMPT = """You are checking whether an answer is supported by a passage.

Passage(s):
{passages}

Question: {question}
Answer given: {answer}

Does the passage text above directly state or clearly imply this answer to this question?
Reply with exactly one line: either "SUPPORTED" or "NOT_SUPPORTED", then a dash and a short reason.
Say NOT_SUPPORTED if the passage is about a related topic but does not actually give this answer."""

_VERDICT = re.compile(r"^\s*(NOT_SUPPORTED|SUPPORTED)\b", re.IGNORECASE)


def parse_verdict(reply: str) -> str:
    """SUPPORTED, NOT_SUPPORTED, or UNPARSED when the reply does not start with a verdict."""
    m = _VERDICT.match(reply)
    return m.group(1).upper() if m else "UNPARSED"
