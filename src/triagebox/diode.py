"""One-way "diode" export: only an allowlisted summary leaves the box.

The export keeps just the fields in ``ALLOWED`` for each record, so source text,
prompts and anything else the record happens to carry cannot cross the boundary.
In strict mode an unexpected field is an error rather than silently dropped.
"""

from __future__ import annotations

import json

ALLOWED = ("id", "summary", "citations", "timestamp")
MAX_SUMMARY_CHARS = 1000


class DiodeViolation(Exception):
    pass


def export_summaries(records: list[dict], strict: bool = False) -> bytes:
    out = []
    for r in records:
        extra = set(r) - set(ALLOWED)
        if extra and strict:
            raise DiodeViolation(f"record {r.get('id')!r} has non-exportable fields: {sorted(extra)}")
        kept = {k: r[k] for k in ALLOWED if k in r}
        if "summary" in kept:
            if not isinstance(kept["summary"], str):
                raise DiodeViolation("summary must be a string")
            kept["summary"] = kept["summary"][:MAX_SUMMARY_CHARS]
        out.append(kept)
    return json.dumps(out, sort_keys=True, separators=(",", ":")).encode("utf-8")
