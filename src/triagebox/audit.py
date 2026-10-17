"""Tamper-evident audit log: a hash chain over every prompt/answer event.

Each entry stores the hash of the previous entry, so editing, reordering or
removing an entry in the middle breaks verification. Truncating the END of the
log cannot be detected from the log alone; keep the latest hash somewhere the box
cannot rewrite (for example, include it in each signed outbound bundle).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

GENESIS = "0" * 64


def _hash(prev: str, seq: int, ts: str, event: dict) -> str:
    body = json.dumps({"seq": seq, "ts": ts, "event": event}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((prev + body).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.entries: list[dict] = []
        if path is not None and path.exists():
            self.entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    @property
    def head(self) -> str:
        return self.entries[-1]["hash"] if self.entries else GENESIS

    def append(self, event: dict, ts: str) -> dict:
        seq = len(self.entries)
        entry = {"seq": seq, "ts": ts, "event": event, "prev": self.head}
        entry["hash"] = _hash(entry["prev"], seq, ts, event)
        self.entries.append(entry)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry


def verify_chain(entries: list[dict]) -> int | None:
    """Return the index of the first bad entry, or ``None`` if the chain is intact."""
    prev = GENESIS
    for i, e in enumerate(entries):
        if e.get("seq") != i or e.get("prev") != prev or e.get("hash") != _hash(prev, i, e.get("ts", ""), e.get("event", {})):
            return i
        prev = e["hash"]
    return None
