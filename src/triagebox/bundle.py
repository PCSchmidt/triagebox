"""Signed store-and-forward bundles for degraded, disconnected links.

A bundle is a set of versioned items plus a manifest of their SHA-256 hashes,
signed with Ed25519. The receiver verifies the signature, every hash and size,
and a strictly increasing sequence number (to reject replay) before accepting
anything. Items are merged by version with a deterministic tie-break.

Limits: the sequence number only stops replay if the receiver persists the
highest accepted value; key distribution and revocation are out of scope.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


class BundleError(Exception):
    """A bundle failed verification and must be rejected whole."""


@dataclass(frozen=True)
class Item:
    id: str
    version: int
    payload: bytes


@dataclass
class Bundle:
    manifest: bytes  # canonical JSON, exactly what was signed
    signature: bytes
    payloads: dict[str, bytes] = field(default_factory=dict)  # sha256 hex -> bytes


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def make_bundle(items: list[Item], key: Ed25519PrivateKey, seq: int, created: str) -> Bundle:
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate item ids")
    manifest = _canonical(
        {
            "seq": seq,
            "created": created,
            "items": [
                {"id": i.id, "version": i.version, "sha256": _sha256(i.payload), "size": len(i.payload)}
                for i in sorted(items, key=lambda i: i.id)
            ],
        }
    )
    return Bundle(manifest, key.sign(manifest), {_sha256(i.payload): i.payload for i in items})


def verify_bundle(bundle: Bundle, public_key: Ed25519PublicKey, last_seq: int) -> list[Item]:
    """Return the items if and only if the whole bundle verifies."""
    try:
        public_key.verify(bundle.signature, bundle.manifest)
    except InvalidSignature as e:
        raise BundleError("bad signature") from e
    try:
        m = json.loads(bundle.manifest)
        seq, entries = m["seq"], m["items"]
    except (ValueError, KeyError, TypeError) as e:
        raise BundleError("malformed manifest") from e
    if not isinstance(seq, int) or seq <= last_seq:
        raise BundleError(f"replayed or out-of-order bundle (seq {seq}, last accepted {last_seq})")

    seen, hashes, items = set(), set(), []
    for e in entries:
        if e["id"] in seen:
            raise BundleError(f"duplicate item id {e['id']!r}")
        seen.add(e["id"])
        payload = bundle.payloads.get(e["sha256"])
        if payload is None:
            raise BundleError(f"missing payload for {e['id']!r}")
        if _sha256(payload) != e["sha256"] or len(payload) != e["size"]:
            raise BundleError(f"payload for {e['id']!r} does not match its manifest entry")
        hashes.add(e["sha256"])
        items.append(Item(e["id"], e["version"], payload))
    if set(bundle.payloads) - hashes:
        raise BundleError("bundle carries payloads not listed in the manifest")
    return items


def merge(local: dict[str, Item], incoming: list[Item]) -> tuple[dict[str, Item], list[str]]:
    """Merge ``incoming`` into ``local``; return (merged, ids with an unresolved-by-version conflict).

    The higher version wins. Equal versions with different payloads are a
    conflict, resolved deterministically (larger SHA-256 wins) so every node
    converges, and reported so a person can review it.
    """
    merged = dict(local)
    conflicts = []
    for it in incoming:
        cur = merged.get(it.id)
        if cur is None or it.version > cur.version:
            merged[it.id] = it
        elif it.version == cur.version and it.payload != cur.payload:
            conflicts.append(it.id)
            if _sha256(it.payload) > _sha256(cur.payload):
                merged[it.id] = it
    return merged, conflicts


def write_bundle(bundle: Bundle, directory: Path) -> None:
    (directory / "payloads").mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_bytes(bundle.manifest)
    (directory / "manifest.sig").write_bytes(bundle.signature)
    for h, data in bundle.payloads.items():
        (directory / "payloads" / h).write_bytes(data)


def read_bundle(directory: Path) -> Bundle:
    payloads = {p.name: p.read_bytes() for p in (directory / "payloads").iterdir()}
    return Bundle((directory / "manifest.json").read_bytes(), (directory / "manifest.sig").read_bytes(), payloads)
