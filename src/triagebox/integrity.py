"""Boot-time integrity check for the model and index files. Fails closed."""

from __future__ import annotations

import hashlib
from pathlib import Path


class IntegrityError(Exception):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_manifest(root: Path, relative_paths: list[str]) -> dict[str, str]:
    return {p: sha256_file(root / p) for p in relative_paths}


def verify_manifest(root: Path, manifest: dict[str, str]) -> list[str]:
    """Problems found: a missing file or a hash mismatch for each. Empty list means intact."""
    problems = []
    for rel, expected in manifest.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif sha256_file(path) != expected:
            problems.append(f"modified: {rel}")
    return problems


def boot_check(root: Path, manifest: dict[str, str]) -> None:
    """Raise unless every listed file is present and unmodified."""
    problems = verify_manifest(root, manifest)
    if problems:
        raise IntegrityError("; ".join(problems))
