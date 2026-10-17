import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from triagebox.audit import GENESIS, AuditLog, verify_chain
from triagebox.bundle import Bundle, BundleError, Item, make_bundle, merge, read_bundle, verify_bundle, write_bundle
from triagebox.diode import DiodeViolation, export_summaries
from triagebox.integrity import IntegrityError, boot_check, build_manifest, verify_manifest

KEY = Ed25519PrivateKey.generate()
PUB = KEY.public_key()
ITEMS = [Item("doc1", 1, b"first document"), Item("doc2", 3, b"second document")]


def bundle(seq=5):
    return make_bundle(ITEMS, KEY, seq=seq, created="2026-10-16T12:00:00Z")


# ------------------------------------------------------------------ bundles
def test_round_trip_returns_the_items():
    got = verify_bundle(bundle(), PUB, last_seq=4)
    assert sorted((i.id, i.version, i.payload) for i in got) == sorted((i.id, i.version, i.payload) for i in ITEMS)


def test_rejects_replay_and_out_of_order():
    with pytest.raises(BundleError, match="replayed"):
        verify_bundle(bundle(seq=5), PUB, last_seq=5)
    with pytest.raises(BundleError, match="replayed"):
        verify_bundle(bundle(seq=4), PUB, last_seq=9)


def test_rejects_wrong_key():
    other = Ed25519PrivateKey.generate().public_key()
    with pytest.raises(BundleError, match="signature"):
        verify_bundle(bundle(), other, last_seq=0)


def test_rejects_tampered_manifest():
    b = bundle()
    b.manifest = b.manifest.replace(b'"version":1', b'"version":9')
    with pytest.raises(BundleError, match="signature"):
        verify_bundle(b, PUB, last_seq=0)


def test_rejects_tampered_payload_and_missing_payload():
    b = bundle()
    h = next(iter(b.payloads))
    b.payloads[h] = b"evil replacement"
    with pytest.raises(BundleError, match="does not match"):
        verify_bundle(b, PUB, last_seq=0)
    b2 = bundle()
    b2.payloads.pop(next(iter(b2.payloads)))
    with pytest.raises(BundleError, match="missing payload"):
        verify_bundle(b2, PUB, last_seq=0)


def test_rejects_unlisted_extra_payload():
    b = bundle()
    b.payloads["f" * 64] = b"smuggled"
    with pytest.raises(BundleError, match="not listed"):
        verify_bundle(b, PUB, last_seq=0)


def test_duplicate_ids_rejected_at_creation():
    with pytest.raises(ValueError):
        make_bundle([Item("x", 1, b"a"), Item("x", 2, b"b")], KEY, 1, "t")


def test_disk_round_trip(tmp_path):
    write_bundle(bundle(), tmp_path)
    got = verify_bundle(read_bundle(tmp_path), PUB, last_seq=0)
    assert {i.id for i in got} == {"doc1", "doc2"}


def test_merge_takes_higher_version_and_adds_new():
    local = {"doc1": Item("doc1", 2, b"local v2")}
    merged, conflicts = merge(local, [Item("doc1", 1, b"old"), Item("doc2", 1, b"new")])
    assert merged["doc1"].payload == b"local v2" and "doc2" in merged and conflicts == []
    merged, _ = merge(local, [Item("doc1", 3, b"newer")])
    assert merged["doc1"].payload == b"newer"


def test_merge_equal_version_conflict_converges_both_ways():
    a, b = Item("d", 1, b"alpha"), Item("d", 1, b"bravo")
    m1, c1 = merge({"d": a}, [b])
    m2, c2 = merge({"d": b}, [a])
    assert m1["d"] == m2["d"]  # same winner regardless of order
    assert c1 == ["d"] and c2 == ["d"]
    assert merge({"d": a}, [Item("d", 1, b"alpha")])[1] == []  # identical content is no conflict


# -------------------------------------------------------------------- diode
def test_diode_exports_only_allowlisted_fields():
    rec = {"id": "r1", "summary": "Three trucks.", "citations": ["d#0"], "timestamp": "t", "source_text": "SECRET", "prompt": "P"}
    out = export_summaries([rec])
    assert b"SECRET" not in out and b'"prompt"' not in out
    assert json.loads(out)[0] == {"id": "r1", "summary": "Three trucks.", "citations": ["d#0"], "timestamp": "t"}


def test_diode_strict_mode_rejects_extra_fields_and_truncates_summaries():
    with pytest.raises(DiodeViolation):
        export_summaries([{"id": "r", "summary": "s", "source_text": "x"}], strict=True)
    long = export_summaries([{"id": "r", "summary": "x" * 5000}])
    assert len(json.loads(long)[0]["summary"]) == 1000
    with pytest.raises(DiodeViolation):
        export_summaries([{"id": "r", "summary": {"nested": "dict"}}])


# -------------------------------------------------------------------- audit
def fill(log, n=4):
    for i in range(n):
        log.append({"q": f"question {i}"}, ts=f"t{i}")
    return log


def test_chain_verifies_and_links_to_genesis():
    log = fill(AuditLog())
    assert log.entries[0]["prev"] == GENESIS
    assert verify_chain(log.entries) is None


def test_edit_in_the_middle_is_detected():
    log = fill(AuditLog())
    log.entries[2]["event"]["q"] = "rewritten"
    assert verify_chain(log.entries) == 2


def test_removed_and_reordered_entries_are_detected():
    log = fill(AuditLog())
    removed = log.entries[:1] + log.entries[2:]
    assert verify_chain(removed) == 1
    swapped = [log.entries[0], log.entries[2], log.entries[1], log.entries[3]]
    assert verify_chain(swapped) == 1


def test_tail_truncation_is_not_detectable_from_the_log_alone():
    log = fill(AuditLog())
    assert verify_chain(log.entries[:2]) is None  # documented limitation
    assert log.head != log.entries[1]["hash"]  # but an externally kept head would expose it


def test_log_persists_and_reloads(tmp_path):
    path = tmp_path / "audit.jsonl"
    fill(AuditLog(path))
    reloaded = AuditLog(path)
    assert len(reloaded.entries) == 4 and verify_chain(reloaded.entries) is None
    reloaded.append({"q": "more"}, ts="t9")
    assert verify_chain(AuditLog(path).entries) is None


# ---------------------------------------------------------------- integrity
def test_boot_check_passes_then_fails_closed_on_change_or_removal(tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"weights")
    (tmp_path / "index.db").write_bytes(b"index")
    manifest = build_manifest(tmp_path, ["model.gguf", "index.db"])
    boot_check(tmp_path, manifest)  # intact

    (tmp_path / "index.db").write_bytes(b"index but altered")
    assert verify_manifest(tmp_path, manifest) == ["modified: index.db"]
    with pytest.raises(IntegrityError, match="modified: index.db"):
        boot_check(tmp_path, manifest)

    (tmp_path / "model.gguf").unlink()
    with pytest.raises(IntegrityError, match="missing: model.gguf"):
        boot_check(tmp_path, manifest)
