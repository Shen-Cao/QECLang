from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3

import pytest

from qeclang.archive.protocol import (EvidenceReader, EvidenceStore, EvidenceStoreError,
                                      EvidenceStoreLimitExceeded, bundle_id)
from qeclang.archive.sqlite_store import SQLiteEvidenceStore


def test_lazy_creation_readonly_missing_and_opaque_keys(tmp_path):
    path = tmp_path / "not-created.sqlite"
    payload = b'{"evidence":"example"}'
    key = bundle_id(payload)
    store = SQLiteEvidenceStore(path)
    readonly = SQLiteEvidenceStore(path, readonly=True)
    assert not path.exists()
    for reader in (store, readonly):
        with pytest.raises(KeyError):
            reader.fetch(key)
        assert not path.exists()
    with pytest.raises(PermissionError):
        readonly.save(key, payload)
    for invalid in ("../outside", key.upper(), "a" * 63, True):
        with pytest.raises(EvidenceStoreError):
            store.fetch(invalid)
        with pytest.raises(EvidenceStoreError):
            store.save(invalid, payload)
        assert not path.exists()
    assert isinstance(store, EvidenceStore) and isinstance(readonly, EvidenceReader)


def test_roundtrip_idempotence_reopen_and_unusual_path(tmp_path):
    path = tmp_path / "evidence ? # café.sqlite"
    store = SQLiteEvidenceStore(path)
    for payload in (b"", b"{}", b"\x00\xff\x80binary"):
        key = bundle_id(payload)
        assert store.save(key, payload) is True
        assert store.save(key, payload) is False
        assert store.fetch(key) == payload
        assert SQLiteEvidenceStore(path, readonly=True).fetch(key) == payload
    with pytest.raises(KeyError):
        store.fetch(bundle_id(b"absent"))
    assert SQLiteEvidenceStore(path).path == path.resolve()


def test_invalid_payloads_fail_before_database_creation(tmp_path):
    path = tmp_path / "absent.sqlite"
    store = SQLiteEvidenceStore(path, max_bundle_bytes=3)
    for payload in (bytearray(b"a"), "a", None):
        with pytest.raises(EvidenceStoreError):
            store.save(bundle_id(b"a"), payload)
    with pytest.raises(EvidenceStoreError, match="match"):
        store.save(bundle_id(b"a"), b"b")
    with pytest.raises(EvidenceStoreLimitExceeded):
        store.save(bundle_id(b"four"), b"four")
    assert not path.exists()
    for limit in (-1, True, 1.0, 1 << 70):
        with pytest.raises(EvidenceStoreError):
            SQLiteEvidenceStore(path, max_bundle_bytes=limit)


def test_empty_blob_with_zero_capacity_roundtrips_and_retries(tmp_path):
    path = tmp_path / "zero-capacity.sqlite"
    store = SQLiteEvidenceStore(path, max_bundle_bytes=0)
    key = bundle_id(b"")
    assert store.save(key, b"") is True
    assert store.fetch(key) == b""
    assert store.save(key, b"") is False
    assert SQLiteEvidenceStore(path, readonly=True, max_bundle_bytes=0).fetch(key) == b""
    with pytest.raises(EvidenceStoreLimitExceeded):
        store.save(bundle_id(b"x"), b"x")


def test_digest_corruption_and_occupied_key_are_never_overwritten(tmp_path):
    path = tmp_path / "corrupted.sqlite"
    store = SQLiteEvidenceStore(path)
    payload, corrupt = b"original", b"different"
    key = bundle_id(payload)
    store.save(key, payload)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE evidence_bundles SET payload=? WHERE bundle_id=?", (corrupt, key))
    for action in (lambda: store.fetch(key), lambda: store.save(key, payload)):
        with pytest.raises(EvidenceStoreError, match="match"):
            action()
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT payload FROM evidence_bundles").fetchone()[0] == corrupt


def test_oversized_malicious_blob_is_refused_before_blob_transfer(tmp_path, monkeypatch):
    import qeclang.archive.sqlite_store as module

    path = tmp_path / "oversized.sqlite"
    store = SQLiteEvidenceStore(path, max_bundle_bytes=16)
    key = bundle_id(b"safe")
    store.save(key, b"safe")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE evidence_bundles SET payload=zeroblob(1000000) WHERE bundle_id=?", (key,))
    commands = []
    original = module._connect

    def tracked(*args, **kwargs):
        connection = original(*args, **kwargs)
        connection.set_trace_callback(commands.append)
        return connection

    monkeypatch.setattr(module, "_connect", tracked)
    with pytest.raises(EvidenceStoreLimitExceeded):
        store.fetch(key)
    assert any("SELECT length(payload)" in sql for sql in commands)
    assert not any("SELECT substr(payload" in sql for sql in commands)


def test_non_blob_payload_rejected_before_length_or_transfer(tmp_path, monkeypatch):
    import qeclang.archive.sqlite_store as module

    path = tmp_path / "wrong-type.sqlite"
    store = SQLiteEvidenceStore(path)
    key = bundle_id(b"safe")
    store.save(key, b"safe")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE evidence_bundles SET payload=? WHERE bundle_id=?", ("not a blob", key))
    original = module._connect
    commands = []

    def tracked(*args, **kwargs):
        connection = original(*args, **kwargs)
        connection.set_trace_callback(commands.append)
        return connection

    monkeypatch.setattr(module, "_connect", tracked)
    with pytest.raises(EvidenceStoreError, match="BLOB"):
        store.fetch(key)
    assert not any("SELECT length(payload)" in sql or "SELECT substr(payload" in sql for sql in commands)


def test_read_length_and_bytes_are_read_in_one_snapshot(tmp_path, monkeypatch):
    import qeclang.archive.sqlite_store as module

    path = tmp_path / "snapshot.sqlite"
    store = SQLiteEvidenceStore(path)
    key = bundle_id(b"safe")
    store.save(key, b"safe")
    original = module._read_payload

    def checked(connection, *args):
        assert connection.in_transaction
        return original(connection, *args)

    monkeypatch.setattr(module, "_read_payload", checked)
    assert store.fetch(key) == b"safe"
    assert store.save(key, b"safe") is False


def test_single_transaction_rolls_back_data_and_new_schema(tmp_path, monkeypatch):
    import qeclang.archive.sqlite_store as module

    existing, fresh = tmp_path / "existing.sqlite", tmp_path / "fresh.sqlite"
    old = b"committed"
    store = SQLiteEvidenceStore(existing)
    store.save(bundle_id(old), old)
    original = module._insert_payload

    def fail(connection, key, payload):
        original(connection, key, payload)
        raise RuntimeError("simulated failure after insert")

    monkeypatch.setattr(module, "_insert_payload", fail)
    new = b"rolled-back"
    for target in (store, SQLiteEvidenceStore(fresh)):
        with pytest.raises(RuntimeError, match="simulated"):
            target.save(bundle_id(new), new)
        with pytest.raises(KeyError):
            target.fetch(bundle_id(new))
    assert store.fetch(bundle_id(old)) == old
    with sqlite3.connect(fresh) as connection:
        assert connection.execute("SELECT name FROM sqlite_master").fetchall() == []


def test_unrelated_database_and_wrong_metadata_untouched(tmp_path):
    unrelated = tmp_path / "unrelated.sqlite"
    with sqlite3.connect(unrelated) as connection:
        connection.execute("CREATE TABLE important (value TEXT)")
        connection.execute("INSERT INTO important VALUES ('keep')")
    before = unrelated.read_bytes()
    store = SQLiteEvidenceStore(unrelated)
    key = bundle_id(b"data")
    for action in (lambda: store.fetch(key), lambda: store.save(key, b"data")):
        with pytest.raises(EvidenceStoreError, match="schema"):
            action()
    assert unrelated.read_bytes() == before
    correct = tmp_path / "metadata.sqlite"
    store = SQLiteEvidenceStore(correct)
    store.save(key, b"data")
    with sqlite3.connect(correct) as connection:
        connection.execute("UPDATE evidence_metadata SET value='different application'")
    with pytest.raises(EvidenceStoreError, match="metadata"):
        store.fetch(key)
    with pytest.raises(EvidenceStoreError, match="metadata"):
        store.save(bundle_id(b"new"), b"new")
    with sqlite3.connect(correct) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evidence_bundles").fetchone()[0] == 1


@pytest.mark.parametrize("statement", [
    "CREATE VIEW extra AS SELECT * FROM evidence_bundles",
    "CREATE TRIGGER extra AFTER INSERT ON evidence_bundles BEGIN DELETE FROM evidence_bundles; END",
    "CREATE INDEX extra ON evidence_bundles(payload)",
    "PRAGMA user_version=17",
    "PRAGMA application_id=17",
])
def test_unexpected_schema_objects_and_metadata_rejected(tmp_path, statement):
    path = tmp_path / "modified.sqlite"
    store = SQLiteEvidenceStore(path)
    key = bundle_id(b"data")
    store.save(key, b"data")
    with sqlite3.connect(path) as connection:
        connection.execute(statement)
    with pytest.raises(EvidenceStoreError):
        store.fetch(key)
    with pytest.raises(EvidenceStoreError):
        store.save(bundle_id(b"different"), b"different")


def test_concurrent_duplicate_and_distinct_saves_are_atomic(tmp_path):
    path = tmp_path / "concurrent.sqlite"
    payload = b"identical concurrent retry"
    key = bundle_id(payload)

    def save_same(_):
        return SQLiteEvidenceStore(path).save(key, payload)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(save_same, range(24)))
    assert results.count(True) == 1 and results.count(False) == 23
    payloads = [f"item {i}".encode() for i in range(24)]

    def save_distinct(item):
        store = SQLiteEvidenceStore(path)
        assert store.save(bundle_id(item), item)
        return store.fetch(bundle_id(item))

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(save_distinct, payloads)) == payloads
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evidence_bundles").fetchone()[0] == 25


def test_file_corruption_and_missing_parent_do_not_create_directories(tmp_path):
    path = tmp_path / "not-sqlite"
    path.write_bytes(b"unrelated raw bytes")
    original = path.read_bytes()
    store = SQLiteEvidenceStore(path)
    for action in (lambda: store.fetch(bundle_id(b"a")), lambda: store.save(bundle_id(b"a"), b"a")):
        with pytest.raises(EvidenceStoreError):
            action()
    assert path.read_bytes() == original
    missing = tmp_path / "missing" / "db.sqlite"
    with pytest.raises(EvidenceStoreError):
        SQLiteEvidenceStore(missing).save(bundle_id(b"a"), b"a")
    assert not Path(missing.parent).exists()
