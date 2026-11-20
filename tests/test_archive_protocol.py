from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256

import pytest

from qeclang.archive.protocol import (
    BUNDLE_DOMAIN, EvidenceReader, EvidenceStore, EvidenceStoreError,
    EvidenceStoreLimitExceeded, EvidenceWriter, InMemoryEvidenceStore,
    bundle_id, validate_key, validate_payload,
)


def test_domain_separated_digest_golden_and_exact_byte_identity():
    assert BUNDLE_DOMAIN == b"qeclang.evidence_bundle.v1\x00"
    assert bundle_id(b"") == "abf10d9299e6c1811d0b677700edd2e34aa80cf6e6230f3b7247246954663f6f"
    assert bundle_id(b"{}") == "1de637a013211eca7978b60b2463b59f79895761aa037c2c74227bb81227f964"
    assert bundle_id(b"{}") != sha256(b"{}").hexdigest()
    assert bundle_id(b'{"a":1,"b":2}') != bundle_id(b'{"b":2,"a":1}')
    assert bundle_id(b"{}") != bundle_id(b"{}\n")
    assert validate_key(bundle_id(b"{}")) == bundle_id(b"{}")


@pytest.mark.parametrize("key", [None, True, 42, b"a"*64, [], "", "a"*63, "a"*65,
                                  "A"*64, "g"*64, "../" + "a"*61,
                                  "a"*63 + "/", "a"*63 + "\n", "a"*63 + "\0"])
def test_keys_are_bounded_lowercase_digest_identifiers_not_paths(key):
    with pytest.raises(EvidenceStoreError):
        validate_key(key)
    store = InMemoryEvidenceStore()
    with pytest.raises(EvidenceStoreError):
        store.fetch(key)
    with pytest.raises(EvidenceStoreError):
        store.save(key, b"{}")
    assert store.bundle_count == store.retained_bytes == 0


@pytest.mark.parametrize("payload", [None, "{}", bytearray(b"{}"), memoryview(b"{}"), 2])
def test_only_immutable_bytes_are_accepted(payload):
    key = bundle_id(b"{}")
    with pytest.raises(EvidenceStoreError):
        bundle_id(payload)
    with pytest.raises(EvidenceStoreError):
        validate_payload(key, payload)
    with pytest.raises(EvidenceStoreError):
        InMemoryEvidenceStore().save(key, payload)


def test_read_and_write_capabilities_are_separate_protocols():
    class Reader:
        def fetch(self, key):
            raise KeyError(key)
    class Writer:
        def save(self, key, payload):
            return True
    assert isinstance(Reader(), EvidenceReader)
    assert not isinstance(Reader(), EvidenceWriter)
    assert isinstance(Writer(), EvidenceWriter)
    assert not isinstance(Writer(), EvidenceReader)
    store = InMemoryEvidenceStore()
    assert isinstance(store, EvidenceStore)
    assert store.max_bundle_bytes == 32_000_000
    assert store.max_retained_bytes == 64_000_000


def test_construction_hashing_and_reads_do_not_insert_entries():
    store = InMemoryEvidenceStore()
    key = bundle_id(b"opaque bytes, not a parsed scientific claim")
    with pytest.raises(KeyError) as error:
        store.fetch(key)
    assert error.value.args == (key,)
    assert store.bundle_count == store.retained_bytes == 0
    validate_key(key)
    assert store.bundle_count == store.retained_bytes == 0


def test_roundtrip_idempotence_and_readonly_byte_snapshots():
    source = bytearray(b"record")
    payload = bytes(source)
    store = InMemoryEvidenceStore(max_bundle_bytes=6, max_retained_bytes=6)
    key = bundle_id(payload)
    assert store.save(key, payload) is True
    source[:] = b"edited"
    first, second = store.fetch(key), store.fetch(key)
    assert type(first) is bytes and first == second == b"record"
    with pytest.raises(TypeError):
        memoryview(first)[0] = 0
    assert store.save(key, payload) is False
    assert store.bundle_count == 1 and store.retained_bytes == 6
    with pytest.raises(AttributeError):
        store.max_bundle_bytes = 1


def test_capacity_counts_only_new_payload_bytes_and_retains_previous_values():
    store = InMemoryEvidenceStore(max_bundle_bytes=3, max_retained_bytes=6)
    for payload in (b"abc", b"xyz"):
        assert store.save(bundle_id(payload), payload)
    assert store.retained_bytes == 6 and store.bundle_count == 2
    assert store.save(bundle_id(b"abc"), b"abc") is False
    with pytest.raises(EvidenceStoreLimitExceeded, match="retained"):
        store.save(bundle_id(b"a"), b"a")
    with pytest.raises(EvidenceStoreLimitExceeded, match="bundle"):
        store.save(bundle_id(b"four"), b"four")
    assert store.fetch(bundle_id(b"abc")) == b"abc"
    assert store.fetch(bundle_id(b"xyz")) == b"xyz"
    assert store.retained_bytes == 6 and store.bundle_count == 2


def test_zero_capacity_accepts_only_empty_payload_and_retry_is_idempotent():
    store = InMemoryEvidenceStore(max_bundle_bytes=0, max_retained_bytes=0)
    key = bundle_id(b"")
    assert store.save(key, b"") is True
    assert store.save(key, b"") is False
    assert store.fetch(key) == b""
    assert store.bundle_count == 1 and store.retained_bytes == 0
    with pytest.raises(EvidenceStoreLimitExceeded):
        store.save(bundle_id(b"x"), b"x")


def test_per_bundle_limit_precedes_hashing_or_copying(monkeypatch):
    import qeclang.archive.protocol as protocol
    key = bundle_id(b"oversized")
    def forbidden(*args, **kwargs):
        raise AssertionError("payload size guard must precede hashing")
    monkeypatch.setattr(protocol, "bundle_id", forbidden)
    store = InMemoryEvidenceStore(max_bundle_bytes=2)
    with pytest.raises(EvidenceStoreLimitExceeded):
        store.save(key, b"oversized")
    with pytest.raises(EvidenceStoreLimitExceeded):
        validate_payload(key, b"oversized", max_bundle_bytes=2)
    assert store.bundle_count == store.retained_bytes == 0


def test_key_content_mismatch_never_inserts_or_replaces():
    store = InMemoryEvidenceStore()
    key = bundle_id(b"original")
    with pytest.raises(EvidenceStoreError, match="match"):
        store.save(key, b"wrong")
    assert store.bundle_count == store.retained_bytes == 0
    assert store.save(key, b"original")
    with pytest.raises(EvidenceStoreError, match="match"):
        store.save(key, b"wrong")
    assert store.fetch(key) == b"original"
    assert store.retained_bytes == 8


def test_corrupted_retained_bytes_are_rejected_and_not_implicitly_repaired():
    store = InMemoryEvidenceStore()
    payload = b"original"
    key = bundle_id(payload)
    store.save(key, payload)
    store._payloads[key] = b"corrupt"  # Deliberate storage-fault injection.
    with pytest.raises(EvidenceStoreError, match="match"):
        store.fetch(key)
    with pytest.raises(EvidenceStoreError, match="match"):
        store.save(key, payload)
    assert store._payloads[key] == b"corrupt"
    assert store.bundle_count == 1 and store.retained_bytes == len(payload)


def test_exact_byte_conflict_check_does_not_depend_only_on_digest_equality(monkeypatch):
    import qeclang.archive.protocol as protocol
    key = "0"*64
    monkeypatch.setattr(protocol, "bundle_id", lambda payload: key)
    store = InMemoryEvidenceStore()
    assert store.save(key, b"first")
    with pytest.raises(EvidenceStoreError, match="conflicting"):
        store.save(key, b"other")
    assert store.fetch(key) == b"first"
    assert store.retained_bytes == 5


def test_concurrent_identical_retries_insert_once():
    store = InMemoryEvidenceStore(max_retained_bytes=4)
    payload = b"same"
    key = bundle_id(payload)
    with ThreadPoolExecutor(max_workers=8) as pool:
        inserted = list(pool.map(lambda _: store.save(key, payload), range(24)))
    assert sum(inserted) == 1
    assert store.bundle_count == 1 and store.retained_bytes == 4


def test_concurrent_distinct_saves_cannot_overrun_capacity():
    store = InMemoryEvidenceStore(max_bundle_bytes=1, max_retained_bytes=1)
    def save(payload):
        try:
            return store.save(bundle_id(payload), payload)
        except EvidenceStoreLimitExceeded:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        inserted = list(pool.map(save, (b"a", b"b")))
    assert sum(inserted) == 1
    assert store.bundle_count == store.retained_bytes == 1


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "8"])
def test_invalid_byte_limits_reject_before_store_use(invalid):
    with pytest.raises(EvidenceStoreError):
        InMemoryEvidenceStore(max_bundle_bytes=invalid)
    with pytest.raises(EvidenceStoreError):
        InMemoryEvidenceStore(max_retained_bytes=invalid)
    with pytest.raises(EvidenceStoreError):
        validate_payload(bundle_id(b"{}"), b"{}", max_bundle_bytes=invalid)
