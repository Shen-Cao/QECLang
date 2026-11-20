"""Minimal content-addressed evidence byte stores with explicit writes.

Stores enforce byte integrity and configured retention limits. They do not
parse bundles, execute programs, or certify mathematical claims; those duties
belong to the archive service and its independent inspector.
"""
from hashlib import sha256
import re
from threading import RLock
from typing import Protocol, runtime_checkable

BUNDLE_DOMAIN = b"qeclang.evidence_bundle.v1\x00"
DEFAULT_MAX_BUNDLE_BYTES = 32_000_000
DEFAULT_MAX_RETAINED_BYTES = 64_000_000
_KEY = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceStoreError(ValueError):
    """Invalid key/payload, conflicting content, or detected storage corruption."""


class EvidenceStoreLimitExceeded(EvidenceStoreError):
    """A configured per-bundle or retained-byte capacity was exceeded."""


def _byte_limit(value, name):
    if type(value) is not int or value < 0:
        raise EvidenceStoreError(f"{name} must be a nonnegative integer")
    return value


def validate_key(key: str) -> str:
    """Accept exactly 64 lowercase hexadecimal characters; never normalize paths."""
    if type(key) is not str or len(key) != 64 or _KEY.fullmatch(key) is None:
        raise EvidenceStoreError("bundle key must be exactly 64 lowercase hexadecimal characters")
    return key


def bundle_id(payload: bytes) -> str:
    """SHA256(b'qeclang.evidence_bundle.v1\\0' + payload), without normalization.

    This hashes the supplied bytes, not a semantic equivalence class. Callers
    apply size guards before hashing untrusted input; store methods do so.
    """
    if type(payload) is not bytes:
        raise EvidenceStoreError("evidence payload must be immutable bytes")
    digest = sha256(BUNDLE_DOMAIN)
    digest.update(payload)
    return digest.hexdigest()


def validate_payload(key: str, payload: bytes, *, max_bundle_bytes=DEFAULT_MAX_BUNDLE_BYTES) -> bytes:
    """Check canonical key shape, byte limit, and key/content agreement before copying."""
    validate_key(key)
    limit = _byte_limit(max_bundle_bytes, "max_bundle_bytes")
    if type(payload) is not bytes:
        raise EvidenceStoreError("evidence payload must be immutable bytes")
    if len(payload) > limit:
        raise EvidenceStoreLimitExceeded("evidence bundle exceeds max_bundle_bytes")
    if bundle_id(payload) != key:
        raise EvidenceStoreError("evidence payload does not match its bundle key")
    return payload


@runtime_checkable
class EvidenceReader(Protocol):
    """Read capability; a missing valid key raises KeyError without creating state."""

    def fetch(self, key: str) -> bytes:
        """Return bounded immutable bytes whose content digest equals key."""
        ...


@runtime_checkable
class EvidenceWriter(Protocol):
    """Explicit write capability, with immutable content and idempotent retries."""

    def save(self, key: str, payload: bytes) -> bool:
        """Return True for insertion, False for identical existing content; never replace."""
        ...


@runtime_checkable
class EvidenceStore(EvidenceReader, EvidenceWriter, Protocol):
    """Combined read/write capability; services may require only one side."""


class InMemoryEvidenceStore:
    """Reference store with atomic, idempotent save and integrity-checked fetch.

    Limits count payload bytes, excluding dictionaries, hashes, temporary copies,
    and Python overhead. Reads and construction never create an entry. No data
    survives process/object lifetime, and no disk or network access occurs.
    """
    __slots__ = ("_max_bundle_bytes", "_max_retained_bytes", "_retained_bytes", "_payloads", "_lock")

    def __init__(self, *, max_bundle_bytes=DEFAULT_MAX_BUNDLE_BYTES,
                 max_retained_bytes=DEFAULT_MAX_RETAINED_BYTES):
        self._max_bundle_bytes = _byte_limit(max_bundle_bytes, "max_bundle_bytes")
        self._max_retained_bytes = _byte_limit(max_retained_bytes, "max_retained_bytes")
        self._retained_bytes = 0
        self._payloads: dict[str, bytes] = {}
        self._lock = RLock()

    @property
    def max_bundle_bytes(self):
        return self._max_bundle_bytes

    @property
    def max_retained_bytes(self):
        return self._max_retained_bytes

    @property
    def retained_bytes(self):
        with self._lock:
            return self._retained_bytes

    @property
    def bundle_count(self):
        with self._lock:
            return len(self._payloads)

    def fetch(self, key: str) -> bytes:
        validate_key(key)
        with self._lock:
            payload = self._payloads[key]
            validate_payload(key, payload, max_bundle_bytes=self._max_bundle_bytes)
            return memoryview(payload).tobytes()

    def save(self, key: str, payload: bytes) -> bool:
        validate_payload(key, payload, max_bundle_bytes=self._max_bundle_bytes)
        with self._lock:
            if key in self._payloads:
                previous = self._payloads[key]
                validate_payload(key, previous, max_bundle_bytes=self._max_bundle_bytes)
                if previous != payload:
                    raise EvidenceStoreError("existing bundle has conflicting bytes; replacement is forbidden")
                return False
            if len(payload) > self._max_retained_bytes - self._retained_bytes:
                raise EvidenceStoreLimitExceeded("evidence store exceeds max_retained_bytes")
            self._payloads[key] = memoryview(payload).tobytes()
            self._retained_bytes += len(payload)
            return True


__all__ = ["BUNDLE_DOMAIN", "DEFAULT_MAX_BUNDLE_BYTES", "DEFAULT_MAX_RETAINED_BYTES",
           "EvidenceStoreError", "EvidenceStoreLimitExceeded", "validate_key", "bundle_id",
           "validate_payload", "EvidenceReader", "EvidenceWriter", "EvidenceStore",
           "InMemoryEvidenceStore"]
