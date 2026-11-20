"""Explicit-write SQLite transport for content-addressed evidence bundle bytes.

This store checks byte integrity, not mathematical evidence or authenticity.
It is a dedicated local SQLite database with ordinary SQLite locking; byte
limits exclude database/page-cache size, SQLite workspace and filesystem races.
The archive codec and scientific inspector remain independent trust boundaries.
"""
from pathlib import Path
import sqlite3

from .protocol import (DEFAULT_MAX_BUNDLE_BYTES, EvidenceStoreError,
                       EvidenceStoreLimitExceeded, validate_key, validate_payload)

SCHEMA = "qeclang.evidence.sqlite.v1"
_METADATA_SQL = "CREATE TABLE evidence_metadata (key TEXT NOT NULL PRIMARY KEY, value TEXT NOT NULL)"
_BUNDLES_SQL = "CREATE TABLE evidence_bundles (bundle_id TEXT NOT NULL PRIMARY KEY, payload BLOB NOT NULL)"
_OBJECTS = {
    "evidence_metadata": ("table", "evidence_metadata", _METADATA_SQL),
    "evidence_bundles": ("table", "evidence_bundles", _BUNDLES_SQL),
    "sqlite_autoindex_evidence_metadata_1": ("index", "evidence_metadata", None),
    "sqlite_autoindex_evidence_bundles_1": ("index", "evidence_bundles", None),
}


def _connect(path, *, readonly):
    mode = "ro" if readonly else "rwc"
    connection = sqlite3.connect(path.as_uri() + "?mode=" + mode,
                                 uri=True, isolation_level=None, timeout=30)
    connection.execute("PRAGMA trusted_schema=OFF")
    if readonly:
        connection.execute("PRAGMA query_only=ON")
    return connection


def _schema(connection):
    """Recognize only this store's schema, or an uninitialized empty database.

    Even metadata/schema text is read in bounded slices. Never execute SQL from
    the database's schema column or admit additional triggers, views or tables.
    """
    for pragma in ("application_id", "user_version"):
        if connection.execute("PRAGMA " + pragma).fetchone()[0] != 0:
            raise EvidenceStoreError("unrelated SQLite application/version metadata")
    rows = connection.execute(
        "SELECT type,substr(name,1,129),length(name),substr(tbl_name,1,129),length(tbl_name),"
        "substr(sql,1,513),length(sql) FROM sqlite_master LIMIT 5").fetchall()
    if not rows:
        return False
    if len(rows) != len(_OBJECTS):
        raise EvidenceStoreError("unrelated or incomplete evidence-store schema")
    seen = set()
    for kind, name, name_length, table, table_length, sql, sql_length in rows:
        if (name_length != len(name) or table_length != len(table) or
                (sql is not None and sql_length != len(sql)) or name in seen or
                _OBJECTS.get(name) != (kind, table, sql)):
            raise EvidenceStoreError("unrelated or modified evidence-store schema")
        seen.add(name)
    if seen != set(_OBJECTS):
        raise EvidenceStoreError("incomplete evidence-store schema")
    metadata = connection.execute(
        "SELECT typeof(key),length(key),substr(key,1,33),"
        "typeof(value),length(value),substr(value,1,65) FROM evidence_metadata LIMIT 2").fetchall()
    expected = [("text", 6, "schema", "text", len(SCHEMA), SCHEMA)]
    if metadata != expected:
        raise EvidenceStoreError("invalid evidence-store metadata")
    return True


def _initialize(connection):
    connection.execute(_METADATA_SQL)
    connection.execute(_BUNDLES_SQL)
    connection.execute("INSERT INTO evidence_metadata VALUES (?,?)", ("schema", SCHEMA))


def _read_payload(connection, key, limit):
    """Inspect BLOB type/length before transferring a bounded value to Python."""
    row = connection.execute("SELECT typeof(payload) FROM evidence_bundles WHERE bundle_id=?",
                              (key,)).fetchone()
    if row is None:
        raise KeyError(key)
    if row[0] != "blob":
        raise EvidenceStoreError("stored evidence payload is not a BLOB")
    length = connection.execute("SELECT length(payload) FROM evidence_bundles WHERE bundle_id=?",
                                 (key,)).fetchone()[0]
    if type(length) is not int or length < 0:
        raise EvidenceStoreError("invalid stored BLOB length")
    if length > limit:
        raise EvidenceStoreLimitExceeded("stored evidence bundle exceeds max_bundle_bytes")
    if length == 0:
        return validate_payload(key, b"", max_bundle_bytes=limit)
    payload = connection.execute(
        "SELECT substr(payload,1,?) FROM evidence_bundles WHERE bundle_id=?", (limit + 1, key)).fetchone()[0]
    if type(payload) is not bytes or len(payload) != length:
        raise EvidenceStoreError("stored BLOB changed or has inconsistent length")
    return validate_payload(key, payload, max_bundle_bytes=limit)


def _insert_payload(connection, key, payload):
    connection.execute("INSERT INTO evidence_bundles VALUES (?,?)", (key, payload))


class SQLiteEvidenceStore:
    """Lazy local store: only save may create or modify the database.

    Construction performs no SQLite connection or initialization. Fetch opens
    an existing database in URI read-only mode and reads one transaction
    snapshot; missing databases/keys raise KeyError. Save validates bytes first,
    then initializes an empty database or checks the complete dedicated schema
    inside BEGIN IMMEDIATE. Identical saves are idempotent; conflicting/corrupt
    occupied keys are never overwritten. Parent directories must already exist.

    Keys are opaque lowercase SHA-256 identifiers, never paths. Stored bytes
    need not be trusted: the archive service must decode and independently check
    any claimed mathematics. This substrate does not parse or normalize JSON.
    """
    __slots__ = ("_path", "_readonly", "_max_bundle_bytes")

    def __init__(self, path, *, readonly=False, max_bundle_bytes=DEFAULT_MAX_BUNDLE_BYTES):
        if type(readonly) is not bool:
            raise EvidenceStoreError("readonly must be boolean")
        if type(max_bundle_bytes) is not int or not 0 <= max_bundle_bytes < (1 << 63) - 1:
            raise EvidenceStoreError("max_bundle_bytes must be a nonnegative SQLite-sized integer")
        self._path = Path(path).resolve()
        self._readonly = readonly
        self._max_bundle_bytes = max_bundle_bytes

    @property
    def path(self):
        return self._path

    @property
    def readonly(self):
        return self._readonly

    @property
    def max_bundle_bytes(self):
        return self._max_bundle_bytes

    def fetch(self, key: str) -> bytes:
        validate_key(key)
        if not self._path.exists():
            raise KeyError(key)
        connection = None
        try:
            connection = _connect(self._path, readonly=True)
            connection.execute("BEGIN")
            if not _schema(connection):
                raise KeyError(key)
            payload = _read_payload(connection, key, self._max_bundle_bytes)
            connection.commit()
            return payload
        except sqlite3.Error as error:
            if not self._path.exists():
                raise KeyError(key) from error
            raise EvidenceStoreError(f"SQLite evidence read failed: {error}") from error
        finally:
            if connection is not None:
                connection.close()

    def save(self, key: str, payload: bytes) -> bool:
        if self._readonly:
            raise PermissionError("evidence store was opened read-only")
        validate_payload(key, payload, max_bundle_bytes=self._max_bundle_bytes)
        connection = None
        try:
            connection = _connect(self._path, readonly=False)
            connection.execute("BEGIN IMMEDIATE")
            if not _schema(connection):
                _initialize(connection)
            try:
                previous = _read_payload(connection, key, self._max_bundle_bytes)
            except KeyError:
                _insert_payload(connection, key, payload)
                inserted = True
            else:
                if previous != payload:
                    raise EvidenceStoreError("existing bundle has conflicting bytes; replacement is forbidden")
                inserted = False
            connection.commit()
            return inserted
        except BaseException as error:
            if connection is not None:
                connection.rollback()
            if isinstance(error, sqlite3.Error):
                raise EvidenceStoreError(f"SQLite evidence write failed: {error}") from error
            raise
        finally:
            if connection is not None:
                connection.close()


__all__ = ["SQLiteEvidenceStore"]
