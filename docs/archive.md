# Explicit evidence archives

The archive stores a program and its reported run as immutable, content-addressed bytes. Publication and retrieval each perform a fresh scientific inspection under the caller's policy. Storage does not make a producer's claims authoritative, and running a workflow never publishes it automatically.

## Publish and retrieve

```python
from qeclang.archive import InMemoryEvidenceStore, publish, retrieve
from qeclang.workflow import Program, execute

program = Program.load("examples/programs/transport.json")
record = execute(program)
store = InMemoryEvidenceStore()
assert store.bundle_count == 0

receipt = publish(program, record, store)
assert receipt["inserted"]
assert not publish(program, record, store)["inserted"]

received = retrieve(store, receipt["bundle_id"],
                    expected_program=program.fingerprint())
assert received["inspection"]["status"] == "checked"
```

`publish(program, record, store, *, limits=ArchiveLimits(), allow_prefix=False)` validates the exact canonical bytes and independently inspects their contents before calling the writer. A failed admission or scientific check makes no write call. The receipt contains `bundle_id`, `inserted`, and the newly derived `inspection`.

`retrieve(store, key, *, limits=ArchiveLimits(), allow_prefix=False, expected_program=None)` checks fetched byte type, size, content identity, canonical bundle schema, program validity, and scientific evidence again. It returns `bundle_id`, the decoded `program`, the reported `run`, and a new `inspection`. The optional `expected_program` pins the program's separate fingerprint when the reader already knows which program is intended.

The inspector checks only its supported relations and finite-instance claims, such as logical upper witnesses and supported symbolic rank certificates. Its `unverified` fields remain important. Publication does not upgrade distance lower bounds, distance exactness, producer assurance, candidate counts, historical ledger usage, or runtime provenance into independently checked facts. See [workflow contracts](workflow.md).

## Bundle bytes and identities

The stored document has exactly three fields:

```json
{
  "schema": "qeclang.evidence_bundle.v1",
  "program": {"schema": "qeclang.program.v1", "nodes": [], "outputs": []},
  "run": {}
}
```

The shape above illustrates the envelope; the run must separately pass its schema and inspection requirements. Actual bytes use the shared canonical JSON convention: sorted object keys, compact separators, ASCII escapes, and no nonfinite numbers. Retrieval rejects noncanonical encodings instead of silently rewriting their identities.

The content key is

```text
SHA256(b"qeclang.evidence_bundle.v1\x00" + canonical_bundle_bytes).hexdigest()
```

Keys are exactly 64 lowercase hexadecimal characters. They never contain filesystem paths. The domain prefix distinguishes this byte format from other digest uses. Hashes bind exact representations, including the run's historical metadata; they do not classify mathematically equivalent programs or codes, authenticate an author, establish discovery or novelty, or certify a theorem. Altering the reported runtime metadata can change a bundle ID even when the independently checkable mathematics is unchanged.

Inspection output is returned separately and is not stored as universal truth in the envelope. A later reader may have different supported predicates, stricter limits, or a newer checker. That reader recomputes its own verdict rather than trusting a saved one.

## Minimal store capabilities

`qeclang.archive.protocol` defines separate structural protocols:

| Capability | Operation | Contract |
|---|---|---|
| `EvidenceReader` | `fetch(key: str) -> bytes` | Return bounded immutable bytes matching the key; a missing valid key raises `KeyError` |
| `EvidenceWriter` | `save(key: str, payload: bytes) -> bool` | Return `True` on insertion and `False` on an identical retry; never replace conflicting content |
| `EvidenceStore` | Both operations | Convenience for providers supporting both capabilities |

`validate_key`, `bundle_id`, and `validate_payload` implement the shared key/content convention. `validate_payload` checks the per-bundle byte limit before hashing or copying. Invalid key/content or detected corruption raises `EvidenceStoreError`; a capacity refusal raises its subclass `EvidenceStoreLimitExceeded`. Protocol conformance describes an interface, not proof of a custom provider's behavior.

The stores treat payloads as opaque bytes. Calling `save` directly checks byte integrity only; mathematical admission belongs to `publish` and `retrieve`. Similarly, fetching bytes directly does not invoke the scientific inspector. A custom reader must bound its transfer before returning bytes; service-side checks cannot undo an oversized allocation already made by a provider.

### In-memory reference store

`InMemoryEvidenceStore(max_bundle_bytes=32_000_000, max_retained_bytes=64_000_000)` retains immutable byte snapshots. Construction and reads insert no entries. Identical retries do not consume capacity twice, including when the store is full. Different content cannot replace an occupied key, and corruption detected on fetch or retry is rejected rather than repaired implicitly.

Save/fetch and capacity accounting are lock-protected. `bundle_count` and `retained_bytes` expose the retained entry/payload counts. Byte caps exclude dictionary/object overhead and temporary copies. This store does not access disk or the network and does not survive process lifetime.

### SQLite reference store

`SQLiteEvidenceStore(path, *, readonly=False, max_bundle_bytes=32_000_000)` is a lazy descriptor. Constructing it does not connect to, create, or initialize a database. Fetch opens an existing database in read-only mode and reads a transaction snapshot; missing databases or keys raise `KeyError`. An empty uninitialized database reads as missing.

Only `save` may create or modify the file. It validates bytes first, then uses an immediate transaction to initialize an empty database or verify the dedicated archive schema before insertion. Identical retries are idempotent. Conflicting bytes, modified metadata, and unrelated database objects are rejected; the store does not repurpose an existing application database. Parent directories must already exist. A `readonly=True` instance rejects save with `PermissionError`.

BLOB type and length are checked before transferring a bounded payload to Python. The per-bundle limit does not cap total database size, SQLite/page-cache workspace, or filesystem race costs. There is no implicit eviction, garbage collection, or overwrite operation. Generated databases belong in ignored output directories.

## Reader policy and partial runs

`ArchiveLimits` contains independent JSON, program-admission, and scientific-inspection policies. Its default outer JSON cap is 32,000,000 bytes. Store limits and service limits both apply; neither is read from the stored run. A reader can therefore refuse a bundle that another policy previously admitted. Byte admission, JSON item/depth bounds, artifact import limits, and rank-verification work budgets are distinct controls, not a total-memory or wall-clock guarantee.

Full completed runs are required by default. With `allow_prefix=True`, publication/retrieval may accept an independently checkable completed prefix and report `checked_prefix`. This certifies only the accepted prefix's supported claims. It does not certify that the producer actually stopped for its reported reason, spent its claimed budget, or reached an authentic runtime failure. The reader must opt into prefix acceptance separately; the bundle cannot enable it for them.

Scientific retrieval is also different from strict replay. It does not reexecute search or require the producer's runtime identity to match the reader's. Strict replay remains available through the workflow API when exact execution reproduction is the intended question.

## Command-line archive

```sh
mkdir -p output
qeclang run examples/programs/transport.json > output/transport-run.json
qeclang archive put output/evidence.sqlite examples/programs/transport.json \
  output/transport-run.json
qeclang archive get output/evidence.sqlite BUNDLE_ID
```

Replace `BUNDLE_ID` with the returned 64-character key. `get` checks the bundle again and prints the ID and fresh inspection by default; `--include-record` also returns the program and run. `--expected-program SHA256` pins the expected program identity. Both verbs support `--allow-prefix`, `--max-bundle-bytes`, and `--max-reader-steps`; the last bounds arithmetic units for symbolic rank-proof verification.

CLI failures distinguish refused scientific evidence, exhausted budgets, missing content, and invalid requests. Neither command weakens verification based on stored metadata. This is an explicit evidence-storage boundary, not an automatic memory service or a general publication platform.
