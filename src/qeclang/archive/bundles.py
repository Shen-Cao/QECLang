"""Explicit publication and independent read-back of immutable program/run bundles."""
from dataclasses import dataclass, field

from qeclang.workflow import InspectionLimits, Program, ProgramLimits, inspect_run
from qeclang.workflow.json_input import (JSONLimits, canonical_json, loads_json,
                                        validate_json_data)
from .protocol import EvidenceStoreError, EvidenceStoreLimitExceeded, bundle_id, validate_key

SCHEMA = 'qeclang.evidence_bundle.v1'


@dataclass(frozen=True)
class ArchiveLimits:
    """Local admission and scientific-verification policies; never loaded from a bundle."""
    json: JSONLimits = field(default_factory=lambda: JSONLimits(max_bytes=32_000_000,
        max_payload_items=8_000_000, max_payload_depth=96, max_integer_bits=4096))
    program: ProgramLimits = field(default_factory=ProgramLimits)
    inspection: InspectionLimits = field(default_factory=InspectionLimits)

    def __post_init__(self):
        if not (isinstance(self.json, JSONLimits) and isinstance(self.program, ProgramLimits)
                and isinstance(self.inspection, InspectionLimits)):
            raise TypeError('archive policy requires JSON, program and inspection limits')


class ArchiveRefused(ValueError):
    """A scientific inspection did not succeed under the caller's admission policy."""
    def __init__(self, report):
        self.inspection = report
        super().__init__(report['reason'] or report['status'])


def _checked(payload, limits, allow_prefix, expected_program=None):
    decoded = loads_json(payload, limits=limits.json)
    if (type(decoded) is not dict or set(decoded) != {'schema', 'program', 'run'}
            or decoded['schema'] != SCHEMA):
        raise EvidenceStoreError('invalid evidence bundle schema or fields')
    if canonical_json(decoded).encode('ascii') != payload:
        raise EvidenceStoreError('evidence bundle is not canonical JSON')
    program = Program.from_dict(decoded['program'], limits=limits.program)
    if expected_program is not None and program.fingerprint() != expected_program:
        raise EvidenceStoreError('bundle contains a different program')
    inspection = inspect_run(program, decoded['run'], limits=limits.inspection, allow_prefix=allow_prefix)
    if inspection['status'] not in (('checked', 'checked_prefix') if allow_prefix else ('checked',)):
        raise ArchiveRefused(inspection)
    return decoded, inspection


def publish(program, record, store, *, limits=ArchiveLimits(), allow_prefix=False):
    """Validate the exact bytes being published, then explicitly call the writer.

    A bundle records a run, including producer-only metadata. The separately
    returned receipt lists only claims independently established by inspection.
    The store cannot upgrade those claims. Failed admission makes no write call.
    """
    if not isinstance(program, Program) or not isinstance(limits, ArchiveLimits):
        raise TypeError('publish requires Program and ArchiveLimits')
    if type(allow_prefix) is not bool:
        raise TypeError('allow_prefix must be boolean')
    document = {'schema': SCHEMA, 'program': program.to_dict(), 'run': record}
    validate_json_data(document, limits=limits.json)
    payload = canonical_json(document).encode('ascii')
    _, inspection = _checked(payload, limits, allow_prefix)
    key = bundle_id(payload)
    inserted = store.save(key, payload)
    if type(inserted) is not bool:
        raise EvidenceStoreError('writer acknowledgement must be a boolean')
    return {'bundle_id': key, 'inserted': inserted, 'inspection': inspection}


def retrieve(store, key, *, limits=ArchiveLimits(), allow_prefix=False, expected_program=None):
    """Fetch by content identity and inspect again using the reader's own policy.

    A reader implementing fetch is responsible for limiting transport size before
    returning bytes. No execution policy, checker version or cached verdict in
    stored data overrides the caller's limits. expected_program optionally pins
    a second identity for use with an independently known program.
    """
    if not isinstance(limits, ArchiveLimits) or type(allow_prefix) is not bool:
        raise TypeError('invalid archive reader policy')
    validate_key(key)
    if expected_program is not None:
        validate_key(expected_program)
    payload = store.fetch(key)
    if type(payload) is not bytes:
        raise EvidenceStoreError('reader returned a non-bytes bundle')
    if len(payload) > limits.json.max_bytes:
        raise EvidenceStoreLimitExceeded('fetched bundle exceeds reader byte cap')
    if bundle_id(payload) != key:
        raise EvidenceStoreError('retrieved bytes differ from the requested content identity')
    decoded, inspection = _checked(payload, limits, allow_prefix, expected_program)
    return {'bundle_id': key, 'program': decoded['program'], 'run': decoded['run'], 'inspection': inspection}
