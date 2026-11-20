from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from qeclang.archive import ArchiveLimits, ArchiveRefused, InMemoryEvidenceStore, bundle_id, publish, retrieve
from qeclang.archive.protocol import EvidenceStoreError
from qeclang.cli import main
from qeclang.constructions.records import canonical_json
from qeclang.workflow import ExecutionLimits, InspectionLimits, Program, execute

BASE = Path(__file__).resolve().parents[1] / 'examples/programs'


def test_local_roundtrip_and_reader_policies():
    p = Program.load(BASE / 'transport.json')
    run = execute(p, verify_relations=False)
    store = InMemoryEvidenceStore()
    receipt = publish(p, run, store)
    key = receipt['bundle_id']
    assert receipt['inserted'] and not publish(p, run, store)['inserted']
    result = retrieve(store, key, expected_program=p.fingerprint())
    assert result['inspection'] == receipt['inspection'] and result['run'] == run
    result['run']['artifacts'].clear()
    assert retrieve(store, key)['run'] == run
    with pytest.raises(EvidenceStoreError, match='different program'):
        retrieve(store, key, expected_program='b'*64)
    with pytest.raises(ArchiveRefused) as failure:
        retrieve(store, key, limits=replace(ArchiveLimits(), inspection=InspectionLimits(max_certificates=0)))
    assert failure.value.inspection['status'] == 'budget_exhausted'


def test_fail_before_write_and_independent_reads():
    class NoWriter:
        def save(self, *a):
            pytest.fail('no store call is permitted before successful verification')
    p = Program.load(BASE / 'transport.json')
    record = execute(p)
    corrupt = deepcopy(record)
    corrupt['artifacts'][record['outputs']['transported']]['payload']['certificate']['dual'] = '0x0'
    with pytest.raises(ArchiveRefused):
        publish(p, corrupt, NoWriter())
    class ForgedReader:
        def fetch(self, key):
            return canonical_json({'schema': 'qeclang.evidence_bundle.v1', 'program': p.to_dict(),
                                   'run': corrupt}).encode()
    reader = ForgedReader()
    with pytest.raises(EvidenceStoreError, match='identity'):
        retrieve(reader, '0'*64)
    # Correct content digest does not establish scientific validity.
    with pytest.raises(ArchiveRefused):
        retrieve(reader, bundle_id(reader.fetch(None)))


def test_refused_packed_expansion_preserves_checked_dimension_in_archive():
    p = Program.load(BASE / 'cyclic_rank.json')
    record = execute(p, limits=ExecutionLimits(max_matrix_bits=1))
    assert record['status'] == 'budget_exhausted'
    store = InMemoryEvidenceStore()
    with pytest.raises(ArchiveRefused):
        publish(p, record, store)
    receipt = publish(p, record, store, allow_prefix=True)
    result = retrieve(store, receipt['bundle_id'], allow_prefix=True)
    inspection = result['inspection']
    assert inspection['status'] == 'checked_prefix'
    assert inspection['claims'][0]['value'] == 2
    assert inspection['unverified_stop']['node'] == 'binary'
    with pytest.raises(ArchiveRefused):
        retrieve(store, receipt['bundle_id'])


def test_archive_cli_put_get_and_missing(tmp_path, capsys):
    program = BASE / 'transport.json'
    record = execute(Program.load(program))
    path = tmp_path / 'record'
    path.write_text(json.dumps(record))
    db = tmp_path / 'archive.sqlite'
    assert main(['archive', 'get', str(db), '0'*64]) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'not_found'
    assert not db.exists()
    assert main(['archive', 'put', str(db), str(program), str(path)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt['inserted']
    assert main(['archive', 'get', str(db), receipt['bundle_id'], '--include-record']) == 0
    saved = json.loads(capsys.readouterr().out)
    assert saved['run'] == record and saved['inspection']['status'] == 'checked'
