from copy import deepcopy
from pathlib import Path

import pytest

from qeclang.analysis.certificate_io import certificate_record
from qeclang.analysis.certificates import certify_css_logical
from qeclang.constructions import steane
from qeclang.workflow import ExecutionLimits, InspectionLimits, Program, execute, inspect_run
from qeclang.workflow.artifacts import describe

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/programs/exact_steane.json'


def test_exact_distance_claim_survives_scientific_inspection_and_tampering_is_rejected(monkeypatch):
    p = Program.load(EXAMPLE)
    result = execute(p, verify_relations=False, limits=ExecutionLimits(max_candidates=0))
    assert result['status'] == 'completed', result['trace'][-1]
    assert result['ledger']['used']['candidates'] == 0
    from qeclang.algebra import BinaryMatrix
    import qeclang.workflow.operations as ops
    def forbidden(*a, **kw):
        raise AssertionError('scientific checker must not run elimination/search')
    for name in ('solve', 'rank', 'row_basis', 'rref', 'nullspace', 'contains'):
        monkeypatch.setattr(BinaryMatrix, name, forbidden)
    monkeypatch.setattr(ops, 'css_distance', forbidden)
    from qeclang.analysis import low_weight
    monkeypatch.setattr(low_weight, 'certify_low_weight', forbidden)
    checked = inspect_run(p, result)
    assert checked['status'] == 'checked', checked
    assert {(c['sector'], c['value']) for c in checked['claims'] if c['predicate'] == 'sector_distance'} == {
        ('x', 3), ('z', 3)}
    full = [c for c in checked['claims'] if c['predicate'] == 'logical_distance']
    assert len(full) == 1 and full[0]['value'] == 3 and full[0]['sector'] == 'pauli'
    bad = deepcopy(result)
    last = bad['trace'][-1]
    old = last['output']
    value = dict(bad['artifacts'][old]['payload'], lower_bound=2, status='bounded')
    desc = describe('distance_claim', value)
    del bad['artifacts'][old]
    bad['artifacts'][desc['digest']] = desc
    last['output'] = bad['outputs'][last['node']] = desc['digest']
    assert inspect_run(p, bad)['status'] == 'invalid_record'


def test_budget_failure_in_lower_bound_producer_and_checker():
    p = Program.load(EXAMPLE)
    record = execute(p, limits=ExecutionLimits(max_exclusion_steps=0))
    assert record['status'] == 'evidence_unavailable'
    assert record['trace'][-1]['op'] == 'css.combine_distance'
    ordinary = execute(p)
    verdict = inspect_run(p, ordinary, limits=InspectionLimits(max_exclusion_verify_steps=0))
    assert verdict['status'] == 'budget_exhausted' and not verdict['claims']
    stopped = execute(p, limits=ExecutionLimits(max_exclusion_verify_steps=0))
    assert stopped['status'] == 'budget_exhausted'
    assert inspect_run(p, stopped, allow_prefix=True)['status'] == 'checked_prefix'


def test_bounds_for_distinct_sectors_or_presentations_cannot_be_joined():
    p = Program.load(EXAMPLE)
    data = p.to_dict()
    imports = [n for n in data['nodes'] if n['op'] == 'css.witness']
    combined = [n for n in data['nodes'] if n['op'] == 'css.combine_distance'][-1]
    combined['inputs'][-1] = next(n['id'] for n in imports if n['id'] != combined['inputs'][-1])
    with pytest.raises(ValueError, match='sector'):
        Program.from_dict(data)


def test_unresolved_interval_is_not_promoted_to_exact():
    # A heavier but valid logical upper witness can leave a gap above the exclusion.
    code = steane()
    proof = certificate_record(code, certify_css_logical(code, 'x', 127))
    data = Program.load(EXAMPLE).to_dict()
    first, exclusion = data['nodes'][:2]
    src = first['id']
    data['nodes'] = [first, exclusion,
        {'id':'heavy','op':'css.import_witness','inputs':[src],'params':{'certificate':proof}},
        {'id':'gap','op':'css.combine_distance','inputs':[src,exclusion['id'],'heavy'],'params':{}}]
    data['outputs'] = ['gap']
    p = Program.from_dict(data)
    record = execute(p)
    assert record['status'] == 'completed'
    claim = inspect_run(p, record)['claims'][-1]
    assert claim['status'] == 'bounded' and claim['value'] is None
    assert (claim['lower_bound'], claim['upper_bound']) == (3, 7)


def test_retrieved_exact_result_is_derived_again_under_reader_limits():
    from dataclasses import replace
    from qeclang.archive import ArchiveRefused, ArchiveLimits, InMemoryEvidenceStore, publish, retrieve
    p = Program.load(EXAMPLE)
    r = execute(p)
    store = InMemoryEvidenceStore()
    key = publish(p, r, store)['bundle_id']
    checked = retrieve(store, key)['inspection']
    assert checked['claims'][-1]['predicate'] == 'logical_distance'
    assert checked['claims'][-1]['value'] == 3
    with pytest.raises(ArchiveRefused) as refused:
        retrieve(store, key, limits=replace(ArchiveLimits(), inspection=InspectionLimits(max_exclusion_verify_steps=0)))
    assert refused.value.inspection['status'] == 'budget_exhausted'
    assert not refused.value.inspection['claims']
