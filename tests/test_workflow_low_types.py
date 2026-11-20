from copy import deepcopy
from hashlib import sha256
import json

import pytest

from qeclang.workflow.artifacts import ARTIFACT_SCHEMA, describe
from qeclang.workflow.import_artifact import (ArtifactImportError, ImportBudget, ImportLimitExceeded,
                                               ImportLimits, import_artifact)
from qeclang.workflow.json_input import JSONLimitExceeded, JSONLimits
from qeclang.workflow.low_weight_payload import DISTANCE_CLAIM_SCHEMA, EXCLUSION_SCHEMA
from qeclang.workflow.program import Program, ProgramLimits, SCHEMA


def witness(*, sector='x', weight=1):
    return {'schema': 'qeclang.certificate.v1', 'source': 'a'*64, 'kind': 'css_logical',
            'claim': {'distance_upper_bound': weight, 'sector': sector},
            'certificate': {'sector': sector, 'operator': '0x1', 'dual': '0x1', 'weight': weight}}


def exclusion(status='excluded'):
    record = {'schema': EXCLUSION_SCHEMA, 'record_id': 'b'*64, 'source': 'a'*64,
              'sector': 'x', 'through': 2, 'status': status, 'steps': 20,
              'claim': None, 'memberships': None, 'witness': None, 'reason': None}
    if status == 'excluded':
        record.update(claim={'predicate': 'nontrivial_weight_exclusion', 'through': 2},
                      memberships=[[0], [1, 2]])
    elif status == 'found':
        record.update(claim={'predicate': 'distance_upper_bound', 'upper_bound': 1}, witness=witness())
    else:
        record['reason'] = 'producer arithmetic budget exhausted'
    return record


def distance_claim(*, sector='x', lower=3, upper=3):
    return {'schema': DISTANCE_CLAIM_SCHEMA, 'source': 'a'*64, 'sector': sector,
            'lower_bound': lower, 'upper_bound': upper,
            'status': 'exact' if lower == upper else 'bounded'}


def program_data(*, full=False):
    code = {'schema': 'qeclang.code.v1', 'kind': 'css', 'n': 7,
            'hx': [[0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6]],
            'hz': [[0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6]]}
    nodes = [{'id': 'code', 'op': 'css.literal', 'inputs': [], 'params': {'code': code}}]
    for sector in ('x', 'z') if full else ('x',):
        nodes.extend([
            {'id': f'{sector}_upper', 'op': 'css.import_witness', 'inputs': ['code'],
             'params': {'certificate': witness(sector=sector, weight=3)}},
            {'id': f'{sector}_absence', 'op': 'css.exclude', 'inputs': ['code'],
             'params': {'sector': sector, 'through': 2, 'max_steps': 10000}},
            {'id': f'{sector}_distance', 'op': 'css.combine_distance',
             'inputs': ['code', f'{sector}_absence', f'{sector}_upper'], 'params': {}},
        ])
    if full:
        nodes.append({'id': 'full', 'op': 'css.full_distance',
                      'inputs': ['code', 'x_distance', 'z_distance'], 'params': {}})
    return {'schema': SCHEMA, 'nodes': nodes, 'outputs': [nodes[-1]['id']]}


def rehashed(kind, payload):
    encoded = json.dumps({'schema': ARTIFACT_SCHEMA, 'kind': kind, 'payload': payload},
                         sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    return {'kind': kind, 'digest': sha256(encoded.encode()).hexdigest(), 'payload': payload}


def test_low_weight_operations_have_nominal_sector_types_and_immutable_params():
    raw = program_data()
    program = Program.from_dict(raw)
    proof, combined = program.nodes[-2:]
    assert proof.result_type == 'exclusion' and combined.result_type == 'distance_claim'
    assert proof.code_id == combined.code_id == 'code'
    assert proof.sector == combined.sector == 'x'
    assert (proof.n, proof.mx, proof.mz) == (7, 3, 3)
    raw['nodes'][-2]['params']['through'] = 1
    copied = proof.params
    copied['sector'] = 'z'
    assert proof.params['through'] == 2 and proof.params['sector'] == 'x'
    assert Program.loads(json.dumps(program.to_dict())).fingerprint() == program.fingerprint()


@pytest.mark.parametrize('change', [{'through': 0}, {'through': 3}, {'through': True},
                                   {'through': 2.0}, {'max_steps': -1}, {'max_steps': False},
                                   {'sector': 'pauli'}, {'sector': 'y'}, {'extra': 1}])
def test_exclusion_parameters_are_closed_and_strict(change):
    raw = program_data()
    raw['nodes'][2]['params'].update(change)
    with pytest.raises(ValueError):
        Program.from_dict(raw)


def test_exclusion_request_requires_all_fields_and_allows_zero_budget():
    raw = program_data()
    raw['nodes'][2]['params']['max_steps'] = 0
    assert Program.from_dict(raw).nodes[2].params['max_steps'] == 0
    for field in ('sector', 'through', 'max_steps'):
        changed = deepcopy(raw)
        del changed['nodes'][2]['params'][field]
        with pytest.raises(ValueError):
            Program.from_dict(changed)


def test_combine_requires_matching_nominal_sources_sectors_and_evidence_kinds():
    raw = program_data()
    raw['nodes'][2]['params']['sector'] = 'z'
    with pytest.raises(ValueError, match='sectors'):
        Program.from_dict(raw)
    for changed_input in (1, 2):
        raw = program_data()
        alias = deepcopy(raw['nodes'][0])
        alias['id'] = 'alias'
        raw['nodes'].insert(1, alias)
        raw['nodes'][changed_input+1]['inputs'] = ['alias']
        with pytest.raises(ValueError, match='bound'):
            Program.from_dict(raw)
    raw = program_data()
    raw['nodes'][2].update(op='css.distance', params={'sector': 'x', 'max_weight': 2, 'max_candidates': 0})
    with pytest.raises(ValueError, match='input types'):
        Program.from_dict(raw)
    raw = program_data()
    raw['nodes'][-1]['inputs'] = ['code', 'x_absence', 'x_absence']
    with pytest.raises(ValueError, match='input types'):
        Program.from_dict(raw)
    raw = program_data()
    raw['nodes'][-1]['params'] = {'allow_no_logicals': True}
    with pytest.raises(ValueError):
        Program.from_dict(raw)


def test_full_distance_requires_one_source_bound_claim_per_css_sector():
    raw = program_data(full=True)
    program = Program.from_dict(raw)
    assert program.nodes[-1].result_type == 'distance_claim'
    assert (program.nodes[-1].code_id, program.nodes[-1].sector) == ('code', 'pauli')
    raw['nodes'][-1]['inputs'][1:] = ['z_distance', 'x_distance']
    assert Program.from_dict(raw).nodes[-1].sector == 'pauli'
    raw['nodes'][-1]['inputs'][1:] = ['x_distance', 'x_distance']
    with pytest.raises(ValueError, match='one x and one z'):
        Program.from_dict(raw)
    raw = program_data(full=True)
    raw['nodes'].append({'id': 'invalid', 'op': 'css.full_distance',
                         'inputs': ['code', 'full', 'z_distance'], 'params': {}})
    with pytest.raises(ValueError, match='one x and one z'):
        Program.from_dict(raw)
    raw = program_data(full=True)
    raw['nodes'][-1]['params'] = {'sector': 'pauli'}
    with pytest.raises(ValueError):
        Program.from_dict(raw)


def test_full_distance_rejects_equal_codes_with_different_nominal_binding():
    raw = program_data(full=True)
    alias = deepcopy(raw['nodes'][0])
    alias['id'] = 'alias'
    raw['nodes'].insert(1, alias)
    for node in raw['nodes']:
        if node['id'].startswith('z_'):
            node['inputs'] = ['alias' if value == 'code' else value for value in node['inputs']]
    with pytest.raises(ValueError, match='bound'):
        Program.from_dict(raw)


@pytest.mark.parametrize('status', ['excluded', 'found', 'budget_exhausted'])
def test_exclusion_status_payloads_roundtrip_without_claim_verification(status, monkeypatch):
    import qeclang.analysis.certificates as proofs
    import qeclang.analysis.certificate_io as certificates
    from qeclang.algebra import BinaryMatrix
    def forbidden(*args, **kwargs):
        raise AssertionError('artifact syntax admission must not verify mathematics')
    monkeypatch.setattr(certificates, 'presentation_fingerprint', forbidden)
    monkeypatch.setattr(certificates, 'verify_certificate_record', forbidden)
    monkeypatch.setattr(proofs, 'verify_css_logical', forbidden)
    for method in ('rank', 'solve', 'matvec'):
        monkeypatch.setattr(BinaryMatrix, method, forbidden)
    value = exclusion(status)
    described = describe('exclusion', value)
    imported = import_artifact(described, 'exclusion')
    assert imported == value
    assert imported is not value and described['payload'] is not value
    # The source and self-ID are intentionally fictitious. Syntax admission must
    # not turn them into a checked exclusion, witness, or existence claim.
    assert imported['record_id'] == 'b'*64
    if status == 'excluded':
        imported['memberships'][0].clear()
        assert described['payload']['memberships'][0] == [0]
    assert 'assurance' not in described


@pytest.mark.parametrize('change', [
    {'schema': 'unsupported'}, {'record_id': 'B'*64}, {'source': 'short'},
    {'sector': 'pauli'}, {'through': True}, {'through': 0}, {'through': 3},
    {'steps': True}, {'steps': -1}, {'status': 'exact'}, {'reason': 'not a success'},
    {'claim': {'predicate': 'distance_lower_bound', 'through': 2}},
    {'claim': {'predicate': 'nontrivial_weight_exclusion', 'through': True}},
    {'claim': {'predicate': 'nontrivial_weight_exclusion', 'through': 1}},
    {'memberships': [[1, 0]]}, {'memberships': [[0, 0]]}, {'memberships': [[True]]},
    {'memberships': [[-1]]}, {'memberships': [0]}, {'memberships': None},
    {'witness': witness()}, {'lower_bound': 3},
])
def test_bad_exclusion_syntax_rejected_by_description_and_rehashed_import(change):
    value = {**exclusion(), **change}
    with pytest.raises(ValueError):
        describe('exclusion', value)
    with pytest.raises(ArtifactImportError):
        import_artifact(rehashed('exclusion', value), 'exclusion')


@pytest.mark.parametrize('change', [
    {'claim': {'predicate': 'distance_upper_bound', 'upper_bound': 3}},
    {'claim': {'predicate': 'distance_upper_bound', 'upper_bound': True}},
    {'claim': {'predicate': 'distance_upper_bound', 'upper_bound': 2}},
    {'memberships': []}, {'witness': None}, {'reason': 'unexpected'},
])
def test_found_status_has_consistent_witness_presence_and_bound(change):
    value = {**exclusion('found'), **change}
    with pytest.raises(ValueError):
        describe('exclusion', value)


def test_found_witness_source_sector_and_canonical_words_are_structural():
    for key, value in (('source', 'c'*64), ('kind', 'stabilizer_logical')):
        changed = exclusion('found')
        changed['witness'][key] = value
        with pytest.raises(ValueError):
            describe('exclusion', changed)
    for key, value in (('sector', 'z'), ('operator', '0x01'), ('dual', '0X1'), ('weight', True)):
        changed = exclusion('found')
        changed['witness']['certificate'][key] = value
        with pytest.raises(ValueError):
            describe('exclusion', changed)
    changed = exclusion('found')
    changed['witness']['certificate']['dual'] = '0x0'
    assert import_artifact(describe('exclusion', changed), 'exclusion')['witness']['certificate']['dual'] == '0x0'


@pytest.mark.parametrize('change', [{'claim': {}}, {'memberships': []}, {'witness': witness()},
                                   {'reason': None}, {'reason': ''}])
def test_budget_status_cannot_publish_partial_evidence(change):
    with pytest.raises(ValueError):
        describe('exclusion', {**exclusion('budget_exhausted'), **change})


@pytest.mark.parametrize('sector', ['x', 'z', 'pauli'])
@pytest.mark.parametrize('lower,upper', [(3, 3), (2, 7)])
def test_distance_claims_preserve_tagged_schema_without_mathematical_assurance(sector, lower, upper):
    value = distance_claim(sector=sector, lower=lower, upper=upper)
    artifact = describe('distance_claim', value)
    assert import_artifact(artifact, 'distance_claim') == value
    assert artifact['payload']['schema'] == DISTANCE_CLAIM_SCHEMA
    assert 'assurance' not in artifact


@pytest.mark.parametrize('change', [
    {'schema': 'unsupported'}, {'source': 'short'}, {'sector': 'y'},
    {'lower_bound': 0}, {'lower_bound': True}, {'upper_bound': False},
    {'lower_bound': 4}, {'status': 'bounded'}, {'status': 'no_logicals'},
    {'status': 'budget_exhausted'}, {'witness': None}, {'through': 2},
])
def test_distance_claim_schema_and_finite_interval_consistency(change):
    value = {**distance_claim(), **change}
    with pytest.raises(ValueError):
        describe('distance_claim', value)
    with pytest.raises(ArtifactImportError):
        import_artifact(rehashed('distance_claim', value), 'distance_claim')


def test_independent_payload_limits_and_cumulative_byte_admission():
    value = describe('exclusion', exclusion())
    with pytest.raises(ImportLimitExceeded, match='row count'):
        import_artifact(value, 'exclusion', limits=ImportLimits(max_axis=1))
    with pytest.raises(ImportLimitExceeded, match='generator index'):
        import_artifact(value, 'exclusion', limits=ImportLimits(max_axis=2))
    with pytest.raises(ImportLimitExceeded, match='indices'):
        import_artifact(value, 'exclusion', limits=ImportLimits(max_low_weight_indices=2))
    assert import_artifact(value, 'exclusion', limits=ImportLimits(max_low_weight_indices=3)) == exclusion()
    found = exclusion('found')
    found['witness']['certificate']['dual'] = '0x8'
    with pytest.raises(ImportLimitExceeded, match='word width'):
        import_artifact(describe('exclusion', found), 'exclusion', limits=ImportLimits(max_axis=3))
    with pytest.raises(ImportLimitExceeded, match='upper bound'):
        import_artifact(describe('distance_claim', distance_claim()), 'distance_claim', limits=ImportLimits(max_axis=2))
    with pytest.raises(JSONLimitExceeded):
        describe('exclusion', exclusion(), limits=JSONLimits(max_payload_items=5))
    with pytest.raises(JSONLimitExceeded):
        describe('distance_claim', distance_claim(), limits=JSONLimits(max_bytes=10))
    budget = ImportBudget(max_artifact_bytes=10)
    with pytest.raises(ImportLimitExceeded):
        import_artifact(value, 'exclusion', budget=budget)
    with pytest.raises(ValueError):
        ImportLimits(max_low_weight_indices=True)


def test_empty_membership_arrays_and_no_logicals_remain_distinct():
    value = exclusion()
    value['memberships'] = []
    assert import_artifact(describe('exclusion', value), 'exclusion', limits=ImportLimits(max_axis=0)) == value
    bad = {**distance_claim(), 'status': 'no_logicals', 'lower_bound': None, 'upper_bound': None}
    with pytest.raises(ValueError):
        describe('distance_claim', bad)


def test_legacy_program_hash_is_not_changed_by_new_closed_operations():
    raw = {'schema': SCHEMA, 'nodes': [{'id': 'c', 'op': 'css.literal', 'inputs': [],
        'params': {'code': {'schema': 'qeclang.code.v1', 'kind': 'css', 'n': 1, 'hx': [], 'hz': []}}}],
        'outputs': ['c']}
    expected = sha256(json.dumps(raw, sort_keys=True, separators=(',', ':'),
                                ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    assert Program.from_dict(raw, limits=ProgramLimits()).fingerprint() == expected


def test_actual_producer_records_match_the_portable_payload_contract(monkeypatch):
    import qeclang.analysis.low_weight as low_weight
    from qeclang.algebra import BinaryMatrix
    from qeclang.codes import CSSCode
    from qeclang.constructions import steane
    produced = [low_weight.certify_low_weight(steane(), 'x'),
                low_weight.certify_low_weight(CSSCode(BinaryMatrix((), 1), BinaryMatrix((), 1)), 'z'),
                low_weight.certify_low_weight(steane(), 'x', max_steps=0)]
    assert [record['status'] for record in produced] == ['excluded', 'found', 'budget_exhausted']
    def forbidden(*args, **kwargs):
        raise AssertionError('artifact admission must not invoke the scientific checker')
    monkeypatch.setattr(low_weight, 'check_low_weight', forbidden)
    monkeypatch.setattr(low_weight, 'certify_low_weight', forbidden)
    for value in produced:
        assert import_artifact(describe('exclusion', value), 'exclusion') == value
    assert Program.from_dict(program_data(full=True)).nodes[-1].sector == 'pauli'


def test_guarded_admission_precedes_decimal_serialization_of_large_integers():
    value = exclusion()
    value['steps'] = 1 << 100_000
    with pytest.raises(JSONLimitExceeded):
        describe('exclusion', value)
    with pytest.raises(ImportLimitExceeded):
        import_artifact({'kind': 'exclusion', 'digest': '0'*64, 'payload': value}, 'exclusion')
