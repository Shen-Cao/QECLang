"""Adversarial checks of scientific inspection, distinct from strict run replay."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from qeclang.algebra import BinaryMatrix
from qeclang.analysis.certificate_io import certificate_record, verify_certificate_record
from qeclang.analysis.certificates import LogicalWitness
from qeclang.codes import CSSCode, ChainComplex, SparseCSSCode
from qeclang.io import from_dict
from qeclang.workflow import ExecutionLimits, Program, execute
from qeclang.workflow.artifacts import ARTIFACT_SCHEMA, describe
from qeclang.workflow.inspect_run import InspectionLimits, inspect_run
from qeclang.workflow.json_input import JSONLimitExceeded, JSONLimits, loads_json, validate_json_data

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/programs/transport.json'


def authentic(*, limits=ExecutionLimits(), verify_relations=False):
    program = Program.load(EXAMPLE)
    record = execute(program, limits=limits, verify_relations=verify_relations)
    assert record['status'] == 'completed'
    return program, record


def step(record, name):
    return next(item for item in record['trace'] if item['node'] == name)


def artifact(record, name):
    return record['artifacts'][step(record, name)['output']]


def replace_artifact(record, old_digest, replacement):
    """Rehash a changed artifact and update every graph/output reference to it."""
    replacement = deepcopy(replacement)
    encoded = json.dumps({'schema': ARTIFACT_SCHEMA, 'kind': replacement['kind'],
                          'payload': replacement['payload']}, sort_keys=True,
                         separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    new_digest = sha256(encoded.encode()).hexdigest()
    replacement['digest'] = new_digest
    del record['artifacts'][old_digest]
    record['artifacts'][new_digest] = replacement
    for item in record['trace']:
        item['inputs'] = [new_digest if ref == old_digest else ref for ref in item['inputs']]
        if item['output'] == old_digest:
            item['output'] = new_digest
    record['outputs'] = {name: new_digest if ref == old_digest else ref
                         for name, ref in record['outputs'].items()}
    return replacement


def assert_refused(result, status='invalid_record'):
    assert result['status'] == status, result
    assert result['claims'] == [] and result['relations'] == []
    assert result['reason']


def assert_only_upper_claims(result):
    assert all(claim['predicate'] == 'distance_upper_bound' for claim in result['claims'])
    assert all(claim['scope'] == 'instance' for claim in result['claims'])
    assert all('lower_bound' not in claim and 'distance' not in claim for claim in result['claims'])


def test_authentic_completed_run_checks_all_edges_despite_unverified_producer_mode():
    program, record = authentic()
    assert step(record, 'complex')['evidence']['assurance'] == 'not_requested'
    result = inspect_run(program, record)
    assert result['status'] == 'checked', result
    assert result['scope'] == 'based_relations_and_css_upper_witnesses'
    assert {item['node'] for item in result['relations']} == {
        'complex', 'original', 'relabeled', 'sparse', 'final', 'transported'}
    assert {claim['node'] for claim in result['claims']} == {'logical', 'transported'}
    assert result['reported_distance_nodes'] == ['distance']
    assert_only_upper_claims(result)
    for claim in result['claims']:
        saved = artifact(record, claim['node'])['payload']
        assert claim['bound'] == saved['claim']['distance_upper_bound'] == 3
        assert claim['source'] == saved['source']
        assert claim['sector'] == saved['claim']['sector'] == 'x'
    _, optional_exhausted = authentic(limits=ExecutionLimits(max_optional_checks=0), verify_relations=True)
    assert inspect_run(program, optional_exhausted)['status'] == 'checked'


def test_inspection_does_not_execute_search_elimination_or_producing_conversions(monkeypatch):
    program, record = authentic()
    import qeclang.analysis as analysis
    import qeclang.analysis.distance as distance
    import qeclang.transforms as transforms
    import qeclang.workflow.executor as executor
    import qeclang.workflow.operations as operations
    def forbidden(*args, **kwargs):
        raise AssertionError('scientific inspection must not rerun the producer or elimination')
    monkeypatch.setattr(executor, 'execute', forbidden)
    monkeypatch.setattr(executor, 'replay', forbidden)
    monkeypatch.setattr(operations, 'evaluate', forbidden)
    monkeypatch.setattr(operations, 'css_distance', forbidden)
    monkeypatch.setattr(analysis, 'css_distance', forbidden)
    monkeypatch.setattr(distance, 'css_distance', forbidden)
    monkeypatch.setattr(transforms, 'permute_qubits', forbidden)
    for method in ('rank', 'solve', 'rref', 'row_basis', 'nullspace'):
        monkeypatch.setattr(BinaryMatrix, method, forbidden)
    monkeypatch.setattr(CSSCode, 'to_chain_complex', forbidden)
    monkeypatch.setattr(ChainComplex, 'to_css', forbidden)
    monkeypatch.setattr(SparseCSSCode, 'to_packed', forbidden)
    monkeypatch.setattr(SparseCSSCode, 'from_packed', forbidden)
    result = inspect_run(program, record)
    assert result['status'] == 'checked', result
    assert_only_upper_claims(result)


def test_recorded_runtime_cost_assurance_and_mode_are_not_scientific_evidence():
    program, record = authentic()
    record['runtime'] = {'package_version': 'future-version', 'source_sha256': '0' * 64,
                          'python_version': 'different'}
    record['ledger'] = {'limits': {'invented': 1}, 'used': {'candidates': 999999999}}
    record['verify_relations'] = True
    for item in record['trace']:
        item['spent'] = {'fabricated_cost': 999999}
        item['evidence'] = {'assurance': 'formally_verified', 'predicate': 'distance_lower_bound',
                            'bound': 999999}
    result = inspect_run(program, record)
    assert result['status'] == 'checked', result
    assert_only_upper_claims(result)
    disclosed = json.dumps(result['unverified']).lower()
    assert 'runtime' in disclosed and 'ledger' in disclosed and 'assurance' in disclosed


def test_forged_exact_lower_bound_does_not_become_a_scientific_claim():
    # Hx=(111), Hz=(101): X on site 1 is a weight-one logical, while the
    # quotient-basis search initially reports the valid weight-two witness 101.
    raw = {'schema': 'qeclang.program.v1', 'nodes': [
        {'id': 'code', 'op': 'css.literal', 'inputs': [], 'params': {'code': {
            'schema': 'qeclang.code.v1', 'kind': 'css', 'n': 3,
            'hx': [[0, 1, 2]], 'hz': [[0, 2]]}}},
        {'id': 'search', 'op': 'css.distance', 'inputs': ['code'],
         'params': {'sector': 'x', 'max_weight': 1, 'max_candidates': 0}},
        {'id': 'upper', 'op': 'css.witness', 'inputs': ['code', 'search'], 'params': {}},
    ], 'outputs': ['search', 'upper']}
    program = Program.from_dict(raw)
    code = from_dict(raw['nodes'][0]['params']['code'])
    true_upper = certificate_record(code, LogicalWitness('x', 2, 3, 1))
    assert verify_certificate_record(code, true_upper)
    assert true_upper['claim']['distance_upper_bound'] == 1
    record = execute(program)
    assert record['status'] == 'completed'
    saved = artifact(record, 'search')
    assert saved['payload']['status'] == 'budget_exhausted'
    assert saved['payload']['lower_bound'] == 1 and saved['payload']['upper_bound'] == 2
    forged = deepcopy(saved)
    forged['payload'].update(status='exact', lower_bound=2, searched_through=1)
    replace_artifact(record, saved['digest'], forged)
    result = inspect_run(program, record)
    assert result['status'] == 'checked', result
    assert len(result['claims']) == 1 and result['claims'][0]['bound'] == 2
    assert result['reported_distance_nodes'] == ['search']
    assert_only_upper_claims(result)
    assert 'lower' in json.dumps(result['unverified']).lower()


def test_distance_report_alone_has_no_scientific_bound_claim():
    data = json.loads(EXAMPLE.read_text())
    data.update(nodes=data['nodes'][:4], outputs=['distance'])
    program = Program.from_dict(data)
    record = execute(program)
    result = inspect_run(program, record)
    assert result['status'] == 'checked' and result['claims'] == []
    assert result['reported_distance_nodes'] == ['distance']


def test_equal_n_k_whole_run_substitution_is_rejected_against_program_literal():
    program, record = authentic()
    alternate = program.to_dict()
    alternate['nodes'][0]['params']['code'] = {
        'schema': 'qeclang.code.v1', 'kind': 'css', 'n': 7,
        'hx': [[0], [1], [2]], 'hz': [[3], [4], [5]]}
    original_code = from_dict(artifact(record, 'original')['payload'])
    other_code = from_dict(alternate['nodes'][0]['params']['code'])
    assert (original_code.n, original_code.k) == (other_code.n, other_code.k) == (7, 1)
    substituted = execute(Program.from_dict(alternate), verify_relations=False)
    assert substituted['status'] == 'completed'
    substituted['program'] = program.fingerprint()
    assert_refused(inspect_run(program, substituted))


def test_rehashed_valid_but_wrong_target_is_not_accepted_on_matching_parameters():
    program, record = authentic()
    original = artifact(record, 'original')
    target = artifact(record, 'relabeled')
    assert original['digest'] != target['digest']
    # All uses are updated, including outputs. Digest integrity is now valid,
    # but the declared permutation still does not map to this valid CSS target.
    replace_artifact(record, target['digest'], original)
    assert_refused(inspect_run(program, record))


@pytest.mark.parametrize('word', ['operator', 'dual'])
def test_other_valid_target_certificate_is_not_the_requested_transport(word):
    program, record = authentic()
    target = from_dict(artifact(record, 'final')['payload'])
    previous = artifact(record, 'transported')
    cert = previous['payload']['certificate']
    operator, dual = int(cert['operator'], 16), int(cert['dual'], 16)
    if word == 'operator':
        operator = next(operator ^ row for row in target.hx.rows
                        if (operator ^ row).bit_count() == cert['weight'] and row)
    else:
        dual ^= target.hz.rows[0]
    alternative = LogicalWitness(cert['sector'], operator, dual, operator.bit_count())
    replacement = certificate_record(target, alternative)
    assert verify_certificate_record(target, replacement)
    assert replacement != previous['payload']
    replace_artifact(record, previous['digest'], describe('witness', replacement))
    assert_refused(inspect_run(program, record))


def test_valid_noncanonical_separating_dual_can_be_checked_without_replaying_solver_choice():
    data = json.loads(EXAMPLE.read_text())
    data.update(nodes=data['nodes'][:5], outputs=['logical'])
    program = Program.from_dict(data)
    record = execute(program)
    code = from_dict(artifact(record, 'original')['payload'])
    previous = artifact(record, 'logical')
    payload = previous['payload']['certificate']
    alternate = LogicalWitness(payload['sector'], int(payload['operator'], 16),
                                int(payload['dual'], 16) ^ code.hz.rows[0], payload['weight'])
    replacement = certificate_record(code, alternate)
    assert verify_certificate_record(code, replacement)
    replace_artifact(record, previous['digest'], describe('witness', replacement))
    result = inspect_run(program, record)
    assert result['status'] == 'checked', result
    assert_only_upper_claims(result)


@pytest.mark.parametrize('attack', ['remove', 'add', 'wrong_kind', 'digest', 'input', 'output', 'trace_order'])
def test_artifact_accounting_types_digests_and_graph_bindings(attack):
    program, record = authentic()
    if attack == 'remove':
        del record['artifacts'][step(record, 'complex')['output']]
    elif attack == 'add':
        extra = describe('css', CSSCode(BinaryMatrix((1,), 1), BinaryMatrix((), 1)))
        assert extra['digest'] not in record['artifacts']
        record['artifacts'][extra['digest']] = extra
    elif attack == 'wrong_kind':
        old = artifact(record, 'original')
        replacement = describe('sparse_css', SparseCSSCode.from_packed(from_dict(old['payload'])))
        replace_artifact(record, old['digest'], replacement)
    elif attack == 'digest':
        artifact(record, 'logical')['payload']['certificate']['dual'] = '0x0'
    elif attack == 'input':
        step(record, 'transported')['inputs'][0] = step(record, 'final')['output']
    elif attack == 'output':
        record['outputs']['transported'] = step(record, 'logical')['output']
    else:
        record['trace'][0], record['trace'][1] = record['trace'][1], record['trace'][0]
    assert_refused(inspect_run(program, record))


def test_rehashed_zero_dual_fails_parity_even_with_reported_verified_assurance():
    program, record = authentic()
    old = artifact(record, 'logical')
    replacement = deepcopy(old)
    replacement['payload']['certificate']['dual'] = '0x0'
    replace_artifact(record, old['digest'], replacement)
    step(record, 'logical')['evidence']['assurance'] = 'independently_checked'
    assert_refused(inspect_run(program, record))


def test_wrong_program_and_malformed_records_fail_without_releasing_partial_claims():
    program, record = authentic()
    other_data = program.to_dict()
    other_data['outputs'] = ['distance']
    assert_refused(inspect_run(Program.from_dict(other_data), record))
    for bad in (None, [], {'schema': 'qeclang.run.v1'}, {**record, 'unknown': 0},
                {**record, 'status': 'unknown'}, {**record, 'runtime': float('nan')}):
        assert_refused(inspect_run(program, bad))


@pytest.mark.parametrize('limits', [InspectionLimits(max_relation_checks=0),
                                   InspectionLimits(max_relation_checks=5),
                                   InspectionLimits(max_certificates=0),
                                   InspectionLimits(max_certificates=1),
                                   InspectionLimits(record=JSONLimits(max_bytes=1))])
def test_inspection_limits_return_no_partially_verified_claims(limits):
    program, record = authentic()
    assert_refused(inspect_run(program, record, limits=limits), 'budget_exhausted')


def test_authentic_bounded_no_logicals_empty_and_partial_outcomes():
    program, bounded = authentic(limits=ExecutionLimits(max_candidates=0))
    assert artifact(bounded, 'distance')['payload']['status'] == 'budget_exhausted'
    assert inspect_run(program, bounded)['status'] == 'checked'
    partial = execute(program, limits=ExecutionLimits(max_validation_work=0))
    assert partial['status'] == 'budget_exhausted'
    assert_refused(inspect_run(program, partial), 'unsupported')
    data = {'schema': 'qeclang.program.v1', 'nodes': [
        {'id': 'code', 'op': 'css.literal', 'inputs': [], 'params': {'code': {
            'schema': 'qeclang.code.v1', 'kind': 'css', 'n': 1, 'hx': [[0]], 'hz': []}}},
        {'id': 'distance', 'op': 'css.distance', 'inputs': ['code'],
         'params': {'sector': 'x', 'max_weight': 1, 'max_candidates': 10}},
    ], 'outputs': ['distance']}
    state_program = Program.from_dict(data)
    state_run = execute(state_program)
    assert artifact(state_run, 'distance')['payload']['status'] == 'no_logicals'
    result = inspect_run(state_program, state_run)
    assert result['status'] == 'checked' and result['claims'] == []
    data['nodes'].append({'id': 'unavailable', 'op': 'css.witness',
                           'inputs': ['code', 'distance'], 'params': {}})
    unavailable_program = Program.from_dict(data)
    unavailable = execute(unavailable_program)
    assert unavailable['status'] == 'evidence_unavailable'
    assert_refused(inspect_run(unavailable_program, unavailable), 'unsupported')
    empty = Program.from_dict({'schema': 'qeclang.program.v1', 'nodes': [], 'outputs': []})
    assert inspect_run(empty, execute(empty))['status'] == 'checked'


def test_json_admission_distinguishes_budget_exhaustion_from_malformed_data():
    with pytest.raises(JSONLimitExceeded):
        loads_json('[0,0,0]', limits=JSONLimits(max_payload_items=3))
    with pytest.raises(JSONLimitExceeded):
        validate_json_data({'large': 256}, limits=JSONLimits(max_integer_bits=8))
    with pytest.raises(ValueError) as failure:
        loads_json('{"duplicate":1,"duplicate":2}')
    assert not isinstance(failure.value, JSONLimitExceeded)


@pytest.mark.parametrize('field,value', [('source', '0' * 64), ('sector', 'z')])
def test_distance_source_and_sector_remain_bound_even_when_reports_are_unverified(field, value):
    program, record = authentic()
    previous = artifact(record, 'distance')
    replacement = deepcopy(previous)
    replacement['payload'][field] = value
    replace_artifact(record, previous['digest'], replacement)
    assert_refused(inspect_run(program, record))


def test_forged_no_logicals_report_does_not_prove_zero_logical_dimension():
    data = json.loads(EXAMPLE.read_text())
    data.update(nodes=data['nodes'][:4], outputs=['distance'])
    program = Program.from_dict(data)
    record = execute(program)
    previous = artifact(record, 'distance')
    replacement = deepcopy(previous)
    replacement['payload'].update(status='no_logicals', lower_bound=None, upper_bound=None,
                                   witness=None, searched_through=0, candidates=0)
    replace_artifact(record, previous['digest'], replacement)
    result = inspect_run(program, record)
    assert result['status'] == 'checked', result
    assert result['claims'] == []
    assert result['reported_distance_nodes'] == ['distance']
