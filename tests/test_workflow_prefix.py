"""Opt-in scientific inspection of retained completed prefixes and stop markers."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from qeclang.algebra import BinaryMatrix
from qeclang.codes import CSSCode
from qeclang.constructions import records
from qeclang.workflow import ExecutionLimits, InspectionLimits, Program, execute, inspect_run
from qeclang.workflow.artifacts import ARTIFACT_SCHEMA, describe

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/programs/transport.json'


def css_prefix():
    program = Program.load(EXAMPLE)
    run = execute(program, limits=ExecutionLimits(max_transports=0), verify_relations=False)
    assert run['status'] == 'budget_exhausted' and run['trace'][-1]['node'] == 'transported'
    return program, run


def lift_program(*, backend='polynomial', trace=True):
    source = {'schema': records.SCHEMA, 'kind': 'abelian_lifted_product',
              'coefficient_field': 'GF(2)', 'orders': [7], 'action': records.ACTION,
              'convention': records.CONVENTION, 'physical_ordering': records.ORDERING,
              'a': {'shape': [1, 1], 'entries': [[[[0], [1]]]]},
              'b': {'shape': [1, 1], 'entries': [[[[0], [2]]]]}}
    return Program.from_dict({'schema': 'qeclang.program.v1', 'nodes': [
        {'id': 'source', 'op': 'lift.literal', 'inputs': [], 'params': {'construction': source}},
        {'id': 'rank', 'op': 'lift.parameters', 'inputs': ['source'],
         'params': {'backend': backend, 'max_steps': 100000, 'record_operations': trace}},
        {'id': 'physical', 'op': 'lift.css', 'inputs': ['source'], 'params': {}},
        {'id': 'later_rank', 'op': 'lift.parameters', 'inputs': ['source'],
         'params': {'backend': 'crt', 'max_steps': 100000, 'record_operations': False}},
    ], 'outputs': ['rank', 'physical', 'later_rank']})


def replace_artifact(run, old, replacement):
    changed = deepcopy(replacement)
    payload = {'schema': ARTIFACT_SCHEMA, 'kind': changed['kind'], 'payload': changed['payload']}
    new = sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    changed['digest'] = new
    del run['artifacts'][old]
    run['artifacts'][new] = changed
    for step in run['trace']:
        step['inputs'] = [new if ref == old else ref for ref in step['inputs']]
        if step.get('output') == old:
            step['output'] = new
    run['outputs'] = {name: new if ref == old else ref for name, ref in run['outputs'].items()}


def refused(result, status='invalid_record'):
    assert result['status'] == status, result
    assert result['claims'] == result['relations'] == []
    assert 'checked_prefix_nodes' not in result and 'unverified_stop' not in result
    assert result['reason']


def test_prefix_scope_preserves_upper_witness_and_all_completed_relations():
    program, run = css_prefix()
    refused(inspect_run(program, run), 'unsupported')
    review = inspect_run(program, run, allow_prefix=True)
    assert review['status'] == 'checked_prefix', review
    assert review['checked_prefix_nodes'] == [node.id for node in program.nodes[:-1]]
    assert review['unverified_stop'] == {'node': 'transported', 'status': 'budget_exhausted'}
    assert {claim['node'] for claim in review['claims']} == {'logical'}
    assert review['claims'][0]['predicate'] == 'distance_upper_bound'
    assert review['claims'][0]['bound'] == 3
    assert {relation['node'] for relation in review['relations']} == {
        'complex', 'original', 'relabeled', 'sparse', 'final'}
    assert review['reported_distance_nodes'] == ['distance']
    assert {'stop_cause', 'stop_spend'} <= set(review['unverified'])
    assert set(run['outputs']) == {'distance', 'final'}


def test_completed_results_are_unchanged_by_prefix_opt_in():
    program = Program.load(EXAMPLE)
    run = execute(program)
    ordinary = inspect_run(program, run)
    permitted = inspect_run(program, run, allow_prefix=True)
    assert ordinary == permitted
    assert ordinary['status'] == 'checked'
    assert 'checked_prefix_nodes' not in ordinary and 'unverified_stop' not in ordinary


def test_traced_dimension_is_verified_before_refused_materialization(monkeypatch):
    program = lift_program()
    run = execute(program, limits=ExecutionLimits(max_matrix_bits=0))
    assert run['trace'][-1]['node'] == 'physical'
    import qeclang.workflow.executor as executor
    import qeclang.workflow.operations as operations
    import qeclang.workflow.lift_operations as lift_operations
    import qeclang.analysis.structured as structured
    import qeclang.symbolic.repeated_root as repeated
    from qeclang.constructions.lifted import LiftedProduct
    from qeclang.symbolic import GroupAlgebraMatrix
    def forbidden(*args, **kwargs):
        raise AssertionError('prefix inspection must not execute producers or materialize binary checks')
    monkeypatch.setattr(executor, 'execute', forbidden)
    monkeypatch.setattr(executor, 'replay', forbidden)
    monkeypatch.setattr(operations, 'evaluate', forbidden)
    monkeypatch.setattr(lift_operations, 'evaluate_lift', forbidden)
    monkeypatch.setattr(structured, 'analyze_lifted', forbidden)
    monkeypatch.setattr(repeated, 'cyclic_rank_general', forbidden)
    monkeypatch.setattr(LiftedProduct, 'lower', forbidden)
    monkeypatch.setattr(GroupAlgebraMatrix, 'lower', forbidden)
    for method in ('rank', 'solve', 'row_basis', 'nullspace'):
        monkeypatch.setattr(BinaryMatrix, method, forbidden)
    review = inspect_run(program, run, allow_prefix=True)
    assert review['status'] == 'checked_prefix', review
    assert review['checked_prefix_nodes'] == ['source', 'rank']
    assert review['unverified_stop'] == {'node': 'physical', 'status': 'budget_exhausted'}
    claim, = review['claims']
    assert (claim['node'], claim['predicate'], claim['value']) == ('rank', 'logical_dimension', 2)
    assert claim['rank_x'] == claim['rank_z'] == 6
    assert review['unverified_parameter_nodes'] == []
    assert set(run['outputs']) == {'rank'}


def test_only_completed_unverified_parameter_nodes_are_listed():
    program = lift_program(backend='crt', trace=False)
    run = execute(program, limits=ExecutionLimits(max_matrix_bits=0))
    review = inspect_run(program, run, allow_prefix=True)
    assert review['status'] == 'checked_prefix'
    assert review['claims'] == [] and review['unverified_parameter_nodes'] == ['rank']
    run = execute(program, limits=ExecutionLimits(max_blocks=0))
    assert run['trace'][-1]['node'] == 'rank'
    review = inspect_run(program, run, allow_prefix=True)
    assert review['status'] == 'checked_prefix' and review['checked_prefix_nodes'] == ['source']
    assert review['unverified_parameter_nodes'] == []


def test_all_recognized_stop_modes_and_empty_completed_prefix(monkeypatch):
    program = Program.load(EXAMPLE)
    first = execute(program, limits=ExecutionLimits(max_matrix_bits=0))
    review = inspect_run(program, first, allow_prefix=True)
    assert review['status'] == 'checked_prefix' and review['checked_prefix_nodes'] == []
    assert review['claims'] == review['relations'] == []
    bad = program.to_dict()
    bad['nodes'][0]['params']['code']['hz'][0] = [0]
    invalid_program = Program.from_dict(bad)
    invalid = execute(invalid_program)
    assert invalid['status'] == 'invalid_input'
    assert inspect_run(invalid_program, invalid, allow_prefix=True)['status'] == 'checked_prefix'
    import qeclang.workflow.operations as operations
    with monkeypatch.context() as patch:
        patch.setattr(operations, 'permute_qubits', lambda code, permutation: code)
        failed = execute(program)
    assert failed['status'] == 'verification_failed' and failed['trace'][-1]['node'] == 'relabeled'
    review = inspect_run(program, failed, allow_prefix=True)
    assert review['status'] == 'checked_prefix'
    assert {claim['node'] for claim in review['claims']} == {'logical'}
    state = {'schema': 'qeclang.program.v1', 'nodes': [
        {'id': 'code', 'op': 'css.literal', 'inputs': [], 'params': {'code': {
            'schema': 'qeclang.code.v1', 'kind': 'css', 'n': 1, 'hx': [[0]], 'hz': []}}},
        {'id': 'distance', 'op': 'css.distance', 'inputs': ['code'],
         'params': {'sector': 'x', 'max_weight': 1, 'max_candidates': 10}},
        {'id': 'witness', 'op': 'css.witness', 'inputs': ['code', 'distance'], 'params': {}},
    ], 'outputs': ['distance', 'witness']}
    state_program = Program.from_dict(state)
    unavailable = execute(state_program)
    assert unavailable['status'] == 'evidence_unavailable'
    review = inspect_run(state_program, unavailable, allow_prefix=True)
    assert review['status'] == 'checked_prefix' and review['claims'] == []
    assert review['checked_prefix_nodes'] == ['code', 'distance']
    assert review['reported_distance_nodes'] == ['distance']


def test_marker_cause_and_cost_are_unverified_not_reproduced():
    program, run = css_prefix()
    run['runtime'] = {'different_version': 'future'}
    run['ledger'] = {'historical_data': 'unverified'}
    stop = run['trace'][-1]
    stop.update(reason='A different producer reported this stop', resource='future_capacity',
                requested=0, available=999999, spent={'other_counter': 123})
    review = inspect_run(program, run, allow_prefix=True)
    assert review['status'] == 'checked_prefix'
    assert review['unverified_stop'] == {'node': 'transported', 'status': 'budget_exhausted'}
    assert review['claims'][0]['bound'] == 3
    stop.pop('requested')
    stop.pop('available')
    stop['resource'] = 'artifact_admission'
    assert inspect_run(program, run, allow_prefix=True)['status'] == 'checked_prefix'


@pytest.mark.parametrize('mutation', [
    lambda s: s.update(node='different'), lambda s: s.update(op='css.distance'),
    lambda s: s.update(inputs=[]), lambda s: s.update(status='invalid_input'),
    lambda s: s.update(status='completed'), lambda s: s.pop('reason'),
    lambda s: s.update(reason=None), lambda s: s.pop('spent'), lambda s: s.update(spent=[]),
    lambda s: s.update(spent={'bad': -1}), lambda s: s.update(spent={'bad': True}),
    lambda s: s.pop('resource'), lambda s: s.update(resource=''), lambda s: s.update(resource=1),
    lambda s: s.pop('requested'), lambda s: s.pop('available'),
    lambda s: s.update(requested=True), lambda s: s.update(available=-1),
    lambda s: s.update(available=1.5), lambda s: s.update(resource='artifact_admission'),
    lambda s: s.update(output='0'*64), lambda s: s.update(evidence={'assurance': 'checked'}),
    lambda s: s.update(extra='unknown'),
])
def test_malformed_terminal_marker_releases_no_prefix_claims(mutation):
    program, run = css_prefix()
    mutation(run['trace'][-1])
    refused(inspect_run(program, run, allow_prefix=True))


def test_nonbudget_marker_cannot_carry_budget_fields():
    program, run = css_prefix()
    run['status'] = run['trace'][-1]['status'] = 'verification_failed'
    refused(inspect_run(program, run, allow_prefix=True))
    for key in ('resource', 'requested', 'available'):
        run['trace'][-1].pop(key)
    assert inspect_run(program, run, allow_prefix=True)['status'] == 'checked_prefix'


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(trace=[]), lambda r: r['trace'].pop(),
    lambda r: r['trace'].append(deepcopy(r['trace'][-1])),
    lambda r: r['trace'].__setitem__(1, deepcopy(r['trace'][-1])),
    lambda r: r['trace'].reverse(), lambda r: r['trace'][0].update(spent={'bad': True}),
    lambda r: r.update(status='completed'), lambda r: r.update(status='cancelled'),
])
def test_nonprefix_traces_and_extra_or_missing_terminal_nodes(mutation):
    program, run = css_prefix()
    mutation(run)
    refused(inspect_run(program, run, allow_prefix=True))


def test_partial_outputs_are_exactly_the_available_requested_node_ids():
    program, original = css_prefix()
    run = deepcopy(original)
    del run['outputs']['distance']
    refused(inspect_run(program, run, allow_prefix=True))
    run = deepcopy(original)
    run['outputs']['base'] = run['trace'][0]['output']
    refused(inspect_run(program, run, allow_prefix=True))
    run = deepcopy(original)
    run['outputs']['transported'] = next(s['output'] for s in run['trace'] if s['node'] == 'logical')
    refused(inspect_run(program, run, allow_prefix=True))


def test_failed_or_suffix_artifacts_and_missing_internal_evidence_are_rejected():
    program, original = css_prefix()
    full = execute(program)
    run = deepcopy(original)
    later = full['artifacts'][full['outputs']['transported']]
    assert later['digest'] not in run['artifacts']
    run['artifacts'][later['digest']] = later
    refused(inspect_run(program, run, allow_prefix=True))
    run = deepcopy(original)
    del run['artifacts'][next(s['output'] for s in run['trace'] if s['node'] == 'complex')]
    refused(inspect_run(program, run, allow_prefix=True))
    run = deepcopy(original)
    extra = describe('css', CSSCode(BinaryMatrix((1,), 1), BinaryMatrix((), 1)))
    run['artifacts'][extra['digest']] = extra
    refused(inspect_run(program, run, allow_prefix=True))


def test_every_retained_prefix_relation_and_certificate_is_checked_after_rehashing():
    program, original = css_prefix()
    run = deepcopy(original)
    source = next(s['output'] for s in run['trace'] if s['node'] == 'original')
    target = next(s['output'] for s in run['trace'] if s['node'] == 'relabeled')
    replace_artifact(run, target, run['artifacts'][source])
    refused(inspect_run(program, run, allow_prefix=True))
    run = deepcopy(original)
    key = next(s['output'] for s in run['trace'] if s['node'] == 'logical')
    bad = deepcopy(run['artifacts'][key])
    bad['payload']['certificate']['dual'] = '0x0'
    replace_artifact(run, key, bad)
    refused(inspect_run(program, run, allow_prefix=True))


@pytest.mark.parametrize('limits', [InspectionLimits(max_relation_checks=4), InspectionLimits(max_certificates=0)])
def test_reader_refusal_does_not_publish_partially_verified_prefix(limits):
    program, run = css_prefix()
    refused(inspect_run(program, run, allow_prefix=True, limits=limits), 'budget_exhausted')
    assert inspect_run(program, run, allow_prefix=True, limits=limits)['reader_usage']['artifact_bytes'] > 0


def test_traced_prefix_claim_is_withheld_when_rank_reader_budget_exhausts():
    program = lift_program()
    run = execute(program, limits=ExecutionLimits(max_matrix_bits=0))
    refused(inspect_run(program, run, allow_prefix=True, limits=InspectionLimits(max_rank_steps=0)),
            'budget_exhausted')


@pytest.mark.parametrize('option', [1, None, 'yes'])
def test_prefix_policy_requires_explicit_boolean(option):
    program, run = css_prefix()
    with pytest.raises(TypeError, match='allow_prefix'):
        inspect_run(program, run, allow_prefix=option)


def test_stopped_empty_program_cannot_invent_a_terminal_node():
    program = Program.from_dict({'schema': 'qeclang.program.v1', 'nodes': [], 'outputs': []})
    run = execute(program)
    assert inspect_run(program, run, allow_prefix=True)['status'] == 'checked'
    run['status'] = 'invalid_input'
    refused(inspect_run(program, run, allow_prefix=True))


def test_rehashed_rank_evidence_in_prefix_cannot_hide_behind_later_refusal():
    program = lift_program()
    run = execute(program, limits=ExecutionLimits(max_matrix_bits=0))
    key = run['outputs']['rank']
    changed = deepcopy(run['artifacts'][key])
    diagonal = changed['payload']['x_evidence']['diagonal']
    index = next(i for i, value in enumerate(diagonal) if value)
    diagonal[index] ^= (1 << 7) | 1
    replace_artifact(run, key, changed)
    refused(inspect_run(program, run, allow_prefix=True))
