from itertools import product

import pytest

from qeclang.constructions import LiftedProduct
from qeclang.constructions.records import to_dict
from qeclang.symbolic import GroupAlgebraElement as G, GroupAlgebraMatrix as M
from qeclang.workflow import Program, ExecutionLimits, execute, inspect_run, InspectionLimits
from qeclang.workflow.lift_relations import check_lift
from qeclang.workflow.relations import RelationError


def construction(orders):
    def terms(*ts):
        return G.from_terms(orders, ts)
    identity = terms((0,)*len(orders))
    shift = terms((1,)*len(orders))
    zero = terms()
    a = M(((identity+shift, zero, identity), (shift, identity, zero)), 3, orders)
    b = M(((identity, shift),), 2, orders)
    return LiftedProduct(a, b)


def program(lift, backend, *, trace=False, expand=True, repeat=False):
    nodes = [dict(id='source', op='lift.literal', inputs=[], params={'construction': to_dict(lift)}),
             dict(id='rank', op='lift.parameters', inputs=['source'],
                  params=dict(backend=backend, max_steps=1_000_000, record_operations=trace))]
    if repeat:
        nodes.append({**nodes[1], 'id':'rank2'})
    if expand:
        nodes += [dict(id='physical', op='lift.css', inputs=['source'], params={}),
                  dict(id='chain', op='css.chain', inputs=['physical'], params={})]
    return Program.from_dict(dict(schema='qeclang.program.v1', nodes=nodes,
                                  outputs=[node['id'] for node in nodes[1:]]))


def payload(result, name):
    return result['artifacts'][result['outputs'][name]]['payload']


@pytest.mark.parametrize('orders,backend', [((5,), 'crt'), ((4,), 'polynomial'),
                                              ((3,3), 'abelian'), ((2,3), 'mixed-abelian')])
def test_symbolic_dimension_vs_packed_and_independent_rank_scope(orders, backend):
    lift = construction(orders)
    expected = lift.lower()
    p = program(lift, backend, trace=backend == 'polynomial')
    record = execute(p)
    assert record['status'] == 'completed', record['trace'][-1]
    # The parameters artifact schema includes all StructuredParameters fields.
    data = payload(record, 'rank')
    assert data['status'] == 'exact' and data['k'] == expected.k
    assert (data['rank_x'], data['rank_z']) == (expected.hx.rank(), expected.hz.rank())
    assert record['ledger']['used']['rank_steps'] == data['steps'] > 0
    assert record['trace'][2]['evidence']['assurance'] == 'relation_checked'
    review = inspect_run(p, record)
    assert review['status'] == 'checked', review
    if backend == 'polynomial':
        claim, = review['claims']
        assert claim['predicate'] == 'logical_dimension' and claim['value'] == expected.k
        assert claim['verification_steps'] > 0
        assert not review['unverified_parameter_nodes']
        scarce = inspect_run(p, record, limits=InspectionLimits(max_rank_steps=1))
        assert scarce['status'] == 'budget_exhausted' and not scarce['claims']
    else:
        assert not review['claims'] and review['unverified_parameter_nodes'] == ['rank']


def test_symbolic_program_never_unconditionally_expands(monkeypatch):
    lift = construction((31,))
    p = program(lift, 'polynomial', trace=True, expand=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('binary expansion is forbidden')
    monkeypatch.setattr(LiftedProduct, 'lower', forbidden)
    monkeypatch.setattr(M, 'lower', forbidden)
    record = execute(p, limits=ExecutionLimits(max_axis=0, max_matrix_bits=0))
    assert record['status'] == 'completed' and record['ledger']['peak_requested']['matrix_bits'] == 0
    assert inspect_run(p, record)['claims'][0]['value'] == payload(record, 'rank')['k']


def test_partial_charge_on_second_rank_and_expansion_refusal():
    lift = construction((7,))
    p = program(lift, 'crt', expand=False, repeat=True)
    steps = payload(execute(p), 'rank')['steps']
    record = execute(p, limits=ExecutionLimits(max_rank_steps=steps+15))
    first, later = payload(record, 'rank'), payload(record, 'rank2')
    assert first['status'] == 'exact' and later['status'] == 'budget_exhausted'
    assert 0 < later['steps'] <= 15 and later['k'] is None
    assert record['ledger']['used']['rank_steps'] == first['steps'] + later['steps']
    p = program(lift, 'crt', expand=True)
    record = execute(p, limits=ExecutionLimits(max_matrix_bits=1))
    assert record['status'] == 'budget_exhausted' and record['trace'][-1]['node'] == 'physical'
    assert payload(record, 'rank')['status'] == 'exact'


def test_direct_expansion_relation_rejects_equal_parameter_mutation(monkeypatch):
    from qeclang.transforms import permute_qubits
    lift = construction((3, 2))
    target = lift.lower()
    permutation = [1, 0, *range(2, target.n)]
    wrong = permute_qubits(target, permutation)
    assert wrong.k == target.k
    def forbidden(*args, **kwargs):
        raise AssertionError('independent relation must not call either lowerer')
    monkeypatch.setattr(LiftedProduct, 'lower', forbidden)
    monkeypatch.setattr(LiftedProduct, 'sparse_checks', forbidden)
    monkeypatch.setattr(LiftedProduct, 'symbolic_checks', forbidden)
    assert 'ordered_check_supports' in check_lift(lift, target)
    with pytest.raises(RelationError):
        check_lift(lift, wrong)


def test_zero_widths_and_refused_extra_checks():
    for m, n in product(range(2), repeat=2):
        lift = LiftedProduct(M.zeros(m,n,(3,)), M.zeros(n,m,(3,)))
        assert check_lift(lift, lift.lower())
    p = program(construction((2,)), 'crt', expand=True)
    record = execute(p, limits=ExecutionLimits(max_lift_comparison_units=0))
    assert record['status'] == 'completed'
    assert payload(record, 'rank')['status'] == 'unsupported'
    assert record['trace'][2]['evidence'] == {'assurance': 'budget_exhausted', 'resource': 'lift_comparison_units'}
    assert inspect_run(p, record)['status'] == 'checked'


def test_rehashed_rank_proof_cannot_change_certified_dimension():
    from copy import deepcopy
    from qeclang.workflow.artifacts import describe
    lift = construction((4,))
    p = program(lift, 'polynomial', trace=True, expand=False)
    record = execute(p)
    old = record['outputs']['rank']
    value = deepcopy(payload(record, 'rank'))
    value.pop('schema')
    # Replace a nonzero diagonal entry with another polynomial having the same
    # gcd with x^4+1. The ranks and all claimed totals can remain unchanged.
    diagonal = value['x_evidence']['diagonal']
    index = next(i for i, v in enumerate(diagonal) if v)
    diagonal[index] ^= 0b10001
    altered = describe('parameters', value)
    new = altered['digest']
    del record['artifacts'][old]
    record['artifacts'][new] = altered
    record['trace'][-1]['output'] = record['outputs']['rank'] = new
    verdict = inspect_run(p, record)
    assert verdict['status'] == 'invalid_record'
    assert not verdict['claims']


def test_unsupported_needs_no_symbolic_check_materialization():
    p = program(construction((2,)), 'crt', expand=False)
    result = execute(p, limits=ExecutionLimits(max_blocks=0, max_symbolic_terms=0))
    assert result['status'] == 'completed'
    assert payload(result, 'rank')['status'] == 'unsupported'
    assert result['ledger']['used']['rank_steps'] == 0


def test_requested_missing_trace_is_internal_failure(monkeypatch):
    from dataclasses import replace
    from qeclang.workflow import lift_operations
    producer = lift_operations.analyze_lifted
    def incomplete(*args, **kwargs):
        result = producer(*args, **kwargs)
        return replace(result, x_evidence=replace(result.x_evidence, operations=None))
    monkeypatch.setattr(lift_operations, 'analyze_lifted', incomplete)
    with pytest.raises(ArithmeticError, match='omitted'):
        execute(program(construction((4,)), 'polynomial', trace=True, expand=False))
