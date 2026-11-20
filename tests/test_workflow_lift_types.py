from hashlib import sha256
import json

import pytest

from qeclang.constructions import records
from qeclang.workflow.lift_types import (PARAMETER_BACKENDS, construction_limits,
                                          validate_lift_literal, validate_parameter_request)
from qeclang.workflow.program import Program, ProgramLimits, SCHEMA


def construction(*, orders=(3,), a_shape=(1, 2), b_shape=(2, 1)):
    def matrix(shape):
        m, n = shape
        return {'shape': [m, n],
                'entries': [[[[0] * len(orders)] for _ in range(n)] for _ in range(m)]}
    return {'schema': records.SCHEMA, 'kind': 'abelian_lifted_product',
            'coefficient_field': 'GF(2)', 'orders': list(orders), 'action': records.ACTION,
            'convention': records.CONVENTION, 'physical_ordering': records.ORDERING,
            'a': matrix(a_shape), 'b': matrix(b_shape)}


def data(record=None, *, backend='crt', max_steps=1000, record_operations=False):
    return {'schema': SCHEMA, 'nodes': [
        {'id': 'seed', 'op': 'lift.literal', 'inputs': [],
         'params': {'construction': construction() if record is None else record}},
        {'id': 'parameters', 'op': 'lift.parameters', 'inputs': ['seed'],
         'params': {'backend': backend, 'max_steps': max_steps, 'record_operations': record_operations}},
        {'id': 'code', 'op': 'lift.css', 'inputs': ['seed'], 'params': {}},
        {'id': 'chain', 'op': 'css.chain', 'inputs': ['code'], 'params': {}},
        {'id': 'roundtrip', 'op': 'chain.css', 'inputs': ['chain'], 'params': {}},
        {'id': 'distance', 'op': 'css.distance', 'inputs': ['roundtrip'],
         'params': {'sector': 'x', 'max_weight': 1, 'max_candidates': 10}},
        {'id': 'witness', 'op': 'css.witness', 'inputs': ['roundtrip', 'distance'], 'params': {}},
    ], 'outputs': ['parameters', 'roundtrip', 'witness']}


def test_lift_types_dimensions_and_existing_css_composition():
    program = Program.from_dict(data())
    assert [node.result_type for node in program.nodes] == [
        'lift', 'parameters', 'css', 'chain', 'css', 'distance', 'witness']
    assert all(node.n == 12 for node in program.nodes)
    assert program.nodes[1].code_id == 'seed'
    assert program.nodes[-1].code_id == 'roundtrip'
    assert program.nodes[1].sector is None and program.nodes[-1].sector == 'x'
    assert Program.loads(json.dumps(program.to_dict())).fingerprint() == program.fingerprint()
    metadata = validate_lift_literal(construction())
    assert metadata.a_shape == (1, 2) and metadata.b_shape == (2, 1)
    assert (metadata.group_size, metadata.n, metadata.x_checks, metadata.z_checks) == (3, 12, 3, 12)
    record = construction(orders=(2, 3), a_shape=(2, 3), b_shape=(1, 2))
    metadata = validate_lift_literal(record)
    assert metadata.orders == (2, 3)
    assert (metadata.group_size, metadata.n, metadata.x_checks, metadata.z_checks) == (6, 48, 24, 18)


def test_large_symbolic_length_parses_without_ring_building_or_expansion(monkeypatch):
    from qeclang.symbolic import GroupAlgebraElement, GroupAlgebraMatrix
    from qeclang.constructions.lifted import LiftedProduct
    from qeclang.algebra import BinaryMatrix
    def forbidden(*args, **kwargs):
        raise AssertionError('structured program admission must not instantiate or lower rings')
    monkeypatch.setattr(records, 'from_dict', forbidden)
    monkeypatch.setattr(GroupAlgebraElement, 'from_terms', forbidden)
    monkeypatch.setattr(GroupAlgebraMatrix, '__post_init__', forbidden)
    monkeypatch.setattr(GroupAlgebraMatrix, 'lower', forbidden)
    monkeypatch.setattr(GroupAlgebraMatrix, 'lower_sparse', forbidden)
    monkeypatch.setattr(LiftedProduct, '__post_init__', forbidden)
    monkeypatch.setattr(BinaryMatrix, 'rank', forbidden)
    record = construction(orders=(1_000_000,), a_shape=(1, 1), b_shape=(1, 1))
    program = Program.from_dict(data(record, backend='polynomial'),
                               limits=ProgramLimits(max_qubits=1, max_symbolic_sites=2_000_000))
    assert program.nodes[0].n == program.nodes[2].n == 2_000_000
    assert program.nodes[2].result_type == 'css'
    with pytest.raises(ValueError, match='max_sites'):
        Program.from_dict(data(record), limits=ProgramLimits(max_symbolic_sites=1_999_999))


def test_parameter_backends_are_explicit_but_capability_decision_is_deferred():
    even_multicyclic = construction(orders=(2, 4))
    for backend in PARAMETER_BACKENDS:
        program = Program.from_dict(data(even_multicyclic, backend=backend))
        assert program.nodes[1].params['backend'] == backend
    assert Program.from_dict(data(backend='polynomial', record_operations=True)).nodes[1].params['record_operations']
    assert Program.from_dict(data(max_steps=0)).nodes[1].params['max_steps'] == 0


@pytest.mark.parametrize('parameters', [
    {'backend': 'packed', 'max_steps': 1, 'record_operations': False},
    {'backend': 'automatic', 'max_steps': 1, 'record_operations': False},
    {'backend': 'crt', 'max_steps': -1, 'record_operations': False},
    {'backend': 'crt', 'max_steps': True, 'record_operations': False},
    {'backend': 'crt', 'max_steps': 1, 'record_operations': 1},
    {'backend': 'crt', 'max_steps': 1, 'record_operations': True},
    {'backend': 'abelian', 'max_steps': 1, 'record_operations': True},
    {'backend': 'mixed-abelian', 'max_steps': 1, 'record_operations': True},
    {'backend': 'crt', 'max_steps': 1},
    {'backend': 'crt', 'max_steps': 1, 'record_operations': False, 'fallback': 'packed'},
])
def test_parameter_request_shape_types_and_trace_restriction(parameters):
    with pytest.raises(ValueError):
        validate_parameter_request(parameters)
    raw = data()
    raw['nodes'][1]['params'] = parameters
    with pytest.raises(ValueError):
        Program.from_dict(raw)


@pytest.mark.parametrize('field,value', [
    ('schema', 'qeclang.construction.v2'), ('kind', 'noncommutative_lifted_product'),
    ('coefficient_field', 'GF(4)'), ('action', 'column-to-row'),
    ('convention', 'other'), ('physical_ordering', 'CC,VV'),
])
def test_literal_rejects_unsupported_mathematical_conventions(field, value):
    record = construction()
    record[field] = value
    with pytest.raises(ValueError, match='semantics'):
        Program.from_dict(data(record))


@pytest.mark.parametrize('terms', [[[0], [0]], [[1], [0]], [[3]], [[-1]], [[True]],
                                  [[]], [[0, 1]], [0], [[0.0]]])
def test_monomials_must_be_unique_ordered_reduced_and_correctly_typed(terms):
    record = construction()
    record['a']['entries'][0][0] = terms
    with pytest.raises(ValueError):
        Program.from_dict(data(record))


def test_record_normalization_produces_accepted_canonical_terms():
    record = construction()
    record['a']['entries'][0][0] = [[2], [0], [1]]
    with pytest.raises(ValueError, match='canonical'):
        Program.from_dict(data(record))
    normalized = records.to_dict(records.from_dict(record))
    assert normalized['a']['entries'][0][0] == [[0], [1], [2]]
    assert Program.from_dict(data(normalized)).nodes[0].params['construction'] == normalized


@pytest.mark.parametrize('change', [
    lambda r: r.update(orders=[]), lambda r: r.update(orders=[0]),
    lambda r: r.update(orders=[True]), lambda r: r.update(orders=[1.5]),
    lambda r: r['a'].update(shape=[-1, 2]), lambda r: r['a'].update(shape=[True, 2]),
    lambda r: r['a'].update(shape=[1]), lambda r: r['a'].update(entries=[]),
    lambda r: r['a']['entries'].__setitem__(0, []), lambda r: r['b'].update(extra=1),
    lambda r: r.update(unknown=0),
])
def test_orders_shapes_and_unknown_fields_are_rejected(change):
    record = construction()
    change(record)
    with pytest.raises(ValueError):
        Program.from_dict(data(record))


@pytest.mark.parametrize('limits', [
    ProgramLimits(max_seed_axis=1), ProgramLimits(max_seed_entries=3),
    ProgramLimits(max_seed_terms=3), ProgramLimits(max_group_factors=0),
    ProgramLimits(max_cyclic_order=2), ProgramLimits(max_group_size=2),
    ProgramLimits(max_symbolic_sites=11),
])
def test_seed_and_group_budgets_cover_both_seeds(limits):
    with pytest.raises(ValueError):
        Program.from_dict(data(), limits=limits)


def test_multi_factor_product_limit_and_zero_axis_shapes():
    with pytest.raises(ValueError, match='max_group_size'):
        Program.from_dict(data(construction(orders=(3, 5))), limits=ProgramLimits(max_group_size=14))
    with pytest.raises(ValueError):
        Program.from_dict(data(construction(orders=(1, 1, 1))), limits=ProgramLimits(max_group_factors=2))
    free_cc = construction(orders=(3,), a_shape=(1, 0), b_shape=(1, 0))
    metadata = validate_lift_literal(free_cc)
    assert (metadata.n, metadata.x_checks, metadata.z_checks) == (3, 0, 0)
    empty_code = construction(orders=(1,), a_shape=(0, 3), b_shape=(2, 0))
    metadata = validate_lift_literal(empty_code)
    assert (metadata.n, metadata.x_checks, metadata.z_checks) == (0, 0, 6)
    with pytest.raises(ValueError, match='max_sites'):
        Program.from_dict(data(empty_code), limits=ProgramLimits(max_symbolic_sites=5))
    assert Program.from_dict(data(empty_code), limits=ProgramLimits(max_symbolic_sites=6)).nodes[0].n == 0
    enormous_empty_axis = construction(a_shape=(0, 1_000_000), b_shape=(0, 0))
    with pytest.raises(ValueError):
        Program.from_dict(data(enormous_empty_axis))


def test_translated_core_caps_and_seed_limits_are_explicit():
    limits = ProgramLimits(max_seed_axis=4, max_seed_entries=12, max_seed_terms=17,
                           max_group_factors=3, max_cyclic_order=7, max_group_size=49,
                           max_symbolic_sites=1234)
    core = construction_limits(limits)
    assert core.max_rows == core.max_columns == 4
    assert core.max_entries == 12 and core.max_terms == 17
    assert core.max_factors == 3 and core.max_order == 7 and core.max_group_size == 49
    assert core.max_sites == 1234
    assert records.from_dict(construction(), limits=core).n == 12


def test_old_css_program_fingerprints_do_not_depend_on_new_admission_caps():
    old = {'schema': SCHEMA, 'nodes': [
        {'id': 'code', 'op': 'css.literal', 'inputs': [], 'params': {'code': {
            'schema': 'qeclang.code.v1', 'kind': 'css', 'n': 2, 'hx': [], 'hz': []}}}
    ], 'outputs': ['code']}
    restrictive = ProgramLimits(max_seed_axis=0, max_seed_entries=0, max_seed_terms=0,
                                max_group_factors=0, max_cyclic_order=0, max_group_size=0,
                                max_symbolic_sites=0)
    expected = sha256(json.dumps(old, sort_keys=True, separators=(',', ':'),
                                ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    assert Program.from_dict(old).fingerprint() == expected
    assert Program.from_dict(old, limits=restrictive).fingerprint() == expected
    expanded = ProgramLimits(max_symbolic_sites=10_000_000_000)
    assert Program.from_dict(data(), limits=expanded).fingerprint() == Program.from_dict(data()).fingerprint()


def test_new_types_do_not_coerce_parameters_or_lifts_into_css_values():
    raw = data()
    raw['nodes'][3]['inputs'] = ['seed']
    with pytest.raises(ValueError, match='input types'):
        Program.from_dict(raw)
    raw = data()
    raw['nodes'][2]['inputs'] = ['parameters']
    with pytest.raises(ValueError, match='input types'):
        Program.from_dict(raw)
    raw = data()
    raw['nodes'][2]['params'] = {'max_bits': 1}
    with pytest.raises(ValueError):
        Program.from_dict(raw)
    raw = data()
    raw['nodes'][0]['params']['code'] = raw['nodes'][0]['params'].pop('construction')
    with pytest.raises(ValueError):
        Program.from_dict(raw)


def test_lift_payload_snapshot_is_immutable():
    raw = data()
    program = Program.from_dict(raw)
    digest = program.fingerprint()
    raw['nodes'][0]['params']['construction']['orders'][0] = 5
    copied = program.nodes[0].params
    copied['construction']['a']['entries'][0][0].append([1])
    assert program.nodes[0].params['construction'] == construction()
    assert program.fingerprint() == digest


@pytest.mark.parametrize('field', ['max_seed_axis', 'max_seed_entries', 'max_seed_terms',
                                   'max_group_factors', 'max_cyclic_order', 'max_group_size',
                                   'max_symbolic_sites'])
def test_structured_limit_fields_reject_boolean_values(field):
    with pytest.raises(ValueError):
        ProgramLimits(**{field: True})
