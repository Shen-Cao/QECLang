from dataclasses import FrozenInstanceError
import io
import json
from pathlib import Path

import pytest

from qeclang.workflow.json_input import JSONLimits, load_json, loads_json, validate_json_data
from qeclang.workflow.program import Program, ProgramLimits, ProgramNode, SCHEMA


def literal(name='source', n=3):
    return {'id': name, 'op': 'css.literal', 'inputs': [],
            'params': {'code': {'schema': 'qeclang.code.v1', 'kind': 'css', 'n': n,
                                'hx': [[0, 1]] if n >= 2 else [], 'hz': []}}}


def node(name, op, inputs, params=None):
    return {'id': name, 'op': op, 'inputs': inputs, 'params': {} if params is None else params}


def program_data():
    return {'schema': SCHEMA, 'nodes': [
        literal(),
        node('chain', 'css.chain', ['source']),
        node('roundtrip', 'chain.css', ['chain']),
        node('sparse', 'css.sparse', ['roundtrip']),
        node('packed', 'sparse.css', ['sparse']),
        node('permuted', 'css.permute', ['source'], {'permutation': [2, 0, 1]}),
        node('distance', 'css.distance', ['source'], {'sector': 'x', 'max_weight': 3, 'max_candidates': 100}),
        node('witness', 'css.witness', ['source', 'distance']),
        node('transported', 'css.transport', ['source', 'permuted', 'witness'], {'permutation': [2, 0, 1]}),
    ], 'outputs': ['packed', 'distance', 'transported']}


def test_all_operations_static_metadata_and_roundtrip(tmp_path):
    data = program_data()
    program = Program.from_dict(data)
    assert [n.result_type for n in program.nodes] == ['css', 'chain', 'css', 'sparse_css',
                                                    'css', 'css', 'distance', 'witness', 'witness']
    assert all(n.n == 3 for n in program.nodes)
    assert program.nodes[4].depth == 5
    assert program.nodes[6].code_id == program.nodes[7].code_id == 'source'
    assert program.nodes[8].code_id == 'permuted' and program.nodes[8].sector == 'x'
    assert program.outputs == ('packed', 'distance', 'transported')
    assert program.to_dict() == data
    loaded = Program.loads(json.dumps(data, indent=2).encode())
    assert loaded == program and loaded.fingerprint() == program.fingerprint()
    path = tmp_path/'program.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    assert Program.load(path) == program
    assert len(program.fingerprint()) == 64


def test_immutable_snapshot_and_fresh_params():
    data = program_data()
    program = Program.from_dict(data)
    fingerprint = program.fingerprint()
    data['nodes'][0]['params']['code']['n'] = 99
    data['outputs'].clear()
    params = program.nodes[0].params
    params['code']['hx'][0].clear()
    output = program.to_dict()
    output['nodes'].clear()
    assert program.nodes[0].params['code']['n'] == 3
    assert program.nodes[0].params['code']['hx'] == [[0, 1]]
    assert program.fingerprint() == fingerprint
    for obj, attr, value in ((program, 'outputs', ()), (program.nodes[0], 'id', 'changed'),
                              (program.limits, 'max_nodes', 0)):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, attr, value)
    with pytest.raises(TypeError, match='validated Program'):
        Program()
    with pytest.raises(TypeError, match='Program.from_dict'):
        ProgramNode()


def test_parse_has_no_core_deserialization_or_mathematical_execution(monkeypatch):
    import qeclang.io as code_io
    from qeclang.codes import CSSCode
    def forbidden(*args, **kwargs):
        raise AssertionError('program admission must not execute core code construction')
    monkeypatch.setattr(code_io, 'from_dict', forbidden)
    monkeypatch.setattr(CSSCode, '__post_init__', forbidden)
    data = {'schema': SCHEMA, 'nodes': [literal(n=1)], 'outputs': ['source']}
    data['nodes'][0]['params']['code'].update(hx=[[0]], hz=[[0]])
    # This literal is structurally well typed but noncommuting. Runtime admission
    # must reject it; parsing intentionally does not execute the commutation test.
    assert Program.from_dict(data).nodes[0].n == 1


@pytest.mark.parametrize('change', [
    lambda d: d.update(schema='qeclang.program.v2'),
    lambda d: d.update(unknown=1),
    lambda d: d['nodes'][0].update(op='python.eval'),
    lambda d: d['nodes'][0].update(id='not a name'),
    lambda d: d['nodes'][0].update(extra=1),
    lambda d: d['nodes'][1].update(inputs=['chain']),
    lambda d: d['nodes'][1].update(inputs=['packed']),
    lambda d: d['nodes'][1].update(inputs=['missing']),
    lambda d: d['nodes'][1].update(id='source'),
    lambda d: d['nodes'][2].update(inputs=['source']),
    lambda d: d['nodes'][1].update(params={'unexpected': True}),
    lambda d: d.update(outputs=['packed', 'packed']),
    lambda d: d.update(outputs=['missing']),
])
def test_schema_unknown_operations_references_and_types(change):
    data = program_data()
    change(data)
    with pytest.raises(ValueError):
        Program.from_dict(data)


@pytest.mark.parametrize('change', [
    lambda c: c.update(kind='stabilizer'),
    lambda c: c.update(n=True),
    lambda c: c.update(n=-1),
    lambda c: c.update(hx=[[0, 0]]),
    lambda c: c.update(hx=[[3]]),
    lambda c: c.update(hx=[[False]]),
    lambda c: c.update(hz=None),
    lambda c: c.update(name=4),
    lambda c: c.update(generators=[]),
])
def test_literal_record_structure_is_checked_without_code_construction(change):
    data = program_data()
    change(data['nodes'][0]['params']['code'])
    with pytest.raises(ValueError):
        Program.from_dict(data)


def test_nominal_evidence_binding_requires_explicit_transport():
    data = program_data()
    data['nodes'][7]['inputs'][0] = 'roundtrip'
    with pytest.raises(ValueError, match='different CSS node'):
        Program.from_dict(data)
    data = program_data()
    data['nodes'][8]['inputs'][0] = 'roundtrip'
    with pytest.raises(ValueError, match='bound to its source'):
        Program.from_dict(data)
    data = program_data()
    data['nodes'].append(node('again', 'css.transport', ['permuted', 'roundtrip', 'transported'],
                              {'permutation': [1, 2, 0]}))
    assert Program.from_dict(data).nodes[-1].code_id == 'roundtrip'
    data = program_data()
    data['nodes'].insert(1, literal('larger', n=4))
    data['nodes'][-1]['inputs'][1] = 'larger'
    with pytest.raises(ValueError, match='equal dimensions'):
        Program.from_dict(data)


@pytest.mark.parametrize('params', [{'permutation': [0, 1]}, {'permutation': [0, 0, 2]},
                                     {'permutation': [0, 1, True]}, {'permutation': [0, 1, 3]}])
def test_permutation_bijection(params):
    data = program_data()
    data['nodes'][5]['params'] = params
    with pytest.raises(ValueError, match='permutation'):
        Program.from_dict(data)


@pytest.mark.parametrize('params', [{'sector': 'y', 'max_weight': 3, 'max_candidates': 100},
                                    {'sector': 'x', 'max_weight': None, 'max_candidates': 100},
                                    {'sector': 'x', 'max_weight': 3, 'max_candidates': True},
                                    {'sector': 'x', 'max_weight': 3},
                                    {'sector': 'x', 'max_weight': -1, 'max_candidates': 100}])
def test_distance_parameters_are_explicit_and_typed(params):
    data = program_data()
    data['nodes'][6]['params'] = params
    with pytest.raises(ValueError):
        Program.from_dict(data)


@pytest.mark.parametrize('limits', [ProgramLimits(max_nodes=8), ProgramLimits(max_depth=4),
    ProgramLimits(max_inputs=2), ProgramLimits(max_outputs=2), ProgramLimits(max_edges=1),
    ProgramLimits(max_fanout=1), ProgramLimits(max_id_length=3), ProgramLimits(max_qubits=2),
    ProgramLimits(max_checks=0), ProgramLimits(max_integer_bits=1), ProgramLimits(max_bytes=20),
    ProgramLimits(max_string_length=5), ProgramLimits(max_payload_items=10),
    ProgramLimits(max_payload_depth=4)])
def test_resource_limits(limits):
    with pytest.raises(ValueError):
        Program.from_dict(program_data(), limits=limits)


def test_empty_program_and_payload_item_boundary():
    data = {'schema': SCHEMA, 'nodes': [], 'outputs': []}
    limits = ProgramLimits(max_nodes=0, max_depth=0, max_outputs=0, max_edges=0,
                            max_fanout=0, max_inputs=0, max_qubits=0, max_checks=0,
                            max_payload_items=7, max_payload_depth=2)
    program = Program.from_dict(data, limits=limits)
    assert program.nodes == program.outputs == ()
    with pytest.raises(ValueError, match='max_payload_items'):
        Program.from_dict(data, limits=ProgramLimits(max_payload_items=6))
    with pytest.raises(ValueError, match='max_payload_items'):
        Program.loads(json.dumps(data), limits=ProgramLimits(max_payload_items=6))


@pytest.mark.parametrize('text', [
    '{"schema":"qeclang.program.v1","schema":"qeclang.program.v1","nodes":[],"outputs":[]}',
    '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e9999}',
    '{"x":1,"x":2}', '{"x":{"a":1,"a":2}}', '{} {}', '{"x":1,}',
])
def test_strict_json_rejects_duplicate_nonfinite_and_invalid_data(text):
    with pytest.raises(ValueError):
        Program.loads(text)


def test_generic_json_admission_is_reusable_and_preflights_before_parser(monkeypatch):
    assert loads_json('{"value":0.25}', limits=JSONLimits()) == {'value': .25}
    import qeclang.workflow.json_input as admission
    def forbidden(*args, **kwargs):
        raise AssertionError('JSON parser reached before lexical admission checks')
    monkeypatch.setattr(admission.json, 'loads', forbidden)
    with pytest.raises(ValueError, match='max_payload_depth'):
        loads_json('[' * 20 + '0' + ']' * 20, limits=JSONLimits(max_payload_depth=4))
    with pytest.raises(ValueError, match='max_payload_items'):
        loads_json('[0,0,0,0]', limits=JSONLimits(max_payload_items=4))
    with pytest.raises(ValueError, match='max_bytes'):
        loads_json(' ' * 50, limits=JSONLimits(max_bytes=10))


def test_direct_payload_rejects_cycles_foreign_values_and_nonfinite_scalars():
    cyclic = []
    cyclic.append(cyclic)
    for data in (cyclic, {'a': object()}, {'a': (1, 2)}, {1: 'key'}, {'a': float('inf')},
                 {'a': '\ud800'}, {'a': 1 << 65}):
        with pytest.raises(ValueError):
            validate_json_data(data)
    with pytest.raises(ValueError, match='max_integer_bits'):
        loads_json('{"a":' + '9' * 1000 + '}')
    with pytest.raises(ValueError):
        Program.loads(b'\xff')


def test_quoted_brackets_and_canonical_unicode_budget():
    data = program_data()
    data['nodes'][0]['params']['code']['name'] = '[{escaped " quotation \\}] λ'
    assert Program.loads(json.dumps(data)).to_dict() == data
    with pytest.raises(ValueError, match='max_bytes'):
        validate_json_data({'a': 'λ'}, limits=JSONLimits(max_bytes=10))
    assert Program.loads(json.dumps(data, ensure_ascii=False)).fingerprint() == Program.from_dict(data).fingerprint()


def test_file_read_is_bounded_before_parsing(monkeypatch):
    requested = []
    class Reader(io.BytesIO):
        def read(self, size=-1):
            requested.append(size)
            return super().read(size)
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: Reader(b'x' * 100))
    with pytest.raises(ValueError, match='max_bytes'):
        load_json('unused', limits=JSONLimits(max_bytes=8))
    assert requested == [9]


@pytest.mark.parametrize('options', [{'max_nodes': True}, {'max_bytes': -1}, {'max_depth': 1.2},
                                     {'max_payload_depth': 129}, {'max_integer_bits': 4097}])
def test_limit_validation(options):
    with pytest.raises(ValueError):
        ProgramLimits(**options)
