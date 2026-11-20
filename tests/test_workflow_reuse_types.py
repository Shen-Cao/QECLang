from hashlib import sha256
import json

import pytest

from qeclang.constructions import records
from qeclang.workflow.program import Program, ProgramLimits, SCHEMA


def css(name, hx, hz, n=4):
    return {'id': name, 'op': 'css.literal', 'inputs': [], 'params': {'code': {
        'schema': 'qeclang.code.v1', 'kind': 'css', 'n': n, 'hx': hx, 'hz': hz}}}


def certificate(sector='x'):
    return {'schema': 'qeclang.certificate.v1', 'source': 'a'*64, 'kind': 'css_logical',
            'claim': {'sector': sector, 'distance_upper_bound': 2},
            'certificate': {'sector': sector, 'operator': '0x5', 'dual': '0x3', 'weight': 2}}


def maps():
    return {'x_forward': [[0, 1], [0], []], 'x_reverse': [[1], [0, 1]],
            'z_forward': [[0], []], 'z_reverse': [[0]]}


def data(*, origin=False):
    params = {'certificate': certificate()}
    if origin:
        params['origin'] = {'bundle_id': 'b'*64, 'node': 'previous.proof'}
    return {'schema': SCHEMA, 'nodes': [
        css('source', [[0, 1], [2, 3]], [[0, 1, 2, 3]]),
        css('target', [[0, 1, 2, 3], [0, 1], []], [[0, 1, 2, 3], []]),
        {'id': 'imported', 'op': 'css.import_witness', 'inputs': ['source'], 'params': params},
        {'id': 'rebased', 'op': 'css.rebase_witness', 'inputs': ['source', 'target', 'imported'], 'params': maps()},
    ], 'outputs': ['imported', 'rebased']}


def test_import_and_rebase_bindings_shapes_and_immutable_records():
    raw = data(origin=True)
    program = Program.from_dict(raw)
    source, target, imported, rebased = program.nodes
    assert (source.n, source.mx, source.mz) == (4, 2, 1)
    assert (target.n, target.mx, target.mz) == (4, 3, 2)
    assert imported.result_type == rebased.result_type == 'witness'
    assert (imported.code_id, imported.sector, imported.mx, imported.mz) == ('source', 'x', 2, 1)
    assert (rebased.code_id, rebased.sector, rebased.mx, rebased.mz) == ('target', 'x', 3, 2)
    fingerprint = program.fingerprint()
    raw['nodes'][2]['params']['certificate']['certificate']['dual'] = '0x0'
    changed = imported.params
    changed['origin']['node'] = 'changed'
    assert imported.params['certificate']['certificate']['dual'] == '0x3'
    assert imported.params['origin']['node'] == 'previous.proof'
    assert program.fingerprint() == fingerprint
    assert Program.loads(json.dumps(program.to_dict())).fingerprint() == fingerprint


def test_parsing_does_not_hash_sources_check_parity_or_build_matrices(monkeypatch):
    from qeclang.algebra import BinaryMatrix
    import qeclang.analysis.certificate_io as certificates
    import qeclang.analysis.certificates as proof
    import qeclang.workflow.row_equivalence as equivalence
    import qeclang.io as code_io
    def forbidden(*args, **kwargs):
        raise AssertionError('source/parity and row identities belong to execution, not parsing')
    monkeypatch.setattr(certificates, 'presentation_fingerprint', forbidden)
    monkeypatch.setattr(certificates, 'verify_certificate_record', forbidden)
    monkeypatch.setattr(proof, 'verify_css_logical', forbidden)
    monkeypatch.setattr(equivalence, 'verify_row_equivalence', forbidden)
    monkeypatch.setattr(code_io, 'from_dict', forbidden)
    for method in ('rank', 'solve', '__matmul__', 'matvec'):
        monkeypatch.setattr(BinaryMatrix, method, forbidden)
    raw = data()
    raw['nodes'][2]['params']['certificate']['source'] = '0'*64
    raw['nodes'][2]['params']['certificate']['certificate']['dual'] = '0x0'
    # Wrong row identities and an even-pairing certificate remain well-typed;
    # mandatory execution/inspection validators must reject them later.
    raw['nodes'][3]['params']['x_reverse'] = [[2], [0]]
    assert Program.from_dict(raw).nodes[-1].code_id == 'target'


@pytest.mark.parametrize('change', [
    lambda c: c.update(schema='qeclang.certificate.v2'),
    lambda c: c.update(kind='stabilizer_logical'),
    lambda c: c.update(source='A'*64), lambda c: c.update(source='short'),
    lambda c: c.update(extra='untrusted'),
    lambda c: c['certificate'].update(sector='pauli'),
    lambda c: c['claim'].update(sector='z'),
    lambda c: c['certificate'].update(weight=True),
    lambda c: c['certificate'].update(weight=0),
    lambda c: c['certificate'].update(weight=5),
    lambda c: c['claim'].update(distance_upper_bound=True),
    lambda c: c['claim'].update(distance_upper_bound=1),
    lambda c: c['claim'].update(assurance='checked'),
    lambda c: c['certificate'].pop('dual'),
])
def test_certificate_schema_sector_and_claim_shape_are_strict(change):
    raw = data()
    change(raw['nodes'][2]['params']['certificate'])
    with pytest.raises(ValueError):
        Program.from_dict(raw)


@pytest.mark.parametrize('word', ['0x', '0X5', '0xA', '0x05', '-0x1', 5, '0x10', '0x' + 'f'*100])
@pytest.mark.parametrize('field', ['operator', 'dual'])
def test_words_are_canonical_and_bounded_before_execution(field, word):
    raw = data()
    raw['nodes'][2]['params']['certificate']['certificate'][field] = word
    with pytest.raises(ValueError, match='hexadecimal'):
        Program.from_dict(raw)


def test_actual_word_width_not_only_hex_digit_count_is_checked():
    raw = {'schema': SCHEMA, 'nodes': [css('source', [], [], n=3),
        {'id': 'witness', 'op': 'css.import_witness', 'inputs': ['source'],
         'params': {'certificate': certificate()}}], 'outputs': ['witness']}
    raw['nodes'][1]['params']['certificate']['certificate']['operator'] = '0x8'
    with pytest.raises(ValueError, match='width'):
        Program.from_dict(raw)
    raw['nodes'][0] = css('source', [], [], n=0)
    with pytest.raises(ValueError):
        Program.from_dict(raw)


@pytest.mark.parametrize('origin', [None, {}, {'bundle_id': 'B'*64, 'node': 'n'},
    {'bundle_id': 'b'*63, 'node': 'n'}, {'bundle_id': 'b'*64, 'node': 'bad node'},
    {'bundle_id': 'b'*64, 'node': 'n'*65}, {'bundle_id': 'b'*64, 'node': 'n', 'path': '/tmp/file'}])
def test_origin_is_optional_bounded_data_not_an_executable_reference(origin):
    raw = data()
    raw['nodes'][2]['params']['origin'] = origin
    with pytest.raises(ValueError):
        Program.from_dict(raw)


def test_origin_node_need_not_exist_in_this_program():
    raw = data(origin=True)
    assert Program.from_dict(raw).nodes[2].params['origin']['node'] == 'previous.proof'
    original = Program.from_dict(raw).fingerprint()
    raw['nodes'][2]['params']['origin']['bundle_id'] = 'c'*64
    assert Program.from_dict(raw).fingerprint() != original


@pytest.mark.parametrize('key,rows', [
    ('x_forward', [[0], [1]]), ('x_reverse', [[0], [1], []]),
    ('z_forward', [[0]]), ('z_reverse', [[0], []]),
    ('x_forward', [[2], [0], []]), ('x_reverse', [[3], [0, 1]]),
    ('z_forward', [[1], []]), ('z_reverse', [[2]]),
    ('x_forward', [[1, 0], [0], []]), ('x_forward', [[0, 0], [0], []]),
    ('x_forward', [[True], [0], []]), ('x_reverse', [[1.0], [0, 1]]),
    ('z_forward', [[-1], []]), ('z_reverse', [0]),
])
def test_rebase_maps_have_exact_directional_shapes_and_canonical_supports(key, rows):
    raw = data()
    raw['nodes'][3]['params'][key] = rows
    with pytest.raises(ValueError):
        Program.from_dict(raw)


def test_map_budget_counts_all_four_maps_and_preserves_zero_widths():
    assert Program.from_dict(data(), limits=ProgramLimits(max_map_entries=8)).nodes[-1].mx == 3
    with pytest.raises(ValueError, match='max_indices'):
        Program.from_dict(data(), limits=ProgramLimits(max_map_entries=7))
    with pytest.raises(ValueError):
        ProgramLimits(max_map_entries=True)
    raw = data()
    raw['nodes'][0] = css('source', [], [], n=4)
    raw['nodes'][1] = css('target', [[], []], [[]], n=4)
    raw['nodes'][3]['params'] = {'x_forward': [[], []], 'x_reverse': [],
                                  'z_forward': [[]], 'z_reverse': []}
    program = Program.from_dict(raw, limits=ProgramLimits(max_map_entries=0))
    assert (program.nodes[0].mx, program.nodes[0].mz) == (0, 0)
    assert (program.nodes[-1].mx, program.nodes[-1].mz) == (2, 1)


def test_rebase_requires_exact_nominal_source_and_same_physical_width():
    raw = data()
    raw['nodes'][2]['inputs'] = ['target']
    with pytest.raises(ValueError, match='source-bound'):
        Program.from_dict(raw)
    raw = data()
    raw['nodes'][1]['params']['code']['n'] = 5
    with pytest.raises(ValueError, match='equal physical widths'):
        Program.from_dict(raw)
    for bad_params in ({}, {**maps(), 'origin': {}}, {k: v for k, v in maps().items() if k != 'z_reverse'}):
        raw = data()
        raw['nodes'][3]['params'] = bad_params
        with pytest.raises(ValueError):
            Program.from_dict(raw)
    raw = data()
    raw['nodes'][2]['params']['execute'] = 'ignored()'
    with pytest.raises(ValueError):
        Program.from_dict(raw)


def test_rebased_witness_can_be_rebased_back_using_target_counts():
    raw = data()
    backward = {key: maps()[key.replace('forward', 'reverse') if 'forward' in key
                            else key.replace('reverse', 'forward')] for key in maps()}
    raw['nodes'].append({'id': 'returned', 'op': 'css.rebase_witness',
                         'inputs': ['target', 'source', 'rebased'], 'params': backward})
    program = Program.from_dict(raw)
    result = program.nodes[-1]
    assert (result.code_id, result.mx, result.mz, result.sector) == ('source', 2, 1, 'x')


def test_check_counts_propagate_through_code_conversion_and_permutation():
    raw = data()
    raw['nodes'].extend([
        {'id': 'chain', 'op': 'css.chain', 'inputs': ['target'], 'params': {}},
        {'id': 'roundtrip', 'op': 'chain.css', 'inputs': ['chain'], 'params': {}},
        {'id': 'sparse', 'op': 'css.sparse', 'inputs': ['roundtrip'], 'params': {}},
        {'id': 'packed', 'op': 'sparse.css', 'inputs': ['sparse'], 'params': {}},
        {'id': 'permuted', 'op': 'css.permute', 'inputs': ['packed'], 'params': {'permutation': [1, 0, 2, 3]}},
        {'id': 'transported', 'op': 'css.transport', 'inputs': ['target', 'permuted', 'rebased'],
         'params': {'permutation': [1, 0, 2, 3]}},
    ])
    program = Program.from_dict(raw)
    assert all((node.mx, node.mz) == (3, 2) for node in program.nodes[4:])
    assert program.nodes[-1].code_id == 'permuted'


def test_check_counts_derive_from_symbolic_seed_shapes_without_expansion(monkeypatch):
    from qeclang.symbolic import GroupAlgebraMatrix
    record = {'schema': records.SCHEMA, 'kind': 'abelian_lifted_product',
              'coefficient_field': 'GF(2)', 'orders': [3], 'action': records.ACTION,
              'convention': records.CONVENTION, 'physical_ordering': records.ORDERING,
              'a': {'shape': [2, 3], 'entries': [[[], [], []], [[], [], []]]},
              'b': {'shape': [1, 2], 'entries': [[[], []]]}}
    raw = {'schema': SCHEMA, 'nodes': [
        {'id': 'lift', 'op': 'lift.literal', 'inputs': [], 'params': {'construction': record}},
        {'id': 'parameters', 'op': 'lift.parameters', 'inputs': ['lift'],
         'params': {'backend': 'crt', 'max_steps': 1, 'record_operations': False}},
        {'id': 'css', 'op': 'lift.css', 'inputs': ['lift'], 'params': {}},
        {'id': 'chain', 'op': 'css.chain', 'inputs': ['css'], 'params': {}},
    ], 'outputs': ['parameters', 'chain']}
    def forbidden(*args, **kwargs):
        raise AssertionError('static count propagation must not instantiate or expand ring matrices')
    monkeypatch.setattr(records, 'from_dict', forbidden)
    monkeypatch.setattr(GroupAlgebraMatrix, '__post_init__', forbidden)
    monkeypatch.setattr(GroupAlgebraMatrix, 'lower', forbidden)
    program = Program.from_dict(raw)
    assert all((node.n, node.mx, node.mz) == (24, 12, 9) for node in program.nodes)


def test_existing_program_hashes_ignore_new_static_metadata_and_map_caps():
    raw = {'schema': SCHEMA, 'nodes': [css('source', [[0, 1]], [], n=2)], 'outputs': ['source']}
    expected = sha256(json.dumps(raw, sort_keys=True, separators=(',', ':'),
                                ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    assert Program.from_dict(raw).fingerprint() == expected
    assert Program.from_dict(raw, limits=ProgramLimits(max_map_entries=0)).fingerprint() == expected
    assert 'mx' not in Program.from_dict(raw).to_dict()['nodes'][0]
