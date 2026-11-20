from copy import deepcopy
from hashlib import sha256
import json

import pytest

from qeclang.workflow.program import Program, ProgramLimits, SCHEMA


def code(name, *, target=False, n=4):
    hx = [[0, 1, 2, 3], [0, 1], []] if target else [[0, 1], [2, 3]]
    hz = [[0, 1, 2, 3], []] if target else [[0, 1, 2, 3]]
    return {'id': name, 'op': 'css.literal', 'inputs': [], 'params': {'code': {
        'schema': 'qeclang.code.v1', 'kind': 'css', 'n': n, 'hx': hx, 'hz': hz}}}


def exclusion(sector='x'):
    return {'schema': 'qeclang.low_weight.v1', 'record_id': 'b'*64, 'source': 'a'*64,
            'sector': sector, 'through': 1, 'status': 'excluded', 'steps': 10,
            'claim': {'predicate': 'nontrivial_weight_exclusion', 'through': 1},
            'memberships': [], 'witness': None, 'reason': None}


def witness(sector):
    return {'schema': 'qeclang.certificate.v1', 'source': 'a'*64, 'kind': 'css_logical',
            'claim': {'sector': sector, 'distance_upper_bound': 2},
            'certificate': {'sector': sector, 'operator': '0x5' if sector == 'x' else '0x3',
                            'dual': '0x3' if sector == 'x' else '0x5', 'weight': 2}}


def maps():
    return {'x_forward': [[0, 1], [0], []], 'x_reverse': [[1], [0, 1]],
            'z_forward': [[0], []], 'z_reverse': [[0]]}


def data(*, sector='x', relation='permutation'):
    nodes = [code('source'), code('target', target=relation == 'generator_spaces')]
    for item in ('x', 'z') if sector == 'pauli' else (sector,):
        nodes.extend([
            {'id': item+'_absence', 'op': 'css.import_exclusion', 'inputs': ['source'],
             'params': {'record': exclusion(item), 'origin': {'bundle_id': 'c'*64, 'node': 'old.absence'}}},
            {'id': item+'_upper', 'op': 'css.import_witness', 'inputs': ['source'],
             'params': {'certificate': witness(item)}},
            {'id': item+'_interval', 'op': 'css.combine_distance',
             'inputs': ['source', item+'_absence', item+'_upper'], 'params': {}},
        ])
    if sector == 'pauli':
        nodes.append({'id': 'full', 'op': 'css.full_distance',
                      'inputs': ['source', 'x_interval', 'z_interval'], 'params': {}})
    claim = 'full' if sector == 'pauli' else sector+'_interval'
    relation_params = ({'relation': relation, 'permutation': [0, 1, 2, 3]}
                       if relation == 'permutation' else {'relation': relation, 'maps': maps()})
    nodes.append({'id': 'transported', 'op': 'css.transport_distance',
                  'inputs': ['source', 'target', claim], 'params': relation_params})
    return {'schema': SCHEMA, 'nodes': nodes, 'outputs': ['transported']}


@pytest.mark.parametrize('sector', ['x', 'z', 'pauli'])
@pytest.mark.parametrize('relation', ['permutation', 'generator_spaces'])
def test_distance_transport_preserves_sector_and_binds_target_counts(sector, relation):
    raw = data(sector=sector, relation=relation)
    program = Program.from_dict(raw)
    output = program.nodes[-1]
    assert (output.result_type, output.code_id, output.sector) == ('distance_claim', 'target', sector)
    assert (output.n, output.mx, output.mz) == ((4, 3, 2) if relation == 'generator_spaces' else (4, 2, 1))
    absence = program.nodes[2]
    assert (absence.result_type, absence.code_id, absence.sector) == ('exclusion', 'source', 'x' if sector == 'pauli' else sector)
    assert Program.loads(json.dumps(raw)).fingerprint() == program.fingerprint()
    raw['nodes'][2]['params']['record']['memberships'].append([0])
    assert program.nodes[2].params['record']['memberships'] == []


def test_import_origin_is_optional_and_does_not_reference_the_current_graph():
    raw = data()
    program = Program.from_dict(raw)
    assert program.nodes[2].params['origin']['node'] == 'old.absence'
    del raw['nodes'][2]['params']['origin']
    assert Program.from_dict(raw).nodes[2].sector == 'x'


@pytest.mark.parametrize('change', [
    lambda r: r.update(status='found'), lambda r: r.update(status='budget_exhausted'),
    lambda r: r.update(status='no_logicals'), lambda r: r.update(through=0),
    lambda r: r.update(through=True), lambda r: r.update(sector='pauli'),
    lambda r: r.update(record_id='B'*64), lambda r: r.update(source='short'),
    lambda r: r.update(schema='qeclang.low_weight.v2'), lambda r: r.update(reason='unexpected'),
    lambda r: r.update(memberships=None), lambda r: r.update(memberships=[[1, 0]]),
    lambda r: r.update(memberships=[[0, 0]]), lambda r: r.update(memberships=[[True]]),
    lambda r: r.update(claim={'predicate': 'distance_lower_bound', 'through': 1}),
    lambda r: r.update(steps=False), lambda r: r.update(witness=witness('x')),
])
def test_imported_exclusion_record_remains_canonical_excluded_only(change):
    raw = data()
    change(raw['nodes'][2]['params']['record'])
    with pytest.raises(ValueError):
        Program.from_dict(raw)


def test_other_well_formed_exclusion_outcomes_cannot_be_imported_as_exclusion():
    raw = data()
    record = raw['nodes'][2]['params']['record']
    record.update(status='budget_exhausted', claim=None, memberships=None,
                  reason='budget exhausted', witness=None)
    with pytest.raises(ValueError, match='excluded-only'):
        Program.from_dict(raw)
    record.update(status='found', claim={'predicate': 'distance_upper_bound', 'upper_bound': 1},
                  reason=None, witness=witness('x'))
    record['witness']['claim']['distance_upper_bound'] = 1
    record['witness']['certificate']['weight'] = 1
    with pytest.raises(ValueError, match='excluded-only'):
        Program.from_dict(raw)


def test_memberships_are_bounded_by_original_sector_rows_and_physical_width():
    raw = data()
    raw['nodes'][2]['params']['record']['memberships'] = [[1]]
    assert Program.from_dict(raw).nodes[2].sector == 'x'  # source has two X rows
    raw['nodes'][2]['params']['record']['memberships'] = [[2]]
    with pytest.raises(ValueError, match='same-sector'):
        Program.from_dict(raw)
    raw = data(sector='z')
    raw['nodes'][2]['params']['record']['memberships'] = [[1]]
    with pytest.raises(ValueError, match='same-sector'):
        Program.from_dict(raw)  # source has only one Z row, even though two X rows exist
    raw = data()
    raw['nodes'][2]['params']['record']['memberships'] = [[]] * 5
    with pytest.raises(ValueError, match='physical width'):
        Program.from_dict(raw)


def test_zero_check_and_zero_site_exclusion_imports_do_not_assert_logical_existence():
    raw = {'schema': SCHEMA, 'nodes': [
        {'id': 'source', 'op': 'css.literal', 'inputs': [], 'params': {'code': {
            'schema': 'qeclang.code.v1', 'kind': 'css', 'n': 0, 'hx': [], 'hz': []}}},
        {'id': 'absence', 'op': 'css.import_exclusion', 'inputs': ['source'],
         'params': {'record': exclusion()}},
    ], 'outputs': ['absence']}
    assert Program.from_dict(raw).nodes[-1].result_type == 'exclusion'
    raw['nodes'][0]['params']['code']['n'] = 4
    raw['nodes'][1]['params']['record']['memberships'] = [[0]]
    with pytest.raises(ValueError, match='same-sector'):
        Program.from_dict(raw)


@pytest.mark.parametrize('origin', [None, {}, {'bundle_id': 'C'*64, 'node': 'old'},
    {'bundle_id': 'c'*64, 'node': 'bad name'}, {'bundle_id': 'c'*64, 'node': 'n'*65},
    {'bundle_id': 'c'*64, 'node': 'old', 'path': '/tmp/anything'}])
def test_import_origin_uses_the_same_bounded_shape_as_witness_import(origin):
    raw = data()
    raw['nodes'][2]['params']['origin'] = origin
    with pytest.raises(ValueError):
        Program.from_dict(raw)


def test_parser_performs_no_source_hash_parity_membership_or_relation_check(monkeypatch):
    import qeclang.analysis.low_weight as evidence
    import qeclang.analysis.certificate_io as certificates
    import qeclang.workflow.row_equivalence as rows
    from qeclang.algebra import BinaryMatrix
    def forbidden(*args, **kwargs):
        raise AssertionError('proof verification is outside static program admission')
    monkeypatch.setattr(evidence, 'certify_low_weight', forbidden)
    monkeypatch.setattr(evidence, 'check_low_weight', forbidden)
    monkeypatch.setattr(evidence, 'low_weight_id', forbidden)
    monkeypatch.setattr(certificates, 'presentation_fingerprint', forbidden)
    monkeypatch.setattr(rows, 'verify_row_equivalence', forbidden)
    for method in ('rank', 'solve', 'matvec', '__matmul__'):
        monkeypatch.setattr(BinaryMatrix, method, forbidden)
    raw = data(relation='generator_spaces')
    raw['nodes'][2]['params']['record']['source'] = '0'*64
    raw['nodes'][2]['params']['record']['record_id'] = '0'*64
    raw['nodes'][2]['params']['record']['memberships'] = [[1]]
    raw['nodes'][-1]['params']['maps']['x_reverse'] = [[2], [0]]
    assert Program.from_dict(raw).nodes[-1].code_id == 'target'


@pytest.mark.parametrize('params', [
    {}, {'relation': 'homology'}, {'relation': 'permutation'},
    {'relation': 'permutation', 'permutation': [0, 1, 2]},
    {'relation': 'permutation', 'permutation': [0, 0, 2, 3]},
    {'relation': 'permutation', 'permutation': [0, 1, 2, True]},
    {'relation': 'permutation', 'permutation': [0, 1, 2, 3], 'maps': maps()},
    {'relation': 'generator_spaces'}, {'relation': 'generator_spaces', 'maps': {}},
    {'relation': 'generator_spaces', 'maps': maps(), 'permutation': [0, 1, 2, 3]},
])
def test_transport_relation_parameters_are_closed_and_exact(params):
    raw = data()
    raw['nodes'][-1]['params'] = params
    with pytest.raises(ValueError):
        Program.from_dict(raw)


def test_transport_checks_nominal_source_widths_and_retained_shapes():
    raw = data()
    alias = deepcopy(raw['nodes'][0])
    alias['id'] = 'alias'
    raw['nodes'].insert(1, alias)
    raw['nodes'][-1]['inputs'][0] = 'alias'
    with pytest.raises(ValueError, match='source-bound'):
        Program.from_dict(raw)
    raw = data()
    raw['nodes'][1]['params']['code']['n'] = 5
    with pytest.raises(ValueError, match='equal physical widths'):
        Program.from_dict(raw)
    raw = data()
    raw['nodes'][1]['params']['code']['hx'].append([])
    with pytest.raises(ValueError, match='check counts'):
        Program.from_dict(raw)
    raw = data()
    raw['nodes'][-1]['inputs'][-1] = 'x_upper'
    with pytest.raises(ValueError, match='input types'):
        Program.from_dict(raw)


def test_generator_space_transport_honors_map_shapes_and_shared_limits():
    raw = data(relation='generator_spaces')
    assert Program.from_dict(raw, limits=ProgramLimits(max_map_entries=8)).nodes[-1].mx == 3
    with pytest.raises(ValueError, match='max_indices'):
        Program.from_dict(raw, limits=ProgramLimits(max_map_entries=7))
    for key, support in (('x_forward', [[0], [1]]), ('z_reverse', [[2]]),
                         ('x_reverse', [[1.0], [0, 1]])):
        changed = deepcopy(raw)
        changed['nodes'][-1]['params']['maps'][key] = support
        with pytest.raises(ValueError):
            Program.from_dict(changed)


def test_transported_pauli_interval_can_be_transported_again():
    raw = data(sector='pauli', relation='generator_spaces')
    backwards = {key: maps()[key.replace('forward', 'reverse') if 'forward' in key
                            else key.replace('reverse', 'forward')] for key in maps()}
    raw['nodes'].append({'id': 'returned', 'op': 'css.transport_distance',
                         'inputs': ['target', 'source', 'transported'],
                         'params': {'relation': 'generator_spaces', 'maps': backwards}})
    result = Program.from_dict(raw).nodes[-1]
    assert (result.code_id, result.sector, result.mx, result.mz) == ('source', 'pauli', 2, 1)


def test_legacy_program_fingerprints_do_not_include_new_registry_metadata():
    raw = {'schema': SCHEMA, 'nodes': [code('source')], 'outputs': ['source']}
    expected = sha256(json.dumps(raw, sort_keys=True, separators=(',', ':'),
                                ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    assert Program.from_dict(raw).fingerprint() == expected
