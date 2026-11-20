from copy import deepcopy
import json
from pathlib import Path

import pytest

from qeclang.analysis.certificate_io import verify_certificate_record
from qeclang.cli import main
from qeclang.io import from_dict
from qeclang.workflow import ExecutionLimits, Program, execute, replay

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/programs/transport.json'


def data():
    return json.loads(EXAMPLE.read_text())


def artifact(record, node):
    return record['artifacts'][record['outputs'][node]]['payload']


def test_transport_replay_and_assurance_modes():
    program = Program.load(EXAMPLE)
    run = execute(program)
    assert run['status'] == 'completed', run['trace'][-1]
    assert run['ledger']['used']['optional_checks'] == 5
    assert verify_certificate_record(from_dict(artifact(run, 'final')), artifact(run, 'transported'))
    assert artifact(run, 'transported')['claim']['distance_upper_bound'] == 3
    assert run['trace'][-1]['evidence']['assurance'] == 'independently_checked'
    assert replay(program, json.loads(json.dumps(run)))
    corrupt = deepcopy(run)
    corrupt['trace'][-1]['evidence']['bound'] = 0
    assert not replay(program, corrupt)
    for optional in (execute(program, verify_relations=False),
                     execute(program, limits=ExecutionLimits(max_optional_checks=0))):
        assert optional['outputs'] == run['outputs']
        assert optional['artifacts'] == run['artifacts']
        assert optional['ledger']['used']['optional_checks'] == 0
        assert optional['trace'][1]['evidence']['assurance'] != 'relation_checked'
        assert optional['trace'][-1]['evidence']['assurance'] == 'independently_checked'
    assert not replay(program, optional)


def test_cumulative_limits_keep_partial_results_and_verified_upper():
    raw = data()
    second = deepcopy(raw['nodes'][3])
    second['id'] = 'again'
    raw['nodes'].append(second)
    raw['outputs'].append('again')
    result = execute(Program.from_dict(raw), limits=ExecutionLimits(max_candidates=29))
    assert result['status'] == 'completed'
    assert result['ledger']['used']['candidates'] == 29
    assert result['trace'][3]['evidence']['search_status'] == 'exact'
    later = result['trace'][-1]
    assert later['spent']['candidates'] == 1
    assert later['evidence']['search_status'] == 'budget_exhausted'
    assert later['evidence']['lower_bound'] < later['evidence']['upper_bound']
    raw['outputs'].append('original')
    result = execute(Program.from_dict(raw), limits=ExecutionLimits(max_validation_work=0))
    assert result['status'] == 'budget_exhausted' and result['trace'][-1]['node'] == 'sparse'
    assert set(result['outputs']) == {'distance', 'original'}
    assert result['ledger']['used']['candidates'] == 28


def test_refuse_large_input_before_construction(monkeypatch):
    import qeclang.workflow.operations as operations
    def forbidden(*a, **kw):
        raise AssertionError('kernel called before guard')
    monkeypatch.setattr(operations, 'from_dict', forbidden)
    result = execute(Program.load(EXAMPLE), limits=ExecutionLimits(max_matrix_bits=0))
    assert result['status'] == 'budget_exhausted' and not result['outputs']
    assert result['trace'][0]['resource'] == 'matrix_bits'


def test_checked_edge_detects_valid_but_unrelated_target(monkeypatch):
    import qeclang.workflow.operations as operations
    monkeypatch.setattr(operations, 'permute_qubits', lambda code, permutation: code)
    for assurance, expected_failure in ((True, 'relabeled'), (False, 'transported')):
        result = execute(Program.load(EXAMPLE), verify_relations=assurance)
        assert result['status'] == 'verification_failed'
        assert result['trace'][-1]['node'] == expected_failure
        assert 'transported' not in result['outputs']


def test_cli_and_failed_replay(tmp_path, capsys):
    assert main(['run', str(EXAMPLE)]) == 0
    record = json.loads(capsys.readouterr().out)
    path = tmp_path / 'record.json'
    path.write_text(json.dumps(record))
    assert main(['run', str(EXAMPLE), '--replay', str(path)]) == 0
    assert json.loads(capsys.readouterr().out)['matches']
    record['ledger'] = None
    path.write_text(json.dumps(record))
    assert main(['run', str(EXAMPLE), '--replay', str(path)]) == 2
    assert json.loads(capsys.readouterr().out)['matches'] is False
    assert main(['run', str(EXAMPLE), '--max-candidates', '-1']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'invalid_request'


def test_expected_kernel_failure_is_a_value_but_internal_bug_propagates(monkeypatch):
    import qeclang.workflow.operations as operations
    raw = data()
    raw['nodes'][0]['params']['code']['hz'][0] = [0]
    result = execute(Program.from_dict(raw))
    assert result['status'] == 'invalid_input'
    def failure(*a, **kw):
        raise ArithmeticError('broken kernel')
    monkeypatch.setattr(operations, 'css_distance', failure)
    with pytest.raises(ArithmeticError, match='broken kernel'):
        execute(Program.load(EXAMPLE))


def test_scalar_limits_and_empty_logical_set():
    raw = data()
    literal = raw['nodes'][0]
    literal['params']['code'] = {'schema': 'qeclang.code.v1', 'kind': 'css',
                                 'n': 1, 'hx': [[0]], 'hz': []}
    distance, witness = raw['nodes'][3:5]
    distance['inputs'] = ['base']
    witness['inputs'] = ['base', 'distance']
    raw.update(nodes=[literal, distance, witness], outputs=['distance', 'logical'])
    program = Program.from_dict(raw)
    result = execute(program)
    assert result['status'] == 'evidence_unavailable'
    assert set(result['outputs']) == {'distance'}
    assert result['trace'][1]['evidence']['search_status'] == 'no_logicals'
    assert result['trace'][1]['evidence']['upper_bound'] is None
    too_small = execute(program, limits=ExecutionLimits(max_artifact_bytes=1))
    assert too_small['status'] == 'budget_exhausted'
    assert not too_small['artifacts'] and not too_small['ledger']['used']['artifact_bytes']
    assert replay(program, too_small, limits=ExecutionLimits(max_artifact_bytes=1))
