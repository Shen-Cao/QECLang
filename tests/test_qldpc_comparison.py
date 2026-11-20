import importlib.util
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qeclang.algebra import BinaryMatrix
from qeclang.catalog.references import build_reference
from qeclang.codes import CSSCode

_PATH = Path(__file__).parents[1] / 'scripts' / 'compare_qldpc.py'
_SPEC = importlib.util.spec_from_file_location('qeclang_qldpc_comparison_script', _PATH)
comparison = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = comparison
_SPEC.loader.exec_module(comparison)


def has_pinned_qldpc():
    try:
        return metadata.version('qldpc') == comparison.PINNED_VERSION
    except metadata.PackageNotFoundError:
        return False


def observation(identifier):
    code = build_reference(identifier)
    baseline = comparison.preflight(code, comparison.ComparisonLimits())
    known = {item.id: item for item in comparison.reference_cases()}[identifier]
    return {'status': 'observed', 'n': code.n, 'k': code.k,
            'rank_x': baseline['rank_x'], 'rank_z': baseline['rank_z'],
            'source': comparison.presentation_fingerprint(code),
            'distances': {'x': known.expected_dx, 'z': known.expected_dz}}


def test_shape_and_exponential_preflight_precede_foreign_calls(monkeypatch):
    code = build_reference('steane')
    def forbidden(*args, **kwargs):
        raise AssertionError('rank/foreign work reached before dimension guard')
    with monkeypatch.context() as patch:
        patch.setattr(BinaryMatrix, 'rank', forbidden)
        with pytest.raises(ValueError, match='axes'):
            comparison.preflight(code, comparison.ComparisonLimits(max_n=6))
    free = CSSCode(BinaryMatrix((), 4), BinaryMatrix((), 4))
    with pytest.raises(ValueError, match='candidate cap'):
        comparison.preflight(free, comparison.ComparisonLimits(max_foreign_candidates=14))
    assert comparison.preflight(free, comparison.ComparisonLimits())['foreign_candidate_bounds'] == {'x': 15, 'z': 15}
    monkeypatch.setattr(comparison, 'foreign_call', forbidden)
    assert comparison.compare_case('toric_3')['status'] == 'skipped_budget'
    assert comparison.compare_family('steane', limits=comparison.ComparisonLimits(max_n=1))['status'] == 'skipped_budget'
    assert comparison.compare_case('steane', limits=comparison.ComparisonLimits(max_native_candidates=0))['status'] == 'native_incomplete'


def test_driver_does_not_promote_mismatches_errors_or_zero_logical_distances(monkeypatch):
    monkeypatch.setattr(comparison, 'foreign_call', lambda *a: observation('steane'))
    result = comparison.compare_case('steane')
    assert result['status'] == 'matched'
    assert result['assurance'] == 'differential_agreement_not_formal_proof'
    bad = observation('steane')
    bad['distances']['x'] = 1
    monkeypatch.setattr(comparison, 'foreign_call', lambda *a: bad)
    assert comparison.compare_case('steane')['status'] == 'mismatch'
    monkeypatch.setattr(comparison, 'foreign_call', lambda *a: {'status': 'timeout'})
    assert comparison.compare_case('steane')['status'] == 'external_timeout'
    monkeypatch.setattr(comparison, 'foreign_call', lambda *a: observation('bell_redundant'))
    result = comparison.compare_case('bell_redundant')
    assert result['status'] == 'matched_no_logicals'
    assert result['distance_comparison'] == 'not_applicable_no_logicals'
    assert all(v['status'] == 'no_logicals' for v in result['native_distances'].values())
    json.dumps(result, allow_nan=False)


def test_worker_timeout_and_invalid_output_are_explicit(monkeypatch):
    observed = {}
    def timeout(command, **options):
        observed.update(options)
        assert command[0] == sys.executable and command[-1] == '--worker'
        raise subprocess.TimeoutExpired(command, options['timeout'])
    monkeypatch.setattr(comparison.subprocess, 'run', timeout)
    result = comparison.foreign_call('matched', {}, comparison.ComparisonLimits(timeout_seconds=.01))
    assert result['status'] == 'timeout' and observed['timeout'] == .01
    monkeypatch.setattr(comparison.subprocess, 'run', lambda *a, **kw:
                         SimpleNamespace(returncode=0, stdout='not JSON', stderr=''))
    assert comparison.foreign_call('matched', {}, comparison.ComparisonLimits())['status'] == 'error'
    monkeypatch.setattr(comparison.subprocess, 'run', lambda *a, **kw:
                         SimpleNamespace(returncode=1, stdout='', stderr='foreign failure'))
    assert comparison.foreign_call('matched', {}, comparison.ComparisonLimits())['reason'] == 'foreign failure'


@pytest.mark.parametrize('options', [{'max_n': 21}, {'max_n': True}, {'max_checks': -1},
                                    {'max_elements': 1.5}, {'max_native_candidates': -1},
                                    {'max_foreign_candidates': False}, {'timeout_seconds': 0},
                                    {'timeout_seconds': float('inf')}])
def test_invalid_policies(options):
    with pytest.raises(ValueError):
        comparison.ComparisonLimits(**options)


@pytest.mark.skipif(not has_pinned_qldpc(), reason='optional pinned qldpc==0.4.1 required')
def test_real_sector_calls_are_fresh_uncached_and_exhaustive(monkeypatch):
    from qldpc.codes import CSSCode as ForeignCSS
    calls = []
    original = ForeignCSS.get_distance_exact
    def checked(self, pauli=None, **kwargs):
        assert type(self) is ForeignCSS
        assert kwargs == {'cutoff': 0, 'method': 'brute_force'}
        assert self.get_distance_if_known(pauli) is None
        calls.append((self, pauli))
        return original(self, pauli, **kwargs)
    monkeypatch.setattr(ForeignCSS, 'get_distance_exact', checked)
    result = comparison.foreign_worker({'task': 'matched',
        'payload': comparison.to_dict(build_reference('hgp_open_2_3')),
        'limits': comparison.asdict(comparison.ComparisonLimits())})
    assert result['distances'] == {'x': 3, 'z': 2}
    assert [pauli for _, pauli in calls] == ['X', 'Z']
    assert calls[0][0] is not calls[1][0]
    assert result['assurance'] == 'external_computational_result_not_a_certificate'
    assert result['versions']['qldpc'] == '0.4.1'


@pytest.mark.skipif(not has_pinned_qldpc(), reason='optional pinned qldpc==0.4.1 required')
def test_real_family_constructors_use_raw_matrices_not_cached_distances(monkeypatch):
    from qldpc.codes import CSSCode as ForeignCSS
    def forbidden(*args, **kwargs):
        raise AssertionError('family-supplied distance metadata must not be used')
    monkeypatch.setattr(ForeignCSS, 'get_distance_exact', forbidden)
    for identifier in comparison.FAMILIES:
        result = comparison.foreign_worker({'task': 'family', 'payload': identifier,
                    'limits': comparison.asdict(comparison.ComparisonLimits())})
        assert result['status'] == 'observed' and result['family_distance_metadata_used'] is False
        assert result['code']['kind'] == 'css'


def test_cli_small_axis_cap_emits_both_refusals_without_foreign_work(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('small-cap CLI refusal must precede foreign work')
    monkeypatch.setattr(comparison, 'foreign_call', forbidden)
    monkeypatch.setattr(sys, 'argv', ['compare_qldpc.py', '--cases', 'steane', '--max-n', '1'])
    assert comparison.main() == 2
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {row['kind'] for row in rows} == {'matched_presentation', 'independent_family'}
    assert all(row['status'] == 'skipped_budget' for row in rows)
