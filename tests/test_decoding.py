import pytest

from qeclang.algebra import BinaryMatrix as M
from qeclang.codes import CSSCode
from qeclang.decoding import (css_decode, css_logical_success, minimum_weight_decode,
                              simulate_css_memory, wilson_interval)


def steane():
    h = M((0b1010101, 0b1100110, 0b1111000), 7)
    return CSSCode(h, h)


def test_all_single_qubit_paulis_corrected():
    code = steane()
    for i in range(7):
        for label in (1, 2, 3):
            x, z = (label & 1) << i, (label >> 1) << i
            correction = css_decode(code, code.hx.matvec(z), code.hz.matvec(x))
            assert correction.success
            assert css_logical_success(code, x, z, correction)


def test_degenerate_success_differs_from_exact_error_recovery():
    code = steane()
    correction = css_decode(code, 0, 0)
    assert correction.x.error == 0
    assert css_logical_success(code, code.hx.rows[0], 0, correction)
    assert not css_logical_success(code, 0b1111111, 0, correction)


def test_decoder_bounds_invalid_syndrome_and_empty_checks():
    checks = M((3, 6), 3)
    assert minimum_weight_decode(checks, 0).error == 0
    result = minimum_weight_decode(checks, 3)
    assert result.error == 2 and result.weight == 1
    assert minimum_weight_decode(checks, 3, max_weight=0).status == 'weight_limit'
    result = minimum_weight_decode(checks, 3, max_candidates=2)
    assert result.status == 'budget_exhausted' and result.candidates == 2
    assert minimum_weight_decode(M((1, 1), 1), 1).status == 'inconsistent'
    assert minimum_weight_decode(M((), 0), 0).error == 0
    with pytest.raises(ValueError):
        minimum_weight_decode(checks, 4)
    with pytest.raises(ValueError):
        minimum_weight_decode(checks, True)


def test_reproducible_simulation_and_failure_accounting():
    code = steane()
    a = simulate_css_memory(code, 0.2, trials=100, seed=42)
    b = simulate_css_memory(code, 0.2, trials=100, seed=42)
    assert a == b
    assert a.failures == a.decoder_failures + a.logical_failures
    assert a.confidence_interval[0] <= a.failure_rate <= a.confidence_interval[1]
    perfect = simulate_css_memory(code, 0, trials=30, seed=2)
    assert perfect.failures == 0 and perfect.distinct_syndromes == 1
    assert perfect.confidence_interval[1] > 0
    failed = simulate_css_memory(code, 0, trials=30, seed=2, max_candidates_per_sector=0)
    assert failed.failures == failed.decoder_failures == 30


def test_wilson_and_invalid_experiments():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(1-high)
    assert low < 0.5 < high
    for kwargs in ({'probability': -0.1, 'trials': 1, 'seed': 0},
                   {'probability': 0.1, 'trials': 0, 'seed': 0},
                   {'probability': float('nan'), 'trials': 1, 'seed': 0}):
        with pytest.raises(ValueError):
            simulate_css_memory(steane(), **kwargs)


def test_erasure_decoding_all_steane_errors_on_two_sites():
    from itertools import combinations, product
    from qeclang.decoding import css_erasure_decode
    code = steane()
    for sites in combinations(range(code.n), 2):
        for labels in product(range(4), repeat=2):
            x = sum((label & 1) << i for i, label in zip(sites, labels))
            z = sum((label >> 1) << i for i, label in zip(sites, labels))
            result = css_erasure_decode(code, code.hx.matvec(z), code.hz.matvec(x), sites)
            assert result.success and result.region_correctable
            assert code.hx.contains(x ^ result.x.error)
            assert code.hz.contains(z ^ result.z.error)
            mask = sum(1 << i for i in sites)
            assert not (result.x.error | result.z.error) & ~mask


def test_erasure_ambiguity_consistency_and_degeneracy():
    from qeclang.decoding import css_erasure_decode, erasure_decode
    code = steane()
    result = css_erasure_decode(code, 0, 0, range(code.n))
    assert result.success and not result.region_correctable
    assert not result.x.unique_solution and not result.z.unique_solution
    assert css_erasure_decode(code, 1, 0, []).z.status == 'inconsistent'
    assert not css_erasure_decode(code, 1, 0, []).success
    assert erasure_decode(M((3, 6), 3), 3, [1]).error == 2
    assert erasure_decode(M((3, 6), 3), 3, [1]).unique_solution
    bell = CSSCode(M((3,), 2), M((3,), 2))
    degenerate = css_erasure_decode(bell, 1, 1, [0, 1])
    assert degenerate.success and degenerate.region_correctable
    assert not degenerate.x.unique_solution
    with pytest.raises(ValueError):
        css_erasure_decode(code, 0, 0, [0, 0])
    with pytest.raises(ValueError):
        erasure_decode(code.hx, 1 << code.hx.nrows, [0])
