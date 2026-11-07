from random import Random

import pytest

from qeclang.algebra import BinaryMatrix as M
from qeclang.analysis.certificates import verify_css_logical
from qeclang.analysis.distance import css_distance
from qeclang.analysis.heuristic import css_logical_upper_bound
from qeclang.analysis.logicals import css_logicals
from qeclang.analysis.milp import MILPDistanceResult
from qeclang.analysis.portfolio import css_distance_portfolio
from qeclang.codes import CSSCode, SparseCSSCode
from qeclang.constructions import shor, steane, toric


def improvement_code():
    return CSSCode(M((176, 154, 106), 8), M((55, 166, 159), 8))


def test_deterministic_greedy_improvement_preserves_logical_coset():
    code = improvement_code()
    result = css_logical_upper_bound(code, "x", seed=2, restarts=0)
    assert result.initial_upper_bound == 3 and result.upper_bound == 2
    assert result.upper_bound == css_distance(code, "x").distance
    assert verify_css_logical(code, result.certificate)
    assert any(code.hx.contains(result.witness ^ representative) for representative in css_logicals(code).x)
    assert result.restarts_completed == 0
    assert result == css_logical_upper_bound(code, "x", seed=2, restarts=0)


def test_seeded_reproducibility_and_exact_small_oracles():
    rng = Random(872)
    codes = [steane(), shor(), toric(2), improvement_code()]
    for _ in range(30):
        hz = M(tuple(rng.randrange(64) for _ in range(2)), 6)
        kernel = hz.nullspace().rows
        hx = []
        for _ in range(2):
            value = 0
            for row in kernel:
                if rng.getrandbits(1):
                    value ^= row
            hx.append(value)
        codes.append(CSSCode(M(tuple(hx), 6), hz))
    for code in codes:
        for sector in ("x", "z"):
            result = css_logical_upper_bound(code, sector, seed=17, restarts=4,
                                             max_sweeps=4, max_evaluations=200)
            assert result == css_logical_upper_bound(code, sector, seed=17, restarts=4,
                                                      max_sweeps=4, max_evaluations=200)
            exact = css_distance(code, sector)
            if code.k:
                assert verify_css_logical(code, result.certificate)
                assert result.upper_bound >= exact.distance
                assert result.witness.bit_count() == result.upper_bound
            else:
                assert result.status == "no_logicals" and result.witness is None
            assert result.evaluations <= 200
            assert result.sweeps <= 5 * 4


def test_every_published_witness_is_independently_valid(monkeypatch):
    import qeclang.analysis.heuristic as module

    code = steane()
    initial = css_logicals(code).x[0]
    observed = []

    def check(candidate_code, certificate):
        assert verify_css_logical(candidate_code, certificate)
        assert code.hx.contains(certificate.operator ^ initial)
        observed.append(certificate)
        return True

    monkeypatch.setattr(module, "verify_css_logical", check)
    result = css_logical_upper_bound(code, "x", seed=5, restarts=8)
    assert observed and result.certificate in observed


def test_early_candidate_budgets_and_no_logicals():
    code = steane()
    zero = css_logical_upper_bound(code, "x", max_evaluations=0)
    assert zero.status == "budget_exhausted" and zero.upper_bound is None
    one = css_logical_upper_bound(code, "x", max_evaluations=1)
    assert one.status == "budget_exhausted" and one.evaluations == 1
    assert verify_css_logical(code, one.certificate)
    bell = CSSCode(M((3,), 2), M((3,), 2))
    empty = CSSCode(M((), 0), M((), 0))
    for no_logicals in (bell, empty):
        result = css_logical_upper_bound(no_logicals, "x", max_evaluations=0)
        assert result.status == "no_logicals"
        assert result.upper_bound is None and result.certificate is None
        portfolio = css_distance_portfolio(no_logicals, "x")
        assert portfolio.status == "no_logicals" and portfolio.distance is None


def test_basis_preflight_precedes_elimination(monkeypatch):
    import qeclang.analysis.heuristic as module

    def forbidden(*args, **kwargs):
        raise AssertionError("basis construction must not run past preflight")

    monkeypatch.setattr(module, "css_logicals", forbidden)
    code = steane()
    for limits in ({"max_basis_bits": 0}, {"max_basis_dimension": 6}):
        result = css_logical_upper_bound(code, "x", **limits)
        assert result.status == "basis_budget_exhausted" and result.certificate is None
        portfolio = css_distance_portfolio(code, "x", **limits)
        assert portfolio.status == "basis_budget_exhausted" and portfolio.enumeration is None


def test_portfolio_bounds_are_rigorous_and_can_close_gap():
    code = improvement_code()
    bounded = css_distance_portfolio(code, "x", seed=2, restarts=0, max_weight=0)
    assert bounded.status == "bounded"
    assert (bounded.lower_bound, bounded.upper_bound) == (1, 2)
    exact = css_distance_portfolio(code, "x", seed=2, restarts=0, max_weight=1)
    assert exact.status == "exact" and exact.distance == 2
    assert exact.enumeration.searched_through == 1
    assert verify_css_logical(code, exact.certificate)
    enumeration_only = css_distance_portfolio(steane(), "z", max_evaluations=0)
    assert enumeration_only.status == "exact" and enumeration_only.distance == 3


def test_numerical_solver_lower_never_becomes_a_rigorous_bound(monkeypatch):
    import qeclang.analysis.portfolio as module

    code = steane()
    witness = css_logicals(code).x[0]
    simulated = MILPDistanceResult("solver_optimal", witness.bit_count(), witness, 99.0, 1, 1, ())
    monkeypatch.setattr(module, "css_distance_milp", lambda *a, **kw: simulated)
    result = css_distance_portfolio(code, "x", max_weight=0, use_milp=True)
    assert result.solver_lower_bound == 99.0
    assert result.lower_bound == 1 and result.status == "bounded"
    assert result.milp_witness_status == "verified"
    assert result.distance is None


@pytest.mark.parametrize("witness,upper", [(0, 0), (1, 1), (1 << 7, 1), (0b1111111, 1)])
def test_portfolio_rejects_invalid_or_misweighted_milp_witnesses(monkeypatch, witness, upper):
    import qeclang.analysis.portfolio as module

    simulated = MILPDistanceResult("solver_optimal", upper, witness, 1.0, 1, 1, ())
    monkeypatch.setattr(module, "css_distance_milp", lambda *a, **kw: simulated)
    result = css_distance_portfolio(steane(), "x", max_weight=0, use_milp=True)
    assert result.milp_witness_status == "rejected"
    assert result.lower_bound == 1 and result.upper_bound == 3
    assert verify_css_logical(steane(), result.certificate)


def test_optional_real_milp_zero_budget_needs_no_solver():
    result = css_distance_portfolio(steane(), "x", max_weight=0, use_milp=True,
                                    milp_time_limit=0)
    assert result.milp_witness_status == "verified"
    assert result.lower_bound == 1 and result.status == "bounded"


def test_invalid_heuristic_and_portfolio_options():
    code = steane()
    for kwargs in ({"seed": 1.0}, {"restarts": -1}, {"max_evaluations": True},
                   {"max_sweeps": -1}, {"max_basis_bits": float("nan")}):
        with pytest.raises(ValueError):
            css_logical_upper_bound(code, "x", **kwargs)
    with pytest.raises(ValueError):
        css_logical_upper_bound(code, "y")
    with pytest.raises(TypeError, match="explicit"):
        css_logical_upper_bound(SparseCSSCode.from_packed(code), "x")
    with pytest.raises(ValueError):
        css_distance_portfolio(code, "x", use_milp=1)
    with pytest.raises(ValueError):
        css_distance_portfolio(code, "x", max_weight=-1)
