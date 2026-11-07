import json

import pytest

np = pytest.importorskip("numpy")
cp = pytest.importorskip("cvxpy")
from qeclang.recovery import optimize_recovery, recovery_objective  # noqa: E402
from qeclang.semantic import IsometryCode  # noqa: E402
from qeclang.semantic.channels import (KrausChannel, dephasing, entanglement_fidelity,  # noqa: E402
                                      validate_choi)

pytestmark = pytest.mark.skipif("SCS" not in cp.installed_solvers(), reason="optional SCS solver unavailable")


def identity_code(dimension=2):
    return IsometryCode((dimension,), np.eye(dimension))


def assert_optimum(result, expected):
    assert result.status == "numerical_optimal", result.summary()
    assert result.primal_objective == pytest.approx(expected, abs=2e-6)
    assert result.dual_objective == pytest.approx(expected, abs=2e-6)
    assert result.primal_diagnostics.feasible(1e-6)
    assert result.dual_diagnostics.feasible(1e-6)
    assert abs(result.numerical_gap) < 2e-6
    assert validate_choi(result.choi, result.received_dimension,
                         result.logical_dimension, tolerance=1e-6).satisfied
    json.dumps(result.summary(), allow_nan=False)


def test_complex_unitary_recovery_and_independent_kraus_fidelity():
    u = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    noise = KrausChannel((u,))
    result = optimize_recovery(identity_code(), noise)
    assert_optimum(result, 1.0)
    with pytest.raises(ValueError):
        result.choi.flags.writeable = True
    vector = u.conj().T.reshape(-1, order="F")
    assert np.max(np.abs(result.choi - np.outer(vector, vector.conj()))) < 2e-6
    recovery = result.as_channel(repair_tolerance=1e-6, tolerance=1e-6)
    assert recovery.conversion_report is not None
    assert entanglement_fidelity(noise.then(recovery)) == pytest.approx(1, abs=2e-6)


def test_rectangular_encoding_and_complete_erasure():
    code = IsometryCode((3,), np.eye(3)[:, :2])
    result = optimize_recovery(code, KrausChannel((np.eye(3),)))
    assert_optimum(result, 1.0)
    assert result.choi.shape == (6, 6)
    erasure = KrausChannel(([[1, 0]], [[0, 1]]))
    result = optimize_recovery(identity_code(), erasure)
    assert result.received_dimension == 1
    assert_optimum(result, 0.25)
    assert_optimum(optimize_recovery(identity_code(1), KrausChannel((np.eye(1),))), 1)


@pytest.mark.skipif("CLARABEL" not in cp.installed_solvers(), reason="optional comparison solver unavailable")
def test_independent_solver_agreement():
    noise = dephasing(0.2)
    result = optimize_recovery(identity_code(), noise, solver="CLARABEL")
    assert_optimum(result, 0.8)


@pytest.mark.parametrize("probability", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_phase_flip_optimum_is_larger_coset_probability(probability):
    result = optimize_recovery(identity_code(), dephasing(probability))
    assert_optimum(result, max(probability, 1-probability))


def test_exactly_correctable_repetition_error_set():
    encoding = np.zeros((8, 2))
    encoding[0, 0] = encoding[7, 1] = 1
    code = IsometryCode((2, 2, 2), encoding)
    operators = [np.sqrt(0.7)*np.eye(8)]
    for bit in range(3):
        matrix = np.zeros((8, 8))
        for j in range(8):
            matrix[j ^ (1 << bit), j] = np.sqrt(0.1)
        operators.append(matrix)
    noise = KrausChannel(operators)
    assert code.check_correctable(operators).satisfied
    result = optimize_recovery(code, noise)
    assert_optimum(result, 1)


def test_objective_matches_direct_random_complex_rectangular_channels():
    rng = np.random.default_rng(214)
    def isometry(rows, columns):
        q, _ = np.linalg.qr(rng.normal(size=(rows, columns)) + 1j*rng.normal(size=(rows, columns)))
        return q
    v = isometry(4, 2)
    noise_stinespring = isometry(6, 4)
    noise = KrausChannel((noise_stinespring[:3], noise_stinespring[3:]))
    recovery_stinespring = isometry(4, 3)
    recovery = KrausChannel((recovery_stinespring[:2], recovery_stinespring[2:]))
    coefficient = recovery_objective(IsometryCode((4,), v), noise)
    direct = sum(abs(np.trace(r @ n @ v))**2 for r in recovery.kraus for n in noise.kraus) / 4
    choi_value = np.trace(coefficient @ recovery.to_choi())
    assert choi_value.imag == pytest.approx(0, abs=1e-12)
    assert choi_value.real == pytest.approx(direct, abs=1e-12)


def test_explicit_primal_only_and_solver_failure(monkeypatch):
    code, noise = identity_code(), dephasing(0.2)
    result = optimize_recovery(code, noise, solve_dual=False)
    assert result.status == "primal_only"
    assert result.dual_objective is None and result.numerical_gap is None
    def failure(*args, **kwargs):
        raise cp.error.SolverError("test solver failure")
    monkeypatch.setattr(cp.Problem, "solve", failure)
    result = optimize_recovery(code, noise)
    assert result.status == "solver_error"
    assert result.choi is None and result.primal_objective is None
    with pytest.raises(ValueError):
        result.as_channel()


def test_bounded_inaccurate_solver_is_not_certified():
    with pytest.warns(UserWarning, match="inaccurate"):
        result = optimize_recovery(identity_code(), dephasing(0.2), solver_options={"max_iters": 1})
    assert result.status in ("numerical_inaccurate", "invalid_solution", "solver_error")
    assert result.status != "numerical_optimal"


def test_input_scope_and_preallocation_guards(monkeypatch):
    code, noise = identity_code(), dephasing(0.2)
    with pytest.raises(ValueError, match="gauge"):
        optimize_recovery(IsometryCode((2,), np.eye(2), gauge_dimension=2), noise)
    with pytest.raises(ValueError, match="dimension"):
        optimize_recovery(code, KrausChannel((np.eye(3),)))
    def forbidden(*args, **kwargs):
        raise AssertionError("guard must precede solver-variable allocation")
    monkeypatch.setattr(cp, "Variable", forbidden)
    for options in ({"max_choi_dimension": 3}, {"max_elements": 15},
                    {"validation_tolerance": float("nan")}, {"solve_dual": 1},
                    {"solver_options": {"solver": "SCS"}}, {"max_elements": True}):
        with pytest.raises(ValueError):
            optimize_recovery(code, noise, **options)
