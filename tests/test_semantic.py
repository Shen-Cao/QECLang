import pytest

from qeclang.codes import Pauli
from qeclang.constructions import five_qubit, steane

np = pytest.importorskip("numpy")
from qeclang.semantic import IsometryCode, from_stabilizer, pauli_matrix  # noqa: E402


def test_five_qubit_knill_laflamme():
    code = from_stabilizer(five_qubit())
    errors = [np.eye(32)]
    for j in range(5):
        errors.extend(pauli_matrix(p, 5) for p in (
            Pauli(1 << j, 0), Pauli(0, 1 << j), Pauli(1 << j, 1 << j, 1)))
    result = code.check_correctable(errors)
    assert code.logical_dimension == 2
    assert result.satisfied and result.pairs_checked == 256


def test_noncorrectable_and_subsystem():
    code = IsometryCode((2,), np.eye(2))
    assert not code.check_correctable([np.eye(2), np.diag([1, -1])]).satisfied
    gauge = IsometryCode((2, 2), np.eye(4), gauge_dimension=2)
    assert gauge.check_correctable([np.eye(4), np.kron(np.eye(2), np.diag([1, -1]))]).satisfied
    assert not gauge.check_correctable([np.eye(4), np.kron(np.diag([1, -1]), np.eye(2))]).satisfied


def test_dense_guard_and_validation():
    with pytest.raises(ValueError):
        from_stabilizer(steane().to_stabilizer(), max_dimension=16)
    with pytest.raises(ValueError):
        IsometryCode((2,), [[1, 1], [0, 0]])
    with pytest.raises(ValueError):
        IsometryCode((2,), np.eye(2)).check_correctable([])
    with pytest.raises(ValueError):
        IsometryCode((2,), np.eye(2), tolerance=float("nan"))


def test_nonfinite_arithmetic_rejected():
    code = IsometryCode((2,), np.eye(2))
    with pytest.raises(ArithmeticError):
        code.check_correctable([np.diag([1e200, 1e200])])
    with pytest.raises(ValueError):
        from_stabilizer(five_qubit(), max_dimension=-1)
