from dataclasses import FrozenInstanceError
from math import sqrt

import pytest
np = pytest.importorskip('numpy')

from qeclang.semantic.channels import (KrausChannel, amplitude_damping, dephasing,  # noqa: E402
                                      depolarizing, entanglement_fidelity, validate_choi)


def test_immutable_kraus_and_dimension_conventions():
    operator = np.eye(2, dtype=complex)
    channel = KrausChannel((operator,))
    operator[0, 0] = 0
    assert channel.input_dim == channel.output_dim == 2
    assert channel.tp_residual == 0
    assert channel.kraus[0][0, 0] == 1
    with pytest.raises(ValueError):
        channel.kraus[0][0, 0] = 2
    with pytest.raises(ValueError):
        channel.kraus[0].flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        channel.tolerance = 1
    assert np.allclose(channel.apply([[0, 1j], [0, 0]]), [[0, 1j], [0, 0]])


def test_rectangular_choi_is_input_major_column_vectorization():
    isometry = np.array([[1, 0], [0, 0], [0, 1j]])
    channel = KrausChannel((isometry,))
    choi = channel.to_choi()
    vector = isometry.reshape(-1, order='F')
    assert choi.shape == (6, 6)
    assert np.array_equal(choi, np.outer(vector, vector.conj()))
    blocks = choi.reshape(2, 3, 2, 3)
    for i in range(2):
        for j in range(2):
            matrix_unit = np.zeros((2, 2), complex)
            matrix_unit[i, j] = 1
            assert np.allclose(blocks[i, :, j, :], channel.apply(matrix_unit))
    assert np.allclose(np.trace(blocks, axis1=1, axis2=3), np.eye(2))
    assert validate_choi(choi, 2, 3).satisfied
    reconstructed = KrausChannel.from_choi(choi, 2, 3, repair_tolerance=1e-12)
    assert np.allclose(reconstructed.to_choi(), choi)
    with pytest.raises(ValueError, match='square'):
        entanglement_fidelity(channel)


def test_noise_probability_conventions_and_fidelities():
    one = np.diag([0, 1])
    plus = np.full((2, 2), 0.5)
    for p in (0, 0.2, 0.5, 0.75, 1):
        damping = amplitude_damping(p)
        phase = dephasing(p)
        depol = depolarizing(p)
        assert np.allclose(damping.apply(one), np.diag([p, 1-p]))
        assert np.allclose(phase.apply(plus), [[0.5, 0.5-p], [0.5-p, 0.5]])
        assert np.allclose(depol.apply(one), np.diag([2*p/3, 1-2*p/3]))
        assert entanglement_fidelity(damping) == pytest.approx((1+sqrt(1-p))**2/4)
        assert entanglement_fidelity(phase) == pytest.approx(1-p)
        assert entanglement_fidelity(depol) == pytest.approx(1-p)
    assert np.allclose(depolarizing(.75).apply(plus), np.eye(2)/2)


def test_composition_order_and_tensor_factor_order():
    flip = KrausChannel((np.array([[0, 1], [1, 0]]),))
    reset = amplitude_damping(1)
    state = np.diag([1, 0])
    assert np.allclose(flip.then(reset).apply(state), state)
    assert np.allclose(reset.then(flip).apply(state), np.diag([0, 1]))
    assert np.allclose(reset.compose(flip).to_choi(), flip.then(reset).to_choi())
    identity = KrausChannel((np.eye(2),))
    outer_flip = flip.tensor(identity)
    inner_flip = identity.tensor(flip)
    zero_zero = np.diag([1, 0, 0, 0])
    assert np.allclose(outer_flip.apply(zero_zero), np.diag([0, 0, 1, 0]))
    assert np.allclose(inner_flip.apply(zero_zero), np.diag([0, 1, 0, 0]))
    first, second = depolarizing(.2), amplitude_damping(.3)
    assert entanglement_fidelity(first.tensor(second)) == pytest.approx(
        entanglement_fidelity(first)*entanglement_fidelity(second))
    rectangular = KrausChannel((np.array([[1], [0]]),))
    with pytest.raises(ValueError, match='dimensions'):
        first.then(rectangular)


def test_choi_roundtrip_and_positive_eigenvalues_are_not_truncated():
    delta = 1e-14
    choi = np.diag([1-delta, delta, delta, 1-delta])
    channel = KrausChannel.from_choi(choi, 2, 2)
    assert len(channel.kraus) == 4
    assert np.allclose(channel.to_choi(), choi, atol=1e-16, rtol=0)
    report = channel.conversion_report
    assert report.minimum_eigenvalue == delta
    assert not report.symmetrized and report.clipped_negative_eigenvalues == 0
    assert report.repair_tolerance is None
    with pytest.raises(ValueError, match='max_kraus'):
        KrausChannel.from_choi(choi, 2, 2, max_kraus=3)


def test_negative_eigenvalue_repair_is_explicit_and_not_tp_normalization():
    eps = 1e-12
    choi = np.diag([1+eps, -eps, 0, 1])
    assert validate_choi(choi, 2, 2, tolerance=1e-9).satisfied
    with pytest.raises(ValueError, match='negative'):
        KrausChannel.from_choi(choi, 2, 2, tolerance=1e-9)
    with pytest.raises(ValueError, match='negative'):
        KrausChannel.from_choi(choi, 2, 2, tolerance=1e-9, repair_tolerance=eps/2)
    channel = KrausChannel.from_choi(choi, 2, 2, tolerance=1e-9, repair_tolerance=2*eps)
    assert channel.conversion_report.clipped_negative_eigenvalues == 1
    assert channel.conversion_report.reconstruction_residual == pytest.approx(eps)
    assert channel.tp_residual > 0
    assert channel.to_choi()[0, 0] > 1
    assert not validate_choi(np.diag([2, -1, 0, 1]), 2, 2).satisfied


def test_asymmetry_requires_explicit_repair():
    choi = depolarizing(.3).to_choi()
    choi[0, 1] = 1e-12j
    before = choi.copy()
    with pytest.raises(ValueError, match='Hermitian'):
        KrausChannel.from_choi(choi, 2, 2)
    channel = KrausChannel.from_choi(choi, 2, 2, repair_tolerance=1e-10)
    assert channel.conversion_report.symmetrized
    assert channel.conversion_report.hermitian_residual == pytest.approx(1e-12)
    assert np.array_equal(choi, before)


def test_invalid_channels_states_and_nonfinite_intermediates():
    for ops in ((), (np.zeros((2, 2)),), (np.eye(2), np.eye(3)),
                (np.array([[float('nan')]]),), (np.ones((2, 2, 2)),)):
        with pytest.raises((ValueError, TypeError)):
            KrausChannel(ops)
    with pytest.raises(ArithmeticError):
        KrausChannel((np.eye(2)*1e200,))
    hadamard = KrausChannel((np.array([[1, 1], [1, -1]])/sqrt(2),))
    with pytest.raises(ArithmeticError):
        hadamard.apply(np.full((2, 2), 1e308))
    with pytest.raises(ValueError):
        hadamard.apply(np.eye(3))
    with pytest.raises(ValueError):
        validate_choi(np.full((4, 4), np.nan), 2, 2)
    with pytest.raises(ArithmeticError):
        validate_choi(np.full((4, 4), 1e308), 2, 2)
    with pytest.raises(ArithmeticError):
        KrausChannel.from_choi(np.full((4, 4), 1e308), 2, 2)
    with pytest.raises(ValueError, match='trace preserving'):
        KrausChannel.from_choi(np.eye(4), 2, 2)


def test_allocation_guards_precede_expansion(monkeypatch):
    channel = amplitude_damping(.2)
    with pytest.raises(ValueError):
        KrausChannel((np.eye(2),), max_elements=3)
    with pytest.raises(ValueError):
        KrausChannel((np.eye(2),), max_dimension=1)
    with pytest.raises(ValueError):
        amplitude_damping(.2, max_kraus=1)
    with pytest.raises(ValueError):
        channel.to_choi(max_elements=15)
    with pytest.raises(ValueError):
        channel.apply(np.eye(2), max_elements=3)
    with pytest.raises(ValueError):
        channel.then(channel, max_kraus=3)
    with pytest.raises(ValueError):
        KrausChannel.from_choi(np.eye(4), 2, 2, max_elements=15)
    def forbidden(*args, **kwargs):
        raise AssertionError('tensor expanded before guard')
    monkeypatch.setattr(np, 'kron', forbidden)
    with pytest.raises(ValueError, match='max_dimension'):
        channel.tensor(channel, max_dimension=3)
    with pytest.raises(ValueError, match='max_elements'):
        channel.tensor(channel, max_elements=63)


@pytest.mark.parametrize('p', [-0.1, 1.1, float('nan'), float('inf')])
def test_invalid_noise_probabilities(p):
    for factory in (amplitude_damping, dephasing, depolarizing):
        with pytest.raises(ValueError):
            factory(p)


def test_public_limits_and_tolerances_reject_invalid_types():
    from qeclang.semantic.channels import ChannelLimits
    limits = ChannelLimits()
    for shape in ((-1, 2, 1), (2, 0, 1), (2, 2, -1), (2, 2, True), (2, 2, 1.5)):
        with pytest.raises(ValueError):
            limits.channel(*shape)
    with pytest.raises(ValueError):
        limits.array(-1, 2)
    with pytest.raises(ValueError):
        KrausChannel((np.eye(2),), tolerance=True)
    with pytest.raises(ValueError):
        validate_choi(np.eye(4), 2, 2, tolerance=True)
