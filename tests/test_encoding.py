from dataclasses import FrozenInstanceError
import json

import pytest

np = pytest.importorskip("numpy")

from qeclang.semantic import IsometryCode  # noqa: E402
from qeclang.semantic.encoding import encoding_channel, prepare_encoding  # noqa: E402


def subsystem():
    rng = np.random.default_rng(902)
    v, _ = np.linalg.qr(rng.normal(size=(6, 4)) + 1j*rng.normal(size=(6, 4)))
    return IsometryCode((2, 3), v, gauge_dimension=2)


def test_subspace_isometry_channel_and_default_trivial_gauge():
    v = np.array([[1, 0], [0, 1j], [0, 0]])
    code = IsometryCode((3,), v)
    preparation = prepare_encoding(code)
    assert preparation.trivial_gauge_default
    assert preparation.logical_dimension == 2 and preparation.gauge_dimension == 1
    assert len(preparation.channel.kraus) == 1
    assert np.array_equal(preparation.channel.kraus[0], v)
    assert np.array_equal(encoding_channel(code).to_choi(), preparation.channel.to_choi())
    explicit = prepare_encoding(code, [[1]])
    assert not explicit.trivial_gauge_default
    assert np.array_equal(explicit.channel.to_choi(), preparation.channel.to_choi())


@pytest.mark.parametrize("sigma", [np.diag([1., 0.]), np.diag([0., 1.]),
                                    np.diag([.3, .7]), np.array([[.5, -.5j], [.5j, .5]])])
def test_fixed_gauge_agrees_with_independent_tensor_and_partial_trace(sigma):
    code = subsystem()
    prep = prepare_encoding(code, sigma)
    assert prep.channel.input_dim == 2 and prep.channel.output_dim == 6
    assert prep.physical_dimensions == (2, 3)
    for rho in (np.diag([1., 0.]), np.array([[.4, .1j], [-.1j, .6]]),
                np.array([[0, 1+2j], [0, 0]])):
        encoded = prep.channel.apply(rho)
        direct = code.isometry @ np.kron(rho, sigma) @ code.isometry.conj().T
        assert np.allclose(encoded, direct, atol=1e-12, rtol=0)
        decoded_with_gauge = code.isometry.conj().T @ encoded @ code.isometry
        reduced = np.trace(decoded_with_gauge.reshape(2, 2, 2, 2), axis1=1, axis2=3)
        assert np.allclose(reduced, rho, atol=1e-12, rtol=0)
    json.dumps(prep.summary(), allow_nan=False)


def test_gauge_preparation_is_explicit_and_can_change_physical_output():
    code = IsometryCode((2, 2), np.eye(4), gauge_dimension=2)
    with pytest.raises(ValueError, match="explicit gauge_state"):
        prepare_encoding(code)
    with pytest.raises(ValueError, match="explicit gauge_state"):
        encoding_channel(code)
    zero = encoding_channel(code, np.diag([1., 0.]))
    one = encoding_channel(code, np.diag([0., 1.]))
    assert np.array_equal(zero.apply(np.diag([1., 0.])), np.diag([1., 0., 0., 0.]))
    assert np.array_equal(one.apply(np.diag([1., 0.])), np.diag([0., 1., 0., 0.]))


def test_states_and_metadata_are_immutable_copies():
    sigma = np.diag([.2, .8])
    prep = prepare_encoding(subsystem(), sigma)
    summary = prep.summary()
    sigma[0, 0] = .9
    assert prep.input_gauge_state[0, 0] == .2
    for matrix in (prep.input_gauge_state, prep.gauge_state):
        with pytest.raises(ValueError):
            matrix.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        prep.gauge_dimension = 3
    summary["gauge_dimension"] = 99
    assert prep.summary()["gauge_dimension"] == 2
    assert len(summary["input_gauge_sha256"]) == 64


def test_positive_eigenvalues_retained_and_count_guarded(monkeypatch):
    code = IsometryCode((2, 2), np.eye(4), gauge_dimension=2)
    sigma = np.diag([1-1e-14, 1e-14])
    preparation = prepare_encoding(code, sigma)
    assert len(preparation.channel.kraus) == 2
    assert preparation.repair_report.positive_eigenvalues_retained == 2
    assert len(prepare_encoding(code, np.diag([1., 0.]), max_kraus=1).channel.kraus) == 1
    def forbidden(*args, **kwargs):
        raise AssertionError("Kraus contraction must follow count/storage guards")
    monkeypatch.setattr(np, "einsum", forbidden)
    with pytest.raises(ValueError, match="max_kraus"):
        prepare_encoding(code, sigma, max_kraus=1)


def test_explicit_negative_clipping_reports_no_normalization():
    code = IsometryCode((2, 2), np.eye(4), gauge_dimension=2)
    eps = 1e-12
    sigma = np.diag([1+eps, -eps])
    with pytest.raises(ValueError, match="negative"):
        prepare_encoding(code, sigma, tolerance=1e-9)
    with pytest.raises(ValueError, match="negative"):
        prepare_encoding(code, sigma, tolerance=1e-9, repair_tolerance=eps/2)
    prep = prepare_encoding(code, sigma, tolerance=1e-9, repair_tolerance=2*eps)
    assert prep.repair_report.clipped_negative_eigenvalues == 1
    assert prep.repair_report.reconstruction_residual == pytest.approx(eps)
    assert prep.gauge_state[0, 0] > 1
    assert prep.repair_report.prepared_trace_residual > 0
    assert prep.channel.tp_residual > 0
    with pytest.raises(ValueError, match="trace"):
        prepare_encoding(code, np.diag([1.1, -.1]), repair_tolerance=.2)
    with pytest.raises(TypeError):
        encoding_channel(code, sigma, repair_tolerance=2*eps)


def test_asymmetric_gauge_needs_explicit_repair():
    sigma = np.array([[.2, 1e-12j], [0, .8]])
    with pytest.raises(ValueError, match="Hermitian"):
        prepare_encoding(subsystem(), sigma)
    prep = prepare_encoding(subsystem(), sigma, repair_tolerance=1e-10)
    assert prep.repair_report.symmetrized
    assert prep.gauge_diagnostics.hermiticity_residual == pytest.approx(1e-12)
    assert prep.repair_report.clipped_negative_eigenvalues == 0
    assert np.allclose(prep.channel.apply(np.eye(2)/2),
                       subsystem().isometry @ np.kron(np.eye(2)/2, prep.gauge_state)
                       @ subsystem().isometry.conj().T)


def test_full_rank_complex_gauge_and_nontrivial_logical_dimension():
    rng = np.random.default_rng(13)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
    sigma = q @ np.diag([.2, .3, .5]) @ q.conj().T
    code = IsometryCode((3, 3), np.eye(9), gauge_dimension=3)
    prep = prepare_encoding(code, sigma, repair_tolerance=1e-12)
    assert len(prep.channel.kraus) == 3
    rho = np.eye(3)/3
    assert np.allclose(prep.channel.apply(rho), np.kron(rho, sigma))


@pytest.mark.parametrize("sigma", [np.eye(3), np.diag([.2, .7]), np.zeros((2, 2)),
    np.array([[1, 0], [0, float("nan")]]), np.array([.5, .5]),
    [[.5, 0], [0]], [[.5, 1], [0, .5]], np.diag([1.2, -.2])])
def test_malformed_or_invalid_gauge_states(sigma):
    with pytest.raises((ValueError, TypeError)):
        prepare_encoding(subsystem(), sigma)


def test_preallocation_guards_and_revalidation(monkeypatch):
    code = IsometryCode((2, 2), np.eye(4), gauge_dimension=2)
    sigma = np.eye(2)/2
    def forbidden(*args, **kwargs):
        raise AssertionError("eigendecomposition must follow size guards")
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    for options in ({"max_elements": 15}, {"max_dimension": 3},
                    {"max_kraus": 0}, {"tolerance": True}, {"repair_tolerance": -1},
                    {"max_elements": True}):
        with pytest.raises(ValueError):
            prepare_encoding(code, sigma, **options)
    corrupted = code.isometry.copy()
    corrupted[0, 0] = 2
    object.__setattr__(code, "isometry", corrupted)
    with pytest.raises(ValueError, match="isometric"):
        prepare_encoding(code, sigma)


def test_rank_one_logical_subsystem():
    code = IsometryCode((2,), np.eye(2), gauge_dimension=2)
    sigma = np.diag([.4, .6])
    channel = encoding_channel(code, sigma)
    assert channel.input_dim == 1 and channel.output_dim == 2
    assert np.allclose(channel.apply([[1]]), sigma)


def test_trace_invalid_after_repair_rejects_before_kraus_contraction(monkeypatch):
    code = IsometryCode((2, 2), np.eye(4), gauge_dimension=2)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid repaired density must reject before Kraus construction")
    monkeypatch.setattr(np, "einsum", forbidden)
    with pytest.raises(ValueError, match="trace"):
        prepare_encoding(code, np.diag([1.1, -.1]), repair_tolerance=.2)
