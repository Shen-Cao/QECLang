"""Finite encoding channels with an explicitly prepared gauge state.

For logical-major columns V[:, logical*G+gauge], this realizes
rho -> V (rho tensor sigma_G) V†. A subsystem requires an explicit sigma_G.
Numerical density validation is not a formal positivity certificate.
"""
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from math import prod, sqrt

import numpy as np

from .channel_validation import (ChannelLimits, finite, matrix_copy, positive_integer,
                                 readonly, tolerance_value)
from .channels import KrausChannel
from .dense import IsometryCode


@dataclass(frozen=True)
class GaugeDiagnostics:
    hermiticity_residual: float
    trace_residual: float
    minimum_eigenvalue: float


@dataclass(frozen=True)
class GaugeRepairReport:
    repair_tolerance: float | None
    symmetrized: bool
    clipped_negative_eigenvalues: int
    reconstruction_residual: float
    prepared_trace_residual: float
    positive_eigenvalues_retained: int


@dataclass(frozen=True)
class EncodingPreparation:
    """Encoding channel plus immutable input/prepared gauge states and evidence.

    gauge_state is the state reconstructed from the retained eigensystem,
    including the numerical reconstruction discrepancy reported below.
    """
    channel: KrausChannel
    logical_dimension: int
    gauge_dimension: int
    physical_dimensions: tuple[int, ...]
    tolerance: float
    isometry_residual: float
    trivial_gauge_default: bool
    gauge_diagnostics: GaugeDiagnostics
    repair_report: GaugeRepairReport
    input_gauge_state: np.ndarray = field(repr=False, compare=False)
    gauge_state: np.ndarray = field(repr=False, compare=False)

    def summary(self):
        """JSON-compatible provenance; state hashes identify numerical arrays."""
        def fingerprint(matrix):
            return sha256(matrix.astype("<c16", copy=False).tobytes(order="C")).hexdigest()
        return {"method": "fixed-gauge encoding preparation",
                "logical_dimension": self.logical_dimension,
                "gauge_dimension": self.gauge_dimension,
                "physical_dimensions": list(self.physical_dimensions),
                "tolerance": self.tolerance, "isometry_residual": self.isometry_residual,
                "trivial_gauge_default": self.trivial_gauge_default,
                "gauge_diagnostics": asdict(self.gauge_diagnostics),
                "repair_report": asdict(self.repair_report),
                "input_gauge_sha256": fingerprint(self.input_gauge_state),
                "prepared_gauge_sha256": fingerprint(self.gauge_state),
                "basis_order": "logical_index*gauge_dimension+gauge_index",
                "channel_tp_residual": self.channel.tp_residual,
                "channel_limits": asdict(self.channel.limits)}


def _encoding_input(code, limits, tolerance):
    if not isinstance(code, IsometryCode):
        raise TypeError("encoding preparation requires IsometryCode")
    gauge = positive_integer(code.gauge_dimension, "gauge dimension")
    physical = tuple(positive_integer(d, "physical dimension") for d in code.physical_dimensions)
    if not physical:
        raise ValueError("physical factorization must be nonempty")
    output_dim = prod(physical)
    if not isinstance(code.isometry, np.ndarray) or code.isometry.ndim != 2:
        raise ValueError("encoding isometry must be a matrix")
    rows, columns = code.isometry.shape
    if rows != output_dim or columns < 1 or columns % gauge or columns > output_dim:
        raise ValueError("isometry dimensions do not match logical/gauge/physical factors")
    logical = columns // gauge
    limits.channel(output_dim, logical, 1)
    limits.array(output_dim, columns)
    limits.array(columns, columns)
    limits.array(gauge, gauge)
    v = matrix_copy(code.isometry, max_elements=limits.max_elements,
                    shape=(output_dim, columns), max_axis=limits.max_dimension)
    with np.errstate(over="raise", invalid="raise"):
        gram = finite(v.conj().T @ v)
        residual = float(np.max(finite(np.abs(gram - np.eye(columns)))))
    if residual > tolerance:
        raise ValueError("encoding columns are not isometric at preparation tolerance")
    return v, physical, logical, gauge, residual


def prepare_encoding(code, gauge_state=None, *, tolerance=1e-10, repair_tolerance=None,
                      max_dimension=1024, max_kraus=4096, max_elements=1_000_000):
    """Prepare a channel from a subspace or explicitly initialized subsystem.

    Strict mode rejects asymmetry and any computed negative gauge eigenvalues.
    Explicit repair permits bounded symmetrization and negative-only clipping;
    every positive eigenvalue remains, and trace is never normalized. Input and
    reconstructed trace must both equal one within tolerance. No maximally mixed
    gauge, gauge reset, or average over unspecified gauge inputs is inferred.

    Size guards cover isometry/Gram/state arrays and total Kraus storage before
    allocation; they exclude eigensolver workspace and peak memory overhead.
    """
    limits = ChannelLimits(max_dimension, max_kraus, max_elements)
    tol = tolerance_value(tolerance)
    repair = None if repair_tolerance is None else tolerance_value(repair_tolerance,
                                                                   "repair_tolerance")
    v, physical, logical, gauge, isometry_residual = _encoding_input(code, limits, tol)
    defaulted = gauge_state is None
    if defaulted:
        if gauge != 1:
            raise ValueError("subsystem encoding requires an explicit gauge_state density matrix")
        gauge_state = [[1]]
    original = matrix_copy(gauge_state, max_elements=limits.max_elements,
                           shape=(gauge, gauge), max_axis=limits.max_dimension)
    with np.errstate(over="raise", invalid="raise"):
        asymmetry = float(np.max(finite(np.abs(finite(original - original.conj().T)))))
        trace_residual = float(abs(finite(np.trace(original)) - 1))
        if trace_residual > tol:
            raise ValueError("gauge state trace must equal one; normalization is not automatic")
        if asymmetry and (repair is None or asymmetry > repair):
            raise ValueError("gauge state is not exactly Hermitian; explicit bounded repair is required")
        hermitian = finite(original/2 + original.conj().T/2) if asymmetry else original
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
        finite(eigenvalues)
        finite(eigenvectors)
        minimum = float(eigenvalues[0])
        negatives = int(np.count_nonzero(eigenvalues < 0))
        if negatives and (repair is None or minimum < -repair):
            raise ValueError("gauge eigenvalues are negative; explicit bounded repair is required")
        positive = tuple((i, float(value)) for i, value in enumerate(eigenvalues) if value > 0)
        limits.channel(v.shape[0], logical, len(positive))
        retained = np.where(eigenvalues > 0, eigenvalues, 0)
        prepared = finite((eigenvectors * retained) @ eigenvectors.conj().T)
        reconstruction = float(np.max(finite(np.abs(finite(prepared - original)))))
        prepared_trace = float(abs(finite(np.trace(prepared)) - 1))
        if prepared_trace > tol:
            raise ValueError("repaired gauge state trace exceeds tolerance; no normalization is performed")
        # Contract the gauge coordinate; no identity tensor gauge-vector is allocated.
        blocks = v.reshape(v.shape[0], logical, gauge)
        operators = tuple(finite(sqrt(value) * np.einsum("dkg,g->dk", blocks,
                                                         eigenvectors[:, i], optimize=False))
                          for i, value in positive)
    channel = KrausChannel(operators, tolerance=tol, max_dimension=limits.max_dimension,
                           max_kraus=limits.max_kraus, max_elements=limits.max_elements)
    diagnostics = GaugeDiagnostics(asymmetry, trace_residual, minimum)
    report = GaugeRepairReport(repair, bool(asymmetry), negatives, reconstruction,
                                prepared_trace, len(positive))
    return EncodingPreparation(channel, logical, gauge, physical, tol, isometry_residual,
                               defaulted, diagnostics, report, readonly(original), readonly(prepared))


def encoding_channel(code, gauge_state=None, *, tolerance=1e-10, max_dimension=1024,
                     max_kraus=4096, max_elements=1_000_000):
    """Strict convenience returning only the KrausChannel.

    Use prepare_encoding for explicit repair, gauge diagnostics and provenance.
    """
    return prepare_encoding(code, gauge_state, tolerance=tolerance,
                            max_dimension=max_dimension, max_kraus=max_kraus,
                            max_elements=max_elements).channel


__all__ = ["GaugeDiagnostics", "GaugeRepairReport", "EncodingPreparation",
           "prepare_encoding", "encoding_channel"]
