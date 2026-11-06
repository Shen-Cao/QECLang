"""Finite-dimensional semantic reference; requires the ``numeric`` extra.

Dense matrices are exponentially expensive for many-site systems. Numerical
residuals here are checks at a chosen tolerance, never formal certificates.
The logical basis index is logical * gauge_dimension + gauge.
"""

from dataclasses import dataclass
from math import prod
from operator import index

import numpy as np


def _tolerance(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError("tolerance must be finite and positive")
    return float(value)


def _dimension_limit(value):
    if isinstance(value, bool):
        raise ValueError("max_dimension must be a positive integer")
    value = index(value)
    if value < 1:
        raise ValueError("max_dimension must be a positive integer")
    return value


def _finite(array):
    if not np.isfinite(array).all():
        raise ArithmeticError("nonfinite numerical intermediate in correctability check")
    return array


@dataclass(frozen=True)
class CorrectabilityResult:
    satisfied: bool
    max_residual: float
    tolerance: float
    pairs_checked: int
    method: str = "numerical operator Knill-Laflamme"


class IsometryCode:
    """Embedding of logical tensor gauge space into a specified physical space.

    gauge_dimension=1 describes a subspace code. The local factorization is
    required because Hamming weight and locality are not properties of a bare
    Hilbert space. Arrays are copied and made read-only.
    """

    def __init__(self, physical_dimensions, isometry, *, gauge_dimension=1, tolerance=1e-10):
        self.physical_dimensions = tuple(index(d) for d in physical_dimensions)
        if not self.physical_dimensions or any(d < 1 for d in self.physical_dimensions):
            raise ValueError("physical dimensions must be positive")
        self.gauge_dimension = index(gauge_dimension)
        self.tolerance = _tolerance(tolerance)
        v = np.array(isometry, dtype=complex, copy=True)
        if v.ndim != 2 or v.shape[0] != prod(self.physical_dimensions):
            raise ValueError("isometry shape does not match physical factorization")
        if self.gauge_dimension < 1 or v.shape[1] < 1 or v.shape[1] % self.gauge_dimension:
            raise ValueError("columns must be a positive multiple of gauge dimension")
        if not np.isfinite(v).all():
            raise ValueError("isometry must be finite")
        if not np.allclose(v.conj().T @ v, np.eye(v.shape[1]), atol=tolerance, rtol=0):
            raise ValueError("columns must be orthonormal")
        v.flags.writeable = False
        self.isometry = v

    @property
    def logical_dimension(self):
        return self.isometry.shape[1] // self.gauge_dimension

    @property
    def projector(self):
        return self.isometry @ self.isometry.conj().T

    def check_correctable(self, errors, *, tolerance=None):
        """Check an error-operator span using V†Ea†EbV = I_L tensor M_ab.

        Operators need not be a trace-preserving Kraus list. Include identity
        when it belongs to the error model. Residual is maximum entry magnitude.
        """
        tolerance = self.tolerance if tolerance is None else _tolerance(tolerance)
        errors = tuple(np.asarray(e, dtype=complex) for e in errors)
        dimension = self.isometry.shape[0]
        if not errors:
            raise ValueError("at least one error operator is required")
        if any(e.shape != (dimension, dimension) or not np.isfinite(e).all() for e in errors):
            raise ValueError("error operators must be finite square physical operators")
        with np.errstate(over="ignore", invalid="ignore"):
            images = [_finite(e @ self.isometry) for e in errors]
        k, g = self.logical_dimension, self.gauge_dimension
        residual = 0.0
        for a in images:
            for b in images:
                with np.errstate(over="ignore", invalid="ignore"):
                    compressed = _finite(a.conj().T @ b)
                    blocks = compressed.reshape(k, g, k, g)
                    gauge = _finite(sum(blocks[i, :, i, :] / k for i in range(k)))
                    expected = _finite(np.kron(np.eye(k), gauge))
                    difference = _finite(np.abs(compressed - expected))
                residual = max(residual, float(np.max(difference)))
        return CorrectabilityResult(residual <= tolerance, residual, tolerance, len(errors)**2)


def pauli_matrix(pauli, n, *, max_dimension=1024):
    """Dense i^phase X^x Z^z, with qubit zero the least significant basis bit."""
    n = index(n)
    max_dimension = _dimension_limit(max_dimension)
    if n < 0 or n > max_dimension.bit_length() or (1 << n) > max_dimension:
        raise ValueError("Pauli matrix exceeds max_dimension")
    dimension = 1 << n
    if pauli.x < 0 or pauli.z < 0 or (pauli.x | pauli.z) >= dimension:
        raise ValueError("Pauli support exceeds n")
    matrix = np.zeros((dimension, dimension), dtype=complex)
    for column in range(dimension):
        matrix[column ^ pauli.x, column] = (1j ** pauli.phase) * (-1)**((column & pauli.z).bit_count())
    return matrix


def from_stabilizer(code, *, max_dimension=1024, tolerance=1e-10):
    """Lower a signed stabilizer to a dense isometry with arbitrary logical basis.

    This preserves its subspace, but does not synthesize a circuit or preserve a
    preferred logical frame. Cost includes a dense eigendecomposition.
    """
    max_dimension = _dimension_limit(max_dimension)
    if code.n > max_dimension.bit_length() or (1 << code.n) > max_dimension:
        raise ValueError("dense lowering exceeds max_dimension")
    dimension = 1 << code.n
    projector = np.eye(dimension, dtype=complex)
    for generator in code.generators:
        projector = (projector + pauli_matrix(generator, code.n,
                     max_dimension=max_dimension) @ projector) / 2
    eigenvalues, vectors = np.linalg.eigh(projector)
    selected = vectors[:, eigenvalues > 0.5]
    if selected.shape[1] != 1 << code.k:
        raise ArithmeticError("numerical projector rank disagrees with exact stabilizer rank")
    return IsometryCode((2,) * code.n or (1,), selected, tolerance=tolerance)
