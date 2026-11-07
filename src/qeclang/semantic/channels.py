"""Finite-dimensional CPTP channels with explicit numerical conversion policies.

Kraus matrices have shape (output_dim, input_dim). Choi matrices use
J = sum vec_F(K) vec_F(K)^dagger, with index input*output_dim + output.
See https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.Choi
and Nielsen, https://arxiv.org/abs/quant-ph/9606012 for fidelity conventions.
"""
from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np

from .channel_validation import (ChannelLimits, ChoiValidation, finite, matrix_copy,
                                 positive_integer, readonly, tolerance_value,
                                 validate_choi)


@dataclass(frozen=True)
class ChoiConversionReport:
    hermitian_residual: float
    minimum_eigenvalue: float
    symmetrized: bool
    clipped_negative_eigenvalues: int
    reconstruction_residual: float
    repair_tolerance: float | None


@dataclass(frozen=True, init=False, eq=False)
class KrausChannel:
    """An immutable, numerically validated completely positive TP map.

    Operators are copied into immutable byte-backed arrays. No normalization,
    Kraus compression, or eigenvalue truncation occurs during construction.
    Numerical TP validation uses max-entry absolute residual and zero relative
    tolerance; complete positivity follows from the Kraus representation.
    """
    kraus: tuple[np.ndarray, ...]
    tolerance: float
    tp_residual: float
    limits: ChannelLimits
    conversion_report: ChoiConversionReport | None

    def __init__(self, kraus, *, tolerance=1e-10, max_dimension=1024,
                 max_kraus=4096, max_elements=1_000_000):
        limits = ChannelLimits(max_dimension, max_kraus, max_elements)
        tol = tolerance_value(tolerance)
        operators, shape, stored_elements = [], None, 0
        for item in kraus:
            if len(operators) >= limits.max_kraus:
                raise ValueError("Kraus count exceeds max_kraus")
            remaining = limits.max_elements - stored_elements
            array = matrix_copy(item, max_elements=remaining, shape=shape,
                                max_axis=limits.max_dimension)
            shape = array.shape
            limits.channel(*shape, len(operators) + 1)
            operators.append(readonly(array))
            stored_elements += array.size
        if not operators:
            raise ValueError("at least one Kraus operator is required")
        with np.errstate(over="raise", invalid="raise"):
            completeness = np.zeros((shape[1], shape[1]), dtype=complex)
            for operator in operators:
                term = finite(operator.conj().T @ operator)
                completeness = finite(completeness + term)
            residual = float(np.max(finite(np.abs(finite(completeness - np.eye(shape[1]))))))
        if residual > tol:
            raise ValueError(f"Kraus operators are not trace preserving: residual {residual}")
        object.__setattr__(self, "kraus", tuple(operators))
        object.__setattr__(self, "tolerance", tol)
        object.__setattr__(self, "tp_residual", residual)
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "conversion_report", None)

    @property
    def input_dim(self):
        return self.kraus[0].shape[1]

    @property
    def output_dim(self):
        return self.kraus[0].shape[0]

    def _options(self, max_dimension, max_kraus, max_elements):
        return ChannelLimits(self.limits.max_dimension if max_dimension is None else max_dimension,
                             self.limits.max_kraus if max_kraus is None else max_kraus,
                             self.limits.max_elements if max_elements is None else max_elements)

    def apply(self, operator, *, max_elements=None):
        """Apply linearly to any finite input operator, without state repair."""
        limit = self.limits.max_elements if max_elements is None else positive_integer(max_elements, "max_elements")
        if max(self.input_dim**2, self.output_dim**2, self.input_dim*self.output_dim) > limit:
            raise ValueError("channel application exceeds max_elements")
        array = matrix_copy(operator, max_elements=limit, shape=(self.input_dim, self.input_dim))
        result = np.zeros((self.output_dim, self.output_dim), dtype=complex)
        with np.errstate(over="raise", invalid="raise"):
            for kraus in self.kraus:
                left = finite(kraus @ array)
                result = finite(result + finite(left @ kraus.conj().T))
        return result

    def then(self, after, *, tolerance=None, max_dimension=None, max_kraus=None,
             max_elements=None):
        """Compose in application order: return after(self(operator))."""
        if not isinstance(after, KrausChannel) or self.output_dim != after.input_dim:
            raise ValueError("channel composition dimensions do not match")
        limits = self._options(max_dimension, max_kraus, max_elements)
        limits.channel(after.output_dim, self.input_dim, len(self.kraus)*len(after.kraus))
        with np.errstate(over="raise", invalid="raise"):
            operators = tuple(finite(b @ a) for a in self.kraus for b in after.kraus)
        return KrausChannel(operators, tolerance=max(self.tolerance, after.tolerance) if tolerance is None else tolerance,
                            max_dimension=limits.max_dimension, max_kraus=limits.max_kraus,
                            max_elements=limits.max_elements)

    def compose(self, before, **options):
        """Return self(before(operator)); equivalently before.then(self)."""
        if not isinstance(before, KrausChannel):
            raise TypeError("composition requires another KrausChannel")
        return before.then(self, **options)

    def tensor(self, other, *, tolerance=None, max_dimension=None, max_kraus=None,
               max_elements=None):
        """Tensor self with other; self is the outer/most-significant factor."""
        if not isinstance(other, KrausChannel):
            raise TypeError("tensor product requires another KrausChannel")
        limits = self._options(max_dimension, max_kraus, max_elements)
        limits.channel(self.output_dim*other.output_dim, self.input_dim*other.input_dim,
                       len(self.kraus)*len(other.kraus))
        with np.errstate(over="raise", invalid="raise"):
            operators = tuple(finite(np.kron(a, b)) for a in self.kraus for b in other.kraus)
        return KrausChannel(operators, tolerance=max(self.tolerance, other.tolerance) if tolerance is None else tolerance,
                            max_dimension=limits.max_dimension, max_kraus=limits.max_kraus,
                            max_elements=limits.max_elements)

    def to_choi(self, *, max_elements=None):
        """Return J on input tensor output, with trace(J)=input_dim."""
        limit = self.limits.max_elements if max_elements is None else positive_integer(max_elements, "max_elements")
        size = self.input_dim * self.output_dim
        if size**2 > limit:
            raise ValueError("Choi matrix exceeds max_elements")
        result = np.zeros((size, size), dtype=complex)
        with np.errstate(over="raise", invalid="raise"):
            for kraus in self.kraus:
                vector = kraus.reshape(-1, order="F")
                result = finite(result + finite(np.outer(vector, vector.conj())))
        return result

    @classmethod
    def from_choi(cls, choi, input_dim, output_dim, *, tolerance=1e-10,
                  repair_tolerance=None, max_dimension=1024, max_kraus=4096,
                  max_elements=1_000_000):
        """Convert input-major Choi evidence using an explicit repair policy.

        By default, require exact Hermiticity and nonnegative computed eigenvalues.
        Setting repair_tolerance explicitly permits bounded symmetrization and
        clipping of negative eigenvalues within that tolerance. Every positive
        eigenvalue is retained, however small. TP is checked, never normalized.
        ``conversion_report`` records the intervention and reconstruction residual.
        """
        din = positive_integer(input_dim, "input_dim")
        dout = positive_integer(output_dim, "output_dim")
        limits = ChannelLimits(max_dimension, max_kraus, max_elements)
        limits.channel(dout, din, 1)
        size = din*dout
        limits.array(size, size)
        tol = tolerance_value(tolerance)
        repair = None if repair_tolerance is None else tolerance_value(repair_tolerance, "repair_tolerance")
        original = matrix_copy(choi, max_elements=limits.max_elements, shape=(size, size))
        with np.errstate(over="raise", invalid="raise"):
            asymmetry = float(np.max(finite(np.abs(finite(original - original.conj().T)))))
            if asymmetry and (repair is None or asymmetry > repair):
                raise ValueError("Choi matrix is not exactly Hermitian; explicit bounded repair is required")
            matrix = finite(original/2 + original.conj().T/2) if asymmetry else original
            values, vectors = np.linalg.eigh(matrix)
            finite(values)
            finite(vectors)
            minimum = float(values[0])
            negatives = int(np.count_nonzero(values < 0))
            if negatives and (repair is None or minimum < -repair):
                raise ValueError("Choi eigenvalues are negative; explicit bounded repair is required")
            positive = [(i, value) for i, value in enumerate(values) if value > 0]
            limits.channel(dout, din, len(positive))
            operators = tuple(finite(sqrt(value) * vectors[:, i].reshape(dout, din, order="F"))
                              for i, value in positive)
        channel = cls(operators, tolerance=tol, max_dimension=limits.max_dimension,
                      max_kraus=limits.max_kraus, max_elements=limits.max_elements)
        with np.errstate(over="raise", invalid="raise"):
            residual = float(np.max(finite(np.abs(finite(channel.to_choi() - original)))))
        report = ChoiConversionReport(asymmetry, minimum, bool(asymmetry), negatives, residual, repair)
        object.__setattr__(channel, "conversion_report", report)
        return channel


def _probability(value):
    value = float(value)
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probability must be finite and in [0, 1]")
    return value


def amplitude_damping(probability, **options):
    """Decay |1> to |0> with probability gamma in [0,1]."""
    p = _probability(probability)
    return KrausChannel(([[1, 0], [0, sqrt(1-p)]], [[0, sqrt(p)], [0, 0]]), **options)


def dephasing(probability, **options):
    """Phase-flip convention: (1-p) rho + p Z rho Z; p=1/2 fully dephases."""
    p = _probability(probability)
    return KrausChannel((sqrt(1-p)*np.eye(2), sqrt(p)*np.diag([1, -1])), **options)


def depolarizing(probability, **options):
    """Total nonidentity Pauli probability p; X,Y,Z each occur with p/3.

    This matches the memory simulator. The completely mixing point is p=3/4.
    """
    p = _probability(probability)
    scale = sqrt(p/3)
    return KrausChannel((sqrt(1-p)*np.eye(2), scale*np.array([[0, 1], [1, 0]]),
                         scale*np.array([[0, -1j], [1j, 0]]), scale*np.diag([1, -1])), **options)


def entanglement_fidelity(channel: KrausChannel) -> float:
    """F_e(I/d, channel)=sum_i |Tr(K_i)/d|^2, relative to the identity map.

    A square logical channel is required. The result is not clipped to [0,1];
    numerical discrepancies remain visible. See https://arxiv.org/abs/quant-ph/9606012.
    """
    if not isinstance(channel, KrausChannel) or channel.input_dim != channel.output_dim:
        raise ValueError("entanglement fidelity requires a square KrausChannel")
    with np.errstate(over="raise", invalid="raise"):
        terms = [finite(np.trace(k)/channel.input_dim) for k in channel.kraus]
        value = float(finite(sum(abs(term)**2 for term in terms)))
    if not 0 <= value <= 1 + channel.tolerance:
        raise ArithmeticError("entanglement fidelity lies outside its numerical tolerance")
    return value


__all__ = ["KrausChannel", "ChannelLimits", "ChoiValidation", "ChoiConversionReport",
           "validate_choi", "amplitude_damping", "dephasing", "depolarizing",
           "entanglement_fidelity"]
