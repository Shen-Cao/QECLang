"""Numerical channel validation with explicit dimensions and allocation guards."""
from dataclasses import dataclass
from math import isfinite
from operator import index

import numpy as np


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def tolerance_value(value, name="tolerance"):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class ChannelLimits:
    """Bounds on dimensions, Kraus count, and stored/operator array elements.

    max_elements limits total Kraus entries and each individual working array;
    it is not a peak-memory bound and excludes Python/LAPACK workspace overhead.
    """
    max_dimension: int = 1024
    max_kraus: int = 4096
    max_elements: int = 1_000_000

    def __post_init__(self):
        for name in ("max_dimension", "max_kraus", "max_elements"):
            object.__setattr__(self, name, positive_integer(getattr(self, name), name))

    def channel(self, output_dim, input_dim, count):
        output_dim = positive_integer(output_dim, "output_dim")
        input_dim = positive_integer(input_dim, "input_dim")
        if isinstance(count, (bool, np.bool_)):
            raise ValueError("Kraus count must be a nonnegative integer")
        try:
            count = index(count)
        except TypeError as exc:
            raise ValueError("Kraus count must be a nonnegative integer") from exc
        if count < 0:
            raise ValueError("Kraus count must be a nonnegative integer")
        if max(output_dim, input_dim) > self.max_dimension:
            raise ValueError("channel dimension exceeds max_dimension")
        if count > self.max_kraus:
            raise ValueError("Kraus count exceeds max_kraus")
        if max(count * output_dim * input_dim, input_dim**2) > self.max_elements:
            raise ValueError("channel storage or TP workspace exceeds max_elements")

    def array(self, rows, columns):
        rows = positive_integer(rows, "rows")
        columns = positive_integer(columns, "columns")
        if rows * columns > self.max_elements:
            raise ValueError("channel array exceeds max_elements")


def finite(value):
    if not np.isfinite(value).all():
        raise ArithmeticError("nonfinite numerical intermediate in channel operation")
    return value


def matrix_copy(value, *, max_elements, shape=None, max_axis=None):
    """Inspect array/row-sequence shape before making a complex128 copy."""
    if isinstance(value, np.ndarray):
        actual = value.shape
    elif isinstance(value, (tuple, list)):
        rows = len(value)
        if max_axis is not None and rows > max_axis:
            raise ValueError("matrix dimension exceeds limit")
        try:
            columns = len(value[0]) if rows else 0
            if rows * columns > max_elements:
                raise ValueError("matrix exceeds max_elements")
            if any(len(row) != columns for row in value):
                raise ValueError("matrix rows must have equal lengths")
        except TypeError as exc:
            raise ValueError("matrix must be a two-dimensional row sequence") from exc
        actual = (rows, columns)
    else:
        raise TypeError("matrix must be a NumPy array or a sequence of rows")
    if len(actual) != 2 or min(actual) < 1 or (shape is not None and actual != shape):
        raise ValueError("matrix has an invalid or incompatible shape")
    if max_axis is not None and max(actual) > max_axis:
        raise ValueError("matrix dimension exceeds limit")
    if actual[0] * actual[1] > max_elements:
        raise ValueError("matrix exceeds max_elements")
    with np.errstate(over="raise", invalid="raise"):
        array = np.array(value, dtype=np.complex128, copy=True)
    if array.shape != actual or not np.isfinite(array).all():
        raise ValueError("matrix entries must be finite with the declared shape")
    return array


def readonly(array):
    """Use immutable byte backing so writeability cannot simply be re-enabled."""
    return np.frombuffer(array.tobytes(order="C"), dtype=np.complex128).reshape(array.shape)


@dataclass(frozen=True)
class ChoiValidation:
    satisfied: bool
    input_dim: int
    output_dim: int
    tolerance: float
    hermitian_residual: float
    minimum_eigenvalue: float
    tp_residual: float


def validate_choi(choi, input_dim, output_dim, *, tolerance=1e-10,
                  max_dimension=1024, max_elements=1_000_000) -> ChoiValidation:
    """Numerical CPTP diagnostics for input-major J[i,a,j,b].

    The eigenvalue diagnostic uses the Hermitian part, while asymmetry is
    reported separately. This function does not modify, repair, or normalize
    the caller's matrix. Numerical acceptance is at the supplied tolerance.
    """
    din = positive_integer(input_dim, "input_dim")
    dout = positive_integer(output_dim, "output_dim")
    limits = ChannelLimits(max_dimension=max_dimension, max_elements=max_elements)
    limits.channel(dout, din, 1)
    size = din * dout
    limits.array(size, size)
    tol = tolerance_value(tolerance)
    matrix = matrix_copy(choi, max_elements=limits.max_elements, shape=(size, size))
    with np.errstate(over="raise", invalid="raise"):
        asymmetry = float(np.max(finite(np.abs(finite(matrix - matrix.conj().T)))))
        hermitian = finite(matrix / 2 + matrix.conj().T / 2)
        eigenvalues = finite(np.linalg.eigvalsh(hermitian))
        trace_output = finite(np.trace(matrix.reshape(din, dout, din, dout), axis1=1, axis2=3))
        tp = float(np.max(finite(np.abs(finite(trace_output - np.eye(din))))))
    minimum = float(eigenvalues[0])
    return ChoiValidation(asymmetry <= tol and minimum >= -tol and tp <= tol,
                           din, dout, tol, asymmetry, minimum, tp)
