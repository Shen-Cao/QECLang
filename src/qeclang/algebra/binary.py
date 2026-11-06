"""Exact GF(2) matrices with little-endian, integer-packed rows.

Column ``j`` is bit ``j`` of a row. This backend is compact for moderate
matrices and uses Python's arbitrary-precision XOR; it is not a sparse backend.
"""

from dataclasses import dataclass
from functools import cached_property
from numbers import Integral
from typing import Iterable


def nonnegative_integer(value: int, name: str) -> int:
    """Validate an integer without silently truncating floating-point input."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def validate_vector(value: int, width: int, name: str = "vector") -> int:
    value = nonnegative_integer(value, name)
    if value.bit_length() > width:
        raise ValueError(f"{name} does not fit in {width} bits")
    return value


def set_bits(value: int) -> Iterable[int]:
    """Yield positions of nonzero bits, in increasing order."""
    while value:
        bit = value & -value
        yield bit.bit_length() - 1
        value ^= bit


@dataclass(frozen=True)
class BinaryMatrix:
    """Immutable matrix over GF(2), including explicit empty dimensions."""

    rows: tuple[int, ...]
    ncols: int

    def __post_init__(self) -> None:
        width = nonnegative_integer(self.ncols, "ncols")
        object.__setattr__(self, "ncols", width)
        object.__setattr__(self, "rows", tuple(validate_vector(r, width, "row") for r in self.rows))

    @property
    def nrows(self) -> int:
        return len(self.rows)

    @property
    def shape(self) -> tuple[int, int]:
        return self.nrows, self.ncols

    @classmethod
    def zeros(cls, nrows: int, ncols: int) -> "BinaryMatrix":
        return cls((0,) * nonnegative_integer(nrows, "nrows"), ncols)

    @classmethod
    def identity(cls, size: int) -> "BinaryMatrix":
        size = nonnegative_integer(size, "size")
        return cls(tuple(1 << j for j in range(size)), size)

    @classmethod
    def from_dense(cls, rows: Iterable[Iterable[int]], ncols: int | None = None) -> "BinaryMatrix":
        """Build from binary entries; ``ncols`` disambiguates a zero-row matrix."""
        dense = tuple(tuple(row) for row in rows)
        width = len(dense[0]) if dense else 0
        if ncols is not None:
            width = nonnegative_integer(ncols, "ncols")
        packed = []
        for row in dense:
            if len(row) != width:
                raise ValueError("dense matrix rows must have equal length")
            value = 0
            for j, bit in enumerate(row):
                if not isinstance(bit, Integral) or bit not in (0, 1):
                    raise ValueError("dense matrix entries must be binary integers")
                value |= int(bit) << j
            packed.append(value)
        return cls(tuple(packed), width)

    def to_dense(self) -> list[list[int]]:
        return [[(row >> j) & 1 for j in range(self.ncols)] for row in self.rows]

    @cached_property
    def _elimination(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        rows = list(self.rows)
        pivots: list[int] = []
        lead = 0
        for col in range(self.ncols):
            pivot = next((i for i in range(lead, len(rows)) if (rows[i] >> col) & 1), None)
            if pivot is None:
                continue
            rows[lead], rows[pivot] = rows[pivot], rows[lead]
            for i in range(len(rows)):
                if i != lead and (rows[i] >> col) & 1:
                    rows[i] ^= rows[lead]
            pivots.append(col)
            lead += 1
            if lead == len(rows):
                break
        return tuple(rows), tuple(pivots)

    def rref(self) -> tuple["BinaryMatrix", tuple[int, ...]]:
        """Return reduced row-echelon form and its pivot column indices."""
        rows, pivots = self._elimination
        return BinaryMatrix(rows, self.ncols), pivots

    def rank(self) -> int:
        return len(self._elimination[1])

    def row_basis(self) -> "BinaryMatrix":
        return BinaryMatrix(self._elimination[0][: self.rank()], self.ncols)

    def contains(self, vector: int) -> bool:
        """Whether a packed vector belongs to the row space."""
        vector = validate_vector(vector, self.ncols)
        rows, pivots = self._elimination
        for row, pivot in zip(rows, pivots):
            if (vector >> pivot) & 1:
                vector ^= row
        return vector == 0

    def nullspace(self) -> "BinaryMatrix":
        """Return a row basis for the right kernel of this matrix."""
        reduced, pivots = self._elimination
        pivot_set = set(pivots)
        basis = []
        for free in range(self.ncols):
            if free not in pivot_set:
                value = 1 << free
                for row, pivot in zip(reduced, pivots):
                    if (row >> free) & 1:
                        value |= 1 << pivot
                basis.append(value)
        return BinaryMatrix(tuple(basis), self.ncols)

    def solve(self, rhs: int) -> int:
        """Solve ``self @ x = rhs`` with free coordinates zero.

        Raises ``ValueError`` for inconsistent equations. The right-hand side
        is packed in row order, including redundant equations.
        """
        rhs = validate_vector(rhs, self.nrows, "rhs")
        augmented = BinaryMatrix(
            tuple(row | (((rhs >> i) & 1) << self.ncols) for i, row in enumerate(self.rows)),
            self.ncols + 1,
        )
        rows, pivots = augmented._elimination
        if self.ncols in pivots:
            raise ValueError("linear system is inconsistent")
        return sum(((row >> self.ncols) & 1) << pivot for row, pivot in zip(rows, pivots))

    def matvec(self, vector: int) -> int:
        vector = validate_vector(vector, self.ncols)
        return sum(((row & vector).bit_count() & 1) << i for i, row in enumerate(self.rows))

    def transpose(self) -> "BinaryMatrix":
        columns = [0] * self.ncols
        for i, row in enumerate(self.rows):
            for j in set_bits(row):
                columns[j] |= 1 << i
        return BinaryMatrix(tuple(columns), self.nrows)

    @property
    def T(self) -> "BinaryMatrix":
        return self.transpose()

    def __matmul__(self, other: "BinaryMatrix") -> "BinaryMatrix":
        if not isinstance(other, BinaryMatrix):
            return NotImplemented
        if self.ncols != other.nrows:
            raise ValueError("matrix multiplication dimensions do not match")
        product = []
        for row in self.rows:
            value = 0
            for j in set_bits(row):
                value ^= other.rows[j]
            product.append(value)
        return BinaryMatrix(tuple(product), other.ncols)

    def __xor__(self, other: "BinaryMatrix") -> "BinaryMatrix":
        if not isinstance(other, BinaryMatrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("matrix addition dimensions do not match")
        return BinaryMatrix(tuple(a ^ b for a, b in zip(self.rows, other.rows)), self.ncols)

    __add__ = __xor__

    @property
    def is_zero(self) -> bool:
        return not any(self.rows)

    @property
    def nnz(self) -> int:
        return sum(row.bit_count() for row in self.rows)


def hstack(*matrices: BinaryMatrix) -> BinaryMatrix:
    """Concatenate matrices horizontally, preserving column order."""
    if not matrices:
        raise ValueError("hstack requires at least one matrix")
    if any(m.nrows != matrices[0].nrows for m in matrices):
        raise ValueError("hstack row counts must match")
    rows = [0] * matrices[0].nrows
    offset = 0
    for matrix in matrices:
        for i, row in enumerate(matrix.rows):
            rows[i] |= row << offset
        offset += matrix.ncols
    return BinaryMatrix(tuple(rows), offset)


def vstack(*matrices: BinaryMatrix) -> BinaryMatrix:
    if not matrices:
        raise ValueError("vstack requires at least one matrix")
    if any(m.ncols != matrices[0].ncols for m in matrices):
        raise ValueError("vstack column counts must match")
    return BinaryMatrix(tuple(row for m in matrices for row in m.rows), matrices[0].ncols)


def kron(left: BinaryMatrix, right: BinaryMatrix) -> BinaryMatrix:
    """Kronecker product, with the right index varying fastest."""
    rows = []
    for a in left.rows:
        for b in right.rows:
            value = 0
            for j in set_bits(a):
                value |= b << (j * right.ncols)
            rows.append(value)
    return BinaryMatrix(tuple(rows), left.ncols * right.ncols)
