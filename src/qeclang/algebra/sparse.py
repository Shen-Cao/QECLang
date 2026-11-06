"""Immutable sparse binary rows and explicit interoperability boundaries.

This representation stores O(nrows + nnz) integer indices. It supports sparse
syndrome products and graph construction; exact elimination is available only
after an explicit, resource-guarded lowering to ``BinaryMatrix``.
"""

from dataclasses import dataclass
from numbers import Integral

from .binary import BinaryMatrix, nonnegative_integer, set_bits, validate_vector


def _binary(value) -> int:
    if not isinstance(value, Integral) or value not in (0, 1):
        raise ValueError("coefficients must be binary integers")
    return int(value)


@dataclass(frozen=True)
class SparseBinaryMatrix:
    """Rows of sorted, unique support indices over GF(2).

    Constructor/from_supports input must contain no repeated site in a row.
    Use ``from_coo`` when duplicate coordinates should add modulo two.
    """

    rows: tuple[tuple[int, ...], ...]
    ncols: int

    def __post_init__(self):
        width = nonnegative_integer(self.ncols, "ncols")
        normalized = []
        for row in self.rows:
            support = tuple(sorted(nonnegative_integer(j, "column index") for j in row))
            if support and support[-1] >= width:
                raise ValueError("column index exceeds matrix dimensions")
            if len(support) != len(set(support)):
                raise ValueError("support rows must have distinct column indices; use from_coo for cancellation")
            normalized.append(support)
        object.__setattr__(self, "rows", tuple(normalized))
        object.__setattr__(self, "ncols", width)

    @property
    def nrows(self) -> int:
        return len(self.rows)

    @property
    def shape(self) -> tuple[int, int]:
        return self.nrows, self.ncols

    @property
    def nnz(self) -> int:
        return sum(len(row) for row in self.rows)

    @property
    def is_zero(self) -> bool:
        return not any(self.rows)

    @property
    def row_weights(self) -> tuple[int, ...]:
        return tuple(map(len, self.rows))

    @property
    def column_weights(self) -> tuple[int, ...]:
        weights = [0] * self.ncols
        for row in self.rows:
            for j in row:
                weights[j] += 1
        return tuple(weights)

    @classmethod
    def from_supports(cls, rows, ncols: int) -> "SparseBinaryMatrix":
        return cls(tuple(tuple(row) for row in rows), ncols)

    @classmethod
    def zeros(cls, nrows: int, ncols: int) -> "SparseBinaryMatrix":
        return cls(((),) * nonnegative_integer(nrows, "nrows"), ncols)

    @classmethod
    def from_coo(cls, row_indices, column_indices, data=None, *, shape) -> "SparseBinaryMatrix":
        """Build from coordinate entries, cancelling duplicate 1s over GF(2).

        ``shape`` is ``(nrows,ncols)``; absent ``data`` means all coefficients
        are one. Coefficients must be integer zero/one, not floating-point or
        arbitrary integers. Zero coefficients are checked and then discarded.
        """
        if len(shape) != 2:
            raise ValueError("shape must contain two dimensions")
        nrows = nonnegative_integer(shape[0], "nrows")
        ncols = nonnegative_integer(shape[1], "ncols")
        row_indices, column_indices = tuple(row_indices), tuple(column_indices)
        data = (1,) * len(row_indices) if data is None else tuple(data)
        if len(row_indices) != len(column_indices) or len(row_indices) != len(data):
            raise ValueError("COO coordinate and coefficient lengths must match")
        rows = [set() for _ in range(nrows)]
        for i, j, coefficient in zip(row_indices, column_indices, data):
            i, j = nonnegative_integer(i, "row index"), nonnegative_integer(j, "column index")
            if i >= nrows or j >= ncols:
                raise ValueError("COO coordinate exceeds matrix dimensions")
            if _binary(coefficient):
                if j in rows[i]:
                    rows[i].remove(j)
                else:
                    rows[i].add(j)
        return cls.from_supports(rows, ncols)

    @classmethod
    def from_packed(cls, matrix: BinaryMatrix) -> "SparseBinaryMatrix":
        if not isinstance(matrix, BinaryMatrix):
            raise TypeError("from_packed requires a BinaryMatrix")
        return cls(tuple(tuple(set_bits(row)) for row in matrix.rows), matrix.ncols)

    def to_packed(self, *, max_bits: int = 100_000_000) -> BinaryMatrix:
        """Materialize packed rows with a worst-case payload-bit budget.

        The guard excludes Python object overhead and does not claim subsequent
        Gaussian elimination is sparse or computationally inexpensive.
        """
        max_bits = nonnegative_integer(max_bits, "max_bits")
        if self.nrows * self.ncols > max_bits:
            raise ValueError("packed matrix exceeds max_bits")
        return BinaryMatrix(tuple(sum(1 << j for j in row) for row in self.rows), self.ncols)

    def matvec(self, vector: int) -> int:
        """Packed right-hand-side product; Python bigint bit-access costs apply."""
        vector = validate_vector(vector, self.ncols)
        return sum((sum((vector >> j) & 1 for j in row) & 1) << i
                   for i, row in enumerate(self.rows))

    def matvec_bits(self, vector) -> tuple[int, ...]:
        """Binary sequence product in O(ncols + nrows + nnz) scalar operations."""
        vector = tuple(_binary(value) for value in vector)
        if len(vector) != self.ncols:
            raise ValueError("vector length does not match column count")
        return tuple(sum(vector[j] for j in row) & 1 for row in self.rows)

    def transpose(self) -> "SparseBinaryMatrix":
        rows = [[] for _ in range(self.ncols)]
        for i, row in enumerate(self.rows):
            for j in row:
                rows[j].append(i)
        return type(self).from_supports(rows, self.nrows)

    @property
    def T(self) -> "SparseBinaryMatrix":
        return self.transpose()

    def to_scipy(self, *, format: str = "csr"):
        """Create a SciPy sparse array with uint8 coefficients (optional extra)."""
        import numpy as np
        from scipy.sparse import coo_array

        rows = [i for i, support in enumerate(self.rows) for _ in support]
        columns = [j for support in self.rows for j in support]
        matrix = coo_array((np.ones(self.nnz, dtype=np.uint8),
                            (np.array(rows, dtype=np.int64), np.array(columns, dtype=np.int64))),
                           shape=self.shape)
        return matrix.asformat(format)

    @classmethod
    def from_scipy(cls, matrix) -> "SparseBinaryMatrix":
        """Import stored integer/bool 0/1 entries; duplicates cancel over GF(2).

        Float dtypes are rejected even when entries happen to be zero/one.
        Values previously summed to integers greater than one by SciPy are also
        rejected. Use original binary COO entries for explicit GF(2) addition.
        """
        from scipy.sparse import issparse

        if not issparse(matrix) or len(matrix.shape) != 2:
            raise TypeError("from_scipy requires a two-dimensional SciPy sparse object")
        if matrix.dtype.kind not in "biu":
            raise ValueError("SciPy coefficients must have integer or boolean dtype")
        coo = matrix.tocoo(copy=True)
        return cls.from_coo(coo.row, coo.col, coo.data.tolist(), shape=coo.shape)


__all__ = ["SparseBinaryMatrix"]
