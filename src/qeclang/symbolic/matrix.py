"""Matrices over a fixed commutative finite abelian GF(2) group algebra."""
from dataclasses import dataclass
from itertools import product

from qeclang.algebra import BinaryMatrix, SparseBinaryMatrix
from qeclang.algebra.binary import nonnegative_integer
from .group_algebra import GroupAlgebraElement


@dataclass(frozen=True)
class GroupAlgebraMatrix:
    """Block matrix retaining ring structure before regular-representation lift.

    An empty row list needs explicit ncols. ``adjoint`` transposes blocks and
    inverts monomials, and therefore lowers to binary transpose.
    """
    rows: tuple[tuple[GroupAlgebraElement, ...], ...]
    ncols: int
    orders: tuple[int, ...]

    def __post_init__(self):
        zero = GroupAlgebraElement.from_terms(self.orders, [])
        width = nonnegative_integer(self.ncols, "ncols")
        rows = tuple(tuple(row) for row in self.rows)
        for row in rows:
            if len(row) != width:
                raise ValueError("ring matrix rows must match ncols")
            if any(not isinstance(e, GroupAlgebraElement) or e.orders != zero.orders for e in row):
                raise ValueError("all entries must belong to the declared group algebra")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "orders", zero.orders)
        object.__setattr__(self, "ncols", width)

    @property
    def nrows(self):
        return len(self.rows)

    @property
    def shape(self):
        return self.nrows, self.ncols

    @property
    def group_size(self):
        return GroupAlgebraElement.from_terms(self.orders, []).size

    @property
    def is_zero(self):
        return not any(e.terms for row in self.rows for e in row)

    @property
    def row_weights(self):
        return tuple(sum(len(e.terms) for e in row) for row in self.rows)

    @property
    def column_weights(self):
        return tuple(sum(len(row[j].terms) for row in self.rows) for j in range(self.ncols))

    @classmethod
    def zeros(cls, nrows, ncols, orders):
        nrows = nonnegative_integer(nrows, "nrows")
        ncols = nonnegative_integer(ncols, "ncols")
        zero = GroupAlgebraElement.from_terms(orders, [])
        if nrows == 0:
            return cls((), ncols, zero.orders)
        return cls(((zero,) * ncols,) * nrows, ncols, zero.orders)

    @classmethod
    def identity(cls, size, orders):
        size = nonnegative_integer(size, "size")
        zero = GroupAlgebraElement.from_terms(orders, [])
        one = GroupAlgebraElement.from_terms(orders, [(0,) * len(zero.orders)])
        return cls(tuple(tuple(one if i == j else zero for j in range(size)) for i in range(size)),
                   size, zero.orders)

    def _compatible(self, other):
        if not isinstance(other, GroupAlgebraMatrix) or self.orders != other.orders:
            raise ValueError("matrix coefficient rings must agree")

    def __add__(self, other):
        self._compatible(other)
        if self.shape != other.shape:
            raise ValueError("matrix shapes must agree")
        return type(self)(tuple(tuple(a + b for a, b in zip(left, right))
                                for left, right in zip(self.rows, other.rows)), self.ncols, self.orders)

    def __matmul__(self, other):
        self._compatible(other)
        if self.ncols != other.nrows:
            raise ValueError("matrix multiplication dimensions do not match")
        zero = GroupAlgebraElement.from_terms(self.orders, [])
        rows = []
        for row in self.rows:
            entries = []
            for j in range(other.ncols):
                value = zero
                for i, a in enumerate(row):
                    value = value + a * other.rows[i][j]
                entries.append(value)
            rows.append(tuple(entries))
        return type(self)(tuple(rows), other.ncols, self.orders)

    def adjoint(self):
        return type(self)(tuple(tuple(self.rows[i][j].involution() for i in range(self.nrows))
                                for j in range(self.ncols)), self.nrows, self.orders)

    def _check_lift_dimensions(self, max_sites):
        max_sites = nonnegative_integer(max_sites, "max_sites")
        length = self.group_size
        if max(self.nrows, self.ncols) * length > max_sites:
            raise ValueError("ring matrix lift exceeds max_sites")
        return length

    def lower_sparse(self, *, max_nonzeros=10_000_000, max_sites=1_000_000):
        """Lift directly to support rows, preserving the packed lift ordering.

        Budgets bound both expanded dimensions and the exact expanded nonzero
        count. Zero blocks allocate no site products or packed temporaries.
        The representation has O(expanded rows + expanded nonzeros) index
        storage; sparse lifting does not supply scalable exact rank computation.
        """
        length = self._check_lift_dimensions(max_sites)
        max_nonzeros = nonnegative_integer(max_nonzeros, "max_nonzeros")
        if length * sum(self.row_weights) > max_nonzeros:
            raise ValueError("ring matrix lift exceeds max_nonzeros")
        rows = []
        for block_row in self.rows:
            nonzero_blocks = tuple((j, e) for j, e in enumerate(block_row) if e.terms)
            if not nonzero_blocks:
                rows.extend(((),) * length)
                continue
            for site in product(*(range(order) for order in self.orders)):
                rows.append(tuple(j * length + column for j, element in nonzero_blocks
                                  for column in element._support_at(site)))
        return SparseBinaryMatrix.from_supports(rows, self.ncols * length)

    def lower(self, *, max_bits=100_000_000, max_sites=1_000_000):
        """Lift blocks, with the group site varying inside each block coordinate."""
        limit = nonnegative_integer(max_bits, "max_bits")
        length = self._check_lift_dimensions(max_sites)
        if self.nrows * self.ncols * length**2 > limit:
            raise ValueError("ring matrix lift exceeds max_bits")
        rows = []
        cache = {}
        for block_row in self.rows:
            expanded = [0] * length
            for j, element in enumerate(block_row):
                if not element.terms:
                    continue
                if element not in cache:
                    cache[element] = element.lower(max_bits=limit)
                for i, row in enumerate(cache[element].rows):
                    expanded[i] |= row << (j * length)
            rows.extend(expanded)
        return BinaryMatrix(tuple(rows), self.ncols * length)


def ring_hstack(left, right):
    left._compatible(right)
    if left.nrows != right.nrows:
        raise ValueError("block row counts must agree")
    return GroupAlgebraMatrix(tuple(a+b for a, b in zip(left.rows, right.rows)),
                              left.ncols + right.ncols, left.orders)


def ring_kron(left, right):
    """Kronecker product over the shared commutative coefficient ring."""
    left._compatible(right)
    return GroupAlgebraMatrix(tuple(tuple(a*b for a in ar for b in br)
                                    for ar in left.rows for br in right.rows),
                              left.ncols * right.ncols, left.orders)
