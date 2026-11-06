"""CSS constructions whose commutation follows from algebraic identities."""

from qeclang.algebra import BinaryMatrix, hstack, kron
from qeclang.algebra.binary import nonnegative_integer
from qeclang.codes import CSSCode
from qeclang.symbolic import GroupAlgebraElement


def hypergraph_product(a: BinaryMatrix, b: BinaryMatrix, *, max_bits=100_000_000):
    """Tillich--Zemor product in the (VV, CC) qubit ordering.

    n = n_a n_b + m_a m_b; k = k_a k_b + k_a^T k_b^T.
    Checks may be redundant, and are retained to preserve locality.
    """
    n = a.ncols * b.ncols + a.nrows * b.nrows
    max_bits = nonnegative_integer(max_bits, "max_bits")
    checks = a.nrows * b.ncols + a.ncols * b.nrows
    if n * checks > max_bits:
        raise ValueError("hypergraph product exceeds max_bits")
    if max(a.nrows, a.ncols, b.nrows, b.ncols)**2 > max_bits:
        raise ValueError("hypergraph identity intermediate exceeds max_bits")
    hx = hstack(kron(a, BinaryMatrix.identity(b.ncols)),
                kron(BinaryMatrix.identity(a.nrows), b.T))
    hz = hstack(kron(BinaryMatrix.identity(a.ncols), b),
                kron(a.T, BinaryMatrix.identity(b.nrows)))
    return CSSCode(hx, hz, name="hypergraph product")


def bicycle(a: GroupAlgebraElement, b: GroupAlgebraElement, *, max_bits=100_000_000):
    """Generalized bicycle over an abelian group: Hx=[A B], Hz=[B^T A^T].

    AB=BA is guaranteed by the group algebra. This includes univariate and
    bivariate bicycle presentations, without asserting their distance.
    """
    a._compatible(b)
    max_bits = nonnegative_integer(max_bits, "max_bits")
    if 4 * a.size * a.size > max_bits:
        raise ValueError("bicycle checks exceed max_bits")
    am, bm = a.lower(max_bits=max_bits), b.lower(max_bits=max_bits)
    return CSSCode(hstack(am, bm), hstack(bm.T, am.T), name="abelian bicycle")


def direct_sum(a: CSSCode, b: CSSCode):
    """Disjoint union, preserving the supplied check generators."""
    def block(left, right):
        return BinaryMatrix(left.rows + tuple(row << left.ncols for row in right.rows),
                            left.ncols + right.ncols)
    return CSSCode(block(a.hx, b.hx), block(a.hz, b.hz), name="direct sum")
