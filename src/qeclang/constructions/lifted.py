"""Commutative group-algebra lifted products with explicit expansion budgets.

This is a restricted lifted-product construction, not the general
noncommutative balanced-product machinery or an asymptotic goodness theorem.
See Panteleev and Kalachev, https://arxiv.org/abs/2012.04068.
"""
from dataclasses import dataclass

from qeclang.algebra.binary import nonnegative_integer
from qeclang.codes import CSSCode
from qeclang.symbolic.matrix import GroupAlgebraMatrix, ring_hstack, ring_kron


@dataclass(frozen=True)
class LiftedLocality:
    max_check_weight: int
    max_qubit_degree: int
    x_nonzeros: int
    z_nonzeros: int


@dataclass(frozen=True)
class LiftedProduct:
    """Construction IR retaining two ring-matrix seeds and their group action."""
    a: GroupAlgebraMatrix
    b: GroupAlgebraMatrix

    def __post_init__(self):
        if not isinstance(self.a, GroupAlgebraMatrix):
            raise TypeError("seed matrices must be GroupAlgebraMatrix")
        self.a._compatible(self.b)

    @property
    def n(self):
        return self.a.group_size * (self.a.ncols*self.b.ncols + self.a.nrows*self.b.nrows)

    @property
    def check_counts(self):
        length = self.a.group_size
        return length*self.a.nrows*self.b.ncols, length*self.a.ncols*self.b.nrows

    def locality(self):
        """Exact expanded incidence statistics computed without a matrix lift.

        Uniform bounds on these weighted seed row/column sums certify locality
        for a parameterized family; a finite instance alone is not a family.
        """
        a, b = self.a, self.b
        ar, ac, br, bc = a.row_weights, a.column_weights, b.row_weights, b.column_weights

        def maximum_sum(left, right):
            return max(left) + max(right) if left and right else 0

        return LiftedLocality(
            max(maximum_sum(ar, bc), maximum_sum(ac, br)),
            max(maximum_sum(ac, bc), maximum_sum(ar, br)),
            a.group_size * (b.ncols * sum(ar) + a.nrows * sum(br)),
            a.group_size * (a.ncols * sum(br) + b.nrows * sum(ar)),
        )

    def symbolic_checks(self, *, max_blocks=1_000_000):
        """Hx and Hz over the ring, satisfying Hx @ Hz.adjoint() = 0."""
        a, b = self.a, self.b
        blocks = ((a.ncols*b.ncols + a.nrows*b.nrows) *
                  (a.nrows*b.ncols + a.ncols*b.nrows))
        max_blocks = nonnegative_integer(max_blocks, "max_blocks")
        if blocks > max_blocks:
            raise ValueError("symbolic checks exceed max_blocks")

        def identity(size):
            if size * size > max_blocks:
                raise ValueError("identity intermediate exceeds max_blocks")
            return GroupAlgebraMatrix.identity(size, a.orders)
        hx = ring_hstack(ring_kron(a, identity(b.ncols)),
                        ring_kron(identity(a.nrows), b.adjoint()))
        hz = ring_hstack(ring_kron(identity(a.ncols), b),
                        ring_kron(a.adjoint(), identity(b.nrows)))
        return hx, hz

    def lower(self, *, max_bits=100_000_000, max_blocks=1_000_000):
        limit = nonnegative_integer(max_bits, "max_bits")
        if self.n * sum(self.check_counts) > limit:
            raise ValueError("lifted product exceeds max_bits")
        hx, hz = self.symbolic_checks(max_blocks=max_blocks)
        return CSSCode(hx.lower(max_bits=limit), hz.lower(max_bits=limit),
                       name="commutative lifted product")

    def sparse_checks(self, *, max_nonzeros=10_000_000, max_sites=1_000_000,
                      max_blocks=1_000_000):
        """Return sparse (Hx, Hz) without packed expansion or rank analysis.

        Commutation follows from the symbolic product identity. Consumers
        needing exact dimension analysis must explicitly lower to packed form.
        """
        nonzeros = nonnegative_integer(max_nonzeros, "max_nonzeros")
        sites = nonnegative_integer(max_sites, "max_sites")
        profile = self.locality()
        if profile.x_nonzeros + profile.z_nonzeros > nonzeros:
            raise ValueError("sparse lifted product exceeds max_nonzeros")
        if max(self.n, *self.check_counts) > sites:
            raise ValueError("sparse lifted product exceeds max_sites")
        hx, hz = self.symbolic_checks(max_blocks=max_blocks)
        return (hx.lower_sparse(max_nonzeros=nonzeros, max_sites=sites),
                hz.lower_sparse(max_nonzeros=nonzeros, max_sites=sites))
