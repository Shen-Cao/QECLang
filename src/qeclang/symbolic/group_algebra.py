"""Sparse GF(2) group algebra of finite products of cyclic groups.

Terms are translation exponents. Addition is symmetric difference, so duplicate
monomials cancel. This is exact symbolic algebra, not a distance oracle.
"""

from dataclasses import dataclass
from itertools import product
from math import prod
from operator import index

from qeclang.algebra import BinaryMatrix, SparseBinaryMatrix
from qeclang.algebra.binary import nonnegative_integer


@dataclass(frozen=True)
class GroupAlgebraElement:
    """An element of GF(2)[C_l1 x ... x C_lr] in its monomial basis."""

    orders: tuple[int, ...]
    terms: frozenset[tuple[int, ...]]

    def __post_init__(self):
        orders = tuple(index(v) for v in self.orders)
        if not orders or any(v <= 0 for v in orders):
            raise ValueError("cyclic orders must be positive")
        normalized = set()
        for term in self.terms:
            if len(term) != len(orders):
                raise ValueError("each term must have one exponent per cyclic factor")
            term = tuple(index(v) % order for v, order in zip(term, orders))
            if term in normalized:
                normalized.remove(term)
            else:
                normalized.add(term)
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "terms", frozenset(normalized))

    @classmethod
    def from_terms(cls, orders, terms):
        """Accept a sequence and cancel repetitions before canonicalization."""
        # The constructor deliberately accepts an iterable, preserving repetitions.
        return cls(tuple(orders), (tuple(term) for term in terms))

    @property
    def size(self):
        return prod(self.orders)

    def _compatible(self, other):
        if not isinstance(other, GroupAlgebraElement) or self.orders != other.orders:
            raise ValueError("group algebras must have identical ordered factors")

    def __add__(self, other):
        self._compatible(other)
        return type(self)(self.orders, self.terms ^ other.terms)

    def __mul__(self, other):
        self._compatible(other)
        return type(self).from_terms(self.orders, (
            tuple(a + b for a, b in zip(left, right))
            for left in self.terms for right in other.terms
        ))

    def involution(self):
        """Invert translations; lowering sends this operation to transpose."""
        return type(self).from_terms(self.orders, (tuple(-v for v in t) for t in self.terms))

    def _support_at(self, site):
        """Column coordinates of the regular action at a given group site."""
        support = []
        for term in self.terms:
            column = 0
            for position, shift, order in zip(site, term, self.orders):
                column = column * order + (position + shift) % order
            support.append(column)
        return tuple(support)

    def lower_sparse(self, *, max_nonzeros=10_000_000, max_sites=1_000_000):
        """Regular representation in sparse support rows, without packed lifts.

        ``max_sites`` bounds each output dimension, even for the zero element.
        Distinct normalized translations have disjoint support in each row, so
        the nonzero count is exactly ``size * len(terms)``. Budgets count indices
        and dimensions, not bytes or Python object overhead.
        """
        max_sites = nonnegative_integer(max_sites, "max_sites")
        max_nonzeros = nonnegative_integer(max_nonzeros, "max_nonzeros")
        if self.size > max_sites:
            raise ValueError("regular representation exceeds max_sites")
        if self.size * len(self.terms) > max_nonzeros:
            raise ValueError("regular representation exceeds max_nonzeros")
        if not self.terms:
            return SparseBinaryMatrix.zeros(self.size, self.size)
        rows = (self._support_at(site) for site in product(*(range(order) for order in self.orders)))
        return SparseBinaryMatrix.from_supports(rows, self.size)

    def lower(self, *, max_bits=100_000_000):
        """Lower the regular representation; rows use lexicographic site order.

        max_bits bounds the worst-case bitpacked matrix payload, excluding Python
        object overhead. The matrix remains O(|G|^2) in the worst case.
        """
        max_bits = nonnegative_integer(max_bits, "max_bits")
        if self.size * self.size > max_bits:
            raise ValueError("regular representation exceeds max_bits")
        rows = []
        for site in product(*(range(order) for order in self.orders)):
            rows.append(sum(1 << column for column in self._support_at(site)))
        return BinaryMatrix(tuple(rows), self.size)
