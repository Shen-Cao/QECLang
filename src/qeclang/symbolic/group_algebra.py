"""Sparse GF(2) group algebra of finite products of cyclic groups.

Terms are translation exponents. Addition is symmetric difference, so duplicate
monomials cancel. This is exact symbolic algebra, not a distance oracle.
"""

from dataclasses import dataclass
from itertools import product
from math import prod
from operator import index

from qeclang.algebra import BinaryMatrix


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
        return cls(tuple(orders), tuple(tuple(term) for term in terms))

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

    def lower(self, *, max_bits=100_000_000):
        """Lower the regular representation; rows use lexicographic site order.

        max_bits bounds the worst-case bitpacked matrix payload, excluding Python
        object overhead. The matrix remains O(|G|^2) in the worst case.
        """
        if max_bits < 0 or self.size * self.size > max_bits:
            raise ValueError("regular representation exceeds max_bits")
        rows = []
        for site in product(*(range(order) for order in self.orders)):
            row = 0
            for term in self.terms:
                column = 0
                for position, shift, order in zip(site, term, self.orders):
                    column = column * order + (position + shift) % order
                row ^= 1 << column
            rows.append(row)
        return BinaryMatrix(tuple(rows), self.size)
