"""Exact ranks for odd finite abelian lifts via Frobenius character orbits.

A splitting field diagonalizes the commuting regular shifts. Frobenius-related
characters have equal component rank, so one rank per orbit suffices. This
computes dimensions; the Fourier basis does not preserve physical weights.
See https://arxiv.org/html/2012.04068#A2 for semisimple group-algebra decomposition.
"""
from dataclasses import dataclass
from itertools import product
from math import lcm, prod

from qeclang.algebra.binary import nonnegative_integer
from qeclang.algebra.finite_field import BinaryExtensionField
from qeclang.algebra.polynomial import ArithmeticBudget, WorkLimitExceeded, is_irreducible
from .matrix import GroupAlgebraMatrix


@dataclass(frozen=True)
class AbelianRankComponent:
    character: tuple[int, ...]
    orbit_size: int
    rank: int


@dataclass(frozen=True)
class AbelianRankResult:
    orders: tuple[int, ...]
    shape: tuple[int, int]
    splitting_modulus: int
    primitive_root: int
    root_order: int
    components: tuple[AbelianRankComponent, ...]
    rank: int
    steps: int
    method: str = "exact odd-abelian Frobenius character decomposition"
    evidence: str = "verified splitting field/root order, complete character orbits, exact component ranks"

    @property
    def binary_shape(self):
        size = prod(self.orders)
        return self.shape[0]*size, self.shape[1]*size


def _prime_factors(value, budget):
    factors = []
    candidate = 2
    while candidate*candidate <= value:
        budget.charge()
        if value % candidate == 0:
            factors.append(candidate)
            while value % candidate == 0:
                budget.charge()
                value //= candidate
        candidate += 1
    if value > 1:
        factors.append(value)
    return factors


def _splitting_field(exponent, max_degree, max_candidates, budget):
    degree, power = 1, 2 % exponent if exponent > 1 else 1
    while power != 1:
        budget.charge()
        degree += 1
        if degree > max_degree:
            raise WorkLimitExceeded("splitting field exceeds max_field_degree")
        power = 2*power % exponent
    if degree > max_degree:
        raise WorkLimitExceeded("splitting field exceeds max_field_degree")
    first = (1 << degree) | 1
    for attempt in range(min(max_candidates, 1 << max(0, degree-1))):
        budget.charge()
        modulus = first + 2*attempt
        if is_irreducible(modulus, max_degree=max_degree, budget=budget):
            return BinaryExtensionField(modulus, max_degree=max_degree, budget=budget)
    raise WorkLimitExceeded("irreducible polynomial search exceeds max_candidates")


def _root(field, exponent, max_candidates, budget):
    if exponent == 1:
        return 1
    primes = _prime_factors(exponent, budget)
    for candidate in range(2, min(field.order, max_candidates+2)):
        budget.charge()
        root = field.pow(candidate, (field.order-1)//exponent, budget=budget)
        if (field.pow(root, exponent, budget=budget) == 1
                and all(field.pow(root, exponent//p, budget=budget) != 1 for p in primes)):
            return root
    raise WorkLimitExceeded("root search exceeds max_candidates")


def _orbits(orders, budget):
    visited = set()
    for character in product(*(range(order) for order in orders)):
        budget.charge()
        if character in visited:
            continue
        member, count = character, 0
        while member not in visited:
            budget.charge()
            visited.add(member)
            count += 1
            member = tuple(2*v % order for v, order in zip(member, orders))
        if member != character:
            raise ArithmeticError("character orbits are not disjoint cycles")
        yield character, count


def abelian_rank(matrix: GroupAlgebraMatrix, *, max_characters=10_000,
                 max_field_degree=64, max_candidates=4096, max_steps=10_000_000,
                 max_entries=1_000_000, max_factors=16, _meter=None):
    """Exact binary rank for products of odd cyclic groups, without expansion.

    The maximum field degree bounds multiplicative-order growth, independently
    of the number of characters. Polynomial/root searches and arithmetic have
    separate deterministic budgets. No floating-point roots are used.
    Internal _meter shares an enclosing ceiling; result.steps is invocation-local.
    """
    nonnegative_integer(max_steps, "max_steps")
    if _meter is not None and not isinstance(_meter, ArithmeticBudget):
        raise TypeError("_meter must be ArithmeticBudget")
    budget = ArithmeticBudget(max_steps) if _meter is None else _meter
    start = budget.used
    if not isinstance(matrix, GroupAlgebraMatrix):
        raise TypeError("abelian rank requires GroupAlgebraMatrix")
    limits = (max_characters, max_field_degree, max_candidates, max_entries, max_factors)
    for name, value in zip(("max_characters", "max_field_degree", "max_candidates",
                            "max_entries", "max_factors"), limits):
        nonnegative_integer(value, name)
    if len(matrix.orders) > max_factors:
        raise WorkLimitExceeded("group factors exceed max_factors")
    if any(order % 2 == 0 for order in matrix.orders):
        raise ValueError("character field backend requires odd cyclic factors")
    if prod(matrix.orders) > max_characters:
        raise WorkLimitExceeded("character set exceeds max_characters")
    if max(matrix.nrows, matrix.ncols, matrix.nrows*matrix.ncols) > max_entries:
        raise WorkLimitExceeded("seed matrix exceeds max_entries")
    exponent = lcm(*matrix.orders)
    field = _splitting_field(exponent, max_field_degree, max_candidates, budget)
    root = _root(field, exponent, max_candidates, budget)
    generators = tuple(field.pow(root, exponent//order, budget=budget) for order in matrix.orders)
    components = []
    for character, orbit_size in _orbits(matrix.orders, budget):
        values = tuple(field.pow(generator, t, budget=budget)
                       for generator, t in zip(generators, character))
        cache = {}
        def evaluate(element):
            if element in cache:
                return cache[element]
            total = 0
            for term in element.terms:
                budget.charge()
                monomial = 1
                for value, power in zip(values, term):
                    budget.charge()
                    monomial = field.mul(monomial, field.pow(value, power, budget=budget), budget=budget)
                total ^= monomial
            cache[element] = total
            return total
        rows = []
        for row in matrix.rows:
            entries = []
            for element in row:
                budget.charge()
                entries.append(evaluate(element))
            rows.append(entries)
        rank = field.rank(rows, matrix.ncols, max_entries=max_entries, budget=budget)
        components.append(AbelianRankComponent(character, orbit_size, rank))
    return AbelianRankResult(matrix.orders, matrix.shape, field.modulus, root, exponent,
                             tuple(components), sum(c.orbit_size*c.rank for c in components), budget.used - start)
