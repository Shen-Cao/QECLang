"""Odd/2-primary decomposition of finite abelian GF(2) group algebras.

Each cyclic order L=2**v*o splits as C_(2**v) x C_o. Only the odd subgroup is
diagonalized by characters; the 2-primary subgroup retains its regular action
over a splitting field. Pure 2-groups therefore require full component matrices.
"""
from dataclasses import dataclass
from itertools import product
from math import lcm, prod

from qeclang.algebra.binary import nonnegative_integer
from qeclang.algebra.polynomial import ArithmeticBudget, WorkLimitExceeded

from .abelian_analysis import AbelianRankComponent, _orbits, _root, _splitting_field
from .matrix import GroupAlgebraMatrix


@dataclass(frozen=True)
class MixedAbelianRankResult:
    orders: tuple[int, ...]
    odd_parts: tuple[int, ...]
    two_parts: tuple[int, ...]
    shape: tuple[int, int]
    splitting_modulus: int
    primitive_root: int
    root_order: int
    components: tuple[AbelianRankComponent, ...]
    rank: int
    steps: int
    method: str = "exact odd-character decomposition with 2-primary regular components"
    evidence: str = "verified odd splitting field/root and Frobenius orbits; exact 2-primary component ranks"

    @property
    def binary_shape(self):
        size = prod(self.orders)
        return self.shape[0] * size, self.shape[1] * size

    @property
    def component_shape(self):
        size = prod(self.two_parts)
        return self.shape[0] * size, self.shape[1] * size


def _coefficient_map(element, values, odd_parts, two_parts, field, budget):
    coefficients = {}
    for term in element.terms:
        budget.charge()
        shift = tuple(e % p for e, p in zip(term, two_parts))
        value = 1
        for eigenvalue, exponent, odd in zip(values, term, odd_parts):
            budget.charge()
            value = field.mul(value, field.pow(eigenvalue, exponent % odd, budget=budget),
                              budget=budget)
        coefficients[shift] = coefficients.get(shift, 0) ^ value
    return tuple((shift, value) for shift, value in coefficients.items() if value)


def _component(matrix, values, odd_parts, two_parts, field, budget):
    size = prod(two_parts)
    m, n = matrix.nrows * size, matrix.ncols * size
    budget.charge(m * n)
    rows = [[0] * n for _ in range(m)]
    if not m or not n:
        return rows
    cache = {}
    for i, row in enumerate(matrix.rows):
        for j, element in enumerate(row):
            budget.charge()
            if element not in cache:
                cache[element] = _coefficient_map(element, values, odd_parts, two_parts, field, budget)
            coefficients = cache[element]
            if not coefficients:
                continue
            for site_index, site in enumerate(product(*(range(p) for p in two_parts))):
                for shift, coefficient in coefficients:
                    budget.charge()
                    column = 0
                    for position, translation, order in zip(site, shift, two_parts):
                        budget.charge()
                        column = column * order + (position + translation) % order
                    rows[i * size + site_index][j * size + column] ^= coefficient
    return rows


def abelian_rank_mixed(matrix: GroupAlgebraMatrix, *, max_component_entries=1_000_000,
                       max_component_axis=10_000, max_characters=10_000,
                       max_field_degree=64, max_candidates=4096, max_steps=10_000_000,
                       max_seed_entries=100_000, max_factors=16) -> MixedAbelianRankResult:
    """Exact binary rank for arbitrary finite products of cyclic groups.

    Component dimensions are m*P by n*P, P the order of the 2-primary subgroup.
    Axis/entry guards are checked before allocation, including empty dimensions.
    Arithmetic work is shared across splitting-field search and every component.
    The rank sum uses odd Frobenius orbit sizes, never semisimplicity of 2-groups.
    """
    if not isinstance(matrix, GroupAlgebraMatrix):
        raise TypeError("mixed abelian rank requires a GroupAlgebraMatrix")
    limits = {"max_component_entries": max_component_entries,
              "max_component_axis": max_component_axis, "max_characters": max_characters,
              "max_field_degree": max_field_degree, "max_candidates": max_candidates,
              "max_seed_entries": max_seed_entries, "max_factors": max_factors}
    for name, value in limits.items():
        nonnegative_integer(value, name)
    if len(matrix.orders) > max_factors:
        raise WorkLimitExceeded("group factors exceed max_factors")
    if max(matrix.nrows, matrix.ncols, matrix.nrows * matrix.ncols) > max_seed_entries:
        raise WorkLimitExceeded("seed matrix exceeds max_seed_entries")
    two_parts = tuple(order & -order for order in matrix.orders)
    odd_parts = tuple(order // p for order, p in zip(matrix.orders, two_parts))
    if prod(odd_parts) > max_characters:
        raise WorkLimitExceeded("odd character set exceeds max_characters")
    size = prod(two_parts)
    component_shape = matrix.nrows * size, matrix.ncols * size
    if max(component_shape) > max_component_axis:
        raise WorkLimitExceeded("regular component exceeds max_component_axis")
    if prod(component_shape) > max_component_entries:
        raise WorkLimitExceeded("regular component exceeds max_component_entries")
    budget = ArithmeticBudget(max_steps)
    exponent = lcm(*odd_parts)
    field = _splitting_field(exponent, max_field_degree, max_candidates, budget)
    root = _root(field, exponent, max_candidates, budget)
    generators = tuple(field.pow(root, exponent // order, budget=budget) for order in odd_parts)
    components = []
    for character, orbit_size in _orbits(odd_parts, budget):
        values = tuple(field.pow(generator, t, budget=budget)
                       for generator, t in zip(generators, character))
        rows = _component(matrix, values, odd_parts, two_parts, field, budget)
        rank = field.rank(rows, component_shape[1],
                          max_entries=max(max_component_entries, max_component_axis), budget=budget)
        components.append(AbelianRankComponent(character, orbit_size, rank))
    return MixedAbelianRankResult(matrix.orders, odd_parts, two_parts, matrix.shape,
                                   field.modulus, root, exponent, tuple(components),
                                   sum(c.orbit_size * c.rank for c in components), budget.used)


__all__ = ["MixedAbelianRankResult", "abelian_rank_mixed"]
