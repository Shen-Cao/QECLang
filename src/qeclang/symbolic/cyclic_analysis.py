"""Exact odd-cyclic rank using the semisimple polynomial CRT decomposition.

For odd L, x**L+1 is squarefree over GF(2). Its distinct irreducible factors
decompose GF(2)[C_L] into binary extension fields. The binary regular-action
rank is the sum of component field ranks weighted by factor degree. This
backend never materializes the expanded binary check matrix.
"""
from dataclasses import dataclass

from qeclang.algebra.binary import nonnegative_integer
from qeclang.algebra.finite_field import BinaryExtensionField
from qeclang.algebra.polynomial import (ArithmeticBudget, PolynomialFactorization,
                                       WorkLimitExceeded, factor_squarefree, poly_mod,
                                       verify_factorization)

from .matrix import GroupAlgebraMatrix


@dataclass(frozen=True)
class CyclicRankComponent:
    modulus: int
    degree: int
    rank: int


@dataclass(frozen=True)
class CyclicRankResult:
    lift: int
    shape: tuple[int, int]
    factorization: PolynomialFactorization
    components: tuple[CyclicRankComponent, ...]
    rank: int
    steps: int
    method: str = "exact odd-cyclic CRT finite-field elimination"
    assumptions: tuple[str, ...] = (
        "coefficients are in GF(2)",
        "one cyclic factor of odd positive order",
        "regular representation with the supplied physical site basis",
    )
    evidence: str = "irreducible factors and their product verified exactly; exact component elimination"

    @property
    def binary_shape(self):
        return self.shape[0] * self.lift, self.shape[1] * self.lift


def cyclic_rank(matrix: GroupAlgebraMatrix, *, max_degree=512, max_steps=10_000_000,
                max_entries=1_000_000, factorization=None) -> CyclicRankResult:
    """Exact binary rank without expansion; unsupported rings are rejected.

    ``max_degree`` bounds L, and ``max_entries`` bounds seed dimensions/product.
    A supplied PolynomialFactorization may be reused but is independently
    verified, including its modulus; it is never trusted as an assertion.
    """
    if not isinstance(matrix, GroupAlgebraMatrix):
        raise TypeError("cyclic rank requires a GroupAlgebraMatrix")
    if len(matrix.orders) != 1:
        raise ValueError("cyclic rank requires exactly one cyclic factor")
    lift = matrix.orders[0]
    if lift % 2 == 0:
        raise ValueError("cyclic rank requires odd lift order; repeated-root rings are unsupported")
    limit = nonnegative_integer(max_degree, "max_degree")
    entries = nonnegative_integer(max_entries, "max_entries")
    if lift > limit:
        raise WorkLimitExceeded("cyclic modulus exceeds max_degree")
    if max(matrix.nrows, matrix.ncols, matrix.nrows * matrix.ncols) > entries:
        raise WorkLimitExceeded("cyclic seed matrix exceeds max_entries")
    budget = ArithmeticBudget(max_steps)
    modulus = (1 << lift) | 1
    if factorization is None:
        factorization = factor_squarefree(modulus, max_degree=limit, budget=budget)
    elif not isinstance(factorization, PolynomialFactorization):
        raise TypeError("factorization must be PolynomialFactorization")
    elif factorization.modulus != modulus or not verify_factorization(
            modulus, factorization.factors, max_degree=limit, budget=budget):
        raise ValueError("factorization does not certify the cyclic modulus")
    polynomial_rows = []
    for row in matrix.rows:
        polynomials = []
        for element in row:
            value = 0
            for term in element.terms:
                budget.charge()
                value |= 1 << term[0]
            polynomials.append(value)
        polynomial_rows.append(tuple(polynomials))
    components = []
    for factor in factorization.factors:
        field = BinaryExtensionField(factor, max_degree=limit, budget=budget)
        rows = []
        for row in polynomial_rows:
            reduced = []
            for value in row:
                budget.charge()
                reduced.append(poly_mod(value, factor, budget=budget))
            rows.append(reduced)
        rank = field.rank(rows, matrix.ncols, max_entries=entries, budget=budget)
        components.append(CyclicRankComponent(factor, field.degree, rank))
    return CyclicRankResult(lift, matrix.shape, factorization, tuple(components),
                            sum(c.degree * c.rank for c in components), budget.used)


__all__ = ["CyclicRankComponent", "CyclicRankResult", "cyclic_rank"]
