"""Exact polynomial-basis binary extension fields and bounded matrix rank."""
from dataclasses import InitVar, dataclass

from .binary import nonnegative_integer
from .polynomial import (ArithmeticBudget, WorkLimitExceeded, is_irreducible,
                         poly_inverse_mod, poly_mod, poly_mul, poly_pow_mod)


@dataclass(frozen=True)
class BinaryExtensionField:
    """GF(2)[x]/(modulus), with a verified monic irreducible modulus.

    Elements are canonical packed polynomials of degree below the modulus.
    Use ``reduce`` explicitly for arbitrary polynomial representatives.
    """
    modulus: int
    max_degree: InitVar[int] = 512
    max_steps: InitVar[int] = 10_000_000
    budget: InitVar[ArithmeticBudget | None] = None

    def __post_init__(self, max_degree, max_steps, budget):
        value = nonnegative_integer(self.modulus, "modulus")
        if not is_irreducible(value, max_degree=max_degree, max_steps=max_steps, budget=budget):
            raise ValueError("finite-field modulus must be irreducible of positive degree")
        object.__setattr__(self, "modulus", value)

    @property
    def degree(self):
        return self.modulus.bit_length() - 1

    @property
    def order(self):
        return 1 << self.degree

    def _element(self, value):
        value = nonnegative_integer(value, "field element")
        if value.bit_length() > self.degree:
            raise ValueError("field element is not a canonical polynomial representative")
        return value

    def reduce(self, polynomial, *, budget=None):
        return poly_mod(polynomial, self.modulus, budget=budget)

    def add(self, left, right):
        return self._element(left) ^ self._element(right)

    def mul(self, left, right, *, budget=None):
        product = poly_mul(self._element(left), self._element(right), budget=budget)
        return poly_mod(product, self.modulus, budget=budget)

    def inverse(self, value, *, budget=None):
        return poly_inverse_mod(self._element(value), self.modulus, budget=budget)

    def pow(self, value, exponent, *, budget=None):
        return poly_pow_mod(self._element(value), exponent, self.modulus, budget=budget)

    def _rows(self, rows, ncols, limit):
        width = None if ncols is None else nonnegative_integer(ncols, "ncols")
        if width is not None and width > limit:
            raise WorkLimitExceeded("field matrix dimension exceeds max_entries")
        matrix = []
        for row in rows:
            if len(matrix) >= limit:
                raise WorkLimitExceeded("field matrix dimension exceeds max_entries")
            values = []
            for value in row:
                if len(values) >= limit:
                    raise WorkLimitExceeded("field matrix dimension exceeds max_entries")
                values.append(self._element(value))
            width = len(values) if width is None else width
            if len(values) != width:
                raise ValueError("field matrix rows must have the declared width")
            if (len(matrix) + 1) * width > limit:
                raise WorkLimitExceeded("field matrix exceeds max_entries")
            matrix.append(values)
        return matrix, 0 if width is None else width

    def rank(self, rows, ncols=None, *, max_entries=1_000_000,
             max_steps=10_000_000, budget=None):
        """Gaussian-elimination rank without any binary regular representation.

        ``max_entries`` bounds both axes and their product. ``budget``, if given,
        is shared with surrounding symbolic computations instead of max_steps.
        """
        limit = nonnegative_integer(max_entries, "max_entries")
        matrix, width = self._rows(rows, ncols, limit)
        budget = ArithmeticBudget(max_steps) if budget is None else budget
        rank = 0
        for column in range(width):
            if rank == len(matrix):
                break
            pivot = None
            for i in range(rank, len(matrix)):
                budget.charge()
                if matrix[i][column]:
                    pivot = i
                    break
            if pivot is None:
                continue
            matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
            inverse = self.inverse(matrix[rank][column], budget=budget)
            for j in range(column, width):
                budget.charge()
                matrix[rank][j] = self.mul(matrix[rank][j], inverse, budget=budget)
            for i in range(rank + 1, len(matrix)):
                factor = matrix[i][column]
                budget.charge()
                if factor:
                    for j in range(column, width):
                        budget.charge()
                        matrix[i][j] ^= self.mul(factor, matrix[rank][j], budget=budget)
            rank += 1
        return rank


__all__ = ["BinaryExtensionField"]
