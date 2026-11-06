"""Exact GF(2)[x] arithmetic; coefficient of x**j is bit j.

Deterministic Berlekamp factorization is intended for bounded polynomial
degrees. Work budgets count algebraic loop steps, not elapsed time or bit
complexity. Degree limits separately bound operand widths and kernel storage.
"""
from dataclasses import dataclass, field

from .binary import BinaryMatrix, nonnegative_integer


class WorkLimitExceeded(RuntimeError):
    """An explicit arithmetic or degree budget was exhausted."""


@dataclass
class ArithmeticBudget:
    max_steps: int = 10_000_000
    used: int = field(default=0, init=False)

    def __post_init__(self):
        self.max_steps = nonnegative_integer(self.max_steps, "max_steps")

    def charge(self, steps=1):
        steps = nonnegative_integer(steps, "steps")
        if self.used + steps > self.max_steps:
            raise WorkLimitExceeded("polynomial arithmetic exceeds max_steps")
        self.used += steps


def _charge(budget, steps=1):
    if budget is not None:
        budget.charge(steps)


def poly_degree(value):
    """Degree, with degree(0) = -1."""
    return nonnegative_integer(value, "polynomial").bit_length() - 1


def poly_add(left, right):
    return nonnegative_integer(left, "left") ^ nonnegative_integer(right, "right")


def _mul(left, right, budget):
    result = 0
    while right:
        _charge(budget)
        if right & 1:
            result ^= left
        left <<= 1
        right >>= 1
    return result


def poly_mul(left, right, *, budget=None):
    return _mul(nonnegative_integer(left, "left"), nonnegative_integer(right, "right"), budget)


def _divmod(dividend, divisor, budget):
    if divisor == 0:
        raise ZeroDivisionError("zero polynomial divisor")
    quotient = 0
    while dividend and dividend.bit_length() >= divisor.bit_length():
        _charge(budget)
        shift = dividend.bit_length() - divisor.bit_length()
        quotient ^= 1 << shift
        dividend ^= divisor << shift
    return quotient, dividend


def poly_divmod(dividend, divisor, *, budget=None):
    return _divmod(nonnegative_integer(dividend, "dividend"),
                   nonnegative_integer(divisor, "divisor"), budget)


def poly_mod(dividend, modulus, *, budget=None):
    return poly_divmod(dividend, modulus, budget=budget)[1]


def _gcd(left, right, budget):
    while right:
        _charge(budget)
        left, right = right, _divmod(left, right, budget)[1]
    return left


def poly_gcd(left, right, *, budget=None):
    return _gcd(nonnegative_integer(left, "left"), nonnegative_integer(right, "right"), budget)


def poly_derivative(value, *, budget=None):
    value = nonnegative_integer(value, "polynomial")
    result = 0
    for exponent in range(1, value.bit_length(), 2):
        _charge(budget)
        result |= ((value >> exponent) & 1) << (exponent - 1)
    return result


def poly_pow_mod(base, exponent, modulus, *, budget=None):
    modulus = nonnegative_integer(modulus, "modulus")
    exponent = nonnegative_integer(exponent, "exponent")
    base = poly_mod(base, modulus, budget=budget)
    result = poly_mod(1, modulus, budget=budget)
    while exponent:
        _charge(budget)
        if exponent & 1:
            result = _divmod(_mul(result, base, budget), modulus, budget)[1]
        exponent >>= 1
        if exponent:
            base = _divmod(_mul(base, base, budget), modulus, budget)[1]
    return result


def poly_inverse_mod(value, modulus, *, budget=None):
    modulus = nonnegative_integer(modulus, "modulus")
    if modulus < 2:
        raise ValueError("inverse modulus must have positive degree")
    value = poly_mod(value, modulus, budget=budget)
    old_r, r, old_t, t = modulus, value, 0, 1
    while r:
        _charge(budget)
        quotient, remainder = _divmod(old_r, r, budget)
        old_r, r = r, remainder
        old_t, t = t, old_t ^ _mul(quotient, t, budget)
    if old_r != 1:
        raise ZeroDivisionError("polynomial is not invertible modulo modulus")
    return _divmod(old_t, modulus, budget)[1]


def _degree_limit(value, max_degree):
    degree = poly_degree(value)
    if degree > nonnegative_integer(max_degree, "max_degree"):
        raise WorkLimitExceeded("polynomial exceeds max_degree")
    return degree


def _irreducible(value, budget):
    degree = value.bit_length() - 1
    if degree < 1:
        return False
    factors, remaining, prime = set(), degree, 2
    while prime * prime <= remaining:
        _charge(budget)
        if remaining % prime == 0:
            factors.add(prime)
            while remaining % prime == 0:
                remaining //= prime
        prime += 1
    if remaining > 1:
        factors.add(remaining)
    checkpoints = {degree // prime for prime in factors}
    x = _divmod(2, value, budget)[1]
    power = x
    for i in range(1, degree + 1):
        _charge(budget)
        power = _divmod(_mul(power, power, budget), value, budget)[1]
        if i in checkpoints and _gcd(power ^ x, value, budget) != 1:
            return False
    return power == x


def is_irreducible(value, *, max_degree=512, max_steps=10_000_000, budget=None):
    """Rabin's exact finite-field test; no random or probable-prime outcome."""
    value = nonnegative_integer(value, "polynomial")
    _degree_limit(value, max_degree)
    budget = ArithmeticBudget(max_steps) if budget is None else budget
    return _irreducible(value, budget)


@dataclass(frozen=True)
class PolynomialFactorization:
    modulus: int
    factors: tuple[int, ...]
    steps: int
    method: str = "deterministic GF(2) Berlekamp with Rabin verification"

    def verify(self, *, max_degree=512, max_steps=10_000_000):
        return verify_factorization(self.modulus, self.factors, max_degree=max_degree,
                                    max_steps=max_steps)


def verify_factorization(modulus, factors, *, max_degree=512, max_steps=10_000_000,
                         budget=None):
    """Check distinct irreducibles whose product is exactly the modulus."""
    modulus = nonnegative_integer(modulus, "modulus")
    _degree_limit(modulus, max_degree)
    budget = ArithmeticBudget(max_steps) if budget is None else budget
    factors = tuple(nonnegative_integer(f, "factor") for f in factors)
    if modulus == 0 or len(factors) != len(set(factors)):
        return False
    value = 1
    for factor in factors:
        if poly_degree(factor) > poly_degree(modulus) or not _irreducible(factor, budget):
            return False
        value = _mul(value, factor, budget)
        if value.bit_length() > modulus.bit_length():
            return False
    return value == modulus


def factor_squarefree(modulus, *, max_degree=512, max_steps=10_000_000, budget=None):
    """Factor a squarefree binary polynomial; repeated factors are rejected.

    The result carries an independently recheckable irreducible factor list.
    A work-limit exception provides no partial factorization claim.
    """
    modulus = nonnegative_integer(modulus, "modulus")
    degree = _degree_limit(modulus, max_degree)
    if modulus == 0:
        raise ValueError("zero polynomial cannot be factored")
    budget = ArithmeticBudget(max_steps) if budget is None else budget
    start = budget.used
    if _gcd(modulus, poly_derivative(modulus, budget=budget), budget) != 1:
        raise ValueError("factor_squarefree requires a squarefree polynomial")
    if degree == 0:
        return PolynomialFactorization(modulus, (), budget.used - start)
    columns = []
    for j in range(degree):
        _charge(budget)
        columns.append(_divmod(1 << (2 * j), modulus, budget)[1] ^ (1 << j))
    # Conservative count for transpose, elimination, and kernel extraction;
    # each counted operation acts on an at-most-degree-bit packed row.
    budget.charge(4 * degree * degree + degree)
    kernel = BinaryMatrix(tuple(columns), degree).T.nullspace()
    factors = [modulus]
    for splitter in kernel.rows:
        refined = []
        for factor in factors:
            _charge(budget)
            divisor = _gcd(factor, splitter, budget)
            if divisor not in (1, factor):
                quotient, remainder = _divmod(factor, divisor, budget)
                if remainder:
                    raise ArithmeticError("inexact Berlekamp division")
                refined.extend((divisor, quotient))
            else:
                refined.append(factor)
        factors = refined
        if len(factors) == kernel.nrows:
            break
    factors = tuple(sorted(factors, key=lambda f: (f.bit_length(), f)))
    if not verify_factorization(modulus, factors, max_degree=max_degree, budget=budget):
        raise ArithmeticError("Berlekamp factorization failed exact verification")
    return PolynomialFactorization(modulus, factors, budget.used - start)
