from itertools import product

import pytest

from qeclang.algebra.finite_field import BinaryExtensionField
from qeclang.algebra.polynomial import (ArithmeticBudget, PolynomialFactorization,
                                       WorkLimitExceeded, factor_squarefree, is_irreducible,
                                       poly_add, poly_degree, poly_derivative, poly_divmod,
                                       poly_gcd, poly_inverse_mod, poly_mod, poly_mul,
                                       poly_pow_mod, verify_factorization)


def brute_multiply(a, b):
    coefficients = [0] * (a.bit_length() + b.bit_length())
    for i in range(a.bit_length()):
        for j in range(b.bit_length()):
            coefficients[i + j] ^= ((a >> i) & 1) * ((b >> j) & 1)
    return sum(bit << i for i, bit in enumerate(coefficients))


def test_small_polynomial_arithmetic_against_coefficient_convolution():
    for a, b in product(range(64), repeat=2):
        assert poly_mul(a, b) == brute_multiply(a, b)
        assert poly_add(a, b) == a ^ b
        if b:
            quotient, remainder = poly_divmod(a, b)
            assert poly_mul(quotient, b) ^ remainder == a
            assert poly_degree(remainder) < poly_degree(b)
        gcd = poly_gcd(a, b)
        if gcd:
            assert poly_mod(a, gcd) == poly_mod(b, gcd) == 0
    assert poly_degree(0) == -1
    assert poly_gcd(0, 0) == 0
    assert poly_derivative(0b11111) == 0b0101


def test_irreducibility_against_exhaustive_trial_division():
    assert not is_irreducible(0)
    assert not is_irreducible(1)
    for value in range(2, 512):
        degree = poly_degree(value)
        brute = all(poly_mod(value, divisor) for divisor in range(2, 1 << (degree // 2 + 1)))
        assert is_irreducible(value) == brute


def test_all_squarefree_polynomials_through_degree_eight():
    for value in range(1, 512):
        if poly_gcd(value, poly_derivative(value)) == 1:
            certificate = factor_squarefree(value)
            assert certificate.verify()
            expected = tuple(divisor for divisor in range(2, value + 1)
                             if is_irreducible(divisor) and poly_mod(value, divisor) == 0)
            assert set(certificate.factors) == set(expected)
        else:
            with pytest.raises(ValueError, match="squarefree"):
                factor_squarefree(value)


def test_modular_powers_and_inverse():
    for modulus in range(2, 32):
        for a in range(32):
            product_value = poly_mod(1, modulus)
            for exponent in range(9):
                assert poly_pow_mod(a, exponent, modulus) == product_value
                product_value = poly_mod(poly_mul(product_value, a), modulus)
            if poly_gcd(a, modulus) == 1:
                inverse = poly_inverse_mod(a, modulus)
                assert poly_mod(poly_mul(a, inverse), modulus) == 1
            else:
                with pytest.raises(ZeroDivisionError):
                    poly_inverse_mod(a, modulus)


def test_factorization_certificates_and_limits():
    certificate = factor_squarefree((1 << 15) | 1)
    assert certificate.verify()
    assert sum(poly_degree(f) for f in certificate.factors) == 15
    assert not PolynomialFactorization(certificate.modulus, certificate.factors[:-1], 0).verify()
    assert not verify_factorization(9, (3, 3, 3))
    with pytest.raises(WorkLimitExceeded, match="degree"):
        factor_squarefree(1 << 1000, max_degree=100)
    with pytest.raises(WorkLimitExceeded, match="steps"):
        factor_squarefree(0b111, max_steps=0)
    with pytest.raises(WorkLimitExceeded):
        is_irreducible(0b111, max_steps=1)
    with pytest.raises(ValueError):
        factor_squarefree(0)
    with pytest.raises(ZeroDivisionError):
        poly_divmod(1, 0)
    with pytest.raises(ValueError):
        poly_inverse_mod(1, 1)
    budget = ArithmeticBudget(3)
    with pytest.raises(ValueError):
        budget.charge(-1)
    assert budget.used == 0


@pytest.mark.parametrize("bad", [-1, True, 1.0, float("nan")])
def test_polynomial_validation(bad):
    for operation in (poly_degree, is_irreducible, factor_squarefree):
        with pytest.raises(ValueError):
            operation(bad)
    with pytest.raises(ValueError):
        ArithmeticBudget(bad)


def test_small_extension_field_axioms_and_rank():
    for modulus in (2, 3, 7, 11, 19):
        field = BinaryExtensionField(modulus)
        for a, b in product(range(field.order), repeat=2):
            assert field.mul(a, b) == field.mul(b, a)
            assert field.add(a, a) == 0
            if a:
                assert field.mul(a, field.inverse(a)) == 1
                assert field.pow(a, field.order - 1) == 1
            assert field.rank(((a, b),), ncols=2) == int(bool(a or b))
        assert field.rank((), ncols=3) == 0
        assert field.rank(((), ()), ncols=0) == 0
        assert field.rank(((1, 0), (0, 1)), ncols=2) == 2
        assert field.rank(((1, 1), (1, 1)), ncols=2) == 1
    field = BinaryExtensionField(7)
    assert field.mul(2, 2) == 3
    assert field.rank(((1, 2), (2, 3)), ncols=2) == 1
    with pytest.raises(ValueError, match="irreducible"):
        BinaryExtensionField(5)
    with pytest.raises(ValueError, match="canonical"):
        field.mul(4, 1)
    with pytest.raises(ZeroDivisionError):
        field.inverse(0)
    with pytest.raises(ValueError, match="width"):
        field.rank(((1,), (1, 0)))
    with pytest.raises(WorkLimitExceeded):
        field.rank(((1, 0),), max_entries=1)
    with pytest.raises(WorkLimitExceeded):
        field.rank(((1,),), max_steps=0)
