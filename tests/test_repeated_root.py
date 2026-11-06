from dataclasses import replace
from itertools import product
from random import Random

import pytest

from qeclang.algebra.polynomial import WorkLimitExceeded
from qeclang.symbolic import GroupAlgebraElement as G
from qeclang.symbolic.cyclic_analysis import cyclic_rank
from qeclang.symbolic.matrix import GroupAlgebraMatrix as M
from qeclang.symbolic.repeated_root import (PolynomialOperation, cyclic_rank_general,
                                           verify_rank_certificate)


def element(word, lift):
    return G.from_terms((lift,), ((j,) for j in range(lift) if (word >> j) & 1))


def test_all_two_by_two_binary_and_repeated_root_c2_matrices():
    for lift in (1, 2):
        elements = tuple(element(word, lift) for word in range(1 << lift))
        for a, b, c, d in product(elements, repeat=4):
            matrix = M(((a, b), (c, d)), 2, (lift,))
            result = cyclic_rank_general(matrix, record_operations=True)
            assert result.rank == matrix.lower().rank()
            assert verify_rank_certificate(matrix, result)


def test_random_rectangular_even_and_odd_lifts():
    rng = Random(12382)
    for lift in (1, 2, 3, 4, 6, 8, 9, 10, 12, 16):
        for nrows, ncols in product(range(4), repeat=2):
            for _ in range(4):
                matrix = M(tuple(tuple(element(rng.randrange(1 << lift), lift)
                                       for _ in range(ncols)) for _ in range(nrows)), ncols, (lift,))
                result = cyclic_rank_general(matrix, record_operations=True)
                assert result.rank == matrix.lower().rank()
                assert verify_rank_certificate(matrix, result)
                if lift % 2:
                    assert result.rank == cyclic_rank(matrix).rank


def test_repeated_root_nilpotent_information_is_retained():
    for lift in (2, 4, 8, 16):
        x_plus_one = element(3, lift)
        matrix = M(((x_plus_one,),), 1, (lift,))
        result = cyclic_rank_general(matrix, record_operations=True)
        assert result.rank == lift - 1
        assert result.gcds == (3,)
        assert verify_rank_certificate(matrix, result)
    # At L=4, x^2+1=(x+1)^2 has binary rank two although its residue at x=1 vanishes.
    matrix = M(((element(5, 4),),), 1, (4,))
    assert cyclic_rank_general(matrix).rank == 2


def test_zero_diagonal_and_non_smith_diagonal():
    zero = element(0, 4)
    a, b = element(3, 4), element(7, 4)
    matrix = M(((a, zero, zero), (zero, b, zero), (zero, zero, zero)), 3, (4,))
    result = cyclic_rank_general(matrix, record_operations=True)
    assert result.diagonal == (3, 7, 0)
    assert result.rank == 7
    assert result.gcds == (3, 1, 17)
    assert verify_rank_certificate(matrix, result)


def test_independent_replay_and_no_binary_expansion(monkeypatch):
    import qeclang.symbolic.repeated_root as module

    matrix = M(((element(3, 6), element(5, 6)),
                (element(9, 6), element(1, 6))), 2, (6,))
    expected = matrix.lower().rank()

    def forbidden(*args, **kwargs):
        raise AssertionError("binary expansion and recomputation are forbidden")

    for cls in (G, M):
        monkeypatch.setattr(cls, "lower", forbidden)
        monkeypatch.setattr(cls, "lower_sparse", forbidden)
    result = cyclic_rank_general(matrix, record_operations=True)
    assert result.rank == expected
    monkeypatch.setattr(module, "cyclic_rank_general", forbidden)
    assert verify_rank_certificate(matrix, result)


def test_tampered_rank_trace_and_diagonal_are_rejected():
    matrix = M(((element(1, 4), element(3, 4)),
                (element(5, 4), element(2, 4))), 2, (4,))
    result = cyclic_rank_general(matrix, record_operations=True)
    assert result.operations
    assert not verify_rank_certificate(matrix, replace(result, rank=result.rank + 1))
    assert not verify_rank_certificate(matrix, replace(result, diagonal=(0, 0)))
    assert not verify_rank_certificate(matrix, replace(result, gcds=(3, 1)))
    assert not verify_rank_certificate(matrix, replace(result, operation_count=0))
    assert not verify_rank_certificate(matrix, replace(result, operations=None))
    assert not verify_rank_certificate(matrix, replace(result, operations=2))
    assert not verify_rank_certificate(matrix, replace(result, rank=True))
    assert not verify_rank_certificate(matrix, replace(result, operation_count=True))
    bad_operation = PolynomialOperation("row", "add", 0, 0, 1)
    assert not verify_rank_certificate(matrix, replace(result, operations=(bad_operation,), operation_count=1))
    first = replace(result.operations[0], multiplier=0)
    bad_trace = (first,) + result.operations[1:]
    assert not verify_rank_certificate(matrix, replace(result, operations=bad_trace))


def test_degree_growth_and_other_explicit_limits():
    matrix = M(((element(1, 3), element(4, 3)),
                (element(4, 3), element(0, 3))), 2, (3,))
    with pytest.raises(WorkLimitExceeded, match="intermediate"):
        cyclic_rank_general(matrix, max_polynomial_degree=3)
    assert cyclic_rank_general(matrix, max_polynomial_degree=4).rank == 6
    for argument in ({"max_lift": 2}, {"max_entries": 3}, {"max_steps": 0},
                     {"max_operations": 0}):
        with pytest.raises(WorkLimitExceeded):
            cyclic_rank_general(matrix, **argument)
    result = cyclic_rank_general(matrix, record_operations=True)
    with pytest.raises(WorkLimitExceeded, match="operations"):
        verify_rank_certificate(matrix, result, max_operations=0)
    with pytest.raises(ValueError, match="one cyclic"):
        cyclic_rank_general(M.zeros(1, 1, (2, 3)))
    with pytest.raises(ValueError):
        cyclic_rank_general(matrix, record_operations=1)
    for bad in (True, -1, float("nan")):
        with pytest.raises(ValueError):
            cyclic_rank_general(matrix, max_polynomial_degree=bad)
