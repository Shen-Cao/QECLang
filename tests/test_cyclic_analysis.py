from itertools import product
from random import Random

import pytest

from qeclang.algebra.polynomial import PolynomialFactorization, WorkLimitExceeded
from qeclang.symbolic import GroupAlgebraElement as G
from qeclang.symbolic.cyclic_analysis import cyclic_rank
from qeclang.symbolic.matrix import GroupAlgebraMatrix as M


def element(word, lift):
    return G.from_terms((lift,), ((j,) for j in range(lift) if (word >> j) & 1))


def test_all_scalars_and_two_by_two_matrices_over_c3():
    elements = tuple(element(word, 3) for word in range(8))
    cached = cyclic_rank(M(((elements[0],),), 1, (3,))).factorization
    for a, b, c, d in product(elements, repeat=4):
        matrix = M(((a, b), (c, d)), 2, (3,))
        result = cyclic_rank(matrix, factorization=cached)
        assert result.rank == matrix.lower().rank()
        assert result.rank == sum(component.degree * component.rank for component in result.components)
        assert result.binary_shape == (6, 6)


def test_rectangular_random_and_empty_lifts():
    rng = Random(412)
    for lift in (1, 3, 5, 7, 9, 15, 21):
        for nrows, ncols in product(range(4), repeat=2):
            for _ in range(3):
                matrix = M(tuple(tuple(element(rng.randrange(1 << lift), lift)
                                       for _ in range(ncols)) for _ in range(nrows)), ncols, (lift,))
                result = cyclic_rank(matrix)
                assert result.rank == matrix.lower().rank()
                assert result.factorization.verify()


def test_no_binary_expansion_and_nonsemisimple_rejection(monkeypatch):
    a, b = element(0b11, 31), element(0b101, 31)
    matrix = M(((a, b), (b, a)), 2, (31,))

    def forbidden(*args, **kwargs):
        raise AssertionError("symbolic rank must not expand binary blocks")

    monkeypatch.setattr(G, "lower", forbidden)
    monkeypatch.setattr(G, "lower_sparse", forbidden)
    monkeypatch.setattr(M, "lower", forbidden)
    monkeypatch.setattr(M, "lower_sparse", forbidden)
    result = cyclic_rank(matrix)
    assert result.rank == 60
    assert result.factorization.verify()
    with pytest.raises(ValueError, match="odd"):
        cyclic_rank(M.zeros(1, 1, (2,)))
    with pytest.raises(ValueError, match="one cyclic"):
        cyclic_rank(M.zeros(1, 1, (3, 5)))


def test_cyclic_rank_limits_and_untrusted_factorization():
    matrix = M(((element(1, 7),),), 1, (7,))
    with pytest.raises(WorkLimitExceeded, match="degree"):
        cyclic_rank(matrix, max_degree=6)
    with pytest.raises(WorkLimitExceeded, match="entries"):
        cyclic_rank(matrix, max_entries=0)
    with pytest.raises(WorkLimitExceeded, match="steps"):
        cyclic_rank(matrix, max_steps=0)
    with pytest.raises(ValueError, match="certify"):
        cyclic_rank(matrix, factorization=PolynomialFactorization(129, (3,), 0))
    with pytest.raises(ValueError, match="certify"):
        cyclic_rank(matrix, factorization=PolynomialFactorization(3, (3,), 0))
    for bad in (float("nan"), -1, True):
        with pytest.raises(ValueError):
            cyclic_rank(matrix, max_steps=bad)
