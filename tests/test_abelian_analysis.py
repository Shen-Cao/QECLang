from itertools import product
from random import Random

import pytest

from qeclang.algebra.polynomial import WorkLimitExceeded
from qeclang.symbolic import GroupAlgebraElement as G, GroupAlgebraMatrix as M
from qeclang.symbolic.abelian_analysis import abelian_rank
from qeclang.symbolic.cyclic_analysis import cyclic_rank


def test_exhaustive_scalar_c3_c3():
    sites = tuple(product(range(3), repeat=2))
    for bits in range(512):
        element = G.from_terms((3, 3), (site for j, site in enumerate(sites) if (bits >> j) & 1))
        matrix = M(((element,),), 1, (3, 3))
        result = abelian_rank(matrix)
        assert result.rank == matrix.lower().rank()
        assert sum(c.orbit_size for c in result.components) == 9


def test_random_abelian_matrices_and_empty_axes():
    rng = Random(17)
    for orders in ((1,), (3,), (5,), (3, 3), (3, 5), (1, 3, 3), (5, 7)):
        sites = tuple(product(*(range(order) for order in orders)))
        for m, n in ((0, 3), (2, 0), (1, 2), (2, 3), (3, 2)):
            rows = tuple(tuple(G.from_terms(orders, rng.sample(sites, min(3, len(sites))))
                               for _ in range(n)) for _ in range(m))
            matrix = M(rows, n, orders)
            result = abelian_rank(matrix)
            assert result.rank == matrix.lower().rank()
            if len(orders) == 1:
                assert result.rank == cyclic_rank(matrix).rank


def test_no_expansion_and_limits(monkeypatch):
    element = G.from_terms((3, 5), [(0, 0), (1, 2), (2, 4)])
    matrix = M(((element,),), 1, (3, 5))
    expected = matrix.lower().rank()
    def forbidden(*args, **kwargs):
        raise AssertionError("symbolic analysis must not expand")
    monkeypatch.setattr(M, "lower", forbidden)
    monkeypatch.setattr(M, "lower_sparse", forbidden)
    assert abelian_rank(matrix).rank == expected
    for kwargs in ({"max_characters": 1}, {"max_field_degree": 1},
                   {"max_candidates": 0}, {"max_steps": 1}, {"max_entries": 0}):
        with pytest.raises(WorkLimitExceeded):
            abelian_rank(matrix, **kwargs)
    with pytest.raises(ValueError, match="odd"):
        abelian_rank(M.zeros(1, 1, (2, 3)))
