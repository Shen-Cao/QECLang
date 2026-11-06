from itertools import product
from random import Random

import pytest

from qeclang.algebra.polynomial import WorkLimitExceeded
from qeclang.symbolic import GroupAlgebraElement as G, GroupAlgebraMatrix as M
from qeclang.symbolic.abelian_analysis import abelian_rank
from qeclang.symbolic.mixed_abelian import abelian_rank_mixed
from qeclang.symbolic.repeated_root import cyclic_rank_general


def test_all_scalar_c2_c2_and_c2_c3_elements():
    for orders in ((2, 2), (2, 3), (4,), (3, 3)):
        sites = tuple(product(*(range(order) for order in orders)))
        for word in range(1 << len(sites)):
            a = G.from_terms(orders, (site for j, site in enumerate(sites) if (word >> j) & 1))
            matrix = M(((a,),), 1, orders)
            result = abelian_rank_mixed(matrix)
            assert result.rank == matrix.lower().rank()
            assert result.rank == sum(c.orbit_size * c.rank for c in result.components)


def test_random_rectangular_and_empty_mixed_orders():
    rng = Random(721)
    for orders in ((1,), (2,), (4,), (6,), (8,), (12,), (2, 2), (2, 3),
                   (4, 6), (6, 10), (2, 2, 3), (1, 4, 3), (3, 5)):
        sites = tuple(product(*(range(order) for order in orders)))
        for m, n in ((0, 3), (2, 0), (1, 2), (2, 3), (3, 2)):
            for _ in range(3):
                matrix = M(tuple(tuple(G.from_terms(orders, rng.sample(sites, min(4, len(sites))))
                                       for _ in range(n)) for _ in range(m)), n, orders)
                result = abelian_rank_mixed(matrix)
                assert result.rank == matrix.lower().rank()
                assert sum(c.orbit_size for c in result.components) == len(sites) // product_size(result.two_parts)
                if len(orders) == 1:
                    assert result.rank == cyclic_rank_general(matrix).rank
                if all(order % 2 for order in orders):
                    assert result.rank == abelian_rank(matrix).rank


def product_size(values):
    result = 1
    for value in values:
        result *= value
    return result


def test_pure_two_group_has_full_regular_component():
    orders = (4, 2)
    a = G.from_terms(orders, [(0, 0), (1, 0)])
    matrix = M(((a,),), 1, orders)
    result = abelian_rank_mixed(matrix)
    assert result.two_parts == (4, 2)
    assert result.odd_parts == (1, 1)
    assert result.component_shape == result.binary_shape == (8, 8)
    assert len(result.components) == 1
    assert result.components[0].orbit_size == 1
    assert result.rank == 6


def test_no_existing_lower_calls_and_component_limits(monkeypatch):
    a = G.from_terms((4, 3), [(0, 0), (1, 1), (2, 2)])
    matrix = M(((a, a.involution()),), 2, (4, 3))
    expected = matrix.lower().rank()

    def forbidden(*args, **kwargs):
        raise AssertionError("existing binary lower paths must not be called")

    for cls in (G, M):
        monkeypatch.setattr(cls, "lower", forbidden)
        monkeypatch.setattr(cls, "lower_sparse", forbidden)
    assert abelian_rank_mixed(matrix).rank == expected
    for kwargs in ({"max_component_axis": 7}, {"max_component_entries": 31},
                   {"max_characters": 2}, {"max_field_degree": 1},
                   {"max_candidates": 0}, {"max_steps": 0}, {"max_seed_entries": 1},
                   {"max_factors": 1}):
        with pytest.raises(WorkLimitExceeded):
            abelian_rank_mixed(matrix, **kwargs)


def test_large_zero_axes_cannot_bypass_allocation_guard():
    for matrix in (M.zeros(1, 0, (2**100,)), M.zeros(0, 1, (2**100,))):
        with pytest.raises(WorkLimitExceeded, match="axis"):
            abelian_rank_mixed(matrix)
    empty = M.zeros(0, 0, (2**100,))
    result = abelian_rank_mixed(empty, max_component_axis=0, max_component_entries=0)
    assert result.rank == 0 and result.component_shape == (0, 0)


@pytest.mark.parametrize("bad", [True, -1, 1.5, float("nan")])
def test_strict_mixed_limits(bad):
    with pytest.raises(ValueError):
        abelian_rank_mixed(M.zeros(1, 1, (2,)), max_component_entries=bad)
