from itertools import product

import pytest

from qeclang.algebra import SparseBinaryMatrix
from qeclang.symbolic import GroupAlgebraElement as G
from qeclang.symbolic.matrix import GroupAlgebraMatrix as M


def test_all_small_group_elements_sparse_and_packed_lifts_agree():
    for orders in ((1,), (2,), (3,), (2, 2), (2, 3)):
        terms = tuple(product(*(range(order) for order in orders)))
        for word in range(1 << len(terms)):
            element = G.from_terms(orders, (term for j, term in enumerate(terms) if (word >> j) & 1))
            sparse = element.lower_sparse()
            assert sparse.to_packed() == element.lower()
            assert sparse.nnz == element.size * len(element.terms)
            assert sparse.row_weights == (len(element.terms),) * element.size
            assert sparse.column_weights == sparse.row_weights
            assert element.involution().lower_sparse() == sparse.T


def test_matrix_sparse_lift_adjoint_and_products():
    elements = tuple(G.from_terms((3,), ((j,) for j in range(3) if (v >> j) & 1))
                     for v in range(8))
    for a, b in product(elements, repeat=2):
        left = M(((a, b), (b.involution(), a)), 2, (3,))
        right = M(((b,), (a,)), 1, (3,))
        sparse = left.lower_sparse()
        assert sparse.to_packed() == left.lower()
        assert sparse.T == left.adjoint().lower_sparse()
        assert (left @ right).lower_sparse().to_packed() == sparse.to_packed() @ right.lower()
        assert sparse.nnz == left.group_size * sum(left.row_weights)


def test_sparse_pipeline_does_not_use_packed_lowering(monkeypatch):
    element = G.from_terms((17, 19), ((0, 0), (1, 0), (0, 1)))
    matrix = M(((element, element.involution()),), 2, element.orders)

    def forbidden(*args, **kwargs):
        raise AssertionError("sparse lift must not call packed lowering")

    monkeypatch.setattr(G, "lower", forbidden)
    monkeypatch.setattr(M, "lower", forbidden)
    assert element.lower_sparse().nnz == 323 * 3
    assert matrix.lower_sparse().nnz == 323 * 6


def test_sparse_lift_budgets_count_dimensions_and_nonzeros():
    element = G.from_terms((1000,), ((0,), (1,), (2,)))
    assert element.lower_sparse(max_nonzeros=3000, max_sites=1000).nnz == 3000
    with pytest.raises(ValueError, match="nonzeros"):
        element.lower_sparse(max_nonzeros=2999)
    with pytest.raises(ValueError, match="sites"):
        element.lower_sparse(max_sites=999)
    matrix = M(((element, element),), 2, (1000,))
    assert matrix.lower_sparse(max_nonzeros=6000, max_sites=2000).nnz == 6000
    with pytest.raises(ValueError, match="nonzeros"):
        matrix.lower_sparse(max_nonzeros=5999)
    with pytest.raises(ValueError, match="sites"):
        matrix.lower_sparse(max_sites=1999)


def test_huge_zero_lifts_reject_before_enumeration():
    zero = G.from_terms((10**12,), [])
    with pytest.raises(ValueError, match="sites"):
        zero.lower_sparse(max_nonzeros=0)
    for matrix in (M.zeros(1, 0, (10**12,)), M.zeros(0, 1, (10**12,))):
        with pytest.raises(ValueError, match="sites"):
            matrix.lower_sparse(max_nonzeros=0)
        with pytest.raises(ValueError, match="sites"):
            matrix.lower(max_bits=0)
    empty = M.zeros(0, 0, (10**12,))
    assert empty.lower_sparse(max_nonzeros=0, max_sites=0).shape == (0, 0)
    no_rows = M.zeros(0, 10**12, (1,))
    assert no_rows.rows == ()
    with pytest.raises(ValueError, match="sites"):
        no_rows.lower_sparse()


def test_empty_dimensions_and_generator_orders():
    for m, n in product(range(3), repeat=2):
        matrix = M.zeros(m, n, iter((2, 3)))
        assert matrix.orders == (2, 3)
        assert matrix.lower_sparse() == SparseBinaryMatrix.zeros(m * 6, n * 6)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5, -1, True])
def test_strict_budget_arguments(bad):
    element = G.from_terms((1,), [])
    matrix = M(((element,),), 1, (1,))
    for lower, arguments in ((element.lower, {"max_bits": bad}),
                             (element.lower_sparse, {"max_sites": bad}),
                             (element.lower_sparse, {"max_nonzeros": bad}),
                             (matrix.lower_sparse, {"max_nonzeros": bad}),
                             (matrix.lower_sparse, {"max_sites": bad})):
        with pytest.raises(ValueError):
            lower(**arguments)
