from itertools import product

import pytest

from qeclang.algebra import BinaryMatrix
from qeclang.analysis import css_profile
from qeclang.constructions import bicycle, hypergraph_product
from qeclang.constructions.lifted import LiftedProduct
from qeclang.symbolic import GroupAlgebraElement as G
from qeclang.symbolic.matrix import GroupAlgebraMatrix as M


def test_ring_matrix_representation():
    elements = [G.from_terms((3,), ((j,) for j in range(3) if (v >> j) & 1)) for v in range(8)]
    for a, b in product(elements, repeat=2):
        left = M(((a, b), (b.involution(), a)), 2, (3,))
        right = M(((b,), (a,)), 1, (3,))
        assert (left @ right).lower() == left.lower() @ right.lower()
        assert left.adjoint().lower() == left.lower().T
        hx, hz = LiftedProduct(left, right).symbolic_checks()
        assert (hx @ hz.adjoint()).is_zero
        code = LiftedProduct(left, right).lower()
        assert (code.hx @ code.hz.T).is_zero
        symbolic = LiftedProduct(left, right).locality()
        actual = css_profile(code)
        assert symbolic.max_check_weight == actual.max_check_weight
        assert symbolic.max_qubit_degree == actual.max_qubit_degree
        assert symbolic.x_nonzeros == actual.x.nonzeros
        assert symbolic.z_nonzeros == actual.z.nonzeros


def test_trivial_group_matches_hgp():
    zero, one = G.from_terms((1,), []), G.from_terms((1,), [(0,)])
    a = M(((one, one, zero), (zero, one, one)), 3, (1,))
    b = M(((one, one),), 2, (1,))
    product_code = LiftedProduct(a, b).lower()
    ordinary = hypergraph_product(a.lower(), b.lower())
    assert product_code.hx == ordinary.hx
    assert product_code.hz == ordinary.hz


def test_scalar_bicycle_and_size():
    a = G.from_terms((5,), [(0,), (1,)])
    b = G.from_terms((5,), [(0,), (2,)])
    construction = LiftedProduct(M(((a,),), 1, (5,)), M(((b.involution(),),), 1, (5,)))
    code = construction.lower()
    assert code.hx == bicycle(a, b).hx
    assert code.hz == bicycle(a, b).hz
    assert construction.n == 10
    assert construction.check_counts == (5, 5)
    assert hypergraph_product(a.lower(), b.lower()).n == 50
    sparse_x, sparse_z = construction.sparse_checks()
    assert sparse_x.to_packed() == code.hx
    assert sparse_z.to_packed() == code.hz
    with pytest.raises(ValueError, match="nonzeros"):
        construction.sparse_checks(max_nonzeros=1)
    with pytest.raises(ValueError):
        construction.lower(max_bits=10)
    with pytest.raises(ValueError):
        construction.symbolic_checks(max_blocks=1)


def test_empty_blocks_and_invalid_rings():
    for m, n in product(range(3), repeat=2):
        matrix = M.zeros(m, n, (2,))
        assert matrix.lower() == BinaryMatrix.zeros(2*m, 2*n)
        assert matrix.adjoint().lower() == matrix.lower().T
    code = LiftedProduct(M.zeros(0, 2, (3,)), M.zeros(0, 3, (3,))).lower()
    assert (code.n, code.k) == (18, 18)
    with pytest.raises(ValueError):
        LiftedProduct(M.zeros(1, 1, (2,)), M.zeros(1, 1, (3,)))
    with pytest.raises(ValueError):
        M(((G.from_terms((2,), []),),), 2, (2,))
    with pytest.raises(ValueError, match="intermediate"):
        LiftedProduct(M.zeros(0, 100_000, (1,)), M.zeros(0, 1, (1,))).symbolic_checks()
