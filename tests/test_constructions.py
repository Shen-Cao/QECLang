import pytest

from qeclang.algebra import BinaryMatrix
from qeclang.constructions import (
    bacon_shor, bicycle, direct_sum, five_qubit, hypergraph_product,
    repetition_check, shor, steane, toric,
)
from qeclang.symbolic import GroupAlgebraElement as G


def test_catalog_parameters():
    assert (steane().n, steane().k) == (7, 1)
    assert (shor().n, shor().k) == (9, 1)
    assert (five_qubit().n, five_qubit().k) == (5, 1)
    assert (bacon_shor().n, bacon_shor().k, bacon_shor().r) == (9, 1, 4)
    for length in range(2, 6):
        code = toric(length)
        assert (code.n, code.k) == (2 * length**2, 2)
        assert max(row.bit_count() for row in code.hx.rows) == 4


def test_hgp_dimension_with_redundancy():
    for m in (2, 3, 4):
        for n in (2, 3):
            a = repetition_check(m, cyclic=True)
            b = repetition_check(n)
            code = hypergraph_product(a, b)
            expected = (a.ncols-a.rank())*(b.ncols-b.rank()) + (a.nrows-a.rank())*(b.nrows-b.rank())
            assert code.k == expected
    assert hypergraph_product(BinaryMatrix.zeros(0, 2), BinaryMatrix.zeros(0, 3)).k == 6


def test_direct_sum_and_bounds():
    code = direct_sum(steane(), shor())
    assert (code.n, code.k) == (16, 2)
    with pytest.raises(ValueError):
        toric(100, max_bits=100)
    with pytest.raises(ValueError):
        repetition_check(1)


def test_symbolic_homomorphism():
    a = G.from_terms((3, 4), [(0, 0), (1, 0), (0, 1)])
    b = G.from_terms((3, 4), [(2, 0), (0, 2)])
    assert (a*b).lower() == a.lower() @ b.lower()
    assert a.involution().lower() == a.lower().T
    assert (a+b).terms == a.terms ^ b.terms
    assert (a+a).terms == frozenset()
    assert G.from_terms((3,), [(0,), (3,)]).terms == frozenset()
    assert bicycle(a, b).n == 24
    with pytest.raises(ValueError):
        a.lower(max_bits=10)
    with pytest.raises(ValueError):
        a + G.from_terms((3,), [(0,)])
