from itertools import product

import pytest

from qeclang.algebra import BinaryMatrix, SparseBinaryMatrix


def test_all_matrices_through_three_by_three_roundtrip_and_products():
    for nrows in range(4):
        for ncols in range(4):
            for rows in product(range(1 << ncols), repeat=nrows):
                packed = BinaryMatrix(rows, ncols)
                sparse = SparseBinaryMatrix.from_packed(packed)
                assert sparse.to_packed() == packed
                assert sparse.T.to_packed() == packed.T
                assert sparse.T.T == sparse
                assert sparse.nnz == packed.nnz
                assert sparse.row_weights == tuple(row.bit_count() for row in packed.rows)
                assert sparse.column_weights == tuple(row.bit_count() for row in packed.T.rows)
                for vector in range(1 << ncols):
                    assert sparse.matvec(vector) == packed.matvec(vector)
                    bits = tuple((vector >> j) & 1 for j in range(ncols))
                    result = sparse.matvec_bits(bits)
                    assert sum(bit << i for i, bit in enumerate(result)) == packed.matvec(vector)


def test_support_and_coordinate_semantics():
    assert SparseBinaryMatrix.from_supports(((3, 1), ()), 4).rows == ((1, 3), ())
    with pytest.raises(ValueError, match="distinct"):
        SparseBinaryMatrix.from_supports(((1, 1),), 3)
    matrix = SparseBinaryMatrix.from_coo((0, 0, 0, 1, 1), (1, 1, 2, 0, 1),
                                         (1, 1, 1, 0, 1), shape=(2, 3))
    assert matrix.rows == ((2,), (1,))
    assert SparseBinaryMatrix.from_coo((0, 0), (0, 0), shape=(1, 1)).is_zero
    assert SparseBinaryMatrix.from_coo((), (), shape=(0, 7)).shape == (0, 7)


@pytest.mark.parametrize("row_indices,column_indices,data,shape", [
    ((0,), (0,), (0.5,), (1, 1)),
    ((0,), (0,), (1.0,), (1, 1)),
    ((0,), (0,), (2,), (1, 1)),
    ((0,), (0,), (-1,), (1, 1)),
    ((0,), (1,), (1,), (1, 1)),
    ((1,), (0,), (1,), (1, 1)),
    ((0,), (0,), (), (1, 1)),
    ((0,), (), (1,), (1, 1)),
    ((0.0,), (0,), (1,), (1, 1)),
    ((), (), (), (-1, 1)),
])
def test_invalid_coordinate_data(row_indices, column_indices, data, shape):
    with pytest.raises(ValueError):
        SparseBinaryMatrix.from_coo(row_indices, column_indices, data, shape=shape)


def test_materialization_guard_and_immutability():
    matrix = SparseBinaryMatrix(((1_000_000,),), 1_000_001)
    assert matrix.nnz == 1
    with pytest.raises(ValueError, match="max_bits"):
        matrix.to_packed(max_bits=1_000_000)
    with pytest.raises(ValueError):
        matrix.to_packed(max_bits=-1)
    with pytest.raises(AttributeError):
        matrix.ncols = 2
    with pytest.raises(ValueError):
        matrix.matvec(1 << 1_000_001)
    with pytest.raises(ValueError):
        matrix.matvec_bits((0,))


def test_scipy_roundtrip_formats_and_duplicate_cancellation():
    np = pytest.importorskip("numpy")
    scipy = pytest.importorskip("scipy.sparse")
    original = SparseBinaryMatrix(((0, 2), (), (1,)), 4)
    for format in ("coo", "csr", "csc", "lil", "dok", "bsr", "dia"):
        assert SparseBinaryMatrix.from_scipy(original.to_scipy(format=format)) == original
    duplicate = scipy.coo_matrix((np.array([1, 1, 1]), ([0, 0, 1], [0, 0, 1])), shape=(2, 2))
    assert SparseBinaryMatrix.from_scipy(duplicate).rows == ((), (1,))
    boolean = scipy.csr_array(np.array([[True, False], [False, True]]))
    assert SparseBinaryMatrix.from_scipy(boolean).rows == ((0,), (1,))
    for data in (np.array([1.0]), np.array([0.5]), np.array([2]), np.array([-1])):
        invalid = scipy.coo_matrix((data, ([0], [0])), shape=(1, 1))
        with pytest.raises(ValueError):
            SparseBinaryMatrix.from_scipy(invalid)
    for shape in ((0, 3), (3, 0), (0, 0)):
        empty = SparseBinaryMatrix.zeros(*shape)
        assert SparseBinaryMatrix.from_scipy(empty.to_scipy()) == empty
