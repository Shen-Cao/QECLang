from itertools import product

import pytest

from qeclang.algebra import BinaryMatrix, hstack, kron, vstack


def span(rows):
    values = {0}
    for row in rows:
        values |= {value ^ row for value in values}
    return values


def test_all_binary_matrices_through_three_by_three():
    for nrows in range(4):
        for ncols in range(4):
            for rows in product(range(1 << ncols), repeat=nrows):
                matrix = BinaryMatrix(rows, ncols)
                brute_span = span(rows)
                assert 1 << matrix.rank() == len(brute_span)
                assert span(matrix.row_basis().rows) == brute_span
                kernel = {v for v in range(1 << ncols) if matrix.matvec(v) == 0}
                assert span(matrix.nullspace().rows) == kernel
                assert matrix.nullspace().nrows == ncols - matrix.rank()
                assert matrix.T.T == matrix
                for vector in range(1 << ncols):
                    assert matrix.contains(vector) == (vector in brute_span)
                image = {matrix.matvec(v) for v in range(1 << ncols)}
                for rhs in range(1 << nrows):
                    if rhs in image:
                        assert matrix.matvec(matrix.solve(rhs)) == rhs
                    else:
                        with pytest.raises(ValueError, match="inconsistent"):
                            matrix.solve(rhs)


def test_product_matches_direct_dot_products():
    for left_rows in product(range(8), repeat=2):
        left = BinaryMatrix(left_rows, 3)
        for right_rows in product(range(4), repeat=3):
            right = BinaryMatrix(right_rows, 2)
            result = left @ right
            expected = [[sum(a * b for a, b in zip(row, col)) % 2
                         for col in zip(*right.to_dense())] for row in left.to_dense()]
            assert result == BinaryMatrix.from_dense(expected)


def test_blocks_kronecker_and_zero_dimensions():
    a = BinaryMatrix.from_dense([[1, 0], [1, 1]])
    b = BinaryMatrix.from_dense([[1, 1, 0]])
    assert kron(a, b).to_dense() == [[1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0]]
    assert hstack(a, a).to_dense() == [[1, 0, 1, 0], [1, 1, 1, 1]]
    assert vstack(a, a).rows == a.rows + a.rows
    assert (a + a).is_zero
    assert a.nnz == 3
    assert (BinaryMatrix.zeros(2, 0) @ BinaryMatrix.zeros(0, 3)).shape == (2, 3)
    assert kron(BinaryMatrix.zeros(0, 5), a).shape == (0, 10)
    assert kron(BinaryMatrix.zeros(3, 0), a).shape == (6, 0)
    assert hstack(BinaryMatrix.zeros(0, 2), BinaryMatrix.zeros(0, 3)).shape == (0, 5)
    assert vstack(BinaryMatrix.zeros(2, 0), BinaryMatrix.zeros(3, 0)).shape == (5, 0)
    assert BinaryMatrix.from_dense([], ncols=5).shape == (0, 5)


@pytest.mark.parametrize("rows,width", [((-1,), 2), ((4,), 2), ((1.0,), 1), ((True,), 1), ((), -1), ((), 1.5)])
def test_invalid_packed_data(rows, width):
    with pytest.raises(ValueError):
        BinaryMatrix(rows, width)


@pytest.mark.parametrize("dense", [[[0, 2]], [[0.0]], [[0], [1, 1]]])
def test_invalid_dense_data(dense):
    with pytest.raises(ValueError):
        BinaryMatrix.from_dense(dense)


def test_bad_dimensions_and_vectors():
    a = BinaryMatrix.identity(2)
    with pytest.raises(ValueError):
        a @ BinaryMatrix.identity(3)
    with pytest.raises(ValueError):
        a + BinaryMatrix.identity(3)
    with pytest.raises(ValueError):
        hstack(a, BinaryMatrix.identity(3))
    with pytest.raises(ValueError):
        vstack(a, BinaryMatrix.identity(3))
    for operation in (a.matvec, a.contains, a.solve):
        with pytest.raises(ValueError):
            operation(4)


def test_wide_vectors_and_immutability():
    a = BinaryMatrix((1 | (1 << 200),), 201)
    assert a.rank() == 1
    assert a.nullspace().nrows == 200
    assert a.T.T == a
    with pytest.raises(AttributeError):
        a.ncols = 10
