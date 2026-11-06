from itertools import permutations, product

import pytest

from qeclang.algebra import BinaryMatrix
from qeclang.codes import CSSCode, ChainComplex
from qeclang.transforms import (ChainMap, hadamard_dual, homology_basis,
                                independent_checks, permute_qubits)


def test_site_permutations_are_reversible_and_preserve_weight():
    code = CSSCode(BinaryMatrix((3, 12), 4), BinaryMatrix((15,), 4))
    for permutation in permutations(range(4)):
        changed = permute_qubits(code, permutation)
        inverse = tuple(permutation.index(j) for j in range(4))
        assert permute_qubits(changed, inverse) == code
        assert changed.k == code.k
        assert [r.bit_count() for r in changed.hx.rows] == [2, 2]
        assert changed.hz.rows == (15,)
    moved = permute_qubits(code, (2, 0, 3, 1))
    assert moved.hx.rows == (5, 10)


@pytest.mark.parametrize("permutation", [(0, 0), (0,), (0, 2), (0, 1.0), (0, True), (-1, 0)])
def test_invalid_site_permutations(permutation):
    code = CSSCode(BinaryMatrix.zeros(0, 2), BinaryMatrix.zeros(0, 2))
    with pytest.raises(ValueError):
        permute_qubits(code, permutation)


def test_duality_and_independent_checks():
    code = CSSCode(BinaryMatrix((3, 6, 5), 3), BinaryMatrix.zeros(0, 3))
    assert hadamard_dual(hadamard_dual(code)) == code
    reduced = independent_checks(code)
    assert reduced.k == code.k
    assert reduced.hx.nrows == 2
    assert all(reduced.hx.contains(row) for row in code.hx.rows)
    assert all(code.hx.contains(row) for row in reduced.hx.rows)
    empty = CSSCode(BinaryMatrix.zeros(0, 0), BinaryMatrix.zeros(0, 0))
    assert permute_qubits(empty, ()) == empty


def sample_complex():
    # C1 coordinate 0 is a boundary, coordinate 1 is homological, coordinate 2
    # has nonzero boundary in C0. H0 and H2 each also have dimension one.
    return ChainComplex(BinaryMatrix((4, 0), 3), BinaryMatrix((1, 0, 0), 2))


def test_identity_and_homology_coordinates():
    complex = sample_complex()
    assert complex.betti_numbers == (1, 1, 1)
    assert homology_basis(complex, 0).rows == (2,)
    assert homology_basis(complex, 1).rows == (2,)
    assert homology_basis(complex, 2).rows == (2,)
    identity = ChainMap.identity(complex)
    for degree in range(3):
        assert identity.homology_action(degree) == BinaryMatrix.identity(1)
    assert identity.is_quasi_isomorphism()
    assert identity.then(identity) == identity


def test_homology_action_quotients_out_boundaries():
    complex = sample_complex()
    # Add a boundary to the H1 representative: e1 -> e1 + e0.
    map = ChainMap(complex, complex, BinaryMatrix.identity(2),
                   BinaryMatrix((3, 2, 4), 3), BinaryMatrix.identity(2))
    assert map.f1.matvec(2) == 3
    assert map.homology_action() == BinaryMatrix.identity(1)
    assert map.is_quasi_isomorphism()
    assert map.then(map) == ChainMap.identity(complex)
    # Collapse the logical coordinate to a boundary while retaining chain squares.
    collapse = ChainMap(complex, complex, BinaryMatrix.identity(2),
                        BinaryMatrix((3, 0, 4), 3), BinaryMatrix.identity(2))
    assert collapse.homology_action() == BinaryMatrix.zeros(1, 1)
    assert not collapse.is_quasi_isomorphism()


def test_chain_map_rejects_invalid_shapes_and_squares():
    complex = sample_complex()
    with pytest.raises(ValueError, match="dimensions"):
        ChainMap(complex, complex, BinaryMatrix.identity(3),
                 BinaryMatrix.identity(3), BinaryMatrix.identity(2))
    with pytest.raises(ValueError, match="d1"):
        ChainMap(complex, complex, BinaryMatrix.zeros(2, 2),
                 BinaryMatrix.identity(3), BinaryMatrix.identity(2))
    with pytest.raises(ValueError, match="d2"):
        ChainMap(complex, complex, BinaryMatrix.identity(2),
                 BinaryMatrix.identity(3), BinaryMatrix.zeros(2, 2))
    with pytest.raises(ValueError, match="degree"):
        homology_basis(complex, 3)


def test_functoriality_for_all_maps_on_a_small_complex():
    complex = ChainComplex(BinaryMatrix((0,), 2), BinaryMatrix((1, 0), 1))
    maps = []
    for f0, f1a, f1b, f2 in product(range(2), range(4), range(4), range(2)):
        try:
            maps.append(ChainMap(complex, complex, BinaryMatrix((f0,), 1),
                                 BinaryMatrix((f1a, f1b), 2), BinaryMatrix((f2,), 1)))
        except ValueError:
            pass
    assert len(maps) == 16
    for first in maps:
        for second in maps:
            composed = first.then(second)
            for degree in range(3):
                assert composed.homology_action(degree) == (
                    second.homology_action(degree) @ first.homology_action(degree))


def test_quasi_isomorphism_can_remove_contractible_summands():
    source = sample_complex()
    target = ChainComplex(BinaryMatrix.zeros(1, 1), BinaryMatrix.zeros(1, 1))
    projection = ChainMap(source, target, BinaryMatrix((2,), 2),
                          BinaryMatrix((2,), 3), BinaryMatrix((2,), 2))
    assert projection.is_quasi_isomorphism()
    assert source.dimensions != target.dimensions
    for degree in range(3):
        assert projection.homology_action(degree) == BinaryMatrix.identity(1)


def test_zero_complex_and_name_insensitive_composition():
    zero = ChainComplex(BinaryMatrix.zeros(0, 0), BinaryMatrix.zeros(0, 0))
    assert ChainMap.identity(zero).is_quasi_isomorphism()
    for degree in range(3):
        assert ChainMap.identity(zero).homology_action(degree).shape == (0, 0)
    complex = sample_complex()
    renamed = ChainComplex(complex.d1, complex.d2, "different label")
    combined = ChainMap.identity(complex).then(ChainMap.identity(renamed))
    assert combined.is_quasi_isomorphism()
    with pytest.raises(ValueError, match="intermediate"):
        ChainMap.identity(complex).then(ChainMap.identity(zero))
