from itertools import combinations

import pytest

from qeclang.algebra import BinaryMatrix
from qeclang.codes import CSSCode, ChainComplex, Pauli, StabilizerCode, SubsystemCode


def test_pauli_phase_convention():
    x, y, z = (Pauli.from_label(label) for label in "XYZ")
    assert y == Pauli(1, 1, 1)
    assert x * z == Pauli(1, 1, 0)
    assert z * x == Pauli(1, 1, 2)
    assert x * y == Pauli(0, 1, 1)
    assert y * x == Pauli(0, 1, 3)
    assert x * x == y * y == z * z == Pauli(0, 0)
    assert Pauli.from_label("-YY").phase == 0
    assert Pauli.from_label("IX").x == 2
    assert Pauli.from_label("XYZ").weight == 3
    assert not Pauli(1, 1).is_hermitian


def test_signed_dependent_stabilizers():
    xx, zz, yy = (Pauli.from_label(label) for label in ("XX", "ZZ", "YY"))
    with pytest.raises(ValueError, match="-I"):
        StabilizerCode(2, (xx, zz, yy))
    valid = StabilizerCode(2, (xx, zz, Pauli.from_label("-YY"), Pauli(0, 0)))
    assert valid.k == 0
    assert valid.rank == 2
    assert len(valid.basis) == 2
    assert StabilizerCode(1, (Pauli.from_label("-Z"),)).k == 0
    with pytest.raises(ValueError, match="-I"):
        StabilizerCode(1, (Pauli.from_label("Z"), Pauli.from_label("-Z")))
    with pytest.raises(ValueError, match="Hermitian"):
        StabilizerCode(1, (Pauli(1, 1),))
    with pytest.raises(ValueError, match="commute"):
        StabilizerCode(1, (Pauli.from_label("X"), Pauli.from_label("Z")))
    with pytest.raises(ValueError, match="bits"):
        StabilizerCode(1, (Pauli.from_label("IX"),))


def test_five_qubit_code_and_syndromes():
    code = StabilizerCode(5, tuple(Pauli.from_label(s) for s in ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")))
    assert code.k == 1
    assert code.centralizer().nrows == 6
    syndromes = {code.syndrome(Pauli.from_label("I" * i + letter + "I" * (4 - i)))
                 for i in range(5) for letter in "XYZ"}
    assert syndromes == set(range(1, 16))
    assert code.checks.shape == (4, 10)


def test_css_chain_roundtrip_and_redundant_checks():
    h = BinaryMatrix((0b1010101, 0b1100110, 0b1111000), 7)
    code = CSSCode(h, h, "Steane")
    assert (code.n, code.k) == (7, 1)
    assert code.to_stabilizer().k == 1
    chain = code.to_chain_complex()
    assert chain.to_css() == code
    assert chain.dimensions == (3, 7, 3)
    assert chain.betti_numbers == (0, 1, 0)
    redundant = CSSCode(BinaryMatrix(h.rows + (h.rows[0],), 7), h)
    assert redundant.k == 1
    assert redundant.to_chain_complex().betti_numbers == (1, 1, 0)


def test_invalid_css_chain():
    with pytest.raises(ValueError, match="commute"):
        CSSCode(BinaryMatrix((1,), 1), BinaryMatrix((1,), 1))
    with pytest.raises(ValueError, match="qubits"):
        CSSCode(BinaryMatrix.zeros(0, 1), BinaryMatrix.zeros(0, 2))
    with pytest.raises(ValueError, match="boundary"):
        ChainComplex(BinaryMatrix.identity(2), BinaryMatrix.identity(2))
    with pytest.raises(ValueError, match="dimensions"):
        ChainComplex(BinaryMatrix.identity(2), BinaryMatrix.identity(3))


def test_subsystem_bacon_shor_and_degenerate_limits():
    gauge = tuple(Pauli.from_label(s) for s in ("XXII", "IIXX", "ZIZI", "IZIZ"))
    code = SubsystemCode(4, gauge)
    assert (code.k, code.r) == (1, 1)
    assert code.center.rank() == 2
    assert code.center.contains(Pauli.from_label("XXXX").x)
    assert code.center.contains(Pauli.from_label("ZZZZ").z << 4)
    assert code.stabilizer.k == 2
    assert code.centralizer().nrows == 4
    assert SubsystemCode(1, (Pauli.from_label("X"), Pauli.from_label("Z"))).k == 0
    assert SubsystemCode(1, (Pauli.from_label("X"), Pauli.from_label("Z"))).r == 1
    assert SubsystemCode(3, ()).k == 3
    assert SubsystemCode(3, ()).r == 0
    commuting = SubsystemCode(2, (Pauli.from_label("XX"), Pauli.from_label("ZZ")))
    assert commuting.r == 0 and commuting.k == 0


def test_empty_codes():
    empty = CSSCode(BinaryMatrix.zeros(0, 0), BinaryMatrix.zeros(0, 0))
    assert empty.k == empty.n == 0
    assert empty.to_stabilizer().centralizer().shape == (0, 0)
    assert empty.to_chain_complex().betti_numbers == (0, 0, 0)
    trivial = CSSCode(BinaryMatrix.zeros(0, 3), BinaryMatrix.zeros(0, 3))
    assert trivial.k == 3
    assert StabilizerCode(0, ()).k == 0
    assert SubsystemCode(0, ()).k == 0


def test_subsystem_signed_sector():
    gauge = tuple(Pauli.from_label(s) for s in ("XXII", "IIXX", "ZIZI", "IZIZ"))
    sector = tuple(Pauli.from_label(s) for s in ("-XXXX", "ZZZZ"))
    code = SubsystemCode(4, gauge, sector=sector)
    assert code.stabilizer.generators == sector
    assert (code.k, code.r) == (1, 1)
    with pytest.raises(ValueError, match="exactly"):
        SubsystemCode(4, gauge, sector=sector[:1])
    with pytest.raises(ValueError, match="exactly"):
        SubsystemCode(4, gauge, sector=(Pauli.from_label("XXII"), Pauli.from_label("ZZZZ")))
    with pytest.raises(ValueError, match="-I"):
        SubsystemCode(4, gauge, sector=sector + (Pauli.from_label("XXXX"),))


def test_all_two_qubit_gauge_modules_have_valid_center():
    operators = tuple(Pauli(v & 3, v >> 2) for v in range(1, 16))
    for size in range(4):
        for selected in combinations(operators, size):
            code = SubsystemCode(2, selected)
            assert code.k >= 0 and code.r >= 0
            assert code.checks.rank() == code.center.rank() + 2 * code.r
            assert code.k + code.r + code.center.rank() == 2
            for center in code.center.rows:
                assert code.checks.contains(center)
                assert code.centralizer().contains(center)
            assert code.stabilizer.k == code.k + code.r
