"""Small reference codes and parameterized sparse code families."""

from operator import index

from qeclang.algebra import BinaryMatrix
from qeclang.codes import CSSCode, Pauli, StabilizerCode, SubsystemCode
from .products import hypergraph_product


def repetition_check(length: int, *, cyclic=False):
    length = index(length)
    if length < 2:
        raise ValueError("repetition length must be at least two")
    count = length if cyclic else length - 1
    return BinaryMatrix(tuple((1 << i) ^ (1 << ((i + 1) % length)) for i in range(count)),
                        length)


def steane():
    h = BinaryMatrix((0b1010101, 0b1100110, 0b1111000), 7)
    return CSSCode(h, h, name="Steane [[7,1,3]]")


def shor():
    hx = BinaryMatrix((0b000111111, 0b111111000), 9)
    hz = BinaryMatrix(tuple(3 << (3 * block + offset)
                            for block in range(3) for offset in range(2)), 9)
    return CSSCode(hx, hz, name="Shor [[9,1,3]]")


def _pauli(label):
    x = sum(1 << j for j, char in enumerate(label) if char in "XY")
    z = sum(1 << j for j, char in enumerate(label) if char in "ZY")
    return Pauli(x, z, label.count("Y") % 4)


def five_qubit():
    return StabilizerCode(5, tuple(map(_pauli, ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"))),
                          name="perfect [[5,1,3]]")


def toric(length: int, width: int | None = None, *, max_bits=100_000_000):
    """Periodic square lattice product with n=2*length*width and k=2."""
    width = length if width is None else width
    code = hypergraph_product(repetition_check(length, cyclic=True),
                              repetition_check(width, cyclic=True), max_bits=max_bits)
    return CSSCode(code.hx, code.hz, name=f"toric {length}x{width}")


def bacon_shor(rows=3, columns=3):
    """Rectangular Bacon--Shor gauge presentation with k=1 and r=(r-1)(c-1)."""
    rows, columns = index(rows), index(columns)
    if min(rows, columns) < 2:
        raise ValueError("both lattice dimensions must be at least two")
    gauge = []
    for row in range(rows):
        for col in range(columns - 1):
            gauge.append(Pauli((1 << (row * columns + col)) |
                               (1 << (row * columns + col + 1)), 0))
    for row in range(rows - 1):
        for col in range(columns):
            gauge.append(Pauli(0, (1 << (row * columns + col)) |
                               (1 << ((row + 1) * columns + col))))
    return SubsystemCode(rows * columns, tuple(gauge), name="Bacon-Shor")
