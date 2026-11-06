"""Phase-aware qubit Pauli operators."""

from dataclasses import dataclass

from qeclang.algebra.binary import nonnegative_integer


@dataclass(frozen=True)
class Pauli:
    """The operator ``i**phase X**x Z**z``.

    Bit ``j`` addresses physical qubit ``j``. A Y on one qubit is
    ``Pauli(1, 1, 1)``; ``Pauli(1, 1, 0)`` is anti-Hermitian.
    Width is provided by the enclosing code, so trailing identities need no bits.
    """

    x: int
    z: int
    phase: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", nonnegative_integer(self.x, "x"))
        object.__setattr__(self, "z", nonnegative_integer(self.z, "z"))
        object.__setattr__(self, "phase", nonnegative_integer(self.phase, "phase") % 4)

    @property
    def weight(self) -> int:
        return (self.x | self.z).bit_count()

    @property
    def is_hermitian(self) -> bool:
        return (self.phase - (self.x & self.z).bit_count()) % 2 == 0

    def commutes(self, other: "Pauli") -> bool:
        return ((self.x & other.z).bit_count() + (self.z & other.x).bit_count()) % 2 == 0

    def __mul__(self, other: "Pauli") -> "Pauli":
        if not isinstance(other, Pauli):
            return NotImplemented
        phase = (self.phase + other.phase + 2 * (self.z & other.x).bit_count()) % 4
        return Pauli(self.x ^ other.x, self.z ^ other.z, phase)

    @classmethod
    def from_label(cls, label: str) -> "Pauli":
        """Read an optional sign and I/X/Y/Z, with qubit zero at the left."""
        if not isinstance(label, str):
            raise ValueError("Pauli label must be a string")
        phase = 0
        if label.startswith(("+", "-")):
            phase = 2 if label[0] == "-" else 0
            label = label[1:]
        x = z = 0
        for j, letter in enumerate(label):
            if letter not in "IXYZ":
                raise ValueError("Pauli labels contain only I, X, Y, Z and an optional leading sign")
            if letter in "XY":
                x |= 1 << j
            if letter in "YZ":
                z |= 1 << j
            if letter == "Y":
                phase += 1
        return cls(x, z, phase)


def symplectic_product(left: int, right: int, n: int) -> int:
    """Symplectic form on packed ``x | (z << n)`` vectors."""
    mask = (1 << n) - 1
    return (((left & mask) & (right >> n)).bit_count() +
            ((left >> n) & (right & mask)).bit_count()) & 1
