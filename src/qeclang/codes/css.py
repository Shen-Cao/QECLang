"""Binary CSS checks and three-term chain complexes."""

from dataclasses import dataclass

from qeclang.algebra import BinaryMatrix

from .pauli import Pauli
from .stabilizer import StabilizerCode


@dataclass(frozen=True)
class CSSCode:
    """CSS code with positive X/Z generators in the physical qubit basis.

    Sparse check presentations are retained rather than row-reduced. Both row
    and column weights matter for an LDPC family; sparsity is not a property of
    the abstract row space alone.
    """

    hx: BinaryMatrix
    hz: BinaryMatrix
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.hx, BinaryMatrix) or not isinstance(self.hz, BinaryMatrix):
            raise TypeError("CSS checks must be BinaryMatrix objects")
        if self.hx.ncols != self.hz.ncols:
            raise ValueError("X and Z checks must have the same number of qubits")
        # Transposing a presentation with a large declared width can allocate
        # far more columns than there are stored rows or support bits.
        pairs = self.hx.nrows * self.hz.nrows
        if not pairs:
            return
        if pairs <= self.n:
            noncommuting = any((x & z).bit_count() % 2 for x in self.hx.rows for z in self.hz.rows)
        else:
            noncommuting = not (self.hx @ self.hz.T).is_zero
        if noncommuting:
            raise ValueError("CSS checks do not commute: hx @ hz.T is nonzero")

    @property
    def n(self) -> int:
        return self.hx.ncols

    @property
    def k(self) -> int:
        return self.n - self.hx.rank() - self.hz.rank()

    def to_stabilizer(self) -> StabilizerCode:
        return StabilizerCode(
            self.n,
            tuple(Pauli(row, 0) for row in self.hx.rows) + tuple(Pauli(0, row) for row in self.hz.rows),
            self.name,
        )

    def to_chain_complex(self) -> "ChainComplex":
        return ChainComplex(self.hx, self.hz.T, self.name)


@dataclass(frozen=True)
class ChainComplex:
    """Based binary complex C2 --d2--> C1 --d1--> C0.

    Matrices act on column vectors. The physical basis is the basis of C1;
    arbitrary basis changes preserve homology dimension but need not preserve
    Hamming distance, locality, or the LDPC presentation.
    """

    d1: BinaryMatrix
    d2: BinaryMatrix
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.d1, BinaryMatrix) or not isinstance(self.d2, BinaryMatrix):
            raise TypeError("boundary maps must be BinaryMatrix objects")
        if self.d1.ncols != self.d2.nrows:
            raise ValueError("boundary dimensions do not match at C1")
        if not (self.d1 @ self.d2).is_zero:
            raise ValueError("boundary of a boundary must vanish")

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """Dimensions in degree order (C0, C1, C2)."""
        return self.d1.nrows, self.d1.ncols, self.d2.ncols

    @property
    def betti_numbers(self) -> tuple[int, int, int]:
        c0, c1, c2 = self.dimensions
        r1, r2 = self.d1.rank(), self.d2.rank()
        return c0 - r1, c1 - r1 - r2, c2 - r2

    def to_css(self) -> CSSCode:
        return CSSCode(self.d1, self.d2.T, self.name)
