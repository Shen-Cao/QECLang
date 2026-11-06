"""Physical CSS transformations and algebraic maps of based complexes.

Physical relabelings preserve Pauli weight. A general chain map, even one
inducing homology isomorphisms, need not preserve distance or locality and does
not by itself specify a physical encoding circuit.
"""

from dataclasses import dataclass

from qeclang.algebra import BinaryMatrix, vstack
from qeclang.algebra.binary import nonnegative_integer, set_bits
from qeclang.analysis.logicals import quotient_basis
from qeclang.codes import CSSCode, ChainComplex


def permute_qubits(code: CSSCode, permutation) -> CSSCode:
    """Relabel old site ``i`` as ``permutation[i]`` using a strict bijection."""
    if not isinstance(code, CSSCode):
        raise TypeError("qubit permutation currently supports CSSCode")
    permutation = tuple(nonnegative_integer(i, "site index") for i in permutation)
    if len(permutation) != code.n or set(permutation) != set(range(code.n)):
        raise ValueError("permutation must be a bijection of all physical sites")

    def permute(matrix):
        return BinaryMatrix(tuple(sum(1 << permutation[j] for j in set_bits(row))
                                  for row in matrix.rows), code.n)

    return CSSCode(permute(code.hx), permute(code.hz), code.name)


def hadamard_dual(code: CSSCode) -> CSSCode:
    """Apply Hadamard on every physical qubit, exchanging X and Z checks."""
    if not isinstance(code, CSSCode):
        raise TypeError("Hadamard dual currently supports CSSCode")
    return CSSCode(code.hz, code.hx, code.name)


def independent_checks(code: CSSCode) -> CSSCode:
    """Choose independent RREF checks for the same positive CSS stabilizer.

    This preserves the subspace and distance, but row weights, incidence degrees,
    syndrome bits, and a corresponding measurement schedule may change.
    """
    if not isinstance(code, CSSCode):
        raise TypeError("check reduction currently supports CSSCode")
    return CSSCode(code.hx.row_basis(), code.hz.row_basis(), code.name)


def _degree(degree: int) -> int:
    degree = nonnegative_integer(degree, "degree")
    if degree > 2:
        raise ValueError("degree must be 0, 1, or 2")
    return degree


def _cycles_and_boundaries(complex: ChainComplex, degree: int):
    if degree == 0:
        outgoing = BinaryMatrix.zeros(0, complex.dimensions[0])
        incoming = complex.d1
    elif degree == 1:
        outgoing, incoming = complex.d1, complex.d2
    else:
        outgoing = complex.d2
        incoming = BinaryMatrix.zeros(complex.dimensions[2], 0)
    return outgoing.nullspace(), incoming.T.row_basis()


def homology_basis(complex: ChainComplex, degree: int = 1) -> BinaryMatrix:
    """Deterministic quotient representatives as rows in the physical basis.

    In degree one these represent Z logical operators. These coordinates are
    algebraic choices, not minimum-weight representatives.
    """
    if not isinstance(complex, ChainComplex):
        raise TypeError("homology requires a ChainComplex")
    degree = _degree(degree)
    cycles, boundaries = _cycles_and_boundaries(complex, degree)
    return BinaryMatrix(quotient_basis(cycles, boundaries), complex.dimensions[degree])


@dataclass(frozen=True)
class ChainMap:
    """A degree-preserving map of three-term binary based chain complexes.

    ``fi`` has shape ``(target.dimensions[i], source.dimensions[i])``. Chain
    squares commute exactly over GF(2), so cycles map to cycles and boundaries
    to boundaries. Names do not affect composition compatibility.
    """

    source: ChainComplex
    target: ChainComplex
    f0: BinaryMatrix
    f1: BinaryMatrix
    f2: BinaryMatrix

    def __post_init__(self):
        if not isinstance(self.source, ChainComplex) or not isinstance(self.target, ChainComplex):
            raise TypeError("source and target must be ChainComplex objects")
        for degree, matrix in enumerate((self.f0, self.f1, self.f2)):
            if not isinstance(matrix, BinaryMatrix):
                raise TypeError("chain maps must be BinaryMatrix objects")
            if matrix.shape != (self.target.dimensions[degree], self.source.dimensions[degree]):
                raise ValueError(f"chain map dimensions disagree in degree {degree}")
        if self.target.d1 @ self.f1 != self.f0 @ self.source.d1:
            raise ValueError("chain map does not commute with d1")
        if self.target.d2 @ self.f2 != self.f1 @ self.source.d2:
            raise ValueError("chain map does not commute with d2")

    @classmethod
    def identity(cls, complex: ChainComplex) -> "ChainMap":
        if not isinstance(complex, ChainComplex):
            raise TypeError("identity map requires a ChainComplex")
        return cls(complex, complex, *(BinaryMatrix.identity(n) for n in complex.dimensions))

    def then(self, next_map: "ChainMap") -> "ChainMap":
        """Compose in application order: ``next_map`` after ``self``."""
        if not isinstance(next_map, ChainMap):
            raise TypeError("composition requires a ChainMap")
        if (self.target.d1, self.target.d2) != (next_map.source.d1, next_map.source.d2):
            raise ValueError("composition requires identical intermediate based complexes")
        return ChainMap(self.source, next_map.target, next_map.f0 @ self.f0,
                        next_map.f1 @ self.f1, next_map.f2 @ self.f2)

    def homology_action(self, degree: int = 1) -> BinaryMatrix:
        """Induced map with target coordinates as rows, source as columns.

        Coordinates use ``homology_basis`` on each complex. Each image is solved
        modulo target boundaries rather than compared as a physical vector.
        """
        degree = _degree(degree)
        source_basis = homology_basis(self.source, degree)
        target_basis = homology_basis(self.target, degree)
        _, boundaries = _cycles_and_boundaries(self.target, degree)
        cycle_coordinates = vstack(boundaries, target_basis).T
        matrix = (self.f0, self.f1, self.f2)[degree]
        columns = tuple(cycle_coordinates.solve(matrix.matvec(vector)) >> boundaries.nrows
                        for vector in source_basis.rows)
        return BinaryMatrix(columns, target_basis.nrows).T

    def is_quasi_isomorphism(self) -> bool:
        """Whether all induced homology maps are invertible (no metric claim)."""
        for degree in range(3):
            action = self.homology_action(degree)
            if action.nrows != action.ncols or action.rank() != action.nrows:
                return False
        return True


__all__ = ["permute_qubits", "hadamard_dual", "independent_checks", "homology_basis", "ChainMap"]
