"""Commuting Pauli codes and subsystem gauge modules."""

from dataclasses import dataclass
from functools import cached_property

from qeclang.algebra import BinaryMatrix
from qeclang.algebra.binary import nonnegative_integer, set_bits, validate_vector

from .pauli import Pauli, symplectic_product


def _validate_generators(n: int, generators: tuple[Pauli, ...]) -> tuple[Pauli, ...]:
    generators = tuple(generators)
    for generator in generators:
        if not isinstance(generator, Pauli):
            raise TypeError("generators must be Pauli operators")
        validate_vector(generator.x, n, "Pauli x")
        validate_vector(generator.z, n, "Pauli z")
    return generators


def _centralizer(checks: BinaryMatrix, n: int) -> BinaryMatrix:
    mask = (1 << n) - 1
    dual = BinaryMatrix(tuple((row >> n) | ((row & mask) << n) for row in checks.rows), 2 * n)
    return dual.nullspace()


@dataclass(frozen=True)
class StabilizerCode:
    """Joint +1 eigenspace of a commuting, signed Hermitian Pauli group.

    Redundant generators are allowed, but every relation must have phase +1.
    This rules out an empty codespace caused by a hidden -I relation.
    """

    n: int
    generators: tuple[Pauli, ...]
    name: str = ""

    def __post_init__(self) -> None:
        n = nonnegative_integer(self.n, "n")
        object.__setattr__(self, "n", n)
        generators = _validate_generators(n, self.generators)
        object.__setattr__(self, "generators", generators)
        for i, generator in enumerate(generators):
            if not generator.is_hermitian:
                raise ValueError("stabilizer generators must be Hermitian")
            if any(not generator.commutes(other) for other in generators[:i]):
                raise ValueError("stabilizer generators must commute")
        # Trigger the phase-aware dependency check at construction time.
        self.basis

    @cached_property
    def basis(self) -> tuple[Pauli, ...]:
        pivots: dict[int, Pauli] = {}
        for generator in self.generators:
            reduced = generator
            for pivot in sorted(pivots):
                if ((reduced.x | (reduced.z << self.n)) >> pivot) & 1:
                    reduced = reduced * pivots[pivot]
            vector = reduced.x | (reduced.z << self.n)
            if vector:
                pivot = (vector & -vector).bit_length() - 1
                pivots[pivot] = reduced
            elif reduced.phase:
                raise ValueError("stabilizer relations generate -I (empty codespace)")
        return tuple(pivots[pivot] for pivot in sorted(pivots))

    @cached_property
    def checks(self) -> BinaryMatrix:
        return BinaryMatrix(tuple(p.x | (p.z << self.n) for p in self.generators), 2 * self.n)

    @property
    def rank(self) -> int:
        return len(self.basis)

    @property
    def k(self) -> int:
        return self.n - self.rank

    def centralizer(self) -> BinaryMatrix:
        return _centralizer(self.checks, self.n)

    def syndrome(self, error: Pauli) -> int:
        _validate_generators(self.n, (error,))
        return sum((not error.commutes(check)) << i for i, check in enumerate(self.generators))


@dataclass(frozen=True)
class SubsystemCode:
    """A qubit Pauli gauge module and a signed center eigensector.

    Gauge-generator phases are ignored. The module specifies protected/gauge
    dimensions and dressed logical cosets. ``sector`` optionally supplies signed
    commuting Pauli generators spanning exactly the center. When omitted,
    ``stabilizer`` chooses positive Pauli labels on an independent center basis.
    """

    n: int
    generators: tuple[Pauli, ...]
    name: str = ""
    sector: tuple[Pauli, ...] | None = None

    def __post_init__(self) -> None:
        n = nonnegative_integer(self.n, "n")
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "generators", _validate_generators(n, self.generators))
        if self.sector is not None:
            object.__setattr__(self, "sector", tuple(self.sector))
            if self.stabilizer.checks.row_basis() != self.center:
                raise ValueError("sector generators must span exactly the gauge center")

    @cached_property
    def checks(self) -> BinaryMatrix:
        return BinaryMatrix(tuple(p.x | (p.z << self.n) for p in self.generators), 2 * self.n)

    @cached_property
    def center(self) -> BinaryMatrix:
        """Independent vectors in the radical G intersect G-perp."""
        basis = self.checks.row_basis()
        gram = BinaryMatrix(
            tuple(sum(symplectic_product(a, b, self.n) << j for j, b in enumerate(basis.rows))
                  for a in basis.rows),
            basis.nrows,
        )
        center = []
        for coefficients in gram.nullspace().rows:
            vector = 0
            for j in set_bits(coefficients):
                vector ^= basis.rows[j]
            center.append(vector)
        return BinaryMatrix(tuple(center), 2 * self.n).row_basis()

    @property
    def r(self) -> int:
        return (self.checks.rank() - self.center.nrows) // 2

    @property
    def k(self) -> int:
        return self.n - self.center.nrows - self.r

    @cached_property
    def stabilizer(self) -> StabilizerCode:
        if self.sector is not None:
            return StabilizerCode(self.n, self.sector, self.name)
        mask = (1 << self.n) - 1
        generators = tuple(
            Pauli(row & mask, row >> self.n, ((row & mask) & (row >> self.n)).bit_count())
            for row in self.center.rows
        )
        return StabilizerCode(self.n, generators, self.name)

    def centralizer(self) -> BinaryMatrix:
        """Operators commuting with every gauge generator (bare operators)."""
        return _centralizer(self.checks, self.n)
