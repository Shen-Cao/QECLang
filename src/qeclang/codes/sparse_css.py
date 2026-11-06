"""Sparse CSS presentations with explicit validation and lowering costs."""
from dataclasses import InitVar, dataclass, field
from functools import cached_property

from qeclang.algebra import BinaryMatrix, SparseBinaryMatrix
from qeclang.algebra.binary import nonnegative_integer, set_bits
from .css import CSSCode


@dataclass(frozen=True)
class SparseCSSLocality:
    n: int
    x_checks: int
    z_checks: int
    x_nonzeros: int
    z_nonzeros: int
    max_check_weight: int
    max_qubit_degree: int


@dataclass(frozen=True)
class SparseCSSCode:
    """Positive CSS checks retained as sparse support-index rows.

    Validation uses shared-site incidences and row-local parity sets, never a
    full check-by-check product. ``max_validation_work`` bounds the count
    mX+mZ+nnzX+nnzZ+sum_j(degX[j]*degZ[j]). This is a deterministic work proxy,
    not an elapsed-time or byte limit. It is checked before pair traversal.

    There is deliberately no implicit ``k`` or ``rank`` property. Exact rank,
    logical bases, and distance need an explicitly chosen analysis backend.
    ``to_packed`` provides the guarded reference backend boundary.
    """
    hx: SparseBinaryMatrix
    hz: SparseBinaryMatrix
    name: str = ""
    max_validation_work: InitVar[int] = 10_000_000
    validation_work: int = field(init=False, compare=False)

    def __post_init__(self, max_validation_work):
        budget = nonnegative_integer(max_validation_work, "max_validation_work")
        if not isinstance(self.hx, SparseBinaryMatrix) or not isinstance(self.hz, SparseBinaryMatrix):
            raise TypeError("sparse CSS checks must be SparseBinaryMatrix objects")
        if self.hx.ncols != self.hz.ncols:
            raise ValueError("X and Z checks must have the same number of qubits")
        if not isinstance(self.name, str):
            raise ValueError("name must be a string")
        work = self.hx.nrows + self.hz.nrows + self.hx.nnz + self.hz.nnz
        if work > budget:
            raise ValueError("sparse CSS validation exceeds max_validation_work")
        z_incidence = {}
        for check, row in enumerate(self.hz.rows):
            for site in row:
                z_incidence.setdefault(site, []).append(check)
        for row in self.hx.rows:
            for site in row:
                work += len(z_incidence.get(site, ()))
                if work > budget:
                    raise ValueError("sparse CSS validation exceeds max_validation_work")
        for check, row in enumerate(self.hx.rows):
            odd = set()
            for site in row:
                for other in z_incidence.get(site, ()):
                    if other in odd:
                        odd.remove(other)
                    else:
                        odd.add(other)
            if odd:
                raise ValueError(f"CSS checks do not commute: X check {check}, Z check {min(odd)}")
        object.__setattr__(self, "validation_work", work)

    @property
    def n(self) -> int:
        return self.hx.ncols

    @cached_property
    def locality(self) -> SparseCSSLocality:
        degree = {}
        max_weight = 0
        for matrix in (self.hx, self.hz):
            for support in matrix.rows:
                max_weight = max(max_weight, len(support))
                for site in support:
                    degree[site] = degree.get(site, 0) + 1
        return SparseCSSLocality(self.n, self.hx.nrows, self.hz.nrows,
                                 self.hx.nnz, self.hz.nnz, max_weight,
                                 max(degree.values(), default=0))

    def syndrome(self, x: int, z: int) -> tuple[int, int]:
        """Return (sx, sz) = (Hx z, Hz x), in the retained check order."""
        return self.hx.matvec(z), self.hz.matvec(x)

    def syndrome_bits(self, x, z) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Sequence interface avoiding physical-error bigint bit accesses."""
        return self.hx.matvec_bits(z), self.hz.matvec_bits(x)

    @classmethod
    def from_packed(cls, code: CSSCode, *, max_validation_work=10_000_000):
        if not isinstance(code, CSSCode):
            raise TypeError("from_packed requires a CSSCode")
        return cls(SparseBinaryMatrix.from_packed(code.hx),
                   SparseBinaryMatrix.from_packed(code.hz), code.name,
                   max_validation_work=max_validation_work)

    def to_packed(self, *, max_bits=100_000_000) -> CSSCode:
        """Explicitly lower to the rank-capable reference representation.

        The guard sums both check-matrix payload bounds and the mX*mZ product
        used by the packed CSS validator. It excludes Python object overhead,
        transposition temporaries, and subsequent elimination workspace. It is
        neither a total-memory bound nor a guarantee of inexpensive analysis.
        """
        budget = nonnegative_integer(max_bits, "max_bits")
        payload = (self.hx.nrows + self.hz.nrows) * self.n + self.hx.nrows * self.hz.nrows
        if payload > budget:
            raise ValueError("packed CSS checks and validation product exceed max_bits")
        return CSSCode(self.hx.to_packed(max_bits=budget),
                       self.hz.to_packed(max_bits=budget), self.name)


def same_css_presentation(left, right) -> bool:
    """Compare retained generators across sparse/packed backends, ignoring names.

    This compares sparse supports and never lowers sparse matrices or computes
    ranks. Row order and redundant checks are significant for syndrome decoding.
    """
    if not isinstance(left, (CSSCode, SparseCSSCode)) or not isinstance(right, (CSSCode, SparseCSSCode)):
        return False
    if left.n != right.n:
        return False
    for a, b in ((left.hx, right.hx), (left.hz, right.hz)):
        if a.shape != b.shape:
            return False
        for ra, rb in zip(a.rows, b.rows):
            sa = tuple(set_bits(ra)) if isinstance(a, BinaryMatrix) else ra
            sb = tuple(set_bits(rb)) if isinstance(b, BinaryMatrix) else rb
            if sa != sb:
                return False
    return True


__all__ = ["SparseCSSCode", "SparseCSSLocality", "same_css_presentation"]
