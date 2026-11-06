"""Validated complete CSS logical frames for repeated sparse parity analysis."""
from dataclasses import dataclass

from qeclang.algebra import BinaryMatrix, SparseBinaryMatrix
from qeclang.algebra.binary import nonnegative_integer, validate_vector
from qeclang.codes import CSSCode, SparseCSSCode, same_css_presentation
from .logicals import css_logicals


@dataclass(frozen=True, init=False)
class CSSLogicalFrame:
    """A complete paired logical basis and its exact sparse check presentation.

    Construction uses ``from_packed`` or independently verified portable rank
    evidence via ``import_logical_frame``. Verifying just the
    pairings would not establish completeness: an omitted logical pair could
    hide a logical failure. The factory therefore also checks the exact code
    dimension. No unvalidated field-based constructor or deserializer is exposed.

    After construction, syndrome and logical parity tests need neither rank
    computation nor row-space membership. Logical representatives need not be
    sparse in a good code; their support storage can still be large.
    """
    presentation: SparseCSSCode
    x_logicals: SparseBinaryMatrix
    z_logicals: SparseBinaryMatrix

    def __new__(cls, *args, **kwargs):
        raise TypeError("construct a validated CSSLogicalFrame with from_packed(code)")

    @classmethod
    def from_packed(cls, code: CSSCode, *, max_bits=100_000_000,
                    max_validation_work=10_000_000):
        """Compute and validate a full frame using exact packed GF(2) algebra.

        ``max_bits`` guards the input check payload plus a conservative 2*n*n
        kernel/basis payload proxy before elimination. It does not bound all
        simultaneous temporaries, object overhead, or elapsed time. Sparse
        presentation validation has a separate explicit incidence-work budget.
        The returned object retains only sparse check and logical supports.
        """
        if not isinstance(code, CSSCode):
            raise TypeError("frame construction requires an explicitly packed CSSCode")
        budget = nonnegative_integer(max_bits, "max_bits")
        if (code.hx.nrows + code.hz.nrows) * code.n + 2 * code.n**2 > budget:
            raise ValueError("logical frame construction exceeds max_bits")
        presentation = SparseCSSCode.from_packed(code, max_validation_work=max_validation_work)
        logicals = css_logicals(code)
        if len(logicals.x) != code.k or len(logicals.z) != code.k:
            raise ArithmeticError("logical frame is not complete for the exact code dimension")
        xs = BinaryMatrix(logicals.x, code.n)
        zs = BinaryMatrix(logicals.z, code.n)
        if not (code.hz @ xs.T).is_zero or not (code.hx @ zs.T).is_zero:
            raise ArithmeticError("logical frame has nonzero stabilizer syndrome")
        if xs @ zs.T != BinaryMatrix.identity(code.k):
            raise ArithmeticError("logical frame pairing is not the identity")
        frame = object.__new__(cls)
        object.__setattr__(frame, "presentation", presentation)
        object.__setattr__(frame, "x_logicals", SparseBinaryMatrix.from_packed(xs))
        object.__setattr__(frame, "z_logicals", SparseBinaryMatrix.from_packed(zs))
        return frame

    @property
    def n(self) -> int:
        return self.presentation.n

    @property
    def k(self) -> int:
        return self.x_logicals.nrows

    def matches(self, code: CSSCode | SparseCSSCode) -> bool:
        """Match exact retained checks, including row order and redundancy."""
        return same_css_presentation(self.presentation, code)

    def residual_is_trivial(self, x: int, z: int) -> bool:
        """Check exact stabilizer membership using syndrome and logical parity.

        A residual with nonzero syndrome is never accepted, even if its logical
        parities vanish. Physical phases do not affect recovery success.
        """
        x = validate_vector(x, self.n, "X residual")
        z = validate_vector(z, self.n, "Z residual")
        if self.presentation.syndrome(x, z) != (0, 0):
            return False
        return self.z_logicals.matvec(x) == 0 and self.x_logicals.matvec(z) == 0

    def logical_success(self, x: int, z: int, correction) -> bool:
        """Check a CSS correction with ``success``, ``x.error``, and ``z.error``."""
        x = validate_vector(x, self.n, "X error")
        z = validate_vector(z, self.n, "Z error")
        if not correction.success:
            return False
        recovered_x = validate_vector(correction.x.error, self.n, "X correction")
        recovered_z = validate_vector(correction.z.error, self.n, "Z correction")
        return self.residual_is_trivial(x ^ recovered_x, z ^ recovered_z)


__all__ = ["CSSLogicalFrame"]
