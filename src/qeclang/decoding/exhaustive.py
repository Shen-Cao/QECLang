"""Bounded minimum-weight syndrome decoding reference algorithms."""
from dataclasses import dataclass
from typing import Literal

from qeclang.algebra import BinaryMatrix
from qeclang.codes import CSSCode
from qeclang.analysis.distance import binary_words, _limits


@dataclass(frozen=True)
class DecodeResult:
    error: int | None
    weight: int | None
    candidates: int
    status: Literal["found", "weight_limit", "budget_exhausted", "inconsistent"]


def minimum_weight_decode(checks: BinaryMatrix, syndrome: int, *,
                          max_weight: int | None = None,
                          max_candidates: int = 1_000_000) -> DecodeResult:
    """Find a least-weight error with the supplied perfect syndrome.

    This optimizes individual Hamming weight, not quantum coset likelihood.
    Ties follow increasing support order. The budget counts error vectors,
    including zero; consistency is checked by polynomial-time linear algebra.
    """
    limit = _limits(checks.ncols, max_weight, max_candidates)
    if type(syndrome) is not int or not 0 <= syndrome < (1 << checks.nrows):
        raise ValueError("syndrome must fit the number of checks")
    try:
        checks.solve(syndrome)
    except ValueError:
        return DecodeResult(None, None, 0, "inconsistent")
    count = 0
    for w in range(limit + 1):
        for candidate in binary_words(checks.ncols, w):
            if count == max_candidates:
                return DecodeResult(None, None, count, "budget_exhausted")
            count += 1
            if checks.matvec(candidate) == syndrome:
                return DecodeResult(candidate, w, count, "found")
    return DecodeResult(None, None, count, "weight_limit")


@dataclass(frozen=True)
class CSSDecodeResult:
    x: DecodeResult
    z: DecodeResult

    @property
    def success(self) -> bool:
        """Whether both syndromes were matched; this is not logical success."""
        return self.x.status == self.z.status == "found"


def css_decode(code: CSSCode, sx: int, sz: int, *,
               max_weight: int | None = None,
               max_candidates_per_sector: int = 1_000_000) -> CSSDecodeResult:
    """Decode independently: sx = Hx z and sz = Hz x.

    Independent sectors ignore X/Z correlations under depolarizing noise.
    """
    return CSSDecodeResult(
        minimum_weight_decode(code.hz, sz, max_weight=max_weight,
                              max_candidates=max_candidates_per_sector),
        minimum_weight_decode(code.hx, sx, max_weight=max_weight,
                              max_candidates=max_candidates_per_sector))


def css_logical_success(code: CSSCode, x_error: int, z_error: int,
                        correction: CSSDecodeResult) -> bool:
    """A perfect-syndrome correction succeeds iff its residual is a stabilizer."""
    for v in (x_error, z_error):
        if type(v) is not int or not 0 <= v < (1 << code.n):
            raise ValueError("error must fit the code block")
    if not correction.success:
        return False
    return (code.hx.contains(x_error ^ correction.x.error)
            and code.hz.contains(z_error ^ correction.z.error))
