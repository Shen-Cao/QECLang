"""Exact GF(2) decoding for errors on known erased sites."""
from dataclasses import dataclass
from typing import Literal

from qeclang.algebra import BinaryMatrix
from qeclang.analysis import css_erasure_correctable
from qeclang.analysis.structure import _restrict, _support
from qeclang.codes import CSSCode


@dataclass(frozen=True)
class ErasureDecodeResult:
    error: int | None
    status: Literal["found", "inconsistent"]
    unique_solution: bool
    erased: tuple[int, ...]


def erasure_decode(checks: BinaryMatrix, syndrome: int, erased) -> ErasureDecodeResult:
    """Solve for an error supported on the specified sites, without enumeration.

    ``unique_solution`` refers to uniqueness as a binary error vector. Quantum
    corrections need only be unique modulo stabilizers, which is checked by
    ``css_erasure_decode`` at the region level. An inconsistent syndrome may
    indicate an incorrect erasure set or unreliable syndrome measurements.
    """
    support = _support(checks.ncols, erased)
    if type(syndrome) is not int or not 0 <= syndrome < 1 << checks.nrows:
        raise ValueError("syndrome must fit the number of checks")
    restricted = _restrict(checks, support)
    try:
        local = restricted.solve(syndrome)
    except ValueError:
        return ErasureDecodeResult(None, "inconsistent", False, support)
    error = sum(((local >> j) & 1) << i for j, i in enumerate(support))
    return ErasureDecodeResult(error, "found", restricted.rank() == len(support), support)


@dataclass(frozen=True)
class CSSErasureDecodeResult:
    x: ErasureDecodeResult
    z: ErasureDecodeResult
    region_correctable: bool

    @property
    def success(self) -> bool:
        """Whether both syndromes were matched, independent of logical ambiguity."""
        return self.x.status == self.z.status == "found"


def css_erasure_decode(code: CSSCode, sx: int, sz: int, erased) -> CSSErasureDecodeResult:
    """Correct known-site Pauli errors using perfect syndromes sx=Hx z, sz=Hz x.

    If ``region_correctable`` and ``success`` both hold, every error supported
    on the given region with this syndrome differs from the returned correction
    by a stabilizer. An uncorrectable region may still produce a consistent
    correction; that is explicitly reported without a logical recovery guarantee.
    """
    support = _support(code.n, erased)
    return CSSErasureDecodeResult(erasure_decode(code.hz, sz, support),
                                  erasure_decode(code.hx, sx, support),
                                  css_erasure_correctable(code, support))
