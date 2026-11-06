"""Rank, incidence, and erasure diagnostics (no asymptotic claims)."""
from dataclasses import dataclass
from math import comb

from qeclang.algebra import BinaryMatrix
from qeclang.codes import CSSCode, StabilizerCode


@dataclass(frozen=True)
class MatrixProfile:
    rows: int
    columns: int
    rank: int
    nonzeros: int
    max_row_weight: int
    max_column_weight: int
    zero_rows: int
    zero_columns: int


def matrix_profile(matrix: BinaryMatrix) -> MatrixProfile:
    rw = tuple(r.bit_count() for r in matrix.rows)
    cw = tuple(r.bit_count() for r in matrix.T.rows)
    return MatrixProfile(matrix.nrows, matrix.ncols, matrix.rank(), sum(rw),
                         max(rw, default=0), max(cw, default=0),
                         rw.count(0), cw.count(0))


@dataclass(frozen=True)
class CSSProfile:
    n: int
    k: int
    rate: float | None
    x: MatrixProfile
    z: MatrixProfile
    max_check_weight: int
    max_qubit_degree: int


def css_profile(code: CSSCode) -> CSSProfile:
    x, z = matrix_profile(code.hx), matrix_profile(code.hz)
    degrees = [a.bit_count() + b.bit_count()
               for a, b in zip(code.hx.T.rows, code.hz.T.rows)]
    return CSSProfile(code.n, code.k, code.k / code.n if code.n else None, x, z,
                      max(x.max_row_weight, z.max_row_weight),
                      max(degrees, default=0))


def tanner_four_cycles(matrix: BinaryMatrix) -> int:
    """Count undirected 4-cycles in the specified check presentation."""
    return sum(comb((a & b).bit_count(), 2)
               for i, a in enumerate(matrix.rows) for b in matrix.rows[i + 1:])


def _restrict(matrix: BinaryMatrix, support: tuple[int, ...]) -> BinaryMatrix:
    return BinaryMatrix(tuple(sum(((r >> i) & 1) << j for j, i in enumerate(support))
                              for r in matrix.rows), len(support))


def _support(n: int, erased) -> tuple[int, ...]:
    values = tuple(erased)
    if any(type(i) is not int or not 0 <= i < n for i in values):
        raise ValueError("erased sites must be integer indices within the code")
    if len(set(values)) != len(values):
        raise ValueError("erased sites must be distinct")
    return tuple(sorted(values))


def css_erasure_correctable(code: CSSCode, erased) -> bool:
    """Exact known-location erasure criterion, using supported logical ranks."""
    support = _support(code.n, erased)
    outside = tuple(i for i in range(code.n) if i not in set(support))
    for checks, trivial in ((code.hx, code.hz), (code.hz, code.hx)):
        cycles = len(support) - _restrict(checks, support).rank()
        boundaries = trivial.rank() - _restrict(trivial, outside).rank()
        if cycles != boundaries:
            return False
    return True


def stabilizer_erasure_correctable(code: StabilizerCode, erased) -> bool:
    support = _support(code.n, erased)
    outside = tuple(i for i in range(code.n) if i not in set(support))
    constraints = BinaryMatrix(tuple((p >> code.n) | ((p & ((1 << code.n) - 1)) << code.n)
                                     for p in code.checks.rows), 2 * code.n)
    es = support + tuple(i + code.n for i in support)
    os = outside + tuple(i + code.n for i in outside)
    cycles = len(es) - _restrict(constraints, es).rank()
    trivial = code.checks.rank() - _restrict(code.checks, os).rank()
    return cycles == trivial
