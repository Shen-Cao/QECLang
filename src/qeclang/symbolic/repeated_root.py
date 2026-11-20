"""Exact cyclic rank, including repeated roots, by polynomial diagonalization.

Elementary polynomial row and column operations are unimodular, hence remain
invertible modulo f=x**L+1. A diagonal entry a contributes L-deg(gcd(a,f)) to
binary rank. Divisibility ordering of a full Smith normal form is unnecessary.
No expanded binary matrices or eigenvalue assumptions are used.
"""
from dataclasses import dataclass

from qeclang.algebra.binary import nonnegative_integer
from qeclang.algebra.polynomial import (ArithmeticBudget, WorkLimitExceeded,
                                       poly_divmod, poly_gcd, poly_mul)

from .matrix import GroupAlgebraMatrix


@dataclass(frozen=True)
class PolynomialOperation:
    axis: str
    kind: str
    target: int
    source: int
    multiplier: int = 0


@dataclass(frozen=True)
class CyclicPolynomialRankResult:
    lift: int
    shape: tuple[int, int]
    diagonal: tuple[int, ...]
    gcds: tuple[int, ...]
    rank: int
    steps: int
    operation_count: int
    operations: tuple[PolynomialOperation, ...] | None
    method: str = "exact cyclic rank by polynomial unimodular diagonalization"
    evidence: str = "exact elementary operations and diagonal gcd ranks; optional replay trace"

    @property
    def binary_shape(self):
        return self.shape[0] * self.lift, self.shape[1] * self.lift


def _input(matrix, max_lift, max_polynomial_degree, max_entries, budget):
    if not isinstance(matrix, GroupAlgebraMatrix):
        raise TypeError("cyclic polynomial rank requires a GroupAlgebraMatrix")
    if len(matrix.orders) != 1:
        raise ValueError("cyclic polynomial rank requires exactly one cyclic factor")
    lift = matrix.orders[0]
    if lift > nonnegative_integer(max_lift, "max_lift"):
        raise WorkLimitExceeded("cyclic order exceeds max_lift")
    degree = nonnegative_integer(max_polynomial_degree, "max_polynomial_degree")
    if lift > degree:
        raise WorkLimitExceeded("cyclic modulus exceeds max_polynomial_degree")
    entries = nonnegative_integer(max_entries, "max_entries")
    if max(matrix.nrows, matrix.ncols, matrix.nrows * matrix.ncols) > entries:
        raise WorkLimitExceeded("polynomial seed matrix exceeds max_entries")
    rows = []
    for row in matrix.rows:
        values = []
        for element in row:
            budget.charge()
            value = 0
            for term in element.terms:
                budget.charge()
                value |= 1 << term[0]
            values.append(value)
        rows.append(values)
    return rows, lift, degree


def _multiply(left, right, degree, budget):
    if left and right and left.bit_length() + right.bit_length() - 2 > degree:
        raise WorkLimitExceeded("polynomial intermediate exceeds max_polynomial_degree")
    return poly_mul(left, right, budget=budget)


def _apply(rows, ncols, operation, degree, budget):
    if not isinstance(operation, PolynomialOperation):
        raise ValueError("invalid polynomial operation record")
    if operation.axis not in ("row", "column") or operation.kind not in ("swap", "add"):
        raise ValueError("invalid polynomial operation kind")
    count = len(rows) if operation.axis == "row" else ncols
    target = nonnegative_integer(operation.target, "operation target")
    source = nonnegative_integer(operation.source, "operation source")
    multiplier = nonnegative_integer(operation.multiplier, "operation multiplier")
    if target >= count or source >= count or target == source:
        raise ValueError("elementary operation requires distinct in-range indices")
    if multiplier.bit_length() - 1 > degree:
        raise WorkLimitExceeded("operation multiplier exceeds max_polynomial_degree")
    budget.charge()
    if operation.kind == "swap":
        if multiplier:
            raise ValueError("swap operation cannot carry a multiplier")
        if operation.axis == "row":
            rows[target], rows[source] = rows[source], rows[target]
        else:
            for row in rows:
                budget.charge()
                row[target], row[source] = row[source], row[target]
    elif operation.axis == "row":
        for j in range(ncols):
            budget.charge()
            rows[target][j] ^= _multiply(multiplier, rows[source][j], degree, budget)
    else:
        for row in rows:
            budget.charge()
            row[target] ^= _multiply(multiplier, row[source], degree, budget)


def cyclic_rank_general(matrix: GroupAlgebraMatrix, *, max_lift=512,
                        max_polynomial_degree=4096, max_steps=10_000_000,
                        max_entries=1_000_000, max_operations=100_000,
                        record_operations=False, _meter=None) -> CyclicPolynomialRankResult:
    """Explicit resource-bounded fallback for any positive cyclic order.

    Degree growth can terminate this algorithm before a result is available.
    A limit exception never returns a partial rank. An optional elementary
    operation trace supports independent replay with verify_rank_certificate.
    Internal _meter shares an enclosing arithmetic ceiling instead of max_steps;
    result.steps remains local to this invocation.
    """
    if type(record_operations) is not bool:
        raise ValueError("record_operations must be boolean")
    nonnegative_integer(max_steps, "max_steps")
    if _meter is not None and not isinstance(_meter, ArithmeticBudget):
        raise TypeError("_meter must be ArithmeticBudget")
    budget = ArithmeticBudget(max_steps) if _meter is None else _meter
    start = budget.used
    operation_limit = nonnegative_integer(max_operations, "max_operations")
    rows, lift, degree = _input(matrix, max_lift, max_polynomial_degree, max_entries, budget)
    trace = [] if record_operations else None
    count = 0

    def apply(axis, kind, target, source, multiplier=0):
        nonlocal count
        if count >= operation_limit:
            raise WorkLimitExceeded("diagonalization exceeds max_operations")
        operation = PolynomialOperation(axis, kind, target, source, multiplier)
        _apply(rows, matrix.ncols, operation, degree, budget)
        count += 1
        if trace is not None:
            trace.append(operation)

    for pivot in range(min(matrix.shape)):
        selected = None
        for i in range(pivot, matrix.nrows):
            for j in range(pivot, matrix.ncols):
                budget.charge()
                if rows[i][j] and (selected is None or
                                   rows[i][j].bit_length() < selected[0]):
                    selected = rows[i][j].bit_length(), i, j
        if selected is None:
            break
        _, i, j = selected
        if i != pivot:
            apply("row", "swap", pivot, i)
        if j != pivot:
            apply("column", "swap", pivot, j)
        while True:
            restart = False
            for i in range(pivot + 1, matrix.nrows):
                budget.charge()
                if rows[i][pivot]:
                    quotient, remainder = poly_divmod(rows[i][pivot], rows[pivot][pivot],
                                                      budget=budget)
                    if quotient:
                        apply("row", "add", i, pivot, quotient)
                    if remainder:
                        apply("row", "swap", pivot, i)
                        restart = True
                        break
            if restart:
                continue
            for j in range(pivot + 1, matrix.ncols):
                budget.charge()
                if rows[pivot][j]:
                    quotient, remainder = poly_divmod(rows[pivot][j], rows[pivot][pivot],
                                                      budget=budget)
                    if quotient:
                        apply("column", "add", j, pivot, quotient)
                    if remainder:
                        apply("column", "swap", pivot, j)
                        restart = True
                        break
            if not restart:
                break
    diagonal = tuple(rows[i][i] for i in range(min(matrix.shape)))
    modulus = (1 << lift) | 1
    gcds = tuple(poly_gcd(value, modulus, budget=budget) for value in diagonal)
    rank = sum(lift - (gcd.bit_length() - 1) for gcd in gcds)
    return CyclicPolynomialRankResult(lift, matrix.shape, diagonal, gcds, rank,
                                      budget.used - start, count, None if trace is None else tuple(trace))


def verify_rank_certificate(matrix, result, *, max_lift=512, max_polynomial_degree=4096,
                            max_steps=10_000_000, max_entries=1_000_000,
                            max_operations=100_000, _meter=None):
    """Replay a recorded unimodular trace and recompute the diagonal gcd rank.

    This verifier does not call the diagonalization algorithm or binary lowering.
    Malformed/tampered evidence returns False; resource-limit errors propagate.
    Internal _meter retains accepted proxy charges even if replay is refused.
    """
    nonnegative_integer(max_steps, "max_steps")
    if _meter is not None and not isinstance(_meter, ArithmeticBudget):
        raise TypeError("_meter must be ArithmeticBudget")
    budget = ArithmeticBudget(max_steps) if _meter is None else _meter
    if not isinstance(result, CyclicPolynomialRankResult) or not isinstance(result.operations, tuple):
        return False
    if (type(result.rank) is not int or result.rank < 0 or
            type(result.operation_count) is not int or result.operation_count < 0 or
            type(result.lift) is not int or not isinstance(result.diagonal, tuple) or
            not isinstance(result.gcds, tuple) or
            any(type(value) is not int or value < 0 for value in result.diagonal + result.gcds)):
        return False
    operation_limit = nonnegative_integer(max_operations, "max_operations")
    if len(result.operations) > operation_limit:
        raise WorkLimitExceeded("certificate exceeds max_operations")
    rows, lift, degree = _input(matrix, max_lift, max_polynomial_degree, max_entries, budget)
    if result.shape != matrix.shape or result.lift != lift:
        return False
    if result.operation_count != len(result.operations):
        return False
    try:
        for operation in result.operations:
            _apply(rows, matrix.ncols, operation, degree, budget)
    except (ValueError, TypeError):
        return False
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            budget.charge()
            if i != j and value:
                return False
    diagonal = tuple(rows[i][i] for i in range(min(matrix.shape)))
    if diagonal != result.diagonal:
        return False
    modulus = (1 << lift) | 1
    gcds = tuple(poly_gcd(value, modulus, budget=budget) for value in diagonal)
    return gcds == result.gcds and result.rank == sum(lift - (g.bit_length() - 1) for g in gcds)


__all__ = ["PolynomialOperation", "CyclicPolynomialRankResult", "cyclic_rank_general",
           "verify_rank_certificate"]
