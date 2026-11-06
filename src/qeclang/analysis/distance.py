"""Finite, budgeted distance searches with explicit lower and upper bounds."""
from dataclasses import dataclass
from itertools import combinations, product
from typing import Literal

from qeclang.codes import CSSCode, StabilizerCode
from .logicals import quotient_basis, stabilizer_logicals, symplectic


@dataclass(frozen=True)
class DistanceResult:
    """An upper bound always has a nontrivial logical witness.

    A witness is a binary sector vector for CSS, or x | (z << n) for
    stabilizers. ``no_logicals`` represents an empty minimization set, not
    an ordinary finite-distance code. ``searched_through`` records fully
    excluded weight layers; a mid-layer budget stop never increments it.
    """
    lower_bound: int | None
    upper_bound: int | None
    witness: int | None
    searched_through: int
    candidates: int
    status: Literal["exact", "weight_limit", "budget_exhausted", "no_logicals"]

    @property
    def distance(self) -> int | None:
        return self.upper_bound if self.status == "exact" else None


def binary_words(n: int, weight: int):
    """Generate fixed-weight binary words without scanning all 2**n words."""
    for support in combinations(range(n), weight):
        yield sum(1 << i for i in support)


def _limits(n: int, max_weight: int | None, max_candidates: int):
    if max_weight is None:
        max_weight = n
    if type(max_weight) is not int or max_weight < 0:
        raise ValueError("max_weight must be a nonnegative integer")
    if type(max_candidates) is not int or max_candidates < 0:
        raise ValueError("max_candidates must be a nonnegative integer")
    return min(n, max_weight)


def _search(n, initial, words, is_logical, weight, max_weight, max_candidates):
    max_weight = _limits(n, max_weight, max_candidates)
    if not initial:
        return DistanceResult(None, None, None, 0, 0, "no_logicals")
    witness = min(initial, key=weight)
    upper = weight(witness)
    searched, count = 0, 0
    for w in range(1, min(max_weight, upper - 1) + 1):
        for candidate in words(n, w):
            if count == max_candidates:
                return DistanceResult(searched + 1, upper, witness, searched,
                                      count, "budget_exhausted")
            count += 1
            if is_logical(candidate):
                return DistanceResult(w, w, candidate, searched, count, "exact")
        searched = w
    lower = searched + 1
    status = "exact" if lower == upper else "weight_limit"
    return DistanceResult(lower, upper, witness, searched, count, status)


def css_distance(code: CSSCode, sector: Literal["x", "z"], *,
                 max_weight: int | None = None,
                 max_candidates: int = 1_000_000) -> DistanceResult:
    """Search min wt(ker(opposite checks) / same-sector stabilizers).

    The default candidate cap applies to enumeration, not polynomial-time
    basis construction. An initial quotient representative supplies a
    certified upper bound even when enumeration stops early.
    """
    if sector not in ("x", "z"):
        raise ValueError("sector must be 'x' or 'z'")
    checks, trivial = ((code.hz, code.hx) if sector == "x"
                       else (code.hx, code.hz))
    trivial = trivial.row_basis()
    initial = quotient_basis(checks.nullspace(), trivial)
    return _search(code.n, initial, binary_words,
                   lambda v: checks.matvec(v) == 0 and not trivial.contains(v),
                   int.bit_count, max_weight, max_candidates)


def pauli_words(n: int, weight: int):
    for support in combinations(range(n), weight):
        for labels in product((1, 2, 3), repeat=weight):
            x = sum((label & 1) << i for i, label in zip(support, labels))
            z = sum((label >> 1) << i for i, label in zip(support, labels))
            yield x | (z << n)


def stabilizer_distance(code: StabilizerCode, *, max_weight: int | None = None,
                        max_candidates: int = 1_000_000) -> DistanceResult:
    checks = code.checks.row_basis()
    initial = tuple(v for pair in stabilizer_logicals(code) for v in pair)
    mask = (1 << code.n) - 1
    return _search(code.n, initial, pauli_words,
                   lambda v: all(not symplectic(v, s, code.n) for s in checks.rows)
                   and not checks.contains(v),
                   lambda v: ((v & mask) | (v >> code.n)).bit_count(),
                   max_weight, max_candidates)


def subsystem_distance(code, kind: Literal["dressed", "bare"] = "dressed", *,
                       max_weight: int | None = None,
                       max_candidates: int = 1_000_000) -> DistanceResult:
    """Search subsystem dressed or bare Pauli distance.

    Dressed: centralizer(center) / gauge; bare: centralizer(gauge) / center.
    Phases do not affect these quotient weights. A gauge-only code (k=0)
    has no protected logicals even when it has gauge degrees of freedom.
    """
    if kind == "dressed":
        space = code.stabilizer.centralizer()
        trivial = code.checks.row_basis()
        constraints = code.center
    elif kind == "bare":
        space = code.centralizer()
        trivial = code.center
        constraints = code.checks.row_basis()
    else:
        raise ValueError("kind must be 'dressed' or 'bare'")
    initial = quotient_basis(space, trivial)
    mask = (1 << code.n) - 1
    return _search(code.n, initial, pauli_words,
                   lambda v: all(not symplectic(v, s, code.n)
                                 for s in constraints.rows) and not trivial.contains(v),
                   lambda v: ((v & mask) | (v >> code.n)).bit_count(),
                   max_weight, max_candidates)
