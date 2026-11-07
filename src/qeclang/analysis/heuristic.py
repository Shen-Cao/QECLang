"""Bounded seeded searches for CSS logical upper bounds, with exact witnesses."""
from dataclasses import dataclass
from random import Random

from qeclang.algebra.binary import nonnegative_integer, set_bits
from qeclang.codes import CSSCode
from .certificates import LogicalWitness, verify_css_logical
from .logicals import css_logicals


@dataclass(frozen=True)
class CSSHeuristicResult:
    sector: str
    status: str
    upper_bound: int | None
    witness: int | None
    certificate: LogicalWitness | None
    initial_upper_bound: int | None
    restarts_completed: int
    sweeps: int
    evaluations: int
    seed: int
    reason: str | None = None
    method: str = "seeded logical combinations with greedy stabilizer reduction"


def _basis_preflight(code, sector, max_basis_bits, max_basis_dimension):
    if not isinstance(code, CSSCode):
        raise TypeError("CSS heuristic analysis requires a packed CSSCode; sparse lowering is explicit")
    if sector not in ("x", "z"):
        raise ValueError("sector must be x or z")
    bits = nonnegative_integer(max_basis_bits, "max_basis_bits")
    dimension = nonnegative_integer(max_basis_dimension, "max_basis_dimension")
    m, n = code.hx.nrows + code.hz.nrows, code.n
    if max(n, code.hx.nrows, code.hz.nrows) > dimension:
        return "logical basis axes exceed max_basis_dimension"
    # Allow for both kernel/quotient bases, pairing workspace, and augmented
    # certificate equations. This is a payload proxy, not a peak-memory promise.
    if 8 * n * n + 2 * m * (n + 1) > bits:
        return "logical basis workspace exceeds max_basis_bits"
    return None


def css_logical_upper_bound(code: CSSCode, sector: str, *, seed=0, restarts=32,
                            max_sweeps=10, max_evaluations=10_000,
                            max_basis_bits=100_000_000, max_basis_dimension=4096):
    """Find a verified upper bound; no global optimality/lower-bound claim.

    Paired quotient representatives seed the search. Each additional restart
    chooses a nonzero logical combination and a random stabilizer offset.
    Greedy sweeps add stabilizers only when physical weight strictly decreases.
    Evaluations count candidate weight calculations; restarts/sweeps separately
    bound loop structure. Basis construction is guarded by axes/payload limits,
    not charged to the evaluation count or bounded by elapsed time.
    """
    reason = _basis_preflight(code, sector, max_basis_bits, max_basis_dimension)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    restarts = nonnegative_integer(restarts, "restarts")
    max_sweeps = nonnegative_integer(max_sweeps, "max_sweeps")
    max_evaluations = nonnegative_integer(max_evaluations, "max_evaluations")
    if reason is not None:
        return CSSHeuristicResult(sector, "basis_budget_exhausted", None, None, None,
                                   None, 0, 0, 0, seed, reason)
    if code.k == 0:
        return CSSHeuristicResult(sector, "no_logicals", None, None, None, None, 0, 0, 0, seed)
    if max_evaluations == 0:
        return CSSHeuristicResult(sector, "budget_exhausted", None, None, None,
                                   None, 0, 0, 0, seed, "no candidate evaluations available")
    logicals = css_logicals(code)
    representatives, duals = ((logicals.x, logicals.z) if sector == "x"
                              else (logicals.z, logicals.x))
    same = code.hx if sector == "x" else code.hz
    stabilizer_basis = same.row_basis().rows
    generators = tuple(dict.fromkeys(row for row in same.rows + stabilizer_basis if row))
    rng = Random(seed)
    best = None
    evaluations = sweeps = completed = 0
    exhausted = False

    def evaluate(operator, dual):
        nonlocal best, evaluations, exhausted
        if evaluations == max_evaluations:
            exhausted = True
            return None
        evaluations += 1
        weight = operator.bit_count()
        if best is None or (weight, operator) < (best.weight, best.operator):
            certificate = LogicalWitness(sector, operator, dual, weight)
            if not verify_css_logical(code, certificate):
                raise ArithmeticError("heuristic generated an invalid logical witness")
            best = certificate
        return weight

    for operator, dual in zip(representatives, duals):
        if evaluate(operator, dual) is None:
            break
    initial = best.weight if best is not None else None

    def descend(operator, dual, weight):
        nonlocal sweeps
        for _ in range(max_sweeps):
            if not generators:
                break
            sweeps += 1
            order = list(generators)
            rng.shuffle(order)
            improved = False
            for generator in order:
                candidate = operator ^ generator
                value = evaluate(candidate, dual)
                if value is None:
                    return
                if value < weight:
                    operator, weight, improved = candidate, value, True
            if not improved:
                break

    if not exhausted and best is not None:
        descend(best.operator, best.dual, best.weight)
    for _ in range(restarts):
        if exhausted:
            break
        coefficients = rng.randrange(1, 1 << len(representatives))
        operator = 0
        for j in set_bits(coefficients):
            operator ^= representatives[j]
        dual = duals[(coefficients & -coefficients).bit_length() - 1]
        for j in set_bits(rng.getrandbits(len(stabilizer_basis))):
            operator ^= stabilizer_basis[j]
        weight = evaluate(operator, dual)
        if weight is None:
            break
        descend(operator, dual, weight)
        if not exhausted:
            completed += 1
    return CSSHeuristicResult(sector, "budget_exhausted" if exhausted else "completed",
                               best.weight if best else None, best.operator if best else None,
                               best, initial, completed, sweeps, evaluations, seed)


__all__ = ["CSSHeuristicResult", "css_logical_upper_bound"]
