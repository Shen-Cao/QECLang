"""Optional SciPy/HiGHS distance optimization with verified binary witnesses.

MILP optimality is numerical solver evidence, not an exact arithmetic proof.
Use bounded enumeration for independent small-instance exact verification.
"""
from dataclasses import dataclass
from math import isfinite
from time import monotonic

from qeclang.codes import CSSCode
from .logicals import quotient_basis


@dataclass(frozen=True)
class MILPDistanceResult:
    status: str
    upper_bound: int | None
    witness: int | None
    solver_lower_bound: float | None
    subproblems_solved: int
    subproblems_total: int
    solver_messages: tuple[str, ...]
    method: str = "scipy-highs-milp"


def css_distance_milp(code: CSSCode, sector: str, *, time_limit: float = 30.0,
                      node_limit: int = 100_000) -> MILPDistanceResult:
    """Minimize weight over k odd-logical-parity MILPs.

    Nontrivial homology is equivalent to odd pairing with at least one
    opposite-sector logical representative. Each subproblem imposes one
    such pairing, plus the stabilizer syndrome constraints. ``time_limit``
    is a total solver time budget (assembly is not timed by HiGHS);
    ``node_limit`` applies to each subproblem. A zero budget returns only
    the independently verified algebraic upper bound.

    ``solver_optimal`` means every subproblem reports optimality. Bounds
    from HiGHS remain explicitly distinguished from rigorous enumeration
    bounds; any returned witness is checked in exact GF(2) arithmetic.
    Requires the optional ``scipy`` and ``numpy`` dependencies.
    """
    if sector not in ("x", "z"):
        raise ValueError("sector must be 'x' or 'z'")
    if not isfinite(time_limit) or time_limit < 0:
        raise ValueError("time_limit must be finite and nonnegative")
    if type(node_limit) is not int or node_limit < 0:
        raise ValueError("node_limit must be a nonnegative integer")
    if code.k == 0:
        return MILPDistanceResult("no_logicals", None, None, None, 0, 0, ())
    checks, trivial = ((code.hz, code.hx) if sector == "x" else (code.hx, code.hz))
    checks, trivial = checks.row_basis(), trivial.row_basis()
    representatives = quotient_basis(checks.nullspace(), trivial)
    opposite = quotient_basis(trivial.nullspace(), checks)
    witness = min(representatives, key=int.bit_count)
    upper = witness.bit_count()
    if time_limit == 0 or node_limit == 0:
        return MILPDistanceResult("limit", upper, witness, 1.0, 0, len(opposite), ())
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise ImportError("MILP distance requires qeclang[numeric]") from exc

    deadline = monotonic() + time_limit
    lower_bounds, messages = [], []
    all_optimal = True
    solved = 0
    for logical in opposite:
        remaining = deadline - monotonic()
        if remaining <= 0:
            all_optimal = False
            break
        rows = checks.rows + (logical,)
        nvars = code.n + len(rows)
        ri, ci, data = [], [], []
        slack_bounds = []
        for i, row in enumerate(rows):
            support = row
            while support:
                bit = support & -support
                ri.append(i)
                ci.append(bit.bit_length() - 1)
                data.append(1.0)
                support ^= bit
            ri.append(i)
            ci.append(code.n + i)
            data.append(-2.0)
            slack_bounds.append((row.bit_count() - (i == len(rows)-1)) // 2)
        matrix = coo_matrix((data, (ri, ci)), shape=(len(rows), nvars)).tocsc()
        target = np.zeros(len(rows))
        target[-1] = 1
        objective = np.r_[np.ones(code.n), np.zeros(len(rows))]
        remaining = deadline - monotonic()
        if remaining <= 0:
            all_optimal = False
            break
        result = milp(objective, integrality=np.ones(nvars),
                      bounds=Bounds(np.zeros(nvars), np.r_[np.ones(code.n), slack_bounds]),
                      constraints=LinearConstraint(matrix, target, target),
                      options={"time_limit": remaining, "node_limit": node_limit,
                               "mip_rel_gap": 0.0})
        solved += 1
        messages.append(str(result.message))
        dual = getattr(result, "mip_dual_bound", None)
        lower_bounds.append(max(1.0, float(dual))
                            if dual is not None and isfinite(dual) else 1.0)
        valid = False
        if result.x is not None:
            bits = result.x[:code.n]
            rounded = np.rint(bits)
            if (np.all(np.abs(bits - rounded) < 1e-6)
                    and np.all((rounded == 0) | (rounded == 1))):
                candidate = sum(int(bit) << j for j, bit in enumerate(rounded))
                valid = (checks.matvec(candidate) == 0
                         and (candidate & logical).bit_count() % 2 == 1
                         and not trivial.contains(candidate))
                if valid and candidate.bit_count() < upper:
                    witness, upper = candidate, candidate.bit_count()
        if result.status == 0 and not valid:
            messages.append("Rejected solver solution: exact witness validation failed")
            return MILPDistanceResult("solver_error", upper, witness, None, solved,
                                      len(opposite), tuple(messages))
        all_optimal &= result.status == 0 and valid
        if result.status not in (0, 1):
            return MILPDistanceResult("solver_error", upper, witness, None, solved,
                                      len(opposite), tuple(messages))
    if solved != len(opposite):
        lower_bounds.append(1.0)
    return MILPDistanceResult("solver_optimal" if all_optimal else "limit",
                              upper, witness, min(lower_bounds, default=1.0),
                              solved, len(opposite), tuple(messages))
