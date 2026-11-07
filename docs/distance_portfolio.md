# Distance bounds and heuristic witnesses

`css_logical_upper_bound` searches for low-weight logical operators of a packed
CSS code. Its result contains a distance upper bound and an independently
checkable `LogicalWitness`; it never asserts a lower bound or global minimum.

```python
from qeclang.analysis.heuristic import css_logical_upper_bound
from qeclang.analysis.portfolio import css_distance_portfolio
from qeclang.analysis.certificates import verify_css_logical
from qeclang.constructions import steane

code = steane()
upper = css_logical_upper_bound(code, "x", seed=17, restarts=32,
                                max_sweeps=10, max_evaluations=10_000)
assert verify_css_logical(code, upper.certificate)
result = css_distance_portfolio(code, "x", seed=17, max_weight=2,
                                max_candidates=100_000)
print(result.lower_bound, result.upper_bound, result.status)
```

## What the heuristic preserves

Paired logical bases satisfy `x_i · z_j = delta_ij`. A nonzero combination of
same-sector basis vectors represents a nontrivial logical coset. Any coefficient
equal to one identifies an opposite-sector basis vector with odd pairing,
providing a dual witness. Adding a same-sector stabilizer preserves syndrome,
logical coset, and that pairing.

The initial candidates are individual paired representatives. The best examined
initial representative undergoes greedy sweeps of the supplied stabilizer rows
and their independent row basis. Rows are shuffled with the seeded RNG. Only
strict weight decreases update the current descent point. Additional restarts
choose nonzero logical combinations and random stabilizer offsets before the
same descent. There are no global-optimality or information-set-decoding claims.
The algorithm can stop at local minima and its returned upper bound need not be
the distance.

Every published improvement is verified by exact syndrome and dual-pairing
checks. These checks do not rely on the heuristic search, an optimizer, or
floating-point arithmetic. `initial_upper_bound` describes the best initial
representative actually examined; an evaluation budget can stop before all
representatives are examined.

## Bounds in the portfolio

`css_distance_portfolio` combines three sources:

| Source | Evidence used |
| --- | --- |
| Exhaustive bounded enumeration | Rigorous lower bound from fully excluded weight layers, plus a verified logical upper witness |
| Seeded heuristic | Verified logical upper witness only |
| Optional SciPy/HiGHS MILP | Verified logical upper witness; solver lower bound retained separately as numerical evidence |

The result's rigorous lower bound is always enumeration's lower bound. Its upper
bound is the smallest independently verified witness weight. A rejected MILP
witness contributes no upper bound. `solver_lower_bound` is reported separately
and never changes the rigorous lower bound or the exactness status, even when
the solver labels a solution optimal. An inconsistency between rigorous lower
and verified upper bounds raises an error.

Enumeration need only exclude weights below the smallest known verified upper
bound, so the portfolio caps its search weight accordingly. `status="exact"`
requires equality of the rigorous lower bound and verified upper bound.
`status="bounded"` preserves the gap. Numerical solver optimality alone cannot
close that gap. `distance` is populated only for `exact` results.

For zero logical dimension, including the empty code, the result is
`no_logicals`, with no finite or artificially favorable distance assigned.

## Resource limits

`restarts` bounds additional random starts; `max_sweeps` applies separately to
the initial descent and each restart. `max_evaluations` counts candidate weight
calculations, including initial representatives and proposed stabilizer moves.
Exhaustion retains the best verified witness, if any; zero evaluations may return
no heuristic witness. The portfolio can still obtain one from enumeration.

Before logical bases or certificate equations are constructed, both APIs check
`max_basis_dimension` against physical/check dimensions and `max_basis_bits`
against a conservative algebra-workspace payload proxy. The proxy is
`8*n*n + 2*(mX+mZ)*(n+1)` bits. It is not a measured peak-memory or elapsed-time
guarantee and excludes Python object overhead. Basis construction and certificate
checks are not counted as candidate evaluations. Sparse presentations require
an explicit guarded conversion to the packed backend.

Enumeration has its own weight/candidate limits. Optional MILP has its own
solver time/node controls; model assembly and logical-basis work are outside the
solver's time limit. These components therefore have distinct, visible budgets,
not a single wall-clock deadline. The fixed seed reproduces the current
algorithm's proposal sequence; preserve source/runtime provenance for studies.
