# Fixed-encoding recovery optimization

Install `qeclang[sdp]` to use NumPy, CVXPY and the open-source SCS solver. The
core binary library does not import CVXPY. `optimize_recovery` is a dense
small-system reference for a fixed encoding and a fixed channel, following
[Fletcher, Shor and Win](https://arxiv.org/abs/quant-ph/0606035).

The objective is entanglement fidelity for the maximally mixed logical input.
The recovery outputs logical states, so it includes decoding. It is not a
worst-case fidelity, diamond-distance bound, noise threshold, or circuit-level
fault-tolerance calculation. Joint optimization of encoding and recovery is
outside this convex program. Subsystem isometries with nontrivial gauge factors
are rejected until an explicit gauge preparation/objective is specified.

## Convention and equations

Let V have shape physical×logical with logical dimension K. If the noise has
Kraus matrices N_a of shape received×physical, define A_a=N_a V. A recovery
Kraus matrix R_b has shape logical×received. Define

\[
c_a=\operatorname{vec}_F(A_a^\dagger),\qquad
C=\frac{1}{K^2}\sum_a c_a c_a^\dagger.
\]

The recovery Choi matrix J is ordered as received input tensor logical output.
Index `(p,l)` is `p*K+l`. Then

\[
F_e=\frac{1}{K^2}\sum_{a,b}|\operatorname{Tr}(R_b A_a)|^2
=\operatorname{Tr}(CJ).
\]

The primal maximizes Tr(CJ), subject to J positive semidefinite and its partial
trace over logical output equal to the received identity. The dual minimizes
Tr(Y), with Y Hermitian and Y tensor I_K − C positive semidefinite. The strictly
feasible primal matrix I_received tensor I_K/K establishes strong duality for
the mathematical finite-dimensional problem. This does not turn floating-point
solver results into rigorous optimality certificates.

```python
import numpy as np
from qeclang.semantic import IsometryCode, dephasing
from qeclang.recovery import optimize_recovery

code = IsometryCode((2,), np.eye(2))
result = optimize_recovery(code, dephasing(0.2))
assert result.status == "numerical_optimal"
assert abs(result.primal_objective - 0.8) < 1e-6
print(result.summary())
```

## Numerical evidence and resource contracts

The implementation solves primal and dual explicitly, then recomputes objective
values, trace-preservation residual, Hermiticity residual, primal minimum
eigenvalue, dual minimum slack eigenvalue, and objective gap from the returned
arrays. It rejects nonfinite intermediates. `numerical_optimal` requires both
solver statuses to be optimal, residuals within `validation_tolerance`, and a
gap within ten times that tolerance times the objective scale. Other outcomes
include `numerical_inaccurate`, `invalid_solution`, and `solver_error`.
`solve_dual=False` permits `primal_only`, which makes no independently checked
optimality claim. Inspect raw solver statuses as well as the aggregate status.

`primal_objective` and `dual_objective` are numerical values, not certified lower
and upper bounds. Feasibility accepted within a numerical tolerance is not exact
feasibility. Values are not silently clipped to the physical range. The raw
Choi and dual arrays have immutable byte backing. `summary()` excludes those
arrays and is suitable for JSON experiment records.

`max_choi_dimension` bounds K times the received dimension. `max_elements`
bounds individual model/input/intermediate arrays. These do not bound the sum
of all simultaneously live arrays, CVXPY's canonicalization or solver workspace,
or elapsed time. SCS defaults to `eps=1e-7,max_iters=10000` per solve; options can
be supplied explicitly. Other selected solvers use their own defaults unless
options are provided. No solver is substituted automatically. Both solver calls
and all options are recorded in provenance, along with package/solver versions
and encoding/Kraus-presentation hashes. The hashes identify input representations,
not channel equivalence classes. See [CVXPY solver options](https://www.cvxpy.org/tutorial/solvers/index.html).

## Extracting and evaluating a recovery channel

`result.as_channel()` applies the [channel conversion contract](channels.md).
Strict extraction rejects negative computed eigenvalues or non-Hermiticity;
it can reject a numerically acceptable solver candidate. Passing
`repair_tolerance` explicitly permits bounded symmetrization and clipping of
negative eigenvalues, and records those interventions. Trace preservation is
rechecked at the separately chosen `tolerance`; it is never normalized away.

Extraction may change the objective. Reevaluate the composed logical channel:

```python
from qeclang.semantic import KrausChannel, entanglement_fidelity
noise = dephasing(0.2)
recovery = result.as_channel(repair_tolerance=1e-6, tolerance=1e-6)
logical_channel = KrausChannel((code.isometry,)).then(noise).then(recovery)
print(entanglement_fidelity(logical_channel))
print(recovery.conversion_report)
```

`examples/recovery_sdp.py` emits compact diagnostic records and can explicitly
request extraction with `--extract`. Zero/one probability endpoints, complex
unitaries, rectangular encodings and output channels, exactly correctable error
sets, solver failures, and budget limits have dedicated tests. General channel
interchange, nontrivial gauge preparation, and rigorous SDP certificates remain
separate extensions.
