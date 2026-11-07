# Finite-dimensional channels

The optional `qeclang.semantic.channels` module supplies a numerical reference
for noise and recovery maps beyond Pauli sampling. Install the `numeric` extra
for NumPy. These dense tools have explicit size guards; they are intended for
small-system semantic checks and recovery experiments.

## Kraus maps and composition

A `KrausChannel` contains matrices `K_a` of shape `(output_dim, input_dim)` and
acts by `E(A) = sum_a K_a A K_a†`. Construction checks
`sum_a K_a† K_a = I` numerically at the supplied absolute tolerance. Complete
positivity follows from the Kraus representation. The maximum-entry TP residual
is available as `tp_residual`; it is numerical evidence, not an exact certificate.

Operators are copied into immutable byte-backed arrays. Their writeability
cannot be re-enabled through NumPy flags. Channel attributes are frozen.
Construction does not normalize operators, compress Kraus representations, or
remove zero operators.

```python
import numpy as np
from qeclang.semantic.channels import amplitude_damping, dephasing

noise = amplitude_damping(0.2)
rho = np.diag([0.0, 1.0])
assert np.allclose(noise.apply(rho), np.diag([0.2, 0.8]))
combined = noise.then(dephasing(0.1))
```

`apply` is linear on any finite operator of the correct shape. It does not repair
or normalize its input into a density matrix. `first.then(second)` means
`second(first(A))`; `second.compose(first)` is equivalent. For `first.tensor(second)`,
the first channel is the outer, most-significant tensor factor: a basis index is
`first_index * second_dimension + second_index`. This places qubit zero in the
rightmost factor when factors are interpreted as qubits.

`max_dimension` bounds each channel input/output dimension. `max_kraus` bounds
the number of operators. `max_elements` bounds total stored Kraus entries and
each individual working array. Composition and tensor products check their
result sizes before expansion. These limits exclude simultaneous temporaries,
Python object overhead, and numerical library workspace; they are not total
memory guarantees. Nonfinite inputs are rejected, and nonfinite numerical
intermediates raise an error rather than producing a successful result.

## Choi ordering and conversion

`to_choi()` uses

\[
J(E)=\sum_{i,j}|i\rangle\langle j|\otimes E(|i\rangle\langle j|)
    =\sum_a\operatorname{vec}_F(K_a)\operatorname{vec}_F(K_a)^\dagger.
\]

Thus the composite index is `input * output_dim + output`, and `J` reshapes as
`(input_dim, output_dim, input_dim, output_dim)`. Trace preservation means that
tracing axes 1 and 3 gives the input identity; `trace(J) = input_dim`.
This matches the [IBM Choi convention](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.Choi).

`validate_choi(J, input_dim, output_dim, ...)` reports Hermiticity residual,
minimum eigenvalue of the Hermitian part, and TP residual. Its `satisfied` flag
uses the requested numerical tolerance. It does not modify or repair the input.

`KrausChannel.from_choi(...)` applies a stricter extraction policy by default:
the array must be exactly Hermitian and every computed eigenvalue must be
nonnegative. Every positive eigenvalue is retained, however small. Rank-deficient
physical Choi matrices can acquire tiny negative eigenvalues through numerical
roundoff, so strict extraction may reject them.

An explicit `repair_tolerance` permits symmetrization and clipping of negative
eigenvalues within the chosen bound. The returned channel's `conversion_report`
records the Hermiticity residual, minimum eigenvalue, whether symmetrization
occurred, number of clipped negative eigenvalues, and reconstruction residual.
No TP normalization or positive-eigenvalue truncation is performed. The resulting
Kraus map must independently pass TP validation at `tolerance`. Numerical
acceptance by `validate_choi` does not automatically authorize extraction repair.

## Noise parameters and fidelity

The qubit factories use these conventions:

| Factory | Map and probability meaning |
| --- | --- |
| `amplitude_damping(gamma)` | Excitation `|1>` decays to `|0>` with probability gamma. |
| `dephasing(p)` | Phase flip: `(1-p) rho + p Z rho Z`; complete dephasing occurs at `p=1/2`. |
| `depolarizing(p)` | Total nonidentity Pauli probability p; X, Y, Z each occur with probability `p/3`. Complete mixing occurs at `p=3/4`. |

All probabilities include endpoints zero and one. The depolarizing convention
matches `simulate_css_memory`; it differs from parameterizations where `p` is
the probability of replacing the input by the maximally mixed state.

For a square channel acting on a logical space of dimension d,
`entanglement_fidelity(channel)` computes

\[
F_e(I/d,E)=\sum_a|\operatorname{Tr}(K_a)/d|^2.
\]

This is fidelity relative to the identity channel for maximally mixed logical
input, following [Nielsen's Kraus expression](https://arxiv.org/abs/quant-ph/9606012).
It is neither worst-case fidelity nor a diamond-distance bound. Results are not
clipped to `[0,1]`; small numerical discrepancies remain visible.

For a fixed encoding V, physical noise N, and recovery R, form the effective
logical Kraus operators `R_b @ N_a @ V`, then evaluate their square channel.
When converting an SDP recovery candidate with explicit repair, recompute the
fidelity of the resulting channel: extraction can change its objective.
