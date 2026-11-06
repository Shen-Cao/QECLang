# Mathematical model and research program

## Based chain complexes and CSS codes

Fix binary matrices `Hx` of shape `mx × n` and `Hz` of shape `mz × n`, with `Hx Hzᵀ = 0`. The convention throughout this repository is

```text
C2 = F2^mz  --d2 = Hzᵀ-->  C1 = F2^n  --d1 = Hx-->  C0 = F2^mx.
```

Physical qubits are the specified basis of `C1`. Rows of `Hx` are X checks and rows of `Hz` are Z checks. For error `(x,z)`, the X-check syndrome is `Hx z` and the Z-check syndrome is `Hz x`. Then

```text
k   = n - rank(Hx) - rank(Hz)
LZ  = ker(Hx) / row(Hz)
LX  = ker(Hz) / row(Hx)
dZ  = min { wt(z) : z in ker(Hx), z not in row(Hz) }
dX  = min { wt(x) : x in ker(Hz), x not in row(Hx) }
d   = min(dX, dZ).
```

The quotient dimensions count independent logical degrees of freedom. Their minimum nonzero representative weights are systolic and cosystolic quantities. “Many large holes” is useful intuition, but ordinary homology computes only the count; the chosen basis and weight function supply size. The relevant foundation is homological algebra, finite-field linear algebra, and combinatorics. Algebraic geometry contributes particular constructions, but is not required for every qLDPC family. [Breuckmann and Eberhardt](https://arxiv.org/abs/2103.06309).

These distance formulas measure nontrivial logical operators. When `k=0`, the sets are empty: this software reports no logical operator and infinite logical distance. Some coding-theory conventions assign a different distance to stabilizer states; comparisons must state which convention is used.

## Constructions that expose useful primitives

For binary parity checks `A` (`m1 × n1`) and `B` (`m2 × n2`), a hypergraph product has

```text
Hx = [ A ⊗ I_n2  | I_m1 ⊗ Bᵀ ]
Hz = [ I_n1 ⊗ B  | Aᵀ ⊗ I_m2 ].
```

Multiplying gives two copies of `A ⊗ Bᵀ`, which cancel over GF(2). The blocklength is `n1 n2 + m1 m2`. If `ki = ni-rank(Hi)` and `kiT = mi-rank(Hi)`, the logical count is `k1 k2 + k1T k2T`. Sparse seeds retain controlled local weights; product structure also supports parameter theorems. [Tillich and Zémor](https://arxiv.org/abs/0903.0566).

For commuting square matrices `A,B`, the two-block construction `Hx=[A|B]`, `Hz=[Bᵀ|Aᵀ]` commutes because `AB+BA=0`. Circulant polynomials are one source of commuting matrices. This identity is a reusable validity primitive, but neither a positive rate nor a large distance follows from commutativity alone. Bivariate bicycle codes provide an important structured finite-size target. [Bravyi et al.](https://doi.org/10.1038/s41586-024-07107-7).

Lifted products generalize these operations using algebraic symmetry. Asymptotically good qLDPC families already exist: constant rate and distance proportional to length are achievable together. The research goal is therefore better usable constructions, constants, decoders, and proof automation, not merely resolving that existence question. [Panteleev and Kalachev](https://arxiv.org/abs/2111.03654).

## Analysis must return evidence

Rank, orthogonality, quotient bases, and finite-instance sparsity are exact algebraic computations. In contrast, general stabilizer and CSS minimum-distance computation is NP-hard. A universal representation cannot make every analysis efficient. [Kapshikar and Kundu](https://arxiv.org/abs/2203.04262).

A distance routine should return an interval plus evidence. Exhausting every candidate of weight below `w` certifies a lower bound `w`; finding a nontrivial logical of weight `u` certifies an upper bound `u`. A budget interruption inside a layer does not certify that layer. Logical witnesses must be checked independently for syndrome zero and non-membership in the stabilizer/gauge equivalence space. A timeout never means that distance is infinite.

For CSS erasure on a known support `E`, restrict the opposite check matrix to `E`. A nonzero quotient between supported cycles and supported stabilizers signals uncorrectable erasure. This permits exact rank tests without enumerating all Pauli errors. Unknown-location adversarial correction instead requires `2t < d`.

A minimum-weight syndrome representative is a baseline decoder, not degenerate maximum-likelihood quantum decoding. Quantum decoding succeeds when the residual belongs to the stabilizer, or to the gauge space for a subsystem code. Under a correlated or biased noise model, equal weights need not have equal probability. Independent CSS-sector decoding also ignores X/Z correlations from Y errors.

The optional SciPy MILP adapter distinguishes verified feasibility witnesses from solver-reported bounds and optimality. Further optimization adapters should retain that distinction. SciPy exposes mixed-integer optimization with termination status, bounds, and gaps; a returned floating-point vector must be rounded cautiously and then checked over GF(2). A numerical optimality status is not a formally checked certificate. [SciPy MILP documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).

The optional NumPy `IsometryCode` accepts physical dimensions, an isometry, and a gauge dimension; direct operator Knill–Laflamme residuals provide a numerical oracle for small subspace and subsystem codes. Guarded lowering from signed stabilizers preserves their subspace while selecting an arbitrary logical basis. For general channels, recovery optimization can use semidefinite programming with fixed encoding and an explicit fidelity objective. Joint encoding/recovery optimization is not automatically convex. SDP feasibility tolerances, primal/dual residuals, cutoff assumptions, and objective gaps must accompany results.

## Evaluate more than rate and distance

| Property | Why it matters | Evidence required |
|---|---|---|
| `n,k,dX,dZ`, rate, relative distance | Storage and adversarial protection | Exact ranks; bounds/witnesses or theorem hypotheses |
| Check weight and qubit degree | Measurement complexity and LDPC family conditions | Original generator presentation; uniform family bounds |
| Logical error probability | Performance under a chosen decoder/noise pair | Noise model, trial count, seed, confidence interval |
| Decoder runtime and memory | Practical throughput | Hardware/software versions, size scaling, workload |
| Extraction depth, ancillas, connectivity | Physical overhead | Concrete schedule and interaction graph |
| Circuit distance and correlated faults | Fault-tolerant protection | Circuit-level model; not inferred from code distance |
| Redundant checks and measurement robustness | Syndrome noise and single-shot prospects | Explicit measurement model and decoder |
| Logical gates and encoding cost | Computation rather than memory alone | Verified logical action and implementation resources |
| Expansion, soundness, energy barrier | Structure and possible decoding/memory guarantees | Precisely normalized definitions and proofs/bounds |

A threshold is a scaling claim about an explicit family, noise model, measurement procedure, and decoder. One finite code's Monte Carlo curve does not establish a threshold. Under perfect syndrome, code-capacity simulations must not be labeled circuit-level fault-tolerance results.

## Integrate established tools by capability

Use Stim for supported stabilizer-circuit simulation and detector models, not as an arbitrary-code semantic engine. Use PyMatching when a decoding model admits the appropriate graph representation; generic qLDPC syndrome hyperedges do not automatically become matching instances. Evaluate `ldpc` BP+OSD/BP+LSD for broader binary syndrome problems, retaining a trusted residual verifier. [Stim](https://arxiv.org/abs/2103.02202), [Sparse Blossom/PyMatching](https://arxiv.org/abs/2303.15933), [`ldpc` documentation](https://software.roffe.eu/ldpc/).

## Search protocol

1. Generate candidates in a validity-preserving construction family with explicit parameter bounds.
2. Canonicalize only under transformations known to preserve the objective; presentation hashes are not code-equivalence certificates.
3. Reject invalid or dominated candidates using cheap exact rank and locality checks.
4. Obtain bounded distance evidence and test all witnesses independently.
5. Spend decoding and circuit-simulation budgets on the surviving Pareto set.
6. Preserve construction parameters, source revision, solver versions, random seeds, bounds, and reason for stopping.
7. Promote a numerical pattern to a conjecture, then state and prove the needed family hypotheses separately.

An LLM can propose structured expressions and proof strategies. It must not be the authority for validity, novelty, distance, or experimental success. Searching within valid constructions removes a large invalid search space; it does not eliminate difficult optimization or bias from the chosen family.
