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

## Lifted products over an abelian group algebra

Let `A` and `B` have shapes `ma × na` and `mb × nb` over `R = GF(2)[G]`, for a finite abelian group of order `L`. Here `M*` means transpose combined with inversion of every group monomial. The shared-lift construction is

```text
Hx = [ A ⊗ I_nb | I_ma ⊗ B* ]
Hz = [ I_na ⊗ B | A* ⊗ I_mb ].
```

This is [Panteleev–Kalachev, §III-D, equation (12)](https://arxiv.org/html/2012.04068#S3.SS4) with their second seed replaced by `B*`. The ring identity `Hx Hz* = 0` lowers to binary orthogonality because the regular representation preserves multiplication and sends star to transpose. The length is `L(na nb + ma mb)`. Taking the ordinary hypergraph product **after** expanding both seeds instead gives `L²(na nb + ma mb)`; these are different constructions.

Dimension must be computed over the expanded binary checks or with a specialized module theorem. The field-based hypergraph-product dimension formula does not transfer to a ring with zero divisors. Noncommutative groups require compatible commuting actions; this implementation's abelian restriction supplies the needed commutation automatically.

For each seed, weight an entry by its monomial support size. X-check weights are row sums of A plus column sums of B, and Z-check weights are column sums of A plus row sums of B. Qubit participation uses column sums from both seeds on the variable block, and row sums from both seeds on the check block. Uniformly bounded weighted row/column sums therefore control locality independently of lift size.

## Exact structured dimension analysis

For odd cyclic order `L`, write `f=x^L+1=∏ f_i` over GF(2), with distinct irreducible factors of degrees `d_i`. The Chinese remainder theorem decomposes `R=GF(2)[x]/f` into fields. For a ring matrix `M`, its binary regular-action rank is `Σ d_i rank_(GF(2^d_i))(M mod f_i)`. The implementation verifies irreducibility and the factor product, then performs exact component elimination. For an odd abelian product, it evaluates characters in a verified splitting field, computes one rank per Frobenius orbit, and weights that rank by the orbit length. These methods compute dimension; the change of basis does not preserve physical Hamming weight. [Panteleev–Kalachev, Appendices A–B](https://arxiv.org/html/2012.04068#A2).

The cyclic polynomial backend also handles even `L`. Elementary polynomial row/column operations give a diagonal matrix with entries `a_j` and remain invertible modulo `f`. A scalar multiplication map `h ↦ a_j h mod f` has kernel dimension `deg gcd(a_j,f)`: writing `a_j=g a'` and `f=g f'` with coprime `a',f'`, its kernel consists exactly of multiples of `f'` modulo `f`. Thus binary rank is `Σ_j (L-deg gcd(a_j,f))`, including zero diagonal entries. This derivation needs no square-free assumption or ordered Smith divisibility chain. Recorded elementary operations can be replayed and the final gcd contributions rechecked independently. Unimodular polynomial basis transformations are standard module operations; [Sage's polynomial-matrix reference](https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/matrix_polynomial_dense.html) describes their role.

For mixed-order abelian groups, write each order as `L_i=p_i o_i`, where `p_i` is a power of two and `o_i` is odd. The group isomorphism `C_(L_i) ≅ C_(p_i) × C_(o_i)` permits character decomposition of the odd subgroup while retaining the regular action of the 2-primary subgroup over the splitting field. For `P=∏ p_i`, an `m × n` seed gives component matrices of shape `mP × nP`; their ranks are weighted by odd-character orbit lengths. Pure 2-groups require full-size components, and their repeated-root structure is never treated as a field decomposition.

The matrix-level APIs are `symbolic.abelian_analysis.abelian_rank` for odd factors and `symbolic.mixed_abelian.abelian_rank_mixed` for arbitrary factors. They return exact computational results and component metadata, not standalone replay certificates. The construction-level `analyze_lifted` dispatcher and CLI expose `crt`, `polynomial`, `abelian`, `mixed-abelian`, and `packed` as explicit choices. X/Z symbolic computations share an arithmetic budget; unsupported structure never triggers a hidden backend change. Study proposals remain scalar cyclic even though evaluation backends accept broader rings.

Factor degree, arithmetic steps, seed entries, intermediate polynomial degree, trace length, character count, splitting-field degree, and mixed component dimensions have separate budgets. Polynomial degree growth or field/root searches can exhaust a budget even when binary expansion would be small. No backend silently substitutes for another, and a partial computation is not reported as a rank. Work counts are deterministic arithmetic proxies, not elapsed-time or total-memory bounds.

Cyclic Laurent families keep exponents independent of `L`; collisions after reduction modulo `L` can only reduce incidence counts, supplying uniform locality bounds. Exact finite-instance ranks and sampled dimensions are useful for formulating a family theorem, but do not themselves prove uniform rate or distance growth. Construction manifests and fingerprints retain hypotheses and physical order without claiming novelty.

## Analysis must return evidence

Rank, orthogonality, quotient bases, and finite-instance sparsity are exact algebraic computations. In contrast, general stabilizer and CSS minimum-distance computation is NP-hard. A universal representation cannot make every analysis efficient. [Kapshikar and Kundu](https://arxiv.org/abs/2203.04262).

A distance routine should return an interval plus evidence. Exhausting every candidate of weight below `w` certifies a lower bound `w`; finding a nontrivial logical of weight `u` certifies an upper bound `u`. A budget interruption inside a layer does not certify that layer. Logical witnesses must be checked independently for syndrome zero and non-membership in the stabilizer/gauge equivalence space. A timeout never means that distance is infinite.

The implemented CSS portfolio seeds logical-coset combinations and descends by stabilizer additions to find lower-weight upper witnesses. It combines the smallest independently verified witness with the rigorous bound from completed exhaustive layers; MILP lower estimates remain separate numerical evidence. Exactness requires those rigorous bounds to coincide. Search can stop at local minima, and basis construction, candidate evaluations, enumeration, and optional solver work have distinct budgets. [Portfolio contracts](distance_portfolio.md).

Portable complete logical frames allow sparse perfect-syndrome simulation to test residual logical class without repeated elimination. Frame import checks rank evidence, logical commutation, and identity X/Z pairing against the exact source presentation. This verifies completeness algebraically; it does not validate the decoder or noise assumptions.

For CSS erasure on a known support `E`, restrict the opposite check matrix to `E`. A nonzero quotient between supported cycles and supported stabilizers signals uncorrectable erasure. This permits exact rank tests without enumerating all Pauli errors. Unknown-location adversarial correction instead requires `2t < d`.

A minimum-weight syndrome representative is a baseline decoder, not degenerate maximum-likelihood quantum decoding. Quantum decoding succeeds when the residual belongs to the stabilizer, or to the gauge space for a subsystem code. Under a correlated or biased noise model, equal weights need not have equal probability. Independent CSS-sector decoding also ignores X/Z correlations from Y errors.

The optional SciPy MILP adapter distinguishes verified feasibility witnesses from solver-reported bounds and optimality. Further optimization adapters should retain that distinction. SciPy exposes mixed-integer optimization with termination status, bounds, and gaps; a returned floating-point vector must be rounded cautiously and then checked over GF(2). A numerical optimality status is not a formally checked certificate. [SciPy MILP documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).

The optional NumPy `IsometryCode` accepts physical dimensions, an isometry, and a gauge dimension; direct operator Knill–Laflamme residuals provide a numerical oracle for small subspace and subsystem codes. Guarded lowering from signed stabilizers preserves their subspace while selecting an arbitrary logical basis. Finite `KrausChannel` objects now supply separately validated noise, composition, tensor products, and explicit Choi conversion; numerical validation and optional bounded repair remain distinct operations. The optional recovery backend fixes a subspace isometry `V` or a CPTP encoding channel with Kraus operators `E_s`, together with noise Kraus operators `N_a`, and optimizes maximally mixed logical-input entanglement fidelity. With `A_a=N_a V` and recovery Choi order received⊗logical, it forms `C=Σ vec_F(A_a†)vec_F(A_a†)†/K²`, maximizes `Tr(CJ)` under `J≥0, Tr_logical J=I`, and independently solves the dual `min Tr(Y)` subject to `Y⊗I_K−C≥0`. The conjugation makes the objective equal to `Σ_ab |Tr(R_b A_a)|²/K²`. For a fixed encoding channel, replace the effective family by `A_(a,s)=N_a E_s` and sum over both indices; its input dimension remains the protected logical dimension. [Fletcher, Shor, and Win](https://arxiv.org/abs/quant-ph/0606035).

`numerical_optimal` requires both solver optimality statuses, independent feasibility diagnostics, and a small numerical gap. These remain numerical observations rather than rigorous bounds. The raw Choi candidate is not repaired implicitly; explicit channel conversion rechecks trace preservation and records permitted modifications. Explicit gauge preparation gives `E_s=√λ_s V(I_K⊗|g_s>)` from `σ=Σ_s λ_s|g_s><g_s|`, realizing `ρ↦V(ρ⊗σ)V†`. This protects the specified product preparation, not unknown gauge states or arbitrary logical–gauge correlations. Joint encoding/recovery optimization, worst-case fidelity, gauge-independent recovery, and circuit realization are outside this backend. Choi-dimension/payload guards do not bound CVXPY canonicalization, solver workspace, or elapsed time. See [recovery assumptions](recovery.md) and the official [CVXPY constraint](https://www.cvxpy.org/tutorial/constraints/index.html) and [solver](https://www.cvxpy.org/tutorial/solvers/index.html) interfaces.

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

The native `BPDecoder` is a bounded normalized min-sum baseline with reusable sparse adjacency, damping, and clipped log-likelihood ratios. Prior probabilities zero and one are exact constraints, substituted and propagated through singleton checks without epsilon smoothing. Graph variable/check/edge budgets are enforced before allocation, and the fallback cannot alter fixed bits. Edge-message updates cost O(nonzeros); a full iteration also scans variables/checks. Exact syndrome agreement determines its `success` flag, which does **not** establish logical correction. Optional `osd0` performs a reliability-ordered binary solve after BP fails, with an explicit packed-payload guard. The fallback is neither a minimum-weight search nor a degenerate maximum-likelihood decoder. The CSS wrapper uses marginal X/Z error probabilities and ignores their correlations. BP with ordered-statistics post-processing is an established research direction; this small reference implementation does not claim the full performance of mature implementations. [Panteleev and Kalachev](https://arxiv.org/abs/1904.02703), [Roffe et al.](https://arxiv.org/abs/2005.07016).

## Search protocol

1. Generate candidates in a validity-preserving construction family with explicit parameter bounds.
2. Canonicalize only under transformations known to preserve the objective; presentation hashes are not code-equivalence certificates.
3. Reject invalid or dominated candidates using cheap exact rank and locality checks.
4. Obtain bounded distance evidence and test all witnesses independently.
5. Spend decoding and circuit-simulation budgets on the surviving Pareto set.
6. Preserve construction parameters, source revision, solver versions, random seeds, bounds, and reason for stopping.
7. Promote a numerical pattern to a conjecture, then state and prove the needed family hypotheses separately.

The implemented SQLite study engine covers fixed-term scalar cyclic seeds with deterministic random sampling or greedy mutations, transactional resumption, and provenance-bound caching. It ranks eligible instances by exact logical dimension and conservative distance lower bounds together with locality; witness upper bounds never stand in for achieved distance. Its frontier construction and history queries target modest bounded studies.

An LLM can propose structured expressions and proof strategies. It must not be the authority for validity, novelty, distance, or experimental success. Searching within valid constructions removes a large invalid search space; it does not eliminate difficult optimization or bias from the chosen family.
