# Small reproducible CSS reference cases

`qeclang.catalog.references` contains fourteen fixed positive binary CSS recipes. They exercise construction, exact analysis, presentation matching, and interoperability on small instances. They are not a comprehensive code database, a representative sample of qLDPC performance, or a novelty oracle. The largest case has 18 physical qubits; the remaining cases have at most nine.

```python
from qeclang.catalog.references import reference_cases, build_reference

for case in reference_cases():
    print(case.id, case.version, case.to_dict()["expected"])

code = build_reference("hgp_open_2_3", max_qubits=32,
                       max_checks=32, max_bits=100_000)
```

`reference_cases()` returns an immutable tuple of frozen `ReferenceCase` descriptors in lexical ID order. Each descriptor exposes `id` (`case_id` is an alias), `version`, `expected_n`, `expected_k`, `expected_dx`, `expected_dz`, and `expected_distance`, together with check counts, a regeneration recipe, coordinate conventions, and scoped public citations. `to_dict()` returns detached JSON-compatible metadata under `qeclang.reference_case.v1`, including category, provenance, and `parameter_status="reference_expectations"`.

Reading a descriptor or building a code does **not** independently certify its expected parameters. The builder validates CSS commutation and declared output shape; rank and distance verification are separate operations. No proof output, timing, simulator sample, solver result, or machine configuration is part of the tracked catalog.

## Cases and expected parameters

All distances below concern physical site Hamming weight. A dash means that the logical minimization set is empty, not a finite distance or a numerical failure.

| ID | \(n\) | \(k\) | \(d_X\) | \(d_Z\) | Purpose |
|---|---:|---:|---:|---:|---|
| `bell_redundant` | 2 | 0 | — | — | Repeated and zero checks, zero logical dimension |
| `bicycle_cyclic_3` | 6 | 2 | 2 | 2 | Small cyclic group-algebra construction |
| `css_4_2_2` | 4 | 2 | 2 | 2 | Distance-two side of an equal-parameter comparison |
| `disjoint_checks_4_2_1` | 4 | 2 | 1 | 1 | Distance-one side of that comparison |
| `empty` | 0 | 0 | — | — | Empty physical and check spaces |
| `free_2` | 2 | 2 | 1 | 1 | No checks and two unprotected logical qubits |
| `hgp_open_2_3` | 8 | 1 | 3 | 2 | Rectangular open repetition product |
| `repetition_x_checks_3` | 3 | 1 | 1 | 3 | Explicit X/Z asymmetry |
| `repetition_z_checks_3` | 3 | 1 | 3 | 1 | Opposite asymmetric orientation |
| `shor` | 9 | 1 | 3 | 3 | Standard nine-qubit code |
| `steane` | 7 | 1 | 3 | 3 | Standard seven-qubit CSS code |
| `steane_redundant` | 7 | 1 | 3 | 3 | Same physical code with changed generators |
| `toric_2` | 8 | 2 | 2 | 2 | Small periodic product with dependent checks |
| `toric_3` | 18 | 2 | 3 | 3 | Periodic product beyond the smallest fixtures |

The full CSS distance is \(\min(d_X,d_Z)\) when logical qubits exist. Both asymmetric repetition orientations are retained to expose sector-convention errors. The rectangular product's initial algebraic X representative can be heavier than its minimum-weight logical; a bounded producer search is needed to obtain the weight-three witness with the current basis convention.

`steane` and `steane_redundant` have identical X/Z row spaces in the same physical coordinates. Their ordered presentations differ, and explicit two-way generator maps certify their equivalence. By contrast, `css_4_2_2` and `disjoint_checks_4_2_1` share \(n=4,k=2\) but have different distances. They cannot be related by a physical Hamming-metric-preserving code equivalence. Scalar parameter matching alone cannot decide the relation used by the [fixed-coordinate index](catalog_lookup.md).

## Recipes and physical coordinates

All integer check rows use physical site \(i\) as bit \(i\). The following literal tuples specify generators in order, including dependencies and zero rows:

* `steane`: \(H_X=H_Z=(85,102,120)\). The redundant variant appends `(51,0,85)` to both matrices.
* `shor`: \(H_X=(63,504)\), \(H_Z=(3,6,24,48,192,384)\).
* `css_4_2_2`: both check matrices contain the single row `15`. The distance-one foil uses `Hx=(3,)`, `Hz=(12,)`.
* `bell_redundant`: `Hx=(3,3,0)`, `Hz=(3,0,3)`. `free_2` has two width-two, zero-row matrices; `empty` uses width-zero, zero-row matrices.
* The repetition fixtures use `repetition_check(3)=(3,6)` in one sector and a zero-row width-three matrix in the other.

`hgp_open_2_3` calls `hypergraph_product(repetition_check(2), repetition_check(3))`. For seed shapes \(m_a\times n_a\) and \(m_b\times n_b\), the physical order is the \(n_an_b\) variable-variable sites followed by the \(m_am_b\) check-check sites. Within each block, the second tensor factor varies fastest. `toric_2` and `toric_3` use the same ordering with cyclic repetition checks in both factors. The size-two cyclic seed retains its two identical rows; they are not silently reduced.

`bicycle_cyclic_3` uses \(A=1+x\) and \(B=1+x^{-1}\) over \(\mathbb F_2[C_3]\). The regular representation lists residues `0,1,2`; row \(r\) accesses columns \(r+t\bmod3\) for each translation term \(t\). The two physical blocks correspond to `[A | B]`, and \(H_Z=[B^T|A^T]\). Here \(B=A^T\), giving the dual-containing special case `[C | C^T]`. The tiny seed is an algebraic fixture, not a published threshold or overhead benchmark.

## Provenance and citation scope

`source_type="analytic_literature"` marks the standard Steane, Shor, and periodic toric constructions. Their physical ordering is specified by the native recipes rather than inferred from a paper's drawing. The other cases are `algebraic_fixture` entries, including small structured seeds and altered generator presentations. No entry is classified as measured: this catalog contains no empirical decoder performance, stochastic estimates, or numerical-optimization claims.

The public references support these specific construction scopes:

* Steane's seven-qubit CSS construction and single-error correction: [Multiple Particle Interference and Quantum Error Correction](https://arxiv.org/abs/quant-ph/9601029), *Proceedings of the Royal Society A* 452 (1996), [DOI](https://doi.org/10.1098/rspa.1996.0136).
* Shor's nine-qubit encoding: [Scheme for reducing decoherence in quantum computer memory](https://doi.org/10.1103/PhysRevA.52.R2493), *Physical Review A* 52 (1995).
* Toric-code stabilizers and logical sectors: [Fault-tolerant quantum computation by anyons](https://arxiv.org/abs/quant-ph/9707021), *Annals of Physics* 303 (2003), [DOI](https://doi.org/10.1016/S0003-4916(02)00018-0).
* The hypergraph-product construction: [Quantum LDPC codes with positive rate and minimum distance proportional to \(n^{1/2}\)](https://arxiv.org/abs/0903.0566), *IEEE Transactions on Information Theory* 60 (2014), [DOI](https://doi.org/10.1109/TIT.2013.2292061). This citation does not identify the particular open `2×3` seed as a reported research result.
* The cyclic dual-containing bicycle construction: [Sparse Graph Codes for Quantum Error-Correction](https://arxiv.org/abs/quant-ph/0304161), construction B, *IEEE Transactions on Information Theory* 50 (2004), [DOI](https://doi.org/10.1109/TIT.2004.834737). The six-site fixture does not reproduce a performance experiment from that paper.

The catalog links these sources and regenerates native matrices. It neither scrapes external code tables nor redistributes downloaded matrices or publications. A construction citation is not an independent verification receipt for a particular implementation.

## Bounds, versions, and verification

`build_reference(case_id, *, max_qubits=64, max_checks=64, max_bits=1_000_000)` dispatches only fixed in-repository recipes. `max_checks` bounds the total X and Z generator count. Before constructing matrices, it checks physical width, generator count, and the packed payload proxy

\[
n(m_X+m_Z)+m_Xm_Z,
\]

including the CSS commutation-validation product. Product constructors receive the same `max_bits` ceiling for their own intermediate guards. These are size controls, not peak-memory or elapsed-time guarantees. Unknown IDs raise `KeyError`; invalid types or exhausted size limits raise errors. There is no external path, network read, user-supplied constructor expression, or implicit persistence.

Every descriptor currently has version `1`. Change a case's version when its recipe, physical ordering, or expected mathematical parameters change. The descriptor schema has a separate version. Code and index fingerprints bind their own precise representations; a metadata version is neither an equivalence certificate nor an author signature. Generated matrices, proofs, comparisons, and timings should be written only to ignored output locations when explicitly requested.

`tests/test_reference_cases.py` validates all cases with bounded native evidence. Rank factorizations and one-sided inverse identities establish \(k=n-\operatorname{rank}H_X-\operatorname{rank}H_Z\). For each finite sector distance, an independently checked nontrivial logical witness supplies the upper bound, and [weight-one/two exclusion certificates](low_weight.md) supply the lower bound when needed. A weight-one upper witness already meets the trivial positive-weight lower bound. Full rank sums establish the zero-logical cases without assigning them finite distance.

The tests recheck all produced evidence with rank, solving, search, and certificate-production methods disabled. This checks the native Python certificate path, not a Lean formalization, completeness of the reference corpus, a claim of novelty, or an asymptotic qLDPC-family theorem.
