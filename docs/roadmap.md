# Development and validation roadmap

The initial release is a binary-qubit research foundation. Mathematical generality in the design does not imply implementation coverage. The source API and tests define the executable scope.

## Implemented capabilities

| Layer | Available contracts | Remaining limitation |
|---|---|---|
| Code/algebra | Packed/sparse GF(2), CSS/chains, signed stabilizers, signed-sector gauge codes, validated sparse CSS | Generic exact rank can densify; efficient algorithms depend on structure |
| Construction | Ring elements/matrices, hypergraph/lifted/bicycle products, direct sums, reference codes; bounded construction/family records and uniform locality bounds | Abelian group-ring scope; no general noncommutative action compiler |
| Structured dimension | Odd-cyclic CRT, all-cyclic polynomial diagonalization with optional replay, odd-abelian and mixed-order abelian matrix ranks | Field/degree/component growth; pure 2-groups retain full-size components; backend choice remains explicit |
| Evidence | Binary rank identities, logical witnesses, portable frames, workflow checks for permutation and generator-row equivalence, and polynomial rank trace inspection | Logical upper bounds alone leave minimum weight unresolved |
| Workflow language | Bounded programs, explicit symbolic backends, typed resources, strict replay, independent inspection, and optional evidence archives | Finite instances and explicitly supplied maps; lookup by digest does not classify novelty |
| Analysis/decoding | Exact logical/erasure/locality tools, bounded distance and a verified-witness portfolio, optional numerical MILP; exhaustive/erasure/min-sum+OSD0 decoders with exact deterministic priors | Generic distance remains hard; iterative logical performance is unguaranteed |
| Experiments | Sparse perfect-syndrome simulation, bounded CSS memory circuits with explicit Pauli faults and strict Stim detector models, Wilson intervals, transactional SQLite studies | Circuit initialization remains ideal; sequential one-basis experiments do not establish fault tolerance or thresholds; no novelty classification |
| General semantics | Optional dense isometry/subsystem oracle, numerical Knill–Laflamme checks, finite CPTP channels and explicit fixed-gauge encoding preparation and fixed-CPTP-encoder recovery SDP | Exponential storage; recovery is numerical, assumes maximally mixed logical input and a fixed preparation, and does not optimize the encoder |
| Formal checking | Pinned Lean CSS syndrome/nonmembership theorem, concrete bounded literal replay | Python parsing/translation, code semantics, weight and rank proofs remain outside Lean |

The source APIs and tests define executable coverage. Exact arithmetic, independently replayable evidence, numerical solver status, and formal proof have different guarantees; one does not silently replace another.

## Highest-value next research milestones

1. **Strengthen distance evidence where a short proof is available.** The workflow transports independently checked upper bounds. Add small, independently checkable lower-bound predicates with explicit complexity and degeneracy handling. Combine bounds only when presentation, sector and metric agree. Compare with exact enumeration on small instances.
2. **Compare presentations under declared physical transformations.** Start with specific maps that preserve the property of interest. Compare retrieval and revalidation costs with inexpensive direct algorithms as well as with full recomputation. Keep experiments on generator-sensitive quantities bound to the original checks.
3. **Evaluate the language on held-out research tasks.** Compare direct use of the same kernels with typed-program execution under equal work allowances. Measure invalid claims detected, useful bounds per unit of work, and costs of recording and checking. Run agent proposal strategies only once these controls and a provenance-complete reference set are available.

Distance bounds, external decoder comparisons and fault-tolerant extraction remain experimental targets. Prioritize each when it exercises a concrete transport, comparison or evidence requirement in those programs.

Noncommutative actions, prime-power qudits, bosonic approximations, and dynamical codes should follow a concrete question requiring them. A general optimizing compiler or GPU backend needs a measured bottleneck and stable transformation contracts first.

## Immediate research questions

- Can symbolic quotient-ring operations eliminate enough repeated rank computations across a family to outperform repeated explicit expansion?
- Which construction rewrites preserve not only homology but the physical weight metric and sparse measurement presentation?
- Can independently checked logical witnesses and complete-layer search bounds provide useful short certificates for solver portfolios?
- Which small-size parameter gains survive ancilla, extraction-depth, and connectivity costs?
- How much search quality is gained by preserving a construction grammar compared with unconstrained matrix mutation at equal compute budget?
- Can proof obligations for family-level locality and dimension be generated compositionally, while leaving distance to specialized theorems?

These are proposed experiments and theorem targets, not established contributions.

## Benchmark suite

Use exact known small codes as correctness fixtures: repetition sectors, Steane, five-qubit stabilizer, small toric products, and Bacon–Shor gauge parameters. Include dependent checks, no checks, zero logical qubits, inconsistent signs, noncommuting checks, asymmetric matrices, and degenerate seeds.

For algebra scaling, sweep seed sizes and lift orders. Report rank/elimination time, construction time, peak memory, density before and after elimination, and expanded dimension. Compare packed arithmetic against an independent dense GF(2) oracle on small instances, then against a mature backend on larger ones.

For distance scaling, compare exhaustive enumeration and optimization under equal budgets. Report the entire bound interval and number of verified candidates, not just the best witness. Keep failure and timeout cases in the dataset. For search, compare random valid sampling, parameter-local search, and agent proposals with a shared budget and held-out family parameters.

For decoder scaling, report block failure and logical failure separately, plus residual verification. Sweep physical error probability and block size under the same noise model. Give confidence intervals even when zero failures occur. Preserve failed decode counts and timeouts instead of discarding those trials.

Performance results should be generated into ignored output directories. Commit scripts, schemas, parameters, and small hand-verifiable fixtures. Cite literature with public DOI/arXiv URLs; any downloaded copies are a disposable local cache.

## Formalization boundary

The current Lean project proves the CSS parity-witness checker and replays concrete literals. Next proof targets are faithful parsing/translation, physical weight, rank-certificate identities, and selected construction identities; asymptotic distance theorems require their own hypotheses and mathematical development. Keep toolchains pinned and certificates replayable independently of their producer.

A proved abstract identity does not automatically verify Python parsing, bit ordering, serialization, or a numerical solver. State the trusted boundary explicitly and test the translation between representations.
