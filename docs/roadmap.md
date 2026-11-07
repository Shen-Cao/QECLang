# Development and validation roadmap

The initial release is a binary-qubit research foundation. Mathematical generality in the design does not imply implementation coverage. The source API and tests define the executable scope.

## Implemented capabilities

| Layer | Available contracts | Remaining limitation |
|---|---|---|
| Code/algebra | Packed/sparse GF(2), CSS/chains, signed stabilizers, signed-sector gauge codes, validated sparse CSS | Generic exact rank can densify; efficient algorithms depend on structure |
| Construction | Ring elements/matrices, hypergraph/lifted/bicycle products, direct sums, reference codes; bounded construction/family records and uniform locality bounds | Abelian group-ring scope; no general noncommutative action compiler |
| Structured dimension | Odd-cyclic CRT, all-cyclic polynomial diagonalization with optional replay, odd-abelian and mixed-order abelian matrix ranks | Field/degree/component growth; pure 2-groups retain full-size components; backend choice remains explicit |
| Evidence | Binary rank identities, parity logical witnesses, portable presentation-bound records; certified complete logical frames importable against sparse checks | Upper-bound witnesses do not certify distance optimality |
| Analysis/decoding | Exact logical/erasure/locality tools, bounded distance, optional numerical MILP; exhaustive/erasure/min-sum+OSD0 decoders | Generic distance remains hard; iterative logical performance is unguaranteed |
| Experiments | Sparse perfect-syndrome simulation with certified frame and Wilson intervals; transactional SQLite random/mutation studies | Restricted search domain; no circuit-level fault model or novelty classification |
| General semantics | Optional dense isometry/subsystem oracle, numerical Knill–Laflamme checks, finite CPTP channels and fixed-subspace-encoder recovery SDP | Exponential storage; recovery is numerical, assumes maximally mixed logical input, and does not optimize the encoder |
| Formal checking | Pinned Lean CSS syndrome/nonmembership theorem, concrete bounded literal replay | Python parsing/translation, code semantics, weight and rank proofs remain outside Lean |

The source APIs and tests define executable coverage. Exact arithmetic, independently replayable evidence, numerical solver status, and formal proof have different guarantees; one does not silently replace another.

## Highest-value next research milestones

1. **Distance evidence and comparative evaluation.** Combine stronger logical-witness search with construction-specific lower bounds and, where useful, proof-producing optimization. Cross-check small cases exactly and report unresolved intervals at larger sizes. Compare finalists under fixed noise/decoder settings with a mature decoder. Acceptance requires independently verified witnesses, honest budget handling, and reproducible comparisons; no general efficient distance solver is assumed.
2. **Circuit-aware correction experiments.** Build on the finite-channel and fixed-encoder recovery layer with measurement schedules, noisy syndrome extraction, and a concrete fault model for a selected code family. Acceptance requires validated instrument semantics and fault propagation, explicit ancilla/depth/connectivity costs, and separation of code distance from circuit distance. Dense channel recovery does not determine a fault-tolerant circuit.
3. **Theorem-aware family search.** Extend the existing study engine only where a construction grammar, mathematical bound, or useful equivalence reduction supplies an advantage. Preserve proof hypotheses and benchmark random/local/agent proposals at equal budgets. Acceptance is reproducible improvement on held-out instances or a proved family claim, rather than more generated candidates.

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
