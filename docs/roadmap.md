# Development and validation roadmap

The initial release is a binary-qubit research foundation. Mathematical generality in the design does not imply implementation coverage. The source API and tests define the executable scope.

## Implemented foundation

- Exact packed binary algebra, chain/CSS codes, signed stabilizers, and gauge codes with explicit or automatically chosen center sectors.
- Hypergraph products, commuting-block/bicycle constructions, direct sums, and reference codes; sparse abelian GF(2) group-algebra multiplication, involution, and guarded regular-representation lowering.
- Exact logical frames, sparsity profiles, erasure tests, bounded distance search, and an optional MILP adapter whose solver evidence remains distinct from exhaustive certification.
- Bounded syndrome decoding and seeded perfect-syndrome memory experiments with Wilson intervals.
- Optional NumPy finite isometry/subsystem semantics, numerical Knill–Laflamme checks, and guarded signed-stabilizer dense lowering.
- CSS permutations and duality, check reduction, validated chain maps, composition, and induced homology actions.
- Versioned data-only JSON interchange, CLI entry points, and seeded bicycle candidate records.

The test suite is the source of truth for supported edge cases. Dense numerical semantics is not a scalable generic-code backend; the symbolic group algebra currently covers abelian products of cyclic groups, not arbitrary noncommutative groups.

## Milestones

| Stage | Deliverable | Acceptance condition |
|---|---|---|
| 1. Exact foundation | Binary matrices, CSS/chains, signed stabilizers, gauge parameters, structured examples, bounded distance and decoder baselines | Independent small-instance oracles; invalid-sign and zero-dimension tests; reproducible CLI examples |
| 2. Symbolic construction layer | Finite quotient rings, typed block/product expressions, persistent construction provenance | Expansion agrees with direct construction; involution and module-action tests; size estimates before lowering |
| 3. Analysis portfolio | MILP/SAT bounds, erasure analysis, decoder adapters, benchmark harness | Witness verification, honest timeout/gap handling, cross-backend agreement on small instances |
| 4. Experiment layer | Noise channels, measurement schedules, Stim/detector interchange, repeated rounds | Fault propagation tests; separation of code and circuit distance; confidence intervals |
| 5. Wider code semantics | Prime-power qudits, dense nonadditive oracle, subsystem sectors, bosonic approximations | Capability checks; exact versus approximate conversion contracts; declared truncation error |
| 6. Proof and discovery | Lean certificate checkers, theorem-driven family analysis, constrained search agents | No unverified assumptions promoted to results; reproducible candidate artifacts and proof dependencies |

Stages overlap where a small implementation is useful; they are not claims that all listed features already exist. Noncommutative lifted products, arbitrary qLDPC decoders, bosonic operators, and a universal optimizing compiler require separate work.

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

Start with the theorem that orthogonal CSS checks define a chain complex, rank/nullity quotient dimensions, the two-block commutation identity, and witness validation. Prove executable checkers correct before attempting ambitious asymptotic distance theorems. Export small certificates that a separately versioned checker can replay. A Lean project should pin its toolchain and dependencies; it must not rely on a developer's global installation.

A proved abstract identity does not automatically verify Python parsing, bit ordering, serialization, or a numerical solver. State the trusted boundary explicitly and test the translation between representations.
