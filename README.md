# QECLang

> [!NOTE]
> This project is incomplete and still under development. Most of the code was
> vibe-coded (written with heavy AI assistance) and still awaits thorough human
> review.

Structured primitives for quantum error correction research: exact binary
algebra, representation-aware constructions, bounded analysis, and reproducible
experiments. This is a research foundation, not a complete QEC compiler or a
claim of a new code family.

A universal finite-dimensional **semantic** description exists using an
encoding isometry or logical algebra. It generally costs exponential storage.
Efficient work therefore uses specialized representations with explicit
lowering contracts. CSS codes correspond to **based** chain complexes; their
homology counts logical qubits, while nontrivial cycle and cocycle weights
control distance. General stabilizer, subsystem, bosonic, and dynamical codes
require additional structure. See [the design](docs/design.md) and
[research analysis](docs/research.md) for definitions and primary references.

## Install and run

Python 3.11 or newer is required. The exact binary core has no runtime dependencies.

```sh
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
qeclang inspect --code steane
qeclang distance --code five --max-weight 3
qeclang distance --code steane --method portfolio --heuristic-seed 7
qeclang simulate --code steane --probability 0.01 --trials 1000 --seed 7
qeclang simulate --code toric --size 5 --decoder bp-osd0 --trials 100 --seed 7
```

Use `pip install -e .` for the core alone, or `pip install -e '.[numeric]'` for
dense semantic checks and SciPy/HiGHS optimization. The `dev` extra includes
these numerical dependencies. Install `pip install -e '.[sdp]'` for fixed-encoding
recovery optimization with CVXPY/SCS. No commercial solver or machine-specific
setup is required.
Install `pip install -e '.[circuits]'` for the tested Stim 1.16 circuit backend.

```python
from qeclang.constructions import hypergraph_product, repetition_check
from qeclang.analysis import css_distance, css_logicals, css_profile

code = hypergraph_product(repetition_check(3), repetition_check(4))
assert (code.n, code.k) == (18, 1)
assert code.to_chain_complex().betti_numbers[1] == code.k
print(css_profile(code))
print(css_logicals(code))
print(css_distance(code, "x", max_weight=3, max_candidates=100_000))
```

## Implemented scope

A typed, data-only program can compute a distance interval, certify its logical
witness, and carry that witness through a qubit relabeling. The example also
passes through based-chain and sparse representations. Run it twice to compare
the recorded artifacts, checks, and budget usage under the same runtime:

```sh
mkdir -p results
qeclang run examples/programs/transport.json > results/transport-run.json
qeclang run examples/programs/transport.json --replay results/transport-run.json
```

See [workflow programs](docs/workflow.md) for the supported operations, the
meaning of partial results, and the trust boundary for replay.

| Layer | Available functionality |
|---|---|
| Exact algebra | Immutable packed and sparse GF(2) matrices; exact packed rank, kernel, solve, products; guarded conversions and optional SciPy interchange |
| Code representations | CSS, based three-term complexes, signed Pauli stabilizers, subsystem gauge modules with optional signed center sectors |
| Symbolic constructions | Finite abelian group algebra and block matrices, multiplication/adjoints, regular-representation lowering, symbolic locality, replayable family manifests |
| Code families | Hypergraph and commutative lifted products, abelian bicycle codes, direct sums, toric and Bacon–Shor families; Steane, Shor and five-qubit fixtures |
| Transformations | Physical permutations, Hadamard duality, check reduction, chain maps, induced homology and quasi-isomorphism checks |
| Analysis | Exact cyclic/abelian dimension backends, certified portable logical frames, rank/locality/erasure analysis, bounded distance and verified CSS upper-bound heuristics; optional MILP |
| Decoding | Bounded minimum-weight and erasure decoders; sparse min-sum BP with optional order-zero reliability fallback; independent CSS sectors and perfect-syndrome simulation |
| General semantics | Optional dense isometries, Knill–Laflamme checks, immutable Kraus channels, explicit Choi/gauge preparation, and fixed-encoding recovery SDP |
| Evidence | Independently checked rank/logical certificates, Lean CSS witness theorem and bounded concrete replay |
| Experiments | Versioned JSON, CLI, resumable SQLite search studies, seeded candidate generation and timing scripts |
| Circuits | Bounded sequential CSS extraction, explicit Pauli fault locations, strict Stim detector models, and streamed logical-failure sampling |

`distance.status == "exact"` denotes an exact finite search result. A stopped
search reports a bound interval and a logical witness; it does not report an
unproved distance. Codes with no logical qubits use `no_logicals`. MILP uses
separate `solver_optimal` evidence because numerical optimization is not a
formal arithmetic proof. Read [API conventions](docs/api.md) before interpreting
results.

The [distance portfolio](docs/distance_portfolio.md) combines rigorous exhaustive
lower bounds with verified heuristic/solver upper witnesses. Numerical MILP
lower bounds remain separate; a heuristic score is never reported as a proof.

A finite check-weight profile does not prove that a family is qLDPC. The packed
backend is intended for small and moderate instances, and elimination can
become dense. Enumeration is exponential and bounded; decoding is a reference
baseline, not a scalable general qLDPC decoder. The initial circuit backend models
explicit Pauli faults and ideal initialization; optimized fault-tolerant schedules,
threshold studies, noncommutative lifted products, qudits, and bosonic/dynamical execution remain
research work described in the [roadmap](docs/roadmap.md).

## Reproducible experiments

```sh
mkdir -p results
qeclang export --code steane results/steane.json
qeclang inspect --input results/steane.json
qeclang search --orders 3 3 --terms 3 --trials 10 --seed 7 > results/search.jsonl
python scripts/benchmark.py --sizes 3 5 10 20 > results/benchmark.jsonl
python scripts/benchmark_sparse.py --size 1000 --degree 3 > results/sparse.jsonl
python examples/structured_codes.py
python examples/sparse_lifted.py
python examples/family_analysis.py --manifest results/family.json
qeclang family results/family.json --lift 31 --backend abelian
qeclang study results/study.sqlite3 --trials 20 --strategy mutate --seed 7
```

See [decoding methods](docs/decoding.md) for iterative decoder assumptions and
the distinction between syndrome convergence and logical success.

The optional [circuit-memory workflow](docs/circuits.md) preserves raw detector
hyperedges and explicit noise/boundary assumptions:

```sh
qeclang circuit-memory --code steane --basis z --rounds 3 --trials 100 --seed 7 \
  --two-qubit-error 0.001 --measurement-error 0.001 --fallback osd0 \
  --stim-output results/memory.stim
```

This prepares a known logical state and measures one logical basis per run.
It does not claim arbitrary-input encoding, fault tolerance, or a threshold.

Candidate records include construction parameters, seed, code, algorithm
budgets, bound statuses and witnesses. These are candidates, not automatically
novel or inequivalent codes. Generated results are ignored by Git.

For optional general-noise experiments, see [channels](docs/channels.md) and
[recovery optimization](docs/recovery.md). Run
`python examples/recovery_sdp.py --noise amplitude_damping` after installing the
`sdp` extra. Its primal/dual objectives and residuals are numerical evidence;
conversion to a recovery channel requires an explicit repair policy when needed.

[Family contracts](docs/structured_families.md), [study persistence](docs/search_studies.md),
and [formal verification scope](docs/formal.md) describe the respective trust and
resource boundaries. The abelian backends have separate field/component limits;
pure power-of-two groups can still require full-size components.

## Repository map

- `src/qeclang/`: composable library modules and CLI.
- `tests/`: exact small-instance oracles, regression and integration checks.
- `examples/`, `scripts/`: runnable examples and experiment drivers.
- `docs/`: design, research, API conventions and development guidance.
- `report/`: modular LaTeX research report with public-source bibliography.
- `references/`: optional ignored local literature cache; citations use public URLs.

See [development instructions](docs/development.md) for validation and report
building. No research novelty, asymptotic distance theorem, or end-to-end
formal verification is claimed by the implementation.
