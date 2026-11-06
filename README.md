# QECLang

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
qeclang simulate --code steane --probability 0.01 --trials 1000 --seed 7
qeclang simulate --code toric --size 5 --decoder bp-osd0 --trials 100 --seed 7
```

Use `pip install -e .` for the core alone, or `pip install -e '.[numeric]'` for
dense semantic checks and SciPy/HiGHS optimization. The `dev` extra includes
these numerical dependencies. No commercial solver or machine-specific setup
is required.

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

| Layer | Available functionality |
|---|---|
| Exact algebra | Immutable packed and sparse GF(2) matrices; exact packed rank, kernel, solve, products; guarded conversions and optional SciPy interchange |
| Code representations | CSS, based three-term complexes, signed Pauli stabilizers, subsystem gauge modules with optional signed center sectors |
| Symbolic constructions | Finite abelian group algebra and block matrices, multiplication/adjoints, regular-representation lowering, symbolic locality |
| Code families | Hypergraph and commutative lifted products, abelian bicycle codes, direct sums, toric and Bacon–Shor families; Steane, Shor and five-qubit fixtures |
| Transformations | Physical permutations, Hadamard duality, check reduction, chain maps, induced homology and quasi-isomorphism checks |
| Analysis | Logical frames, rank/rate/locality, known-location erasure tests, bounded exact CSS/stabilizer/subsystem distance; optional MILP |
| Decoding | Bounded minimum-weight and erasure decoders; sparse min-sum BP with optional order-zero reliability fallback; independent CSS sectors and perfect-syndrome simulation |
| General semantics | Optional dense subspace/subsystem isometries, numerical Knill–Laflamme checks, stabilizer-to-isometry lowering |
| Experiments | Versioned data-only JSON, CLI, seeded structured candidate generation and timing scripts |

`distance.status == "exact"` denotes an exact finite search result. A stopped
search reports a bound interval and a logical witness; it does not report an
unproved distance. Codes with no logical qubits use `no_logicals`. MILP uses
separate `solver_optimal` evidence because numerical optimization is not a
formal arithmetic proof. Read [API conventions](docs/api.md) before interpreting
results.

A finite check-weight profile does not prove that a family is qLDPC. The packed
backend is intended for small and moderate instances, and elimination can
become dense. Enumeration is exponential and bounded; decoding is a reference
baseline, not a scalable general qLDPC decoder. Circuit noise, thresholds,
noncommutative lifted products, qudits, and bosonic/dynamical execution remain
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
```

See [decoding methods](docs/decoding.md) for iterative decoder assumptions and
the distinction between syndrome convergence and logical success.

Candidate records include construction parameters, seed, code, algorithm
budgets, bound statuses and witnesses. These are candidates, not automatically
novel or inequivalent codes. Generated results are ignored by Git.

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
