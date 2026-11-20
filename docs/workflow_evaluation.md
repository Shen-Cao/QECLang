# Workflow evaluation: timing scopes and rejected corruption

`scripts/benchmark_workflow.py` evaluates the first typed workflow on small, reproducible CSS cases. It measures execution costs and exercises a specific relation failure. It does not claim a speedup, scalable distance computation, or a general security guarantee.

After installing the package, run:

```sh
mkdir -p output
python scripts/benchmark_workflow.py --repeats 7 --warmups 1 --order-seed 17 \
  > output/workflow-benchmark.jsonl
```

The script writes JSON Lines to stdout only: one timing record per selected case and one corruption record. `--cases` selects `steane-exact`, `steane-bounded`, or `toric2`; `--max-candidates` sets the shared execution candidate allowance. Each record includes the program, source/runtime identity, execution and inspection policies, benchmark-script hash, individual timing observations, and summaries. Generated results belong in ignored output directories.

## Comparison design

Every case uses the same nine-node transport DAG: literal → chain → CSS → distance/witness, followed by physical permutation, sparse/packed round-trip, and witness transport. The direct baseline manually calls those same kernels, including admission checks and mandatory witness/transport verification. It does not invoke the workflow executor or implement a different distance algorithm.

Before timing, the script requires direct and checked/unchecked workflow artifacts and requested outputs to agree. It also checks the reference record with the standalone inspector. Every measured return value is validated outside its timed interval. A source/runtime change during a case invalidates the experiment.

| Timed scope | Included work |
|---|---|
| `program_parse` | Bounded schema/type parsing of an already available Python declaration |
| `direct_kernels` | Fresh literal admission, the same conversions/search, witness production and mandatory transport; no workflow artifact/trace machinery |
| `direct_kernels_and_artifacts` | Direct kernels plus typed artifact descriptions, hashing and deduplication; no workflow trace/ledger or complete-record serialization |
| `program_unchecked` | Executor with optional conversion checks disabled; core validation and mandatory witness/transport checks remain enabled |
| `program_checked` | Executor with optional conversion checks enabled, including provenance, ledger, trace and artifact construction |
| `record_serialization` | Canonical JSON serialization of an existing checked record; no execution or file I/O |
| `strict_replay` | Caller-policy replay of the checked record, including source/runtime checks, reexecution and whole-record comparison |
| `standalone_inspection` | Bounded artifact import and independent relation/upper-witness checking of the existing record; no distance search |

These scopes are deliberately unequal. Kernel-only time is not a serialization or replay time. Strict replay and standalone inspection establish different properties: replay compares historical execution, while inspection derives only supported finite-instance relations and upper-bound claims. `checked` inspection status does not verify distance lower bounds, exactness, candidate counts, historical assurance, or the old ledger.

This is a warm, single-process microbenchmark. Imports and reference preparation occur before measurement even when `--warmups 0` is used. Additional warmup calls are discarded. Each measured repetition shuffles the eight phases using the recorded ordering seed, and every phase runs once per repetition. Garbage collection remains at its ordinary runtime setting. The output retains all observations and reports medians/minima/maxima; no timing threshold or relative-speed assertion is used in tests.

## Illustrative observed run

A seven-repetition run with one additional warmup and ordering seed 17 produced these medians in milliseconds. The measured package source hash was `9b6def816642ecef26122664fc0d9bfedb19f3067701567fe117008cacc64676`, matching package sources at commit `3928e52`, under CPython 3.12.12. Execution and inspection used their defaults. These observations describe one environment and are not performance targets.

| Scope | Steane exact | Steane bounded | Toric size 2 |
|---|---:|---:|---:|
| Program parsing | 0.271 | 0.359 | 0.283 |
| Direct kernels | 0.328 | 0.324 | 0.348 |
| Direct kernels + artifacts | 0.458 | 0.521 | 0.531 |
| Program, optional checks off | 3.642 | 5.367 | 4.608 |
| Program, optional checks on | 4.019 | 4.793 | 4.993 |
| Existing record serialization | 0.061 | 0.070 | 0.071 |
| Strict replay | 8.562 | 9.010 | 8.173 |
| Standalone inspection | 1.778 | 1.801 | 1.805 |

The checked median was below the unchecked median in one case, illustrating why these few noisy samples cannot isolate individual overheads by subtraction. Executor measurements also include package-source hashing and other bookkeeping absent from the direct baseline. The cases have only 7 or 8 physical qubits; they test this integration path, not large-qLDPC scaling.

The Steane exact case excludes weights through two with 28 candidates and reports distance three. The bounded case requests no weight enumeration and retains an interval. The size-two toric case excludes weight one with eight candidates and reports distance two. Candidate caps apply equally to the direct and workflow paths. An independently checked upper witness is available in each case; the inspector does not promote the producer's lower or exact-distance claim.

## Equal-parameter corruption fixture

The script injects a process-local faulty permutation producer under `unittest.mock`, restoring it afterward. No package file is changed. The declared map is the identity on four sites, but the producer returns:

| Presentation | X-check support | Z-check support | n | k |
|---|---|---|---:|---:|
| Source | `{0,1}` | `{2,3}` | 4 | 2 |
| Incorrect target | `{0,2}` | `{1,3}` | 4 | 2 |

Both are valid CSS presentations with the same row weights and scalar parameters. The target is incorrect because it does not implement the **declared identity coordinate map**. A different permutation could relate these presentations; that is not the program being checked.

The executor itself describes the incorrect value and computes its valid content digest, trace references, and ledger entries. Therefore hash consistency and scalar `(n,k)` checks cannot explain the rejection. The observed outcomes are:

- Checked execution stops with `verification_failed` at the optional conversion relation check.
- Execution with optional checks disabled completes and publishes the incorrect target with freshly computed digests.
- Standalone inspection of that completed record returns `invalid_record` because the source/target relationship fails.
- When the program additionally transports a valid source logical witness, execution stops with `verification_failed` at `css.transport` even with optional checks disabled.

This fixture distinguishes an optional conversion check, a separate scientific inspector, and the mandatory witness-transport gate. It demonstrates rejection of one concrete producer defect. It does not establish that every conceivable implementation defect or malicious process compromise will be detected.
