# Typed workflows, artifacts, and replay

`qeclang.program.v1` describes a closed, data-only DAG of QEC operations. The first dialect connects a CSS literal to based-chain and sparse representations, physical permutations, bounded distance analysis, logical upper-bound witnesses, and explicit witness transport. Its interpreter calls the existing reference kernels and records the relation and evidence checks around them.

```sh
mkdir -p output
qeclang run examples/programs/transport.json > output/transport-run.json
qeclang run examples/programs/transport.json --replay output/transport-run.json
```

The first command emits `qeclang.run.v1`; the second emits `qeclang.replay.v1` with `matches`. Exit status zero means execution completed, or replay matched, respectively. A refusal or mismatch uses exit status two. An execution can complete with a bounded distance interval: completion does not imply an exact distance or that every optional check ran.

## Program contract

A program has exactly `schema`, `nodes`, and `outputs`. Each node has `id`, `op`, `inputs`, and `params`. References must name earlier nodes, identifiers must be unique, and requested outputs must exist. Parsing validates the operation signatures and parameter shapes without constructing codes or running analysis.

| Operation | Input → output | Parameters and meaning |
|---|---|---|
| `lift.literal` | none → `lift` | `construction`: canonical finite abelian lifted-product seeds |
| `lift.parameters` | `lift` → `parameters` | `backend`, `max_steps`, `record_operations`: symbolic dimension analysis of this instance |
| `lift.css` | `lift` → `css` | Empty parameters; explicit, guarded binary expansion |
| `css.literal` | none → `css` | `code`: existing `qeclang.code.v1` CSS support lists |
| `css.chain` | `css` → `chain` | Empty parameters; `d1=Hx`, `d2=Hzᵀ` |
| `chain.css` | `chain` → `css` | Empty parameters; inverse based representation |
| `css.sparse` | `css` → `sparse_css` | Empty parameters; preserve ordered check supports |
| `sparse.css` | `sparse_css` → `css` | Empty parameters; guarded packed materialization |
| `css.permute` | `css` → `css` | `permutation`: old site `i` maps to `permutation[i]` |
| `css.distance` | `css` → `distance` | `sector`, `max_weight`, `max_candidates` |
| `css.witness` | `css`, `distance` → `witness` | Empty parameters; verify the supplied upper witness |
| `css.transport` | source `css`, target `css`, `witness` → `witness` | `permutation`: verify the relationship and relabel both operator and dual |

Distance and witness nodes have nominal source bindings. Even identical CSS data reached through a different node is a different nominal source; reuse across nodes needs explicit transport, including an identity permutation when appropriate. Runtime checks also bind evidence to the exact mathematical presentation. Thus deduplicating equal artifacts does not erase the program's source bindings.

`Program.from_dict`, `Program.loads`, and `Program.load` enforce `ProgramLimits`: input/canonical bytes, JSON nesting and item counts, integer/string lengths, node and dependency depth, reference/fan-out counts, and literal dimensions. Unknown operations, unknown fields, forward references, cycles, duplicate JSON keys, and nonfinite numbers are rejected. No expression evaluation, dynamic imports, arbitrary callbacks, or plugin dispatch is part of this dialect. Literal commutation and mathematical conversion relations are checked at execution, not inferred from parsing success.

## Identities and artifact payloads

`workflow.artifacts.describe(kind, value)` returns `kind`, `digest`, and `payload`. The digest is SHA-256 of UTF-8 canonical JSON for

```text
{"schema":"qeclang.workflow.artifact.v1","kind":kind,"payload":payload}
```

Canonical JSON sorts object keys, uses compact separators and ASCII escapes, and rejects nonfinite numbers. Payload schemas are included in that hash. Code, chain, and sparse-code names are omitted from the payload entirely; equal content digests therefore cannot retain conflicting decorative labels. Check order, redundant/zero rows, physical coordinate order, X/Z labels, and empty dimensions remain significant.

| Artifact kind | Payload |
|---|---|
| `css` | `qeclang.code.v1` CSS support lists without `name` |
| `sparse_css` | The same support-list schema, produced directly from sparse rows without packed lowering; its artifact kind gives it a different digest |
| `chain` | `qeclang.workflow.chain.v1`: degree order `[0,1,2]`, dimensions, column-vector action, and shaped `d1`/`d2` support rows |
| `distance` | `qeclang.workflow.distance.v1`: source presentation fingerprint, CSS sector, and every `DistanceResult` field |
| `witness` | Existing `qeclang.certificate.v1` logical-certificate record |

Distance payloads preserve `status`, lower/upper bounds, witness, completed weight layers, and candidate count. Their digests identify that entire result record, including search evidence. `no_logicals` carries no finite distance. The serializer checks record consistency but cannot prove that the claimed search was performed or that a witness pairs correctly with the source checks. Independent verification is recorded by the executor, not earned by hashing a record.

Four identities serve different purposes: node identifiers express nominal dependencies; the program fingerprint binds the canonical declaration; artifact digests bind typed payloads; and certificate/distance `source` fields use the existing presentation fingerprint. These hashes are not interchangeable, code-equivalence classifiers, proof certificates, or authenticity signatures. Program names/identifiers remain part of program identity even when a decorative code name is excluded from artifact identity.

## Completion, claims, and assurance

A run retains a `trace`, an `artifacts` map keyed by digest, and an `outputs` map from available requested node IDs to digests. Equal artifacts are stored once; every executed node still has its own trace entry. Failed or unavailable outputs are omitted. Each trace step records its operation, input digests, execution status, output digest when available, evidence, and counter consumption.

| Dimension | Interpretation |
|---|---|
| Execution status | `completed`, `budget_exhausted`, `verification_failed`, `evidence_unavailable`, or `invalid_input` |
| Mathematical result | Exact sector distance, a bounded interval, no logicals, or a particular upper-bound witness |
| Assurance | `core_validated`, `producer_computed`, `relation_checked`, `independently_checked`, `not_requested`, or optional-check `budget_exhausted` |

Distance results are `producer_computed`, including exact results from exhaustive layer exclusion. A separately constructed witness can earn `independently_checked` assurance for its instance-specific **upper bound**. It does not independently certify the distance lower bound or a family theorem. A zero-logical result is valid distance-analysis output, but asking it for a finite logical witness returns `evidence_unavailable`.

Optional conversion checks compare source and target data without invoking the producing conversion or recomputing rank/distance. Exact boundary/support identities or a checked coordinate bijection imply the listed physical-basis, syndrome, quotient, and Hamming-metric guarantees. These are exact Python relational checks, not Lean-verified implementation proofs. Physical permutation preserves Hamming weight; no hardware embedding or geometric-locality guarantee is added.

`verify_relations=False` skips those optional checks. Exhausting their budget also leaves a usable value with explicit unverified assurance. In either case, source/target admission checks and mandatory witness verification remain in force. Witness transport always verifies the permutation relationship, the original certificate, and the resulting target certificate. It transports an upper witness only.

## Execution budgets

`ExecutionLimits` separates cumulative counters from per-operation size ceilings:

- `max_candidates` is shared across distance nodes; each node also has its own candidate request. A zero remaining candidate allowance can still produce an algebraic upper witness and a conservative interval.
- `max_optional_checks` counts optional conversion relation checks. `max_transports` separately bounds mandatory transport calls, including when optional checks are disabled.
- `max_validation_work` meters the sparse validator's incidence-work proxy.
- `max_artifact_bytes` limits retained canonical artifact bytes, charging identical digests only once. Serialization occurs before this charge; the limit is not a preallocation, trace-size, or peak-memory bound.
- `max_axis`, `max_matrix_bits`, and `max_basis_bits` are individual operation ceilings. `ledger.peak_requested` records the largest attempted values, including refused attempts.

The ledger records the caller's limits and consumed counts. Check/transport counts do not count their internal arithmetic instructions, and candidate limits exclude basis construction. These controls are neither a wall-clock deadline nor a complete process-memory guarantee. A hard refusal stops later nodes while preserving earlier published artifacts and available requested outputs.

## Replay contract

```python
from qeclang.workflow import ExecutionLimits, Program, execute, replay

program = Program.load("examples/programs/transport.json")
limits = ExecutionLimits(max_candidates=28)
record = execute(program, limits=limits)
assert replay(program, record, limits=limits)
```

Replay reruns the program under caller-supplied execution limits and the caller's optional-check policy. It never adopts policy from the supplied record. A match requires the same program fingerprint, execution policy, package-source hash, Python implementation/version, and the entire canonical run record: trace, evidence, budgets, payloads, digests, and available outputs. It can match a recorded refusal or bounded result as well as a successful exact computation.

Supply the original policy explicitly when it differs from defaults, for example matching `--max-candidates 28` or `--no-relation-checks` on both CLI calls. `--max-record-bytes` bounds replay-record file input. Duplicate fields and invalid numeric/JSON data are rejected. Source hashing uses package-relative filenames and file contents, without machine-specific paths.

A replay match establishes strict reproducibility of this run under the recorded implementation. It does not establish that the implementation is mathematically correct, remove a producer's trust assumptions, upgrade an interval into an exact distance, or certify unimplemented workflows. The standalone Lean witness checker is a separate facility; this interpreter does not invoke it implicitly.

## Independent inspection of saved evidence

```sh
qeclang run examples/programs/transport.json --inspect output/transport-run.json
```

`--replay` runs the producer again and compares the complete record. `--inspect`
reads a completed record and checks the predicates it can establish independently.
It checks the literal against the caller's program, validates canonical artifact
payloads and digests, follows each input and output binding, and checks each
conversion's physical-coordinate relationship. For a logical witness it checks
syndrome, dual pairing, weight, sector and source presentation. For a transported
witness it also checks the image of both operator and dual under the specified
site permutation. No distance enumeration, rank elimination, solver or Lean
process is required. The verifier's trusted base includes the Python arithmetic
and admission routines and the supplied program.

An inspection of the Steane example returns two claims of X distance at most
three, one for each presentation. The producer's exact-distance report and the
completeness of the searched weight layers remain outside that verdict. Inspecting
a run from another runtime can verify those same mathematical predicates. This
says nothing about the origin of the file or how much work the producer performed.
The result lists the reported distance nodes separately from independently checked
claims. A distance-only or empty program can therefore yield no verified numerical
claims. The source and evidence digests in each claim allow a reader to select
precisely the object to which the result applies.

Inspection uses the reader's JSON, artifact, matrix-size, incidence-work, relation,
and certificate-count limits. Each unique artifact is imported once. A malformed
binding, a falsified certificate, or a refused check yields no scientific claims
from the run. The default mode requires a completed run. An explicit prefix policy can
verify earlier completed nodes. Strict replay remains available for recorded refusals. The CLI flag
`--no-relation-checks` controls producer execution/replay only; inspection always
checks the relevant relationships. A successful inspection gives no assurance
about producer-reported budgets, historical checks, or code equivalence beyond
the explicit relations in that program.

## Lifted-product programs

`lift.literal` admits a canonical `qeclang.construction.v1` seed record. Its
static type includes the unexpanded blocklength. `lift.parameters` takes this
value and an explicit backend, `max_steps`, and `record_operations`. The `crt`,
`polynomial`, `abelian`, and `mixed-abelian` backends retain their respective
coefficient-domain requirements. An unsupported or exhausted rank analysis
returns its status with no logical count. Candidate enumeration and symbolic
arithmetic use separate additive counters. Symbolic preflight, source hashing and
artifact serialization fall outside the arithmetic counter. The charge on a
refusal is the arithmetic work already accepted by the kernel's meter.

Only `lift.css` requests binary expansion. It estimates shape and payload before
allocating the packed representation. When enabled and within the reader's or
executor's limits, its relation checker reconstructs each expected physical
incidence directly from the two seed translations, including inverse shifts in
the adjoint blocks. Equal rank and blocklength alone would miss a misplaced shift.
The extra check has its own `lift_comparison_units` proxy.

```sh
mkdir -p output
qeclang run examples/programs/cyclic_rank.json > output/cyclic-run.json
qeclang run examples/programs/cyclic_rank.json --inspect output/cyclic-run.json
```

The example uses a shared cyclic group of order six, which has repeated roots
over GF(2). Its twelve physical qubits have X and Z check ranks five each, hence
two logical qubits. For `backend="polynomial"` with `record_operations=true`, the
scientific inspector can replay the supplied elementary operations on both
symbolic checks and verify their diagonal gcd rank contributions without binary
lowering or rank production. The checked claim applies to this finite construction.
An untraced rank computation can still finish exactly under the producer's
semantics. Inspection lists that result under `unverified_parameter_nodes`.
CRT/character component rank metadata alone is insufficient for independent
rank certification in this dialect. No automatic backend fallback occurs.

Rank replay shares the source seed interpretation, symbolic check construction
and polynomial arithmetic with the implementation. The verifier independently
checks the elementary operations, their resulting diagonals, and the gcd/rank
identities. An explicit `lift.css` edge exercises a separate direct physical
incidence check. Neither verification step establishes a distance or an
asymptotic family claim.

## Inspecting a completed prefix

A completed node can establish a claim even when a later node refuses to
allocate its output. Prefix inspection is opt-in. For example, this policy
admits the symbolic rank proof and refuses the binary expansion:

```sh
mkdir -p output
# Expected exit status 2. Keep the JSON refusal record.
qeclang run examples/programs/cyclic_rank.json --max-matrix-bits 1 > output/prefix.json
qeclang run examples/programs/cyclic_rank.json --inspect output/prefix.json --allow-prefix
```

`checked_prefix` identifies the checked nodes and reports the terminal operation
under `unverified_stop`. The verifier checks the prefix against the supplied
program and rejects spliced steps, extraneous artifacts and unavailable outputs.
It cannot establish why the producer stopped or how much work the producer spent.
The rank-evidence claim describes the finite lift even though the packed output
was never published. Without `--allow-prefix`, an incomplete run is unsupported
for scientific inspection.
