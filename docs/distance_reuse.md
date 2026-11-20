# Reusing checked distance intervals

A checked finite distance interval can be reused when an explicit relation preserves the physical Hamming metric and the relevant CSS sector. The reuse helper reconstructs the archived interval from its actual exclusion and upper-witness evidence, then produces a new program that checks the relation to the target. It supports both bounded intervals and exact distances; equality of scalar code parameters alone is insufficient.

## API and example

```python
from qeclang.archive import InMemoryEvidenceStore, publish, make_distance_reuse_program
from qeclang.constructions import steane
from qeclang.transforms import permute_qubits
from qeclang.workflow import ExecutionLimits, Program, execute, inspect_run

source_program = Program.load("examples/programs/exact_steane.json")
source_run = execute(source_program)
store = InMemoryEvidenceStore()
key = publish(source_program, source_run, store)["bundle_id"]

permutation = [1, 2, 3, 4, 5, 6, 0]  # Old physical site i maps to permutation[i].
target = permute_qubits(steane(), permutation)
program = make_distance_reuse_program(
    store, key, "distance", target,
    {"relation": "permutation", "permutation": permutation},
)
run = execute(program, limits=ExecutionLimits(
    max_candidates=0, max_exclusion_steps=0, max_basis_bits=0,
))
inspection = inspect_run(program, run)
assert inspection["status"] == "checked"
claim = next(c for c in inspection["claims"] if c["node"] == "transported_distance")
assert claim["sector"] == "pauli" and claim["value"] == 3
```

`make_distance_reuse_program(store, key, distance_node, target, relation, *, allow_prefix=False, limits=ArchiveLimits())` accepts a packed positive `CSSCode` target. It returns a bounded data-only `Program` with outputs `target` and `transported_distance`. JSON code records must first be loaded explicitly with `qeclang.io.from_dict`.

The selected archive node must be a completed, independently checked native `css.combine_distance` sector claim or `css.full_distance` claim with native sector-combination parents. For the example above, selecting `"x_distance"` or `"z_distance"` reuses only that sector. Selecting `"distance"` reconstructs both sector intervals and their full Pauli-distance join.

Assembly admits relation syntax, dimensions, and proof bindings. **The actual mathematical relation to the target is checked by mandatory execution or inspection of the returned program.** Successfully assembling the program does not establish that the proposed target relation is true. Setting `verify_relations=False` disables optional checks and leaves these transport obligations mandatory.

## Reconstructing the evidence

The helper calls [archive retrieval](archive.md) with the caller's current policy. This checks the stored bytes, program, completed trace, and scientific evidence afresh. It resolves the selected node's exact nominal source and premise edges. The selected distance claim and every lower/upper premise must have a matching independently checked claim whose evidence digest names the completed artifact.

The new program embeds:

1. Source and target CSS literals.
2. For each selected sector, the actual archived exclusion record through `css.import_exclusion` and actual logical certificate through `css.import_witness`.
3. A `css.combine_distance` node for each reconstructed sector interval, and `css.full_distance` when both sectors were selected.
4. A final `css.transport_distance` edge from the checked source interval to the target.

The exclusion memberships and upper certificates stay bound to their original source presentation. They are not relabeled to claim that old generator indices describe the target. The checked relation carries the resulting interval to the target instead. This matters when generator order, dependencies, or check counts change.

`css.import_exclusion` accepts only completed `excluded` evidence and independently verifies its obligation completeness and generator identities. A producer refusal or a `found` upper witness cannot serve as a lower premise. Logical upper certificates separately establish that a finite nontrivial logical exists. See the [low-weight exclusion proof](low_weight.md) for the distinction from a vacuous exclusion at \(k=0\).

No distance search, exclusion producer, certificate producer, elimination, or map discovery occurs in reuse. The example can run with both candidate enumeration and exclusion-production budgets set to zero. Verification still costs work: retrieval inspects the archive, execution checks the embedded premises and relation, and independent inspection checks the new run again. The helper never writes storage. Publishing the new run requires a separate explicit `publish` call.

## Supported metric relations

### Physical permutation

```json
{"relation": "permutation", "permutation": [1, 2, 3, 4, 5, 6, 0]}
```

The list is a bijection of the physical sites, mapping old site \(i\) to its listed destination. The target must equal the declared permutation of both source check matrices, retaining generator order. Permutations preserve centralizer conditions, stabilizer membership, and physical Hamming weight. The X, Z, or full Pauli interval therefore keeps its endpoints and exact/bounded status.

### Equal generator spaces

```python
relation = {
    "relation": "generator_spaces",
    "maps": {
        "x_forward": x_forward,
        "x_reverse": x_reverse,
        "z_forward": z_forward,
        "z_reverse": z_reverse,
    },
}
```

The four explicit support-index arrays certify, separately for X and Z,

\[
H^t=A H^s,\qquad H^s=B H^t.
\]

Forward rows index source generators; reverse rows index target generators. Lists must contain distinct increasing zero-based indices. The full map format and a redundant-check example appear in [upper-witness reuse](reuse.md). The reader checks XOR identities without finding maps or eliminating matrices.

Equal row spaces preserve the positive CSS code on fixed physical coordinates, so both sector distances and their minimum are unchanged. The maps need not be inverse on arbitrary syndrome strings when checks are redundant. Generator weights, Tanner graphs, decoder behavior, and measurement costs may change. This relation does not certify that a new check presentation remains uniformly sparse as a family.

Both relation forms preserve sector labels. X/Z exchange, general chain maps, quasi-isomorphisms, arbitrary changes of physical basis, and circuit-performance equivalence are outside this transport contract. Physical distance preservation alone is not a fault-tolerance theorem.

## Scope, origins, and limits

The selected source must have a supported native proof join. A raw `css.distance` report is not reusable as an independently checked interval, even when its producer labels it exact. An archive node whose interval was itself obtained by `css.transport_distance` is explicitly refused by this helper, including transported sector parents of a full join. It does not recursively reconstruct arbitrary transport ancestry. Native joins with imported exclusion and upper-witness ancestors are supported.

Lower and upper endpoints are reconstructed from the checked premises, not copied from an assurance label. An interval such as \([3,7]\) remains bounded after transport; it does not become exact. Full Pauli intervals use the minimum of the matching X/Z lower endpoints and the minimum of their upper endpoints.

Each embedded premise carries an optional `origin` hint containing the bundle key and its actual archive node. These fields are data-only, syntax-checked provenance metadata, not signatures or verification shortcuts. Executing the new program requires no archive access and checks the embedded mathematics independently of those hints. Archive hashes establish content identity, not authorship, code novelty, or equivalence classes.

`allow_prefix=True` permits extraction from an independently checked completed prefix. The selected join and all its premises must be completed; an unexecuted suffix is unavailable. Historical stop causes and reported expenditure remain unverified. The reader must opt in explicitly even if the bundle was originally published with prefix acceptance.

Retrieval uses the supplied archive JSON, program, and inspection policies. Extraction then imports the source and actual evidence under a shared bounded artifact budget; target and relation admission use the caller's dimension and JSON limits. The assembled program must fit the same program-admission policy. Execution and independent inspection have their own `max_exclusion_verify_steps` and `max_interval_transport_work` budgets. Execution also separates `max_exclusion_bits` for proof workspace from `max_basis_bits` for production. These controls apply at separate stages, not as one total allocation or wall-clock bound. A stricter current reader can refuse a previously published bundle, including because other checked portions of that bundle exceed its policy.

## Command line

```sh
qeclang archive reuse-distance output/evidence.sqlite BUNDLE_ID distance \
  target.json relation.json > output/distance-reuse-program.json
qeclang run output/distance-reuse-program.json > output/distance-reuse-run.json
qeclang run output/distance-reuse-program.json --inspect output/distance-reuse-run.json
```

`target.json` uses `qeclang.code.v1` with `kind="css"`. `relation.json` contains one of the two relation forms above. The archive is opened read-only and the command prints a program, not an archive entry. Reader options include `--allow-prefix`, `--max-bundle-bytes`, `--max-reader-exclusion-steps` for low-weight evidence, and `--max-reader-steps` for symbolic rank-proof verification during archive inspection.
