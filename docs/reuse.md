# Reusing checked logical upper witnesses

An archived CSS logical witness can supply an upper bound for a different check presentation of the same positive CSS code. Reuse requires explicit generator maps in both directions and keeps the physical qubit coordinates fixed. It carries a finite-instance upper witness; it does not import the producer's lower bound or distance-optimality claim.

## Assemble, execute, and inspect

```python
from qeclang.archive import make_reuse_program
from qeclang.workflow import ExecutionLimits, execute, inspect_run

program = make_reuse_program(store, bundle_id, "logical", target, maps)
run = execute(program, limits=ExecutionLimits(max_candidates=0, max_basis_bits=0))
inspection = inspect_run(program, run)
assert inspection["status"] == "checked"
```

Here `store` is an evidence reader, `bundle_id` identifies a previously stored bundle, and `"logical"` is a completed witness node in that bundle's program. `target` must be a `CSSCode`; serialized code records can be loaded explicitly with `qeclang.io.from_dict`. The four-map format is defined below. A complete runnable example is [reuse_upper.py](../examples/reuse_upper.py).

`make_reuse_program(store, key, witness_node, target, maps, *, allow_prefix=False, limits=ArchiveLimits())` first calls [archive retrieval](archive.md), including a fresh independent scientific inspection under the reader's limits. It selects one instance-scoped `distance_upper_bound` claim, resolves the witness's nominal source node, and imports the completed source and certificate artifacts. It checks the source fingerprint, sector, bound, and certificate again. The source need not be the first literal or a requested output of the old program.

The returned data-only program has four nodes:

| Node | Operation | Purpose |
|---|---|---|
| `source` | `css.literal` | Embed the checked source presentation |
| `imported_witness` | `css.import_witness` | Independently check its embedded certificate |
| `target` | `css.literal` | Embed the requested target presentation |
| `rebased_witness` | `css.rebase_witness` | Check the supplied generator maps and rebind the witness |

Its outputs are `target` and `rebased_witness`. Assembly validates map shapes and index syntax. **The mathematical map equations are checked when the returned program is executed or its run is inspected.** Returning a program is therefore not an equivalence verdict. Import and rebasing are mandatory checks even when `execute(..., verify_relations=False)` disables optional relation checks.

No distance operation occurs in this program. Assembly, execution, and inspection do not discover generator maps, compute a logical basis, or rerun the archived search. Execution and inspection use the embedded source and certificate without further archive reads. Creating or running the program makes no storage write; publishing the resulting run is a separate explicit action.

## Bidirectional generator maps

Write the source and target check matrices as \(H_X^s,H_Z^s\) and \(H_X^t,H_Z^t\). For each sector, supply binary matrices satisfying

\[
H_X^t=A_XH_X^s,\qquad H_X^s=B_XH_X^t,
\qquad
H_Z^t=A_ZH_Z^s,\qquad H_Z^s=B_ZH_Z^t.
\]

The `maps` dictionary has exactly these keys:

| Key | Matrix | Number of support rows | Each row indexes |
|---|---|---|---|
| `x_forward` | \(A_X\) | Target X checks | Source X checks |
| `x_reverse` | \(B_X\) | Source X checks | Target X checks |
| `z_forward` | \(A_Z\) | Target Z checks | Source Z checks |
| `z_reverse` | \(B_Z\) | Source Z checks | Target Z checks |

Every map is a list of lists of strictly increasing, distinct, zero-based integers. Each inner list describes a GF(2) sum of generators; `[]` denotes the zero row. For example:

```python
from qeclang.algebra import BinaryMatrix as B
from qeclang.codes import CSSCode

source = CSSCode(B((3, 12), 4), B((15,), 4))
target = CSSCode(B((12, 3, 15, 0), 4), B((15, 15, 0), 4))
maps = {
    "x_forward": [[1], [0], [0, 1], []],
    "x_reverse": [[1], [0]],
    "z_forward": [[0], [0], []],
    "z_reverse": [[0]],
}
```

The packed integers above encode physical supports: `3` is `{0,1}`, `12` is `{2,3}`, and `15` is `{0,1,2,3}`. The target reorders, duplicates, and combines generators while preserving the source row spaces. The checks may have different counts and dependencies; the maps need not be invertible on arbitrary syndrome strings. On syndromes produced by physical errors, forward and reverse maps reproduce the respective valid syndrome coordinates.

The reader verifies these equations with XORs rather than elimination. An optional host-side producer, `qeclang.workflow.row_equivalence.row_maps(source, target, ...)`, can discover maps using exact linear solves. That computation has separate guards and cost; it is not performed implicitly by reuse. The runnable example invokes this producer explicitly.

## What transfers

Two-way row-space inclusion preserves the positive stabilizer subspace, each sector's logical quotient, the physical Hamming metric, and the sector distances. A valid source logical representative remains the same physical operator at the target. Its separating dual still certifies nonmembership, while its centralizer condition is preserved. Rebasing updates the certificate's exact presentation fingerprint and verifies it against the target.

This is stronger than matching scalar values such as \(n,k\), and narrower than arbitrary code equivalence. It does not change physical coordinates, exchange X and Z sectors, or treat signed stabilizers. Check weights, qubit incidence, Tanner graphs, measurement costs, and decoder performance can change under generator replacement. Equality of codes does not imply equality of these presentation-dependent properties.

The transferred claim is only \(d_X\leq w\) or \(d_Z\leq w\) for the verified operator weight \(w\). A producer may have reported that this bound is optimal; reuse does not independently certify that report. Identity maps can also reuse a witness for an unchanged presentation. Decorative code names are omitted from the canonical embedded artifact payloads.

## Reader policy and provenance

A historical publication receipt is not an input to this API and cannot override current inspection. A missing witness, an unexecuted node, a distance report without a checked witness, or a certificate with an invalid separating dual is refused. Recomputing content hashes does not make an invalid certificate pass the scientific check.

`allow_prefix=True` permits extraction from an independently checked completed prefix. It never makes an unexecuted suffix reusable or verifies a producer's claimed reason for stopping. Prefix acceptance must be explicitly enabled again by the reader.

The imported witness records `origin={"bundle_id": key, "node": witness_node}`. This is a provenance hint, not an author signature or authentication mechanism. A self-contained program checks its embedded mathematics without consulting that hint. Use a trusted separately supplied archive identity when historical origin matters.

`ArchiveLimits` governs retrieval, parsing, and scientific inspection. Extraction separately imports its source and witness under bounded artifact limits; target and map admission use the reader/program dimensions and JSON limits. Execution and independent inspection then have their own witness, transport, and `max_row_map_work` budgets. These stage-specific budgets do not jointly bound peak memory or wall-clock time. A source bundle accepted earlier may be refused by a stricter current reader.

## Command-line reuse

```sh
qeclang archive reuse output/evidence.sqlite BUNDLE_ID logical \
  target.json maps.json > output/reuse-program.json
qeclang run output/reuse-program.json > output/reuse-run.json
qeclang run output/reuse-program.json --inspect output/reuse-run.json
```

Replace `BUNDLE_ID` and `logical` with the saved bundle key and witness node. `target.json` uses the positive CSS `qeclang.code.v1` format; `maps.json` contains the four arrays above. `reuse` opens the archive read-only and emits a program, not a new archive entry. It supports `--allow-prefix`, `--max-bundle-bytes`, and `--max-reader-steps` for the fresh archive inspection.

The [reuse evaluation](witness_reuse_evaluation.md) compares witness recomputation, bounded distance search, explicit map production, and archive-read/rebind/inspection as separate measured scopes. Avoid interpreting saved search work as a universal speedup: small instances can cost less to recompute than to import and independently check.
