# Comparing a candidate with fixed-coordinate references

`ReferenceIndex` builds a bounded in-memory index of positive binary CSS check
presentations. `default_index()` uses the fixed [reference recipes](reference_cases.md).
`qeclang catalog list` prints their expected parameters and provenance. The
expectations describe each recipe and acquire no new assurance from a lookup.
`qeclang catalog show steane` emits a support record that can also be used as a
workflow literal.

```sh
mkdir -p output
qeclang catalog show steane_redundant > output/query.json
qeclang catalog lookup output/query.json
```

The snapshot digest binds sorted reference IDs to the mathematical presentations
used for this comparison. It excludes display names, citation text, and measured
performance. The row-space screen computes exact binary reduced bases in both
sectors. The screen key includes `n` and binds each basis to its sector and
physical column order. Two positive CSS stabilizers on the same labeled sites
are equal exactly when their X and Z row spaces agree. A hit on the screening
digest triggers a separate check before the result reports a match.

`identical_ordered_checks` means every retained support, including redundancy
and row order, agrees. `verified_generator_spaces` carries explicit coefficient
maps in both directions and both sectors; these can be independently checked
with `verify_row_equivalence`. A supplied permutation maps query-site indices
to comparison-site indices before these checks. The result retains the original
query fingerprint, comparison fingerprint, permutation, and reference fingerprint.
This lookup performs no search for an unknown permutation or local Clifford.
A generator-space match allows transfer of a mathematical distance claim under
the appropriate checked premises. Tanner graphs, measurement outcomes in a
given row order, schedules, and decoder performance can change.

The `parameter_matches_only` list collects other indexed presentations with
the same computed `n,k` but different row-space screen keys. For example,
`css_4_2_2` and `disjoint_checks_4_2_1` both encode two qubits on four sites
and have different distances. Parameter agreement could also occur between
presentations related by an untested physical relabeling.

`no_match_in_scope` refers to the specified coordinates and catalog snapshot.
`unresolved` reports a screen hit whose actual mapping could not be established
under the map budgets. If one hit was verified and another remained unresolved,
`status` is `matched` and `complete_in_scope` is false. The interface makes no
novelty determination. Axis, payload, candidate and map limits constrain local
calculations, with elapsed time and Python object overhead outside those counts.
