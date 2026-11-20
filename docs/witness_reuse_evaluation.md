# Reusing a logical witness across generator changes

Run `python scripts/benchmark_reuse.py --repeats 5 > output/witness_reuse.jsonl`
after installing the library and creating `output/`. The comparison uses one
sector (X) and one fixed coordinate system. Each target reverses its original
row order and appends the sum of its first two generators in each sector.
The initial construction and search, creation of coefficient maps, and retrieval
of a previously found witness are distinct costs.

| Fixture | Direct target upper bound without enumeration | Target upper bound with bounded enumeration | Candidates tested in that enumeration | Independently checked reused upper bound | New search candidates for reuse |
|---|---:|---:|---:|---:|---:|
| Steane | 3 | 3 | 28 | 3 | 0 |
| Toric, side four | 4 | 4 | 5,488 | 4 | 0 |
| Selected eight-qubit CSS | 3 | 2 | 19 | 2 | 0 |

The eight-qubit example was chosen because a weight-two witness improves on the
initial algebraic representative. It illustrates transfer of earlier search
work. It does not estimate how often a random candidate benefits from reuse.
On the two standard fixtures the direct algebraic witness already gave the
same upper bound at zero enumerated candidates. Fresh target enumeration also
excludes lower weights. The reused evidence in this experiment establishes only
an upper bound on the target.

In a five-repeat local run the end-to-end retrieval, rebinding and inspection
path took roughly 13–18 milliseconds across these fixtures. Direct target
witness generation without enumeration took less than a millisecond in each
case. No general runtime advantage follows. The exposed differences are the
bound obtained, whether a known witness survives a change of presentation, and
the extra work required to establish it from scratch. Per-phase raw timings and
medians, candidate counts, source/runtime identity and limits are emitted by the
script. The map derivation phase is measured separately from reuse and can be
charged when a host has no preexisting map.
