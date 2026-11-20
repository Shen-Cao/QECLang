# Bounded comparison with qldpc 0.4.1

The comparison completed on fourteen small positive binary CSS presentations.
All fourteen matched physical length, X and Z check ranks, and logical dimension. All twelve cases
with protected logicals matched both exact sector distances. The two `k=0`
cases have undefined sector distances because they encode no logical qubits.
Every selected case completed within the limits of the `--max-n 18` campaign.

The foreign calculations supply an independent check on these finite examples.
For the four independent family constructors, native XOR checks also verify
the claimed relations between the raw check matrices.

| Case | n | k | dX | dZ |
| --- | --- | --- | --- | --- |
| Bell, redundant/zero checks | 2 | 0 | undefined | undefined |
| Cyclic bicycle, order 3 | 6 | 2 | 2 | 2 |
| Four-site distance-two fixture | 4 | 2 | 2 | 2 |
| Disjoint-check distance-one foil | 4 | 2 | 1 | 1 |
| Empty presentation | 0 | 0 | undefined | undefined |
| Free two-qubit presentation | 2 | 2 | 1 | 1 |
| Open repetition HGP, lengths 2 and 3 | 8 | 1 | 3 | 2 |
| Three-site X-check repetition | 3 | 1 | 1 | 3 |
| Three-site Z-check repetition | 3 | 1 | 3 | 1 |
| Shor | 9 | 1 | 3 | 3 |
| Steane | 7 | 1 | 3 | 3 |
| Steane, redundant/zero checks | 7 | 1 | 3 | 3 |
| Toric, side 2 | 8 | 2 | 2 | 2 |
| Toric, side 3 | 18 | 2 | 3 | 3 |

The versioned native recipes are in `qeclang.catalog.references`. Their stored
parameters serve as expectations. The foreign constructor receives only the
two check matrices.

## Exact calculation protocol

For each case, `to_qldpc` constructs a **fresh generic `qldpc.codes.CSSCode` for
each sector** from the native Hx/Hz arrays. A round trip checks the exact retained
presentation, including zero rows, duplicates, empty dimensions and physical
column order. The worker checks generic class identity and verifies that the
requested sector has no cached distance before making the following call.

```python
foreign.get_distance_exact("X", cutoff=0, method="brute_force")
```

The Z sector uses another fresh instance. No distance, equal-distance promise,
logical frame, or family-specific formula is supplied. The package may recognize
identical X/Z matrices, but the fresh-sector policy prevents reuse of a previously
computed opposite-sector distance. In version 0.4.1,
`get_distance_exact` consults cached values first, and a cutoff of at least two
can permit an early upper-bound return. The default cutoff one remains exact
for valid nontrivial inputs. This campaign uses zero to disable early return
and explicitly selects brute force. Recorded source hashes identify the
installed implementation. [qLDPC library map](https://qldpc.readthedocs.io/library_map.html)
and [distance API](https://qldpc.readthedocs.io/autoapi/qldpc/codes/distance/index.html).

Native sector calculations use bounded weight-layer enumeration and must return
`exact` before comparison. For `k=0`, qldpc's known-distance query returns NaN.
JSON records normalize this to null and mark the distance comparison inapplicable.
QECLang's corresponding status is `no_logicals`.

## Independently constructed families

Four foreign family constructors were also called independently of the native
matrices. Their supplied or cached distances were not queried.

| Foreign constructor | Verified relation to native case |
| --- | --- |
| `SteaneCode()` | Same X/Z generator spaces in fixed physical coordinates, with a different generator ordering |
| `HGPCode(RepetitionCode(2), RepetitionCode(3), set_logicals=False)` | Identical retained presentation |
| `ToricCode(2, rotated=False)` | Identical retained presentation |
| `BBCode({x:3,y:1}, 1+x, 1+x**2*y)` | Identical retained presentation to the order-three cyclic bicycle |

Each relation records an explicit physical permutation and bidirectional row
maps checked by native XOR identities. Identity physical coordinates sufficed
for all four observed cases. The script additionally permits a declared cyclic
coordinate inversion for the bicycle convention. The two `n=4,k=2`
fixtures have different distances despite encoding the same number of logical
qubits on the same number of physical qubits.

## Limits, observations, and reproduction

Default limits are `n≤13`, at most 64 total check rows, a 1,000,000-element
check/commutation payload proxy, 100,000 native candidates per sector, and 65,536
foreign nontrivial logical/stabilizer combinations per sector. The full campaign
sets `--max-n 18` to include `toric_3`. The driver caps the setting at 20.
Before each foreign search, both sides check the enumeration bound
`(2**k - 1) * 2**rank(H_sector)`. Every foreign task runs in a child process with
a 60-second default timeout covering imports, conversion, and computation.
Total foreign workspace, JIT allocation, and OS memory fall outside these limits.
For an explicitly selected case that exceeds the budget, times out, fails, or
remains incomplete, the CLI emits the outcome and exits with a nonzero status.

The following commands reproduce the comparison in an isolated Python environment.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e . 'qldpc==0.4.1' pytest
mkdir -p output/qldpc-cache/numba output/qldpc-cache/mpl
export XDG_CACHE_HOME="$PWD/output/qldpc-cache"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export MPLCONFIGDIR="$XDG_CACHE_HOME/mpl"
export PYTHONPATH=src
.venv/bin/python scripts/compare_qldpc.py --max-n 18 > output/qldpc_comparison.jsonl
.venv/bin/python -m pytest -q tests/test_qldpc_comparison.py tests/test_qldpc_adapter.py
```

The recorded campaign used qldpc 0.4.1, NumPy 2.4.6, galois 0.4.11 and numba
0.65.1. The exact campaign driver is recoverable with
`git show 6b435a4:scripts/compare_qldpc.py`. Each output row includes its digest.
The JSONL contains fourteen matched-presentation observations and four
independent-family observations, with versions, source hashes, full policies,
raw native bounds, external values, and verified relation maps. Generated outputs
remain outside Git.

Elapsed times are diagnostic only. Each process can reuse disk compilation
caches, and its Z calculation follows an X calculation. The largest code
in this comparison has 18 physical qubits.
