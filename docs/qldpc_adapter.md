# Pinned qLDPC CSS interchange

`qeclang.interop.qldpc` exchanges raw positive binary CSS checks with
[`qldpc==0.4.1`](https://pypi.org/project/qldpc/0.4.1/). Importing this adapter
module does not import qLDPC, NumPy, or Galois. Actual conversions require the
pinned distribution and reject another version rather than selecting a fallback.

```python
from qeclang.constructions import steane
from qeclang.interop.qldpc import to_qldpc, from_qldpc

source = steane()
foreign = to_qldpc(source, max_axis=4096, max_elements=10_000_000)
restored = from_qldpc(foreign, max_axis=4096, max_elements=10_000_000)
assert restored.hx == source.hx
assert restored.hz == source.hz
```

The tested boundary is the 0.4.1 `qldpc.codes.CSSCode` constructor and its
`field`, `matrix_x`, `matrix_z`, and `is_subsystem_code` interfaces. Both packages
interpret rows of `Hx` as X checks measuring Z errors, and rows of `Hz` as Z
checks measuring X errors. Import requires the actual Galois GF(2) array class
with an integer buffer, not merely values that happen to look binary in another
field. Every entry is validated before packing and local commutation is checked
independently. The [upstream API documentation](https://qldpc.readthedocs.io/autoapi/index.html)
describes the broader foreign representation.

## Preserved data

- Physical columns retain their numeric site indices.
- X/Z sector labels and raw generator row order are unchanged.
- Redundant generators and zero rows are retained.
- Explicit zero-row and zero-column matrix dimensions survive round trips.
- Pure X/Z generators use positive eigenspaces; there is no sign reinterpretation.

`to_qldpc` always creates a fresh generic external CSS object. It passes
`promise_equal_distance_xz=False`, copies no distance or dimension hints, and
does not propagate logical bases. qLDPC may recognize identical X/Z arrays on its
own. `from_qldpc` ignores stored distances, dimensions, equality promises and
logical-operator caches. These metadata cannot establish a claim in QECLang.
Imported names are empty; names are decorative and not part of this relation.

Signed/general Pauli stabilizer inputs, non-CSS objects, nonbinary qudits and
subsystem/gauge presentations are rejected. Recognizable sign/phase extensions
on external CSS instances are also rejected. No adapter can infer arbitrary
private semantics attached to an otherwise unsigned Python object.

External constructors may attach layout information or a suggested
`bias_tailoring_qubits` set for an optional later conjugation. The adapter reads
the raw check arrays and does not apply that suggestion. It transports neither
geometric labels nor an encoding circuit, extraction schedule, decoder, noise
model or logical frame. A physically different conjugated object must expose
its actual checks; a non-CSS result is outside this adapter.

## Limits and dependencies

`max_axis` bounds each physical/check dimension. `max_elements` bounds
`(mX+mZ)*n + mX*mZ`, a conservative array/commutation payload proxy. Export checks
these bounds before optional imports and dense allocation. Import checks shapes
before copying buffers or invoking foreign subsystem inference. It retains
integer dtype until binary validation, avoiding lossy casts.

Foreign property getters can themselves allocate before returning shapes. These
objects are executable Python, so the adapter cannot sandbox their getters.
Existing foreign storage, dtype byte widths, temporary arrays, package imports,
JIT compilation, backend caches, and elapsed time are outside these element
limits. Callers should choose an isolated environment and configure dependency
cache directories as appropriate; repository code has no machine-specific paths.

Install the optional package in a separate environment when needed:

```sh
python -m venv tmp/qldpc-env
tmp/qldpc-env/bin/python -m pip install -e . 'qldpc==0.4.1' pytest
tmp/qldpc-env/bin/python -m pytest tests/test_qldpc_adapter.py
```

The real-backend tests cover generic matrices, redundant and empty checks,
Steane/HGP/toric/C4 constructors, qudit and subsystem rejection, invalid buffers,
untrusted distance hints, and conversion limits. Mathematical parameter
comparisons remain a separate experiment; interchange alone proves no distance,
performance advantage or novelty.
