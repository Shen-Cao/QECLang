# Encoding channels with an explicit gauge preparation

`IsometryCode` defines an embedding of logical⊗gauge into the physical space. It does not choose a state for the gauge subsystem. `prepare_encoding` makes that choice explicit and constructs a finite CPTP channel from logical inputs to physical outputs.

For an isometry `V` with column index `logical_index * G + gauge_index`, and a gauge density matrix `σ`, the encoding channel is

```text
Enc(ρ) = V (ρ ⊗ σ) V†.
```

If `σ = Σ_s λ_s |g_s><g_s|`, its Kraus operators are `E_s = √λ_s V (I_K ⊗ |g_s>)`. The implementation contracts the gauge coordinate directly and never constructs `I_K ⊗ |g_s>` as a dense temporary. The logical input dimension remains `K`, while the physical output dimension is the product of `physical_dimensions`.

A subspace code has `G=1`; its unique trivial gauge needs no additional input, and the encoding channel has the single Kraus operator `V`. For `G>1`, omitting `gauge_state` raises an error. No maximally mixed gauge, reset state, or average over unspecified gauge inputs is inferred. Protecting a subsystem independently of gauge preparation is a different property, described by [operator quantum error correction](https://arxiv.org/abs/quant-ph/0504189).

## API and numerical contract

```python
import numpy as np
from qeclang.semantic import IsometryCode
from qeclang.semantic.encoding import prepare_encoding, encoding_channel

code = IsometryCode((2, 2), np.eye(4), gauge_dimension=2)
sigma = np.diag([0.75, 0.25])
preparation = prepare_encoding(code, sigma)
rho = np.array([[0.5, 0.2j], [-0.2j, 0.5]])
assert np.allclose(preparation.channel.apply(rho), np.kron(rho, sigma))
metadata = preparation.summary()
strict_channel = encoding_channel(code, sigma)
```

`prepare_encoding` returns `EncodingPreparation` containing the channel, copied immutable input/prepared gauge matrices, dimensions, isometry residual, gauge diagnostics, and a repair report. Its `summary()` supplies JSON-compatible metadata and SHA-256 hashes of the numerical gauge arrays; it does not serialize the arrays or prove physical-state equivalence. The returned channel uses the same [finite-channel conventions](channels.md) as other `KrausChannel` objects.

The isometry is revalidated at the requested preparation tolerance. The gauge matrix must have shape `G×G`, finite entries, trace one within that tolerance, exact Hermiticity, and no negative computed eigenvalues by default. Numerical validation is not a formal positivity certificate.

An explicit `repair_tolerance` permits bounded symmetrization and clipping only of negative eigenvalues. Every positive eigenvalue is retained, including tiny ones. The report records the original asymmetry/minimum eigenvalue, whether symmetrization occurred, how many negative eigenvalues were clipped, the reconstruction discrepancy, and the prepared trace residual. No trace normalization is performed: a repaired matrix whose trace falls outside the requested tolerance is rejected before Kraus contraction. Even without a repair, spectral reconstruction can introduce a reported floating-point discrepancy.

`encoding_channel` is the strict convenience function returning only `KrausChannel`. It has no repair argument; use `prepare_encoding` when repairs or provenance are needed. Input arrays cannot be changed through the returned snapshots, and caller mutation does not change the prepared channel.

## Resource and interpretation boundaries

`max_dimension`, `max_kraus`, and `max_elements` guard the physical/logical dimensions, eigensystem/state/isometry/Gram arrays, and total retained Kraus payload before their allocation. A rank-one gauge may fit a Kraus-count budget that excludes a full-rank gauge. These guards do not bound peak memory, eigensolver workspace, or elapsed time.

The output describes a **fixed product preparation** `ρ⊗σ`. It does not claim recovery for every gauge state or for arbitrary logical–gauge correlations. When using `preparation.channel` as a fixed encoder for recovery optimization, retain `preparation.summary()` with the recovery record. Optimizing recovery for that fixed preparation does not optimize the encoder or gauge state jointly, and does not establish a convex formulation for such joint optimization. Numerical channel recovery also does not synthesize a fault-tolerant physical encoding circuit.
