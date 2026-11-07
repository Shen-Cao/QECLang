# Syndrome decoding and memory experiments

A decoder solves a recovery problem for a specified check presentation and noise
model. Matching a syndrome is necessary for recovery but does not establish
logical success. For CSS codes, after correcting an error `(x, z)`, the residual
must lie in `row(Hx)` in the X component and `row(Hz)` in the Z component. The
module `qeclang.decoding` exposes separate results for these questions.

## Exact reference algorithms

`minimum_weight_decode(H, syndrome, max_weight=..., max_candidates=...)` enumerates
binary error vectors by increasing Hamming weight. It returns the first vector
matching the syndrome, an explicit weight/budget limit, or an inconsistent
syndrome. This minimizes individual error weight, not the total probability of a
quantum stabilizer coset. Redundant checks remain part of the syndrome.

`css_decode(code, sx, sz, ...)` applies the reference decoder independently to
`Hz x = sz` and `Hx z = sx`. Its candidate budget is per sector.
`css_logical_success(code, x, z, correction)` checks residual membership in the
stabilizer row spaces.

`erasure_decode(H, syndrome, erased)` solves the restricted linear system for
known erased sites using exact GF(2) elimination. `css_erasure_decode` additionally
reports whether the entire erased region is correctable. Its `success` field
means both syndromes were matched; `region_correctable` guarantees logical
recovery for every error on those sites. A nonunique classical solution can
still give certain quantum recovery when the ambiguity is a stabilizer.

## Sparse belief propagation

`BPDecoder` retains deterministic check/variable adjacency lists and reuses them
between calls. It accepts both `BinaryMatrix` and `SparseBinaryMatrix`. The
algorithm is normalized min-sum with a parallel schedule, finite LLR saturation,
and optional damping. Probabilities may lie anywhere in `[0,1]`; a scalar is
broadcast to every site. For `0 < p_j < 1`, the initial log-likelihood ratio is
`L_j = log((1-p_j)/p_j)`. Endpoints are exact constraints: `p_j=0` fixes error
bit j to zero and `p_j=1` fixes it to one. No epsilon smoothing is applied.

Before message passing, fixed bits are substituted into every adjacent check,
and a sparse queue propagates all resulting singleton equations. Fixed variables
are removed from the effective graph. Their entries in `posterior_llrs` use
finite `±llr_clip` display placeholders; these values never participate in
message updates. Thus changing `llr_clip` cannot relax a deterministic prior.

For check `a` and variable `j`, the undamped check message has magnitude
`alpha * min(|q[k->a]| for k != j)` and sign
`(-1)^syndrome[a] * product(sign(q[k->a]) for k != j)`, using the
reduced checks and their adjusted syndrome after fixed-bit substitution.
Damping retains the specified fraction of the old check message. Variable
messages add the prior and all incoming check messages except the receiving
edge. Hard decisions use the sign of the total LLR; a zero LLR chooses zero.
Every successful result respects the fixed bits and is checked against every
original syndrome equation.

Two minima per check avoid a quadratic loop over its neighbors. A full iteration
uses `O(n + m + nnz)` scalar message operations and adjacency/message storage has
the same size. Python integer packing and unpacking add costs that depend on
integer width. Constructing from packed matrices also requires extracting their
supports; sparse input avoids that lowering.

Before allocating priors or adjacency arrays, the constructor checks
`max_variables` (default 1,000,000), `max_checks` (default 1,000,000), and
`max_edges` (default 10,000,000) against the original presentation. Deterministic
priors do not bypass these guards, even if subsequent substitution removes the
whole graph. These are count budgets, not peak-memory or elapsed-time guarantees.
All three limits are recorded in `configuration` and simulation provenance;
`CSSBPDecoder` passes them to each sector.

```python
from qeclang.algebra import SparseBinaryMatrix
from qeclang.decoding import BPDecoder

checks = SparseBinaryMatrix(((0, 1), (1, 2)), 3)
decoder = BPDecoder(checks, probabilities=0.05, max_iterations=30,
                    min_sum_scale=0.8, damping=0.2, fallback="osd0")
result = decoder.decode(0b11)
assert result.success
assert checks.matvec(result.error) == 0b11
```

`BPResult.status` distinguishes `converged`, `osd0`, `not_converged`,
`inconsistent`, `inconsistent_priors`, and `fallback_limit`. Only the first two return an accepted
correction. `iterations`, `posterior_llrs`, the final `last_decision`, and its
`syndrome_residual` remain available for diagnostics. Without the fallback, a
nonconvergence result does not prove that the syndrome is inconsistent.
`inconsistent_priors` means exact propagation or the fallback proved that no
syndrome-consistent error can obey the deterministic priors. Without deterministic
priors, such a contradiction is reported as `inconsistent`. A BP iteration limit
alone always yields `not_converged`, not an impossibility claim.

### Order-zero fallback

With `fallback="osd0"`, a failed BP run sorts only unfixed variables by increasing absolute
posterior LLR, using site index to break ties. GF(2) elimination solves the
remaining syndrome using independent pivot columns in that order; nonpivot
corrections are zero, retaining the corresponding hard decisions. Variables fixed
by endpoint priors or singleton propagation are never changed. This is a
reliability-ordered order-zero postprocessor. It does not enumerate higher-order
candidates or optimize likelihood. The `max_osd_bits` limit guards each packed
matrix before materialization using `number_of_checks * number_of_unfixed_variables`;
the exact original syndrome is verified after solving.
Unlike BP iterations, fallback elimination need not have linear cost.

The combination of BP and ordered-statistics postprocessing is established for
quantum LDPC decoding; see [Roffe et al. (2020)](https://arxiv.org/abs/2005.07016).
The normalized min-sum scaling and parallel schedule also follow the conventions
of the [LDPC decoder documentation](https://software.roffe.eu/ldpc/ldpc/bp_decoder.html).
This implementation provides a small, inspectable baseline; it does not claim
performance parity with those optimized implementations.

## CSS decoding and reproducible simulation

`CSSBPDecoder(code, x_probabilities, z_probabilities, **options)` builds two
reusable decoders. The probabilities describe the binary components, so an IID
depolarizing channel with total Pauli probability `p` has both marginals `2*p/3`.
Independent decoding ignores the correlation introduced by Y errors. The
zero-noise endpoint `p=0` is supported directly, including the BP CLI modes.

```python
from qeclang.constructions import steane
from qeclang.decoding import CSSBPDecoder, simulate_css_memory

code = steane()
p = 0.03
decoder = CSSBPDecoder(code, 2*p/3, 2*p/3, fallback="osd0")
experiment = simulate_css_memory(code, p, trials=1000, seed=42, decoder=decoder)
```

`simulate_css_memory(code, p, trials=1000, seed=42, decoder=decoder)` samples IID
site noise with probabilities `I:1-p`, `X:p/3`, `Y:p/3`, `Z:p/3`, using perfect
syndromes. Omitting `decoder` selects exhaustive minimum-weight decoding. Results
record the decoder name and a snapshot of its complete configuration, failure
counts, seed, and a Wilson confidence interval. Syndromes are cached, so custom
decoders must be deterministic and expose `code`, `decode(sx, sz)`, `name`, and a
JSON-compatible `configuration` dictionary. The exact CSS check presentation
must match. Decoder failures and logical failures are counted separately.

Convergence can coincide with logical failure. For the Steane presentation with
rows `(85, 102, 120)`, equal component priors `0.05`, and the default min-sum scale
`0.8`, an X error on site 6 yields a first-iteration syndrome-correct correction
with packed value `116`; its residual against error `64` is a nontrivial logical.
This is an explicit regression test of the distinction. Neither BP nor OSD0 is
guaranteed to correct every error below half the code distance. Performance
requires evaluation under the chosen noise model, parameters, and presentation.
