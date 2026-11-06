# Exact analysis evidence

`qeclang.analysis.certificates` separates producing a result from checking it.
The checkers use GF(2) matrix identities or parity and never rerun elimination,
distance enumeration, or a numerical optimizer. They establish finite-instance
facts under the mathematical meaning of the binary operations; Python tests do
not themselves formally verify that implementation.

## Rank

`certify_rank(A)` produces matrices U,V,W,Z satisfying

```text
A = U V,    W U = I_r,    V Z = I_r.
```

The first identity bounds rank(A) above by r. The other two show U is injective
and V is surjective onto an r-dimensional space, so rank(A)=r.
`verify_rank(A, certificate)` checks only these identities and dimensions.
Empty dimensions and rank zero are included. Certificate production uses exact
elimination under a conservative matrix-payload guard; the proof does not
require trusting that elimination routine.

## Logical upper bounds

For an X logical candidate x, a certificate provides z such that

```text
Hz x = 0,    Hx z = 0,    x dot z = 1.
```

Every row combination of Hx pairs evenly with z, so x cannot be an X
stabilizer. Its physical weight gives an upper bound on X distance. The Z case
exchanges the matrices. General stabilizer certificates use the same argument
with the symplectic pairing. Phase information remains in the source code,
although the logical distance claim depends on unsigned support spaces.

`certify_css_logical` and `certify_stabilizer_logical` find the dual by solving
linear equations. Their verification functions check parity only. They do not
certify that the displayed logical is minimum weight. A lower-bound certificate
requires separate exhaustive, theorem-based, or proof-producing evidence.

## Portable records

`certificate_io.certificate_record(source, certificate)` produces data-only
`qeclang.certificate.v1` records. `verify_certificate_record(source, record)`
checks the presentation fingerprint, declared claim, resource limits and the
certificate identities. Arbitrary-precision vectors use bounded hexadecimal
strings rather than JSON numbers. Matrix budgets cover parsed certificate
payload, with an independent axis limit for empty matrices.

Fingerprints preserve physical coordinate order, check order and signs, while
ignoring descriptive names. They are presentation identities, not equivalence
classifications or authenticity signatures. This format currently covers packed
binary rank and CSS/stabilizer logical upper bounds. It does not serialize
arbitrary solver output or authorize unsupported evidence kinds.

## Structured parameters

`analysis.structured.analyze_lifted` uses an explicitly selected backend:
`crt` for odd cyclic field decomposition, or `packed` for guarded expansion.
Results distinguish `exact`, `unsupported`, and `budget_exhausted`, and carry a
construction fingerprint. No automatic backend substitution occurs. CRT rank
results carry irreducible factors, component ranks, assumptions, and arithmetic
work counts. Factorization is independently rechecked; component ranks are
currently recomputed by exact elimination rather than carrying standalone
field-rank certificates.

Bounds on polynomial steps, degree, entries and packed payload are distinct.
They are deterministic resource proxies, not wall-clock or total-memory
promises. Unsupported even or multicyclic rings are never treated as fields.
