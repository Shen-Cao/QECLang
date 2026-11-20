# Independently checked exclusion through weight two

A CSS sector has no nontrivial logical operator of weight at most two exactly when every zero-syndrome one-site or two-site vector belongs to that sector's stabilizer row space. Syndrome columns reduce this potentially quadratic set to at most one membership obligation per physical qubit. The reader checks explicit sums of generators; it does not rerun distance search or elimination.

## Column classes and membership evidence

For the X sector, write \(S=H_X\) and \(H=H_Z\); exchange them for the Z sector. Let \(c_i=He_i\) be column \(i\) of the opposite-sector check matrix. All equations are over \(\mathbb F_2\).

* A weight-one vector \(e_i\) has zero syndrome precisely when \(c_i=0\).
* A weight-two vector \(e_i+e_j\) has zero syndrome precisely when \(c_i=c_j\).

For exclusion through weight one, supply membership evidence for each \(e_i\) with \(c_i=0\). For exclusion through weight two, retain those obligations and partition the remaining sites into equal nonzero-column classes. In each class choose its least-index site \(r\), then supply evidence for \(e_r+e_i\) at every other site \(i\) in that class. The certificate describes each required vector as an explicit GF(2) sum of rows of \(S\).

This list is sufficient. Pairs of zero-column sites are sums of two certified unit vectors. Any pair in a nonzero class satisfies

\[
e_i+e_j=(e_r+e_i)+(e_r+e_j),
\]

where the term for \(i=r\) is zero. Thus every zero-syndrome vector of weight one or two belongs to \(\operatorname{row}S\). Conversely, exclusion implies that every listed obligation belongs to that row space, since each is itself a zero-syndrome vector of the permitted weight. Singleton nonzero classes need no evidence.

The obligation order is deterministic: zero-column unit vectors in physical order, followed by nonzero classes ordered by their least site, with differences in site order. The reader reconstructs this complete list from the supplied code. Omitting a required membership is not a weaker accepted certificate. No completeness claim depends on the producer's reported candidate count or timing.

There are at most \(n\) obligations, but this is not a constant-cost certificate. Column construction reads the opposite check presentation, and generator-support lists can contain up to \(n m_S\) indices, where \(m_S\) is the number of same-sector checks. Reading and XORing those supports still costs work. The producer may use exact elimination to find coefficients; the checker uses the submitted coefficients and no elimination. Neither path is a general efficient distance algorithm beyond this restricted weight range.

## Producer and checker contracts

```python
from qeclang.analysis.low_weight import certify_low_weight, check_low_weight
from qeclang.constructions import steane

code = steane()
record = certify_low_weight(code, "x", through=2)
checked = check_low_weight(code, record)
assert checked.status == "checked_exclusion"
assert checked.excluded_through == 2
```

Both functions accept an explicitly packed positive `CSSCode` and keyword limits `max_steps=1_000_000`, `max_axis=4096`, `max_bits=100_000_000`, and `max_indices=1_000_000`. The producer additionally takes a sector and `through=1` or `through=2`. It returns a data-only `qeclang.low_weight.v1` record bound to the exact presentation fingerprint, including generator order and redundancy. The record contains the request, status, claimed predicate, complete membership supports or one source-bound logical witness, reported work, and a content ID.

| Producer status | Evidence | Successful independent check |
|---|---|---|
| `excluded` | Complete ordered list of same-sector generator-index supports | `checked_exclusion`, with `excluded_through` |
| `found` | Nontrivial logical certificate of weight at most `through` | `checked_witness`, with `upper_bound` |
| `budget_exhausted` | No claim or partial evidence | `no_evidence` |

The checker returns a `LowWeightCheckResult` with its own `steps`, optional failure reason, and verified claim fields. Invalid evidence yields `invalid_record`; exhaustion of the reader's own quota yields `budget_exhausted`. Neither status supplies a scientific claim. Invalid API arguments or limit types can raise errors before record verification. The reader recomputes source and content identities, regenerates every required obligation, and checks the submitted identities. A matching digest alone is insufficient.

`max_axis` bounds physical width and check counts. `max_bits` bounds packed input, incidence, coefficient, and workspace estimates; it is not a total allocation bound. `max_indices` caps the total submitted generator references. `max_steps` charges arithmetic proxies, including incidence processing and membership checks; the producer also charges tracked elimination and separating-dual construction. Source/content hashing is streaming bookkeeping outside this arithmetic count. Python object overhead, caller-owned input, and wall-clock time are not bounded by these proxies. Reader limits are supplied independently of the record and its historical work counter.

## Exclusion is separate from a finite upper witness

Successful exclusion through \(t\in\{1,2\}\) establishes

\[
\{v\in\ker H:\ 1\leq\operatorname{wt}(v)\leq t\}
\subseteq\operatorname{row}S.
\]

It does not establish that any nontrivial logical operator exists. In particular, a \(k=0\) CSS code has an empty logical minimization set and can satisfy this predicate without an ordinary finite distance. Do not label such an exclusion alone as an exact distance or as a proof that \(k>0\).

A separate logical upper certificate provides an operator \(v\) and dual \(u\) with \(Hv=0\), \(Su=0\), and \(u^Tv=1\). It proves that \(v\notin\operatorname{row}S\) and hence a finite sector distance at most \(\operatorname{wt}(v)\). Combining it with an independently checked exclusion gives

\[
t+1\leq d_{\rm sector}\leq\operatorname{wt}(v).
\]

Equal endpoints establish the exact finite sector distance. A mismatch of source presentation or sector is invalid; a logical upper bound at or below the excluded weight contradicts the exclusion and must be rejected. A producer that finds a low-weight logical instead of an exclusion supplies only an upper witness, without independently proving that it is minimal.

For a CSS code with logical qubits, the full Pauli distance is \(d=\min(d_X,d_Z)\). A nontrivial mixed Pauli has at least one nontrivial sector component, whose support is contained in its physical support; pure-sector operators attain the reverse bound. Excluding low weights in only one sector cannot establish a full CSS lower bound. The `css.full_distance` operation takes a source and matching finite X/Z distance claims, then forms the interval \([\min(l_X,l_Z),\min(u_X,u_Z)]\). Equal endpoints establish exactness even if one sector's interval remains unresolved. Independently obtaining \(d_X=d_Z=3\) establishes \([[n,k,3]]\) distance, but says nothing by itself about the separately determined value of \(k\).

## Exact Steane instance

The data-only [exact Steane program](../examples/programs/exact_steane.json) requests two `css.exclude` results, one for each sector, through weight two. Its `css.distance` nodes use `max_weight=0` and `max_candidates=0` solely to obtain initial logical representatives. `css.witness` certifies their upper bounds, and `css.combine_distance` combines each upper witness with the matching exclusion. A final `css.full_distance` node combines the X and Z intervals for the same CSS source.

```sh
mkdir -p output
qeclang run examples/programs/exact_steane.json > output/exact-steane-run.json
qeclang run examples/programs/exact_steane.json --inspect output/exact-steane-run.json
```

The independently checked claims are `sector_distance` with `status="exact"`, `lower_bound=upper_bound=value=3`, separately for X and Z, followed by `logical_distance` with `sector="pauli"` and the same exact value. The raw distance reports remain producer reports; exactness is derived at the combination nodes. The Steane opposite-check matrices have seven distinct nonzero columns, so neither units nor equal-column pairs can have zero syndrome. Their exclusion records need no membership coefficients. The weight-three upper witnesses supply the nontrivial logical operators.

Execution records producer work in `exclusion_steps` and mandatory combination checks in `exclusion_verify_steps`; inspection has its own `max_exclusion_verify_steps` and index limits. The example's execution ledger records zero enumerated candidates. Its independent inspection can run with the distance producer and matrix elimination methods disabled. Tests additionally compare both sectors and both supported weight ranges against exhaustive small-code support enumeration, and reject rehashed omitted obligations, false coefficients, false dual pairings, and an attempted upgrade from weight-one to weight-two exclusion.

The independent checks are exact Python arithmetic, not a kernel-formalized proof of the lower-bound implementation. The existing Lean logical-witness checker covers a different, upper-bound obligation. This example establishes a known finite instance; it does not claim a new code or an asymptotic family bound.
