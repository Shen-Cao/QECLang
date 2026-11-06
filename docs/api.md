# API and interpretation conventions

## Exact representations

`BinaryMatrix(rows, ncols)` stores immutable nonnegative integer row masks.
Bit `j` is column `j`; matrix vectors are columns. Empty dimensions are explicit.
`from_dense` accepts binary integers only. Arithmetic is over GF(2), never
floating-point rank. `nullspace()` returns a matrix whose **rows** form a right
kernel basis; `solve(rhs)` returns one solution with free coordinates zero and
raises `ValueError` if inconsistent. Rank and row-space elimination are cached.

`CSSCode(hx, hz)` requires `hx @ hz.T == 0`. All listed X and Z generators have
positive signs. Redundant and zero checks are allowed and retained, so syndrome
lengths refer to the supplied presentation. The chain convention is
`C2 --hz.T--> C1 --hx--> C0`. Here `C1` is the ordered physical qubit space.
`ker(hx)/row(hz)` describes Z logicals; `ker(hz)/row(hx)` describes X logicals.

`Pauli(x, z, phase)` means `i**phase X**x Z**z`; `Y` on one site has phase 1.
`Pauli.from_label("XZZXI")` places its leftmost character on qubit zero.
`StabilizerCode(n, generators)` checks Hermiticity, commutation and dependent
phase relations. Commuting generators alone do not guarantee a nonempty code.
Unsigned symplectic vectors are packed as `x | (z << n)`.

`SubsystemCode(n, generators, sector=None)` stores the gauge module modulo
phases. The optional `sector` specifies signed generators spanning exactly its
center; otherwise `.stabilizer` chooses positive Hermitian representatives of
an independent center basis. The actual eigensector matters for embedding,
though parameters and Pauli distance are independent of this choice. `.r`
counts gauge qubits and `.k` protected qubits.

## Structured lowering

`GroupAlgebraElement.from_terms(orders, terms)` represents a polynomial in
commuting finite cyclic shifts. For example `(3, 4)` imposes `x**3=y**4=1`.
Repeated exponents cancel modulo two, including coincidences after reduction.
`involution()` inverts shifts and lowers to matrix transpose. `lower()` uses
lexicographic sites, last coordinate varying fastest. It is an exact regular
representation; it does not model noncommutative rings.

`hypergraph_product(a,b)` orders physical coordinates as `(VV, CC)` and uses
`Hx = [a tensor I_n_b | I_m_a tensor b.T]` and
`Hz = [I_n_a tensor b | a.T tensor I_m_b]`.
`bicycle(a,b)` uses `[A B]` and `[B.T A.T]`; abelian multiplication guarantees
commutation. `max_bits` limits the worst-case packed output payload before
expansion, excluding Python overhead and temporary arrays.

## Analysis results

`css_logicals` returns paired X and Z quotient bases with identity pairing.
These are algebraic representatives, not guaranteed minimum-weight logicals.
General stabilizer frames are similarly paired using the symplectic product.

`css_distance(code, "x"|"z", max_weight=..., max_candidates=...)` reports:

- `lower_bound`: rigorous bound from fully searched weight layers.
- `upper_bound`, `witness`: an explicitly checked nontrivial logical.
- `searched_through`: largest fully excluded weight, even if interrupted midway through the next layer.
- `status`: `exact`, `weight_limit`, `budget_exhausted`, or `no_logicals`.
- `distance`: the finite distance only if status is `exact`.

CSS overall distance is `min(d_x,d_z)`. For `k=0`, logical distance minimizes
an empty set; finite bounds and witness are `None`. This convention avoids
conflating logical distance with the minimum weight of a stabilizer state.
Initial quotient representatives can certify distance 1 without enumeration.
Candidate budgets exclude polynomial-time basis construction.

`stabilizer_distance` enumerates physical Pauli weight, so Y counts as one
site. `subsystem_distance(kind="dressed")` searches `S-perp / G`, and
`kind="bare"` searches `G-perp / S`. Both use Pauli site weight.

`css_distance_milp` lazily imports SciPy/HiGHS. It solves one odd-pairing
subproblem per opposite logical basis vector, under a total time budget and
per-subproblem node budget. `solver_optimal` is numerical solver evidence;
`solver_lower_bound` is not an exact certificate. Every returned binary upper
witness is rechecked by exact arithmetic. An interrupted run retains bounds
and messages; unvisited subproblems retain the trivial bound. SciPy documents
its termination statuses and bounds in the [MILP API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).

Erasure correctability assumes **known locations**, arbitrary errors within
that region, and perfect operation outside it. Profile row weights and degrees
refer to supplied generators. Tanner cycles are presentation dependent.

## Decoding and experiments

`css_decode(code, sx, sz)` uses `sx=Hx*z`, `sz=Hz*x`. Its `.success` means both
syndromes were matched. Use `css_logical_success` to check that a correction's
residual is a stabilizer. Minimum individual error weight is not maximum
likelihood over quantum logical cosets, and independent sectors discard X/Z
correlations in depolarizing noise.

`simulate_css_memory` draws IID per-site I/X/Y/Z with probabilities
`1-p,p/3,p/3,p/3`. It uses perfect syndromes and one correction cycle. Counts
include separate decoder failures and undetected logical failures, and a
Wilson confidence interval. Zero observed failures does not imply zero true
failure probability. This is not a fault-tolerance threshold experiment.

`IsometryCode(physical_dimensions, isometry, gauge_dimension=1)` is an optional
dense semantic reference. Columns are ordered `logical*gauge_dimension+gauge`.
`check_correctable(errors)` checks the supplied error span with a numerical
residual; the list need not form a trace-preserving channel. Include identity
if the model contains no error. The result is tolerance dependent, not a proof
of exact correction. Dense stabilizer lowering chooses an arbitrary logical
basis and does not synthesize an encoder circuit.

## Interchange and reproducibility

`qeclang.io` stores `qeclang.code.v1` JSON with explicit `n`, kind, name,
support lists, phases, and optional subsystem sector. It retains generator
order and redundant checks. Loading validates the code and applies file,
qubit, check-count and matrix-payload limits. It never executes construction
strings. It does not classify equivalence or serialize every semantic dialect.

Search records use `qeclang.experiment.v1`. Preserve all records, including
`k=0` and unsuccessful candidates. Seeded sampling is reproducible in the same
software environment; long-term replay should use recorded polynomial supports
and code data. Record dependency versions alongside generated datasets when
comparing numerical solver results across environments.
