# Structured construction records and cyclic families

A finite code record stores checks. A construction record instead stores the algebraic seeds and their interpretation, so the instance can be regenerated without storing its expanded matrices. A family manifest keeps Laurent exponents independent of the lift order. These are data-only descriptions, not executable Python expressions.

## Finite construction contract

`qeclang.constructions.records` provides `to_dict`, `from_dict`, `dump`, `load`, and `fingerprint` for `LiftedProduct`. Schema `qeclang.construction.v1` fixes:

- The coefficient field GF(2), ordered cyclic group factors, and both seed shapes, including empty dimensions.
- Canonical monomial support in each ring entry. Exponents are reduced to their cyclic ranges, and repeated monomials are rejected on import.
- The regular action: a row at group site `g` has columns at `g + exponent`.
- Star as block transpose plus inversion of monomials; the CSS construction is `Hx=[A⊗I_nb | I_ma⊗B*]`, `Hz=[I_na⊗B | A*⊗I_mb]`.
- Physical order: variable-variable block followed by check-check block, lexicographic base-coordinate pairs, then the group site inside each block coordinate.

Unknown fields, alternate conventions, and future schemas are rejected. This prevents an apparently successful import that silently changes physical qubit ordering or the code construction. The lift identity and its relation to ordinary hypergraph products follow the commutative specialization of [Panteleev–Kalachev, Section III-D](https://arxiv.org/html/2012.04068#S3.SS4).

`fingerprint` is SHA-256 of the canonical JSON description, with sorted object keys and monomials. It identifies this construction presentation. It does not prove equivalence or inequivalence of quantum codes; different seeds, coordinate choices, or constructions may describe equivalent codes.

## Families before choosing a lift

```python
from qeclang.constructions.family import CyclicLiftFamily, LaurentMatrix, LiftDomain
from qeclang.constructions.records import fingerprint

# Each entry is a tuple of integer powers of x, including negative powers.
a = LaurentMatrix((((0, 1, 3),),), ncols=1)
b = LaurentMatrix((((-1, 0, 2),),), ncols=1)
family = CyclicLiftFamily(a, b, LiftDomain(minimum=3, odd_only=True))
instance = family.instantiate(7)
assert instance.n == 14
print(family.locality_bound())
print(family.fingerprint(), fingerprint(instance))
```

`LaurentMatrix` represents a matrix over `GF(2)[x,x^-1]`. Equal integer exponents cancel at construction. Instantiating order `L` then reduces exponents modulo `L`, with further collisions cancelling over GF(2). Neither step changes the original immutable template.

`LiftDomain` specifies a positive minimum, optional inclusive maximum, and whether only odd orders are allowed. The default is all positive odd integers. An explicitly even-capable domain is valid for construction and commutation; it does not authorize use of an odd-order rank theorem. For odd `L`, `x^L-1` is square-free, so its Chinese-remainder components are fields. Even orders generally introduce repeated factors and require different module analysis. [Panteleev–Kalachev, Appendices A–B](https://arxiv.org/html/2012.04068#A1).

`CyclicLiftFamily.to_dict/from_dict/dump/load` use `qeclang.family.v1`. The manifest records the parameter domain, both Laurent templates, and the same ordering/action conventions as finite construction records. A family's fingerprint includes its domain. An instantiated construction receives its own fingerprint; callers should retain both and the chosen lift order in experiment provenance.

## What the locality certificate states

`locality_bound()` counts monomials before reduction modulo `L`. With seed row-count vectors `rA,rB` and column-count vectors `cA,cB`, its check-weight bound is the larger of `max(rA)+max(cB)` and `max(cA)+max(rB)`. Qubit participation is bounded by the larger of `max(cA)+max(cB)` and `max(rA)+max(rB)`. Empty Cartesian blocks contribute zero.

The result also bounds X/Z check nonzeros divided by `L`. Exponent collisions can only decrease these counts. Consequently the same constants bound every instance in the declared domain, independently of lift order. This is a family-level sparsity statement. It asserts neither positive rate, nonzero logical count, distance growth, nor asymptotic goodness.

## Resource and trust boundaries

`ConstructionLimits` bounds file bytes, group factors/order, seed dimensions, combined entry/term counts, Laurent exponent magnitude, and expanded physical/check axis sizes. Importers inspect both matrices before creating ring objects. Instantiation checks dimensions, term budgets, and prospective expanded axes before specializing templates. No binary checks are expanded by importing or instantiating these records.

```python
from qeclang.constructions.records import ConstructionLimits
limits = ConstructionLimits(max_entries=1000, max_terms=10000, max_sites=100000)
restored = CyclicLiftFamily.from_dict(family.to_dict(), limits=limits)
assert restored.instantiate(7, limits=limits) == instance
```

A file byte limit is checked before JSON parsing; duplicate JSON object keys are rejected. Nested structure and term limits apply to the decoded input before algebra-object allocation. Limits exclude Python object overhead and caller-owned input. Programmatically creating a `LaurentMatrix` is a trusted in-process operation, not an untrusted-input parser. Explicit packed/sparse lowering has its own additional resource guards.

Canonical manifests reject duplicate exponents rather than silently simplifying them. Equivalent term orderings are accepted and normalized. File readers do not execute commands, evaluate expressions, dynamically import modules, or load pickle data.

## Exact analysis before expansion

`symbolic.cyclic_analysis.cyclic_rank` factors `x^L-1` for an odd cyclic lift,
reduces the seed matrix into each extension-field component, and sums component
ranks weighted by factor degrees. It never expands a binary regular matrix.
Factorization uses bounded deterministic Berlekamp splitting with exact Rabin
irreducibility verification. Cached factorizations are reverified when supplied.
Even or multicyclic orders are explicitly unsupported by this backend.

`analysis.structured.analyze_lifted` computes both check ranks and the exact
logical count. A total arithmetic budget covers both rank calls. It returns an
explicit unsupported or budget-exhausted result when appropriate. Selecting
`backend="packed"` permits a separately guarded binary expansion; failure never
silently changes the selected method. This dimension calculation makes no
claim that field-component transformations preserve physical Hamming distance.

```sh
mkdir -p results
python examples/family_analysis.py --manifest results/family.json
qeclang family results/family.json --lift 31
python scripts/benchmark_symbolic.py --lifts 7 31 63 127
```

Small cases can favor packed computation because factoring and extension-field
arithmetic have overhead. Benchmark both methods on the relevant seeds and lift
orders; avoiding expansion is not a blanket speed guarantee.
