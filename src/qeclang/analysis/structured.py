"""Explicit analysis dispatch for construction IRs, with no hidden fallback."""
from dataclasses import dataclass

from qeclang.algebra.binary import nonnegative_integer
from qeclang.algebra.polynomial import ArithmeticBudget, WorkLimitExceeded
from qeclang.constructions.lifted import LiftedProduct
from qeclang.constructions.records import fingerprint
from qeclang.symbolic.cyclic_analysis import CyclicRankResult, cyclic_rank
from qeclang.symbolic.repeated_root import CyclicPolynomialRankResult, cyclic_rank_general
from qeclang.symbolic.abelian_analysis import AbelianRankResult, abelian_rank
from qeclang.symbolic.mixed_abelian import MixedAbelianRankResult, abelian_rank_mixed

BACKENDS = ("crt", "polynomial", "abelian", "mixed-abelian", "packed")
RankEvidence = CyclicRankResult | CyclicPolynomialRankResult | AbelianRankResult | MixedAbelianRankResult


@dataclass(frozen=True)
class StructuredParameters:
    construction_fingerprint: str
    n: int
    status: str
    method: str
    k: int | None = None
    rank_x: int | None = None
    rank_z: int | None = None
    steps: int | None = None
    reason: str | None = None
    x_evidence: RankEvidence | None = None
    z_evidence: RankEvidence | None = None

    @property
    def rate(self):
        return self.k / self.n if self.k is not None and self.n else None


def unsupported_reason(backend, orders):
    if backend == "polynomial" and len(orders) != 1:
        return "polynomial backend requires one cyclic factor"
    if backend == "crt" and (len(orders) != 1 or orders[0] % 2 == 0):
        return "CRT field backend requires one odd cyclic factor"
    if backend == "abelian" and any(order % 2 == 0 for order in orders):
        return "abelian character backend requires odd cyclic factors"
    return None


def analyze_lifted(construction: LiftedProduct, *, backend="crt", max_degree=512,
                   max_steps=10_000_000, max_entries=1_000_000,
                   max_blocks=1_000_000, max_bits=100_000_000,
                   max_polynomial_degree=4096, max_operations=100_000,
                   max_sites=1_000_000, max_characters=10_000,
                   max_field_degree=64, max_field_candidates=4096, max_factors=16,
                   max_component_entries=1_000_000, max_component_axis=10_000,
                   record_operations=False):
    """Analyze n,k exactly, or report unsupported/budget_exhausted explicitly.

    crt supports odd cyclic orders; polynomial supports all cyclic orders;
    abelian supports odd abelian groups; mixed-abelian retains regular 2-primary
    components; packed expands the binary checks. No failure changes backend.

    max_steps is shared across both symbolic ranks, including field/factor
    searches. max_operations is shared for polynomial traces/diagonalization.
    max_blocks bounds symbolic check construction; this work and fingerprinting
    are not charged to max_steps. Packed elimination has axis/payload controls,
    not an arithmetic-work budget. max_sites/max_bits apply only to packed.
    max_degree applies to cyclic backends, field/character/factor limits to
    abelian backends, and component limits only to mixed-abelian. Other inactive
    backend-specific limits are validated but unused. Traces require polynomial.
    steps records accepted arithmetic-proxy charges even on symbolic refusal;
    rejected charge requests are not spent. Packed work remains unmetered (None).
    """
    if not isinstance(construction, LiftedProduct):
        raise TypeError("analysis requires LiftedProduct")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if type(record_operations) is not bool:
        raise ValueError("record_operations must be boolean")
    if record_operations and backend != "polynomial":
        raise ValueError("record_operations requires the polynomial backend")
    for name, value in (("max_degree", max_degree), ("max_steps", max_steps),
                        ("max_entries", max_entries), ("max_blocks", max_blocks),
                        ("max_bits", max_bits), ("max_polynomial_degree", max_polynomial_degree),
                        ("max_operations", max_operations), ("max_sites", max_sites),
                        ("max_characters", max_characters), ("max_field_degree", max_field_degree),
                        ("max_field_candidates", max_field_candidates), ("max_factors", max_factors),
                        ("max_component_entries", max_component_entries),
                        ("max_component_axis", max_component_axis)):
        nonnegative_integer(value, name)
    base = {"construction_fingerprint": fingerprint(construction), "n": construction.n,
            "method": backend}
    zero_steps = None if backend == "packed" else 0
    reason = unsupported_reason(backend, construction.a.orders)
    if reason:
        return StructuredParameters(**base, status="unsupported", steps=zero_steps, reason=reason)
    a, b = construction.a, construction.b
    blocks = (a.ncols*b.ncols + a.nrows*b.nrows) * (a.nrows*b.ncols + a.ncols*b.nrows)
    if blocks > max_blocks or max(a.nrows, a.ncols, b.nrows, b.ncols)**2 > max_blocks:
        return StructuredParameters(**base, status="budget_exhausted", steps=zero_steps,
                                    reason="symbolic block budget")
    if backend == "packed":
        mx, mz = construction.check_counts
        if (construction.n * (mx + mz) + mx*mz > max_bits
                or max(construction.n, mx, mz) > max_sites):
            return StructuredParameters(**base, status="budget_exhausted", reason="packed expansion budget")
        code = construction.lower(max_bits=max_bits, max_blocks=max_blocks, max_sites=max_sites)
        return StructuredParameters(**base, status="exact", k=code.k,
                                    rank_x=code.hx.rank(), rank_z=code.hz.rank())
    hx, hz = construction.symbolic_checks(max_blocks=max_blocks)
    meter = ArithmeticBudget(max_steps)
    try:
        if backend == "polynomial":
            options = dict(max_lift=max_degree, max_polynomial_degree=max_polynomial_degree,
                           max_entries=max_entries, record_operations=record_operations, _meter=meter)
            x = cyclic_rank_general(hx, max_steps=max_steps, max_operations=max_operations, **options)
            z = cyclic_rank_general(hz, max_steps=max_steps,
                                    max_operations=max_operations-x.operation_count, **options)
        elif backend == "crt":
            x = cyclic_rank(hx, max_degree=max_degree, max_steps=max_steps, max_entries=max_entries,
                            _meter=meter)
            z = cyclic_rank(hz, max_degree=max_degree, max_steps=max_steps,
                            max_entries=max_entries, factorization=x.factorization, _meter=meter)
        else:
            options = dict(max_characters=max_characters, max_field_degree=max_field_degree,
                           max_candidates=max_field_candidates, max_factors=max_factors, _meter=meter)
            if backend == "abelian":
                rank_function = abelian_rank
                options["max_entries"] = max_entries
            else:
                rank_function = abelian_rank_mixed
                options.update(max_seed_entries=max_entries,
                               max_component_entries=max_component_entries,
                               max_component_axis=max_component_axis)
            x = rank_function(hx, max_steps=max_steps, **options)
            z = rank_function(hz, max_steps=max_steps, **options)
    except WorkLimitExceeded as error:
        return StructuredParameters(**base, status="budget_exhausted", reason=str(error), steps=meter.used)
    return StructuredParameters(**base, status="exact", k=construction.n-x.rank-z.rank,
                                rank_x=x.rank, rank_z=z.rank, steps=meter.used,
                                x_evidence=x, z_evidence=z)
