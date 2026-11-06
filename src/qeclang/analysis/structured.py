"""Explicit analysis dispatch for construction IRs, with no hidden fallback."""
from dataclasses import dataclass

from qeclang.algebra.binary import nonnegative_integer
from qeclang.algebra.polynomial import WorkLimitExceeded
from qeclang.constructions.lifted import LiftedProduct
from qeclang.constructions.records import fingerprint
from qeclang.symbolic.cyclic_analysis import CyclicRankResult, cyclic_rank


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
    x_evidence: CyclicRankResult | None = None
    z_evidence: CyclicRankResult | None = None

    @property
    def rate(self):
        return self.k / self.n if self.k is not None and self.n else None


def analyze_lifted(construction: LiftedProduct, *, backend="crt", max_degree=512,
                   max_steps=10_000_000, max_entries=1_000_000,
                   max_blocks=1_000_000, max_bits=100_000_000):
    """Analyze n,k exactly, or report unsupported/budget_exhausted explicitly.

    crt uses odd cyclic finite-field decomposition. packed explicitly permits
    expansion and exact binary elimination. The caller chooses a fallback;
    this function never changes the requested backend on failure. max_steps
    is shared across both CRT rank computations, including factor verification.
    """
    if not isinstance(construction, LiftedProduct):
        raise TypeError("analysis requires LiftedProduct")
    if backend not in ("crt", "packed"):
        raise ValueError("backend must be crt or packed")
    for name, value in (("max_degree", max_degree), ("max_steps", max_steps),
                        ("max_entries", max_entries), ("max_blocks", max_blocks),
                        ("max_bits", max_bits)):
        nonnegative_integer(value, name)
    base = {"construction_fingerprint": fingerprint(construction), "n": construction.n,
            "method": backend}
    if backend == "crt" and (len(construction.a.orders) != 1 or construction.a.orders[0] % 2 == 0):
        return StructuredParameters(**base, status="unsupported",
                                    reason="CRT field backend requires one odd cyclic factor")
    a, b = construction.a, construction.b
    blocks = (a.ncols*b.ncols + a.nrows*b.nrows) * (a.nrows*b.ncols + a.ncols*b.nrows)
    if blocks > max_blocks or max(a.nrows, a.ncols, b.nrows, b.ncols)**2 > max_blocks:
        return StructuredParameters(**base, status="budget_exhausted", reason="symbolic block budget")
    if backend == "packed":
        mx, mz = construction.check_counts
        if (construction.n * (mx + mz) + mx*mz > max_bits
                or max(construction.n, mx, mz) > 1_000_000):
            return StructuredParameters(**base, status="budget_exhausted", reason="packed expansion budget")
        code = construction.lower(max_bits=max_bits, max_blocks=max_blocks)
        return StructuredParameters(**base, status="exact", k=code.k,
                                    rank_x=code.hx.rank(), rank_z=code.hz.rank())
    hx, hz = construction.symbolic_checks(max_blocks=max_blocks)
    try:
        x = cyclic_rank(hx, max_degree=max_degree, max_steps=max_steps, max_entries=max_entries)
        z = cyclic_rank(hz, max_degree=max_degree, max_steps=max_steps-x.steps,
                        max_entries=max_entries, factorization=x.factorization)
    except WorkLimitExceeded as error:
        return StructuredParameters(**base, status="budget_exhausted", reason=str(error))
    return StructuredParameters(**base, status="exact", k=construction.n-x.rank-z.rank,
                                rank_x=x.rank, rank_z=z.rank, steps=x.steps+z.steps,
                                x_evidence=x, z_evidence=z)
