"""Combine rigorous enumeration lower bounds and independently checked witnesses."""
from dataclasses import dataclass
from math import isfinite

from .certificates import LogicalWitness, certify_css_logical, verify_css_logical
from .distance import DistanceResult, _limits, css_distance
from .heuristic import CSSHeuristicResult, _basis_preflight, css_logical_upper_bound
from .milp import MILPDistanceResult, css_distance_milp


@dataclass(frozen=True)
class CSSDistancePortfolio:
    sector: str
    status: str
    lower_bound: int | None
    upper_bound: int | None
    witness: int | None
    certificate: LogicalWitness | None
    enumeration: DistanceResult | None
    heuristic: CSSHeuristicResult
    milp: MILPDistanceResult | None = None
    solver_lower_bound: float | None = None
    milp_witness_status: str | None = None
    reason: str | None = None

    @property
    def distance(self):
        return self.upper_bound if self.status == "exact" else None


def _certificate(code, sector, witness, claimed_weight):
    if witness is None:
        return None
    try:
        certificate = certify_css_logical(code, sector, witness)
    except (ValueError, TypeError):
        return None
    if certificate.weight != claimed_weight or not verify_css_logical(code, certificate):
        return None
    return certificate


def css_distance_portfolio(code, sector, *, seed=0, restarts=32, max_sweeps=10,
                           max_evaluations=10_000, max_weight=None, max_candidates=1_000_000,
                           max_basis_bits=100_000_000, max_basis_dimension=4096,
                           use_milp=False, milp_time_limit=30.0, milp_node_limit=100_000):
    """Keep exact and numerical evidence distinct while combining bounds.

    Only exhaustive weight-layer exclusion supplies the rigorous lower bound.
    The minimum verified witness weight supplies the upper bound. A numerical
    MILP dual bound is retained separately and never changes exact/bounded status.
    Optional MILP assembly/solver costs follow that backend's documented limits.
    """
    _basis_preflight(code, sector, max_basis_bits, max_basis_dimension)
    max_weight = _limits(code.n, max_weight, max_candidates)
    if type(use_milp) is not bool:
        raise ValueError("use_milp must be boolean")
    if not isfinite(milp_time_limit) or milp_time_limit < 0:
        raise ValueError("milp_time_limit must be finite and nonnegative")
    if type(milp_node_limit) is not int or milp_node_limit < 0:
        raise ValueError("milp_node_limit must be a nonnegative integer")
    heuristic = css_logical_upper_bound(
        code, sector, seed=seed, restarts=restarts, max_sweeps=max_sweeps,
        max_evaluations=max_evaluations, max_basis_bits=max_basis_bits,
        max_basis_dimension=max_basis_dimension)
    if heuristic.status in ("basis_budget_exhausted", "no_logicals"):
        return CSSDistancePortfolio(sector, heuristic.status, None, None, None, None,
                                     None, heuristic, reason=heuristic.reason)
    certificates = []
    if heuristic.certificate is not None:
        if not verify_css_logical(code, heuristic.certificate):
            raise ArithmeticError("heuristic witness failed independent verification")
        certificates.append(heuristic.certificate)
    milp = None
    numerical_lower = milp_status = None
    if use_milp:
        milp = css_distance_milp(code, sector, time_limit=milp_time_limit, node_limit=milp_node_limit)
        certificate = _certificate(code, sector, milp.witness, milp.upper_bound)
        milp_status = "missing" if milp.witness is None else "rejected"
        if certificate is not None:
            certificates.append(certificate)
            milp_status = "verified"
        numerical_lower = milp.solver_lower_bound
    if certificates:
        max_weight = min(max_weight, min(c.weight for c in certificates) - 1)
    enumeration = css_distance(code, sector, max_weight=max_weight, max_candidates=max_candidates)
    certificate = _certificate(code, sector, enumeration.witness, enumeration.upper_bound)
    if certificate is None:
        raise ArithmeticError("enumeration failed to provide a valid logical witness")
    certificates.append(certificate)
    best = min(certificates, key=lambda c: (c.weight, c.operator))
    lower = enumeration.lower_bound
    if lower is None or lower > best.weight:
        raise ArithmeticError("rigorous distance bounds are inconsistent")
    return CSSDistancePortfolio(sector, "exact" if lower == best.weight else "bounded",
                                 lower, best.weight, best.operator, best, enumeration, heuristic,
                                 milp, numerical_lower, milp_status)


__all__ = ["CSSDistancePortfolio", "css_distance_portfolio"]
