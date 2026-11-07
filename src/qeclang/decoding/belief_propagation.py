"""Sparse normalized min-sum syndrome decoding with optional OSD order zero.

The BP/OSD design follows the algorithmic framework of Roffe et al.,
https://arxiv.org/abs/2005.07016 ; this is an independent reference implementation.
Parameters use normalized min-sum conventions documented by LDPC:
https://software.roffe.eu/ldpc/ldpc/bp_decoder.html . No likelihood optimality or
quantum coset maximum-likelihood guarantee is implied.
"""
from collections import deque
from dataclasses import dataclass
from math import isfinite, log, log1p
from numbers import Real
from typing import Literal

from qeclang.algebra import BinaryMatrix
from qeclang.algebra.binary import nonnegative_integer, set_bits
from qeclang.codes import CSSCode, SparseCSSCode


@dataclass(frozen=True)
class BPResult:
    error: int | None
    status: Literal["converged", "osd0", "not_converged", "inconsistent",
                    "inconsistent_priors", "fallback_limit"]
    iterations: int
    last_decision: int
    syndrome_residual: int
    posterior_llrs: tuple[float, ...]

    @property
    def success(self) -> bool:
        """Whether a correction matching every syndrome check was returned."""
        return self.status in ("converged", "osd0")


def _nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _probabilities(value, n):
    if isinstance(value, Real):
        probability = float(value)
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probability must be finite and in [0, 1]")
        values = (probability,) * n
    else:
        values = tuple(map(float, value))
    if len(values) != n or any(not isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("probabilities must provide one finite value in [0, 1] per site")
    return values


class BPDecoder:
    """Reusable adjacency lists for bounded normalized min-sum belief propagation.

    ``probabilities`` is a scalar or per-site binary error probabilities.
    ``damping`` is the fraction of the *old check-to-variable message* retained;
    zero gives undamped parallel updates. ``min_sum_scale`` scales each new
    min-sum message. All LLRs are saturated to ``llr_clip`` (at most 1e6).
    Probabilities zero/one fix bits exactly, with no epsilon smoothing. Fixed
    variables and recursively implied singleton variables are removed from the
    effective check graph. Their reported finite LLRs are display placeholders,
    never messages. A zero-valued LLR on an unfixed variable chooses bit zero.

    Constructor budgets ``max_variables``, ``max_checks``, and ``max_edges``
    bound the original graph before allocating priors or adjacency arrays,
    including variables later fixed by deterministic priors. These are count
    limits, not peak-memory or elapsed-time guarantees.

    No dense elimination is performed unless ``fallback='osd0'`` is requested
    and BP fails. That fallback is bounded by ``max_osd_bits`` per packed matrix,
    and retains the most reliable free hard decisions while solving less
    reliable pivot corrections. It is not minimum-weight or maximum-likelihood
    optimization. The original generator presentation is retained for BP.
    """

    def __init__(self, checks, probabilities=0.1, *, max_iterations=50,
                 min_sum_scale=0.8, damping=0.0, llr_clip=50.0,
                 fallback="none", max_osd_bits=10_000_000,
                 max_variables=1_000_000, max_checks=1_000_000, max_edges=10_000_000):
        from qeclang.algebra.sparse import SparseBinaryMatrix
        if not isinstance(checks, (BinaryMatrix, SparseBinaryMatrix)):
            raise TypeError("checks must be a BinaryMatrix or SparseBinaryMatrix")
        self.checks = checks
        self.n, self.m = checks.ncols, checks.nrows
        self.max_variables = nonnegative_integer(max_variables, "max_variables")
        self.max_checks = nonnegative_integer(max_checks, "max_checks")
        self.max_edges = nonnegative_integer(max_edges, "max_edges")
        if self.n > self.max_variables:
            raise ValueError("BP graph exceeds max_variables")
        if self.m > self.max_checks:
            raise ValueError("BP graph exceeds max_checks")
        if checks.nnz > self.max_edges:
            raise ValueError("BP graph exceeds max_edges")
        self.max_iterations = _nonnegative_integer(max_iterations, "max_iterations")
        self.max_osd_bits = _nonnegative_integer(max_osd_bits, "max_osd_bits")
        if not isfinite(min_sum_scale) or not 0 < min_sum_scale <= 1:
            raise ValueError("min_sum_scale must be in (0, 1]")
        if not isfinite(damping) or not 0 <= damping < 1:
            raise ValueError("damping must be in [0, 1)")
        if not isfinite(llr_clip) or not 0 < llr_clip <= 1_000_000:
            raise ValueError("llr_clip must be in (0, 1000000]")
        if fallback not in ("none", "osd0"):
            raise ValueError("fallback must be 'none' or 'osd0'")
        self.min_sum_scale = float(min_sum_scale)
        self.damping, self.llr_clip = float(damping), float(llr_clip)
        self.fallback = fallback
        self.probabilities = _probabilities(probabilities, self.n)
        self.prior_pins = tuple(0 if p == 0 else 1 if p == 1 else None
                                for p in self.probabilities)
        self.priors = tuple(self.llr_clip if p == 0 else -self.llr_clip if p == 1
                            else self._clip(log1p(-p) - log(p)) for p in self.probabilities)
        self.supports = (tuple(tuple(set_bits(row)) for row in checks.rows)
                         if isinstance(checks, BinaryMatrix) else checks.rows)
        edges_by_check, edge_variables, edge_checks = [], [], []
        edges_by_variable = [[] for _ in range(self.n)]
        for check, support in enumerate(self.supports):
            edges = []
            for site in support:
                edge = len(edge_variables)
                edges.append(edge)
                edge_variables.append(site)
                edge_checks.append(check)
                edges_by_variable[site].append(edge)
            edges_by_check.append(tuple(edges))
        self.edges_by_check = tuple(edges_by_check)
        self.edge_variables = tuple(edge_variables)
        self.edge_checks = tuple(edge_checks)
        self.edges_by_variable = tuple(tuple(edges) for edges in edges_by_variable)

    @property
    def configuration(self):
        """JSON-compatible parameters sufficient to reconstruct this decoder."""
        return {"probabilities": self.probabilities, "max_iterations": self.max_iterations,
                "min_sum_scale": self.min_sum_scale, "damping": self.damping,
                "llr_clip": self.llr_clip, "fallback": self.fallback,
                "max_osd_bits": self.max_osd_bits, "schedule": "parallel",
                "max_variables": self.max_variables, "max_checks": self.max_checks,
                "max_edges": self.max_edges}

    def _clip(self, value):
        return max(-self.llr_clip, min(self.llr_clip, value))

    def _syndrome(self, bits):
        return sum((sum(bits[j] for j in support) % 2) << i
                   for i, support in enumerate(self.supports))

    def _result(self, status, iterations, bits, syndrome, llrs, *, error=None):
        word = sum(bit << j for j, bit in enumerate(bits))
        return BPResult(error, status, iterations, word,
                        syndrome ^ self._syndrome(bits), tuple(llrs))

    def _reduce_checks(self, syndrome):
        """Substitute deterministic priors and propagate all unit equations."""
        target = [(syndrome >> i) & 1 for i in range(self.m)]
        fixed = list(self.prior_pins)
        degree = [len(row) for row in self.supports]
        for site, bit in enumerate(fixed):
            if bit is not None:
                for edge in self.edges_by_variable[site]:
                    check = self.edge_checks[edge]
                    target[check] ^= bit
                    degree[check] -= 1
        pending = deque(i for i, count in enumerate(degree) if count <= 1)
        while pending:
            check = pending.popleft()
            if degree[check] == 0:
                if target[check]:
                    return target, fixed, True
                continue
            site = next(j for j in self.supports[check] if fixed[j] is None)
            fixed[site] = bit = target[check]
            for edge in self.edges_by_variable[site]:
                other = self.edge_checks[edge]
                target[other] ^= bit
                degree[other] -= 1
                if degree[other] <= 1:
                    pending.append(other)
        return target, fixed, False

    def decode(self, syndrome: int) -> BPResult:
        if type(syndrome) is not int or not 0 <= syndrome < 1 << self.m:
            raise ValueError("syndrome must fit the number of checks")
        target, fixed, contradiction = self._reduce_checks(syndrome)
        failure_status = ("inconsistent_priors" if any(p is not None for p in self.prior_pins)
                          else "inconsistent")
        llrs = [prior if bit is None else -self.llr_clip if bit else self.llr_clip
                for prior, bit in zip(self.priors, fixed)]
        bits = [int(v < 0) for v in llrs]
        if contradiction:
            return self._result(failure_status, 0, bits, syndrome, llrs)
        if self._syndrome(bits) == syndrome:
            word = sum(bit << j for j, bit in enumerate(bits))
            return self._result("converged", 0, bits, syndrome, llrs, error=word)
        active_checks = tuple(tuple(edge for edge in edges
                                    if fixed[self.edge_variables[edge]] is None)
                              for edges in self.edges_by_check)
        variable_messages = [llrs[site] for site in self.edge_variables]
        check_messages = [0.0] * len(variable_messages)
        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            for check, edges in enumerate(active_checks):
                minimum = second = self.llr_clip
                minimum_edge = -1
                negative = target[check]
                for edge in edges:
                    value = variable_messages[edge]
                    negative ^= value < 0
                    magnitude = abs(value)
                    if magnitude < minimum:
                        second, minimum, minimum_edge = minimum, magnitude, edge
                    elif magnitude < second:
                        second = magnitude
                for edge in edges:
                    magnitude = second if edge == minimum_edge else minimum
                    if len(edges) > 1:
                        magnitude *= self.min_sum_scale
                    sign = negative ^ (variable_messages[edge] < 0)
                    fresh = -magnitude if sign else magnitude
                    check_messages[edge] = self._clip(
                        self.damping * check_messages[edge] + (1-self.damping) * fresh)
            for site, edges in enumerate(self.edges_by_variable):
                if fixed[site] is not None:
                    continue
                total = self.priors[site] + sum(check_messages[edge] for edge in edges)
                llrs[site] = self._clip(total)
                bits[site] = int(llrs[site] < 0)
                for edge in edges:
                    variable_messages[edge] = self._clip(total - check_messages[edge])
            if self._syndrome(bits) == syndrome:
                word = sum(bit << j for j, bit in enumerate(bits))
                return self._result("converged", iteration, bits, syndrome, llrs, error=word)
        if self.fallback == "none":
            return self._result("not_converged", iteration, bits, syndrome, llrs)
        free_sites = tuple(site for site, bit in enumerate(fixed) if bit is None)
        if len(free_sites) * self.m > self.max_osd_bits:
            return self._result("fallback_limit", iteration, bits, syndrome, llrs)
        return self._osd0(syndrome, bits, llrs, iteration, free_sites, failure_status)

    def _osd0(self, syndrome, bits, llrs, iterations, free_sites, failure_status):
        order = sorted(free_sites, key=lambda site: (abs(llrs[site]), site))
        inverse = {}
        for j, site in enumerate(order):
            inverse[site] = j
        permuted = BinaryMatrix(tuple(sum(1 << inverse[site] for site in support
                                          if site in inverse)
                                      for support in self.supports), len(order))
        try:
            delta = permuted.solve(syndrome ^ self._syndrome(bits))
        except ValueError:
            return self._result(failure_status, iterations, bits, syndrome, llrs)
        corrected = list(bits)
        for j in set_bits(delta):
            corrected[order[j]] ^= 1
        if self._syndrome(corrected) != syndrome:
            raise ArithmeticError("order-zero fallback did not satisfy the exact syndrome")
        word = sum(bit << j for j, bit in enumerate(corrected))
        return self._result("osd0", iterations, corrected, syndrome, llrs, error=word)


@dataclass(frozen=True)
class CSSBPResult:
    x: BPResult
    z: BPResult

    @property
    def success(self) -> bool:
        return self.x.success and self.z.success


class CSSBPDecoder:
    """Independent CSS sectors, reusing both adjacency graphs between calls.

    Inputs are marginal binary probabilities: p_X+p_Y for the X component and
    p_Z+p_Y for the Z component. Correlations between components are ignored.
    ``decode(sx, sz)`` uses sx=Hx z and sz=Hz x.
    """
    def __init__(self, code: CSSCode | SparseCSSCode, x_probabilities=0.1,
                 z_probabilities=0.1, **options):
        if not isinstance(code, (CSSCode, SparseCSSCode)):
            raise TypeError("CSSBPDecoder requires a CSSCode or SparseCSSCode")
        self.code = code
        self.x = BPDecoder(code.hz, x_probabilities, **options)
        self.z = BPDecoder(code.hx, z_probabilities, **options)

    name = "css_normalized_min_sum"

    @property
    def configuration(self):
        return {"x": self.x.configuration, "z": self.z.configuration}

    def decode(self, sx: int, sz: int) -> CSSBPResult:
        return CSSBPResult(self.x.decode(sz), self.z.decode(sx))
