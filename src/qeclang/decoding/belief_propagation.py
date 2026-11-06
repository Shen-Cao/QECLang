"""Sparse normalized min-sum syndrome decoding with optional OSD order zero.

The BP/OSD design follows the algorithmic framework of Roffe et al.,
https://arxiv.org/abs/2005.07016 ; this is an independent reference implementation.
Parameters use normalized min-sum conventions documented by LDPC:
https://software.roffe.eu/ldpc/ldpc/bp_decoder.html . No likelihood optimality or
quantum coset maximum-likelihood guarantee is implied.
"""
from dataclasses import dataclass
from math import isfinite, log, log1p
from numbers import Real
from typing import Literal

from qeclang.algebra import BinaryMatrix
from qeclang.algebra.binary import set_bits
from qeclang.codes import CSSCode, SparseCSSCode


@dataclass(frozen=True)
class BPResult:
    error: int | None
    status: Literal["converged", "osd0", "not_converged", "inconsistent", "fallback_limit"]
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
        if not isfinite(probability) or not 0 < probability < 1:
            raise ValueError("probability must be finite and strictly in (0, 1)")
        values = (probability,) * n
    else:
        values = tuple(map(float, value))
    if len(values) != n or any(not isfinite(p) or not 0 < p < 1 for p in values):
        raise ValueError("probabilities must provide one finite value strictly in (0, 1) per site")
    return values


class BPDecoder:
    """Reusable adjacency lists for bounded normalized min-sum belief propagation.

    ``probabilities`` is a scalar or per-site binary error probabilities.
    ``damping`` is the fraction of the *old check-to-variable message* retained;
    zero gives undamped parallel updates. ``min_sum_scale`` scales each new
    min-sum message. All LLRs are saturated to ``llr_clip`` (at most 1e6).
    Singleton checks fix bits exactly. A zero-valued LLR chooses bit zero.

    No dense elimination is performed unless ``fallback='osd0'`` is requested
    and BP fails. That fallback is bounded by ``max_osd_bits`` per packed matrix,
    and retains the most reliable free hard decisions while solving less
    reliable pivot corrections. It is not minimum-weight or maximum-likelihood
    optimization. The original generator presentation is retained for BP.
    """

    def __init__(self, checks, probabilities=0.1, *, max_iterations=50,
                 min_sum_scale=0.8, damping=0.0, llr_clip=50.0,
                 fallback="none", max_osd_bits=10_000_000):
        from qeclang.algebra.sparse import SparseBinaryMatrix
        if not isinstance(checks, (BinaryMatrix, SparseBinaryMatrix)):
            raise TypeError("checks must be a BinaryMatrix or SparseBinaryMatrix")
        self.checks = checks
        self.n, self.m = checks.ncols, checks.nrows
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
        self.priors = tuple(self._clip(log1p(-p) - log(p)) for p in self.probabilities)
        self.supports = (tuple(tuple(set_bits(row)) for row in checks.rows)
                         if isinstance(checks, BinaryMatrix) else checks.rows)
        edges_by_check, edge_variables = [], []
        edges_by_variable = [[] for _ in range(self.n)]
        for support in self.supports:
            edges = []
            for site in support:
                edge = len(edge_variables)
                edges.append(edge)
                edge_variables.append(site)
                edges_by_variable[site].append(edge)
            edges_by_check.append(tuple(edges))
        self.edges_by_check = tuple(edges_by_check)
        self.edge_variables = tuple(edge_variables)
        self.edges_by_variable = tuple(tuple(edges) for edges in edges_by_variable)

    @property
    def configuration(self):
        """JSON-compatible parameters sufficient to reconstruct this decoder."""
        return {"probabilities": self.probabilities, "max_iterations": self.max_iterations,
                "min_sum_scale": self.min_sum_scale, "damping": self.damping,
                "llr_clip": self.llr_clip, "fallback": self.fallback,
                "max_osd_bits": self.max_osd_bits, "schedule": "parallel"}

    def _clip(self, value):
        return max(-self.llr_clip, min(self.llr_clip, value))

    def _syndrome(self, bits):
        return sum((sum(bits[j] for j in support) % 2) << i
                   for i, support in enumerate(self.supports))

    def _result(self, status, iterations, bits, syndrome, llrs, *, error=None):
        word = sum(bit << j for j, bit in enumerate(bits))
        return BPResult(error, status, iterations, word,
                        syndrome ^ self._syndrome(bits), tuple(llrs))

    def decode(self, syndrome: int) -> BPResult:
        if type(syndrome) is not int or not 0 <= syndrome < 1 << self.m:
            raise ValueError("syndrome must fit the number of checks")
        target = tuple((syndrome >> i) & 1 for i in range(self.m))
        pinned = {}
        llrs = list(self.priors)
        for check, support in enumerate(self.supports):
            if not support and target[check]:
                bits = [int(v < 0) for v in llrs]
                return self._result("inconsistent", 0, bits, syndrome, llrs)
            if len(support) == 1:
                site = support[0]
                if site in pinned and pinned[site] != target[check]:
                    bits = [int(v < 0) for v in llrs]
                    return self._result("inconsistent", 0, bits, syndrome, llrs)
                pinned[site] = target[check]
                llrs[site] = -self.llr_clip if target[check] else self.llr_clip
        bits = [int(v < 0) for v in llrs]
        if self._syndrome(bits) == syndrome:
            word = sum(bit << j for j, bit in enumerate(bits))
            return self._result("converged", 0, bits, syndrome, llrs, error=word)
        variable_messages = [llrs[site] for site in self.edge_variables]
        check_messages = [0.0] * len(variable_messages)
        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            for check, edges in enumerate(self.edges_by_check):
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
                total = self.priors[site] + sum(check_messages[edge] for edge in edges)
                llrs[site] = self._clip(total)
                if site in pinned:
                    llrs[site] = -self.llr_clip if pinned[site] else self.llr_clip
                bits[site] = int(llrs[site] < 0)
                for edge in edges:
                    variable_messages[edge] = (llrs[site] if site in pinned else
                                               self._clip(total - check_messages[edge]))
            if self._syndrome(bits) == syndrome:
                word = sum(bit << j for j, bit in enumerate(bits))
                return self._result("converged", iteration, bits, syndrome, llrs, error=word)
        if self.fallback == "none":
            return self._result("not_converged", iteration, bits, syndrome, llrs)
        if self.n * self.m > self.max_osd_bits:
            return self._result("fallback_limit", iteration, bits, syndrome, llrs)
        return self._osd0(syndrome, bits, llrs, iteration)

    def _osd0(self, syndrome, bits, llrs, iterations):
        order = sorted(range(self.n), key=lambda site: (abs(llrs[site]), site))
        inverse = [0] * self.n
        for j, site in enumerate(order):
            inverse[site] = j
        permuted = BinaryMatrix(tuple(sum(1 << inverse[site] for site in support)
                                      for support in self.supports), self.n)
        try:
            delta = permuted.solve(syndrome ^ self._syndrome(bits))
        except ValueError:
            return self._result("inconsistent", iterations, bits, syndrome, llrs)
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
