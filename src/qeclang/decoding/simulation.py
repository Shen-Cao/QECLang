"""Reproducible ideal-syndrome depolarizing memory experiments."""
from dataclasses import dataclass
from math import sqrt, isfinite
from random import Random
from statistics import NormalDist

from qeclang.codes import CSSCode
from .exhaustive import css_decode, css_logical_success


@dataclass(frozen=True)
class MemoryExperiment:
    trials: int
    failures: int
    decoder_failures: int
    logical_failures: int
    physical_error_probability: float
    seed: int
    failure_rate: float
    confidence: float
    confidence_interval: tuple[float, float]
    distinct_syndromes: int


def wilson_interval(failures: int, trials: int,
                    confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson score interval for independent Bernoulli trials."""
    if (type(trials) is not int or type(failures) is not int
            or not 0 <= failures <= trials or trials == 0):
        raise ValueError("require integer 0 <= failures <= trials and trials > 0")
    if not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    z = -NormalDist().inv_cdf((1 - confidence) / 2)
    p = failures / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    radius = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def simulate_css_memory(code: CSSCode, probability: float, *, trials: int,
                        seed: int, max_weight: int | None = None,
                        max_candidates_per_sector: int = 1_000_000,
                        confidence: float = 0.95) -> MemoryExperiment:
    """IID site noise I:(1-p), X:p/3, Y:p/3, Z:p/3; perfect syndromes.

    Decoder failures count as memory failures and are separately reported.
    This finite-block experiment is not a threshold estimate. Correction
    outcomes are cached by syndrome; budgets apply per distinct syndrome.
    """
    if not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be finite and in [0, 1]")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    wilson_interval(0, trials, confidence)
    # Validate decoder limits even if all trials produce the same syndrome.
    from qeclang.analysis.distance import _limits
    _limits(code.n, max_weight, max_candidates_per_sector)
    rng = Random(seed)
    cache = {}
    logical_failures = decoder_failures = 0
    for _ in range(trials):
        x = z = 0
        for i in range(code.n):
            r = rng.random()
            if r < probability:
                label = rng.randrange(1, 4)
                x |= (label & 1) << i
                z |= (label >> 1) << i
        key = code.hx.matvec(z), code.hz.matvec(x)
        if key not in cache:
            cache[key] = css_decode(code, *key, max_weight=max_weight,
                                    max_candidates_per_sector=max_candidates_per_sector)
        correction = cache[key]
        if not correction.success:
            decoder_failures += 1
        elif not css_logical_success(code, x, z, correction):
            logical_failures += 1
    failures = logical_failures + decoder_failures
    return MemoryExperiment(trials, failures, decoder_failures, logical_failures,
                            probability, seed, failures / trials, confidence,
                            wilson_interval(failures, trials, confidence), len(cache))
