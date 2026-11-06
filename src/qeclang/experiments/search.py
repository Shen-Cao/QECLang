"""Reproducible structured candidate generation, with bounded evaluation."""
from dataclasses import asdict
from itertools import product
from math import prod
from random import Random

from qeclang import __version__
from qeclang.analysis import css_distance, css_profile
from qeclang.analysis.distance import _limits
from qeclang.constructions import bicycle
from qeclang.io import to_dict
from qeclang.symbolic import GroupAlgebraElement


def bicycle_candidates(orders=(3, 3), *, terms=3, trials=10, seed=0,
                       max_weight=2, max_candidates=10_000, max_bits=10_000_000):
    """Yield independently checkable candidates, including unsuccessful ones.

    Sampling is without replacement within a polynomial, with replacement
    between trials; no code-equivalence classification is claimed. All candidate
    parameters and RNG seed are recorded. This is a baseline search harness,
    not a proof of optimality or an autonomous theorem prover.
    """
    orders = tuple(orders)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not orders or any(type(o) is not int or o < 1 for o in orders):
        raise ValueError("orders must be positive integers")
    if type(trials) is not int or trials < 0:
        raise ValueError("trials must be nonnegative")
    if type(terms) is not int or not 0 <= terms <= prod(orders):
        raise ValueError("invalid number of monomials")
    _limits(2 * prod(orders), max_weight, max_candidates)
    if 4 * prod(orders)**2 > max_bits:
        raise ValueError("candidate family exceeds max_bits")
    sites = list(product(*(range(order) for order in orders)))
    rng = Random(seed)
    for trial in range(trials):
        a = GroupAlgebraElement.from_terms(orders, rng.sample(sites, terms))
        b = GroupAlgebraElement.from_terms(orders, rng.sample(sites, terms))
        code = bicycle(a, b, max_bits=max_bits)
        yield {
            "schema": "qeclang.experiment.v1", "qeclang_version": __version__,
            "seed": seed, "trial": trial,
            "construction": {"kind": "abelian_bicycle", "orders": orders,
                             "a": sorted(a.terms), "b": sorted(b.terms)},
            "limits": {"max_weight": max_weight, "max_candidates": max_candidates,
                       "max_bits": max_bits},
            "code": to_dict(code), "profile": asdict(css_profile(code)),
            "distance_x": asdict(css_distance(code, "x", max_weight=max_weight,
                                               max_candidates=max_candidates)),
            "distance_z": asdict(css_distance(code, "z", max_weight=max_weight,
                                               max_candidates=max_candidates)),
        }
