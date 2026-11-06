"""Run with python examples/structured_codes.py after installing qeclang."""
from dataclasses import asdict
import json

from qeclang.analysis import css_distance, css_profile
from qeclang.constructions import bicycle, hypergraph_product, repetition_check
from qeclang.symbolic import GroupAlgebraElement as Polynomial


def main():
    product = hypergraph_product(repetition_check(3), repetition_check(4))
    a = Polynomial.from_terms((3, 3), [(0, 0), (1, 0), (0, 1)])
    b = Polynomial.from_terms((3, 3), [(0, 0), (2, 0), (0, 2)])
    for code in (product, bicycle(a, b)):
        result = {"name": code.name, "profile": asdict(css_profile(code)),
                  "distance": {s: asdict(css_distance(code, s, max_weight=3))
                               for s in ("x", "z")}}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
