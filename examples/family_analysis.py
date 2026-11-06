"""Analyze a cyclic family symbolically; optionally save its replayable manifest."""
import argparse
from dataclasses import asdict
import json

from qeclang.analysis.structured import analyze_lifted
from qeclang.constructions.family import CyclicLiftFamily, LaurentMatrix


def example_family():
    a = LaurentMatrix((((0,), (1,), ()), ((), (0,), (-1,))), 3)
    b = LaurentMatrix((((0,), (2,)),), 2)
    return CyclicLiftFamily(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifts", nargs="+", type=int, default=[3, 7, 31, 127])
    parser.add_argument("--manifest", help="optional output JSON path")
    args = parser.parse_args()
    family = example_family()
    if args.manifest:
        family.dump(args.manifest)
    for lift in args.lifts:
        result = analyze_lifted(family.instantiate(lift))
        print(json.dumps({"family": family.fingerprint(), "lift": lift,
                          "parameters": asdict(result),
                          "uniform_locality": asdict(family.locality_bound())}))


if __name__ == "__main__":
    main()
