"""Emit reproducible parameterized timing observations, not stored benchmarks."""
import argparse
import json
import platform
from time import perf_counter

from qeclang import __version__
from qeclang.analysis import css_profile
from qeclang.constructions import toric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[3, 5, 10, 20])
    args = parser.parse_args()
    for size in args.sizes:
        start = perf_counter()
        code = toric(size)
        constructed = perf_counter()
        profile = css_profile(code)
        finished = perf_counter()
        print(json.dumps({"qeclang_version": __version__, "python": platform.python_version(),
                          "family": "toric", "size": size, "n": code.n, "k": profile.k,
                          "construction_seconds": constructed-start,
                          "analysis_seconds": finished-constructed,
                          "max_check_weight": profile.max_check_weight,
                          "max_qubit_degree": profile.max_qubit_degree}))


if __name__ == "__main__":
    main()
