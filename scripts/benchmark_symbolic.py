"""Compare exact symbolic rank and explicit binary expansion at requested lifts."""
import argparse
import json
from random import Random
from time import perf_counter

from qeclang.symbolic import GroupAlgebraElement as G, GroupAlgebraMatrix as M
from qeclang.symbolic.cyclic_analysis import cyclic_rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifts", nargs="+", type=int, default=[7, 31, 63, 127])
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--terms", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.rows, args.columns, args.terms) < 0:
        parser.error("dimensions and term count must be nonnegative")
    for lift in args.lifts:
        if lift < 1 or lift % 2 == 0 or args.terms > lift:
            parser.error("lifts must be positive odd and at least the term count")
        rng = Random(args.seed)
        rows = tuple(tuple(G.from_terms((lift,), ((e,) for e in rng.sample(range(lift), args.terms)))
                           for _ in range(args.columns)) for _ in range(args.rows))
        matrix = M(rows, args.columns, (lift,))
        start = perf_counter()
        symbolic = cyclic_rank(matrix)
        symbolic_time = perf_counter() - start
        start = perf_counter()
        binary = matrix.lower()
        expanded_rank = binary.rank()
        expanded_time = perf_counter() - start
        assert symbolic.rank == expanded_rank
        print(json.dumps({"lift": lift, "rows": args.rows, "columns": args.columns,
                          "terms": args.terms, "seed": args.seed,
                          "rank": expanded_rank, "symbolic_seconds": symbolic_time,
                          "expansion_and_rank_seconds": expanded_time,
                          "symbolic_steps": symbolic.steps,
                          "factor_degrees": [component.degree for component in symbolic.components]}))


if __name__ == "__main__":
    main()
