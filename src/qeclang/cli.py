"""Command-line entry points for bounded, reproducible small-code experiments."""
import argparse
from dataclasses import asdict
import json
import sqlite3
import sys

from qeclang.analysis import css_distance, css_profile, stabilizer_distance
from qeclang.codes import CSSCode
from qeclang.constructions import five_qubit, shor, steane, toric
from qeclang.decoding import simulate_css_memory
from qeclang.experiments import bicycle_candidates
from qeclang.io import dump, load


def _code(args):
    if args.input:
        return load(args.input)
    return {"steane": steane, "shor": shor, "five": five_qubit,
            "toric": lambda: toric(args.size)}[args.code]()


def _json(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    from qeclang import cli_circuits, cli_structured, cli_study, cli_workflow
    cli_workflow.register(commands)
    cli_circuits.register(commands)
    cli_structured.register(commands)
    cli_study.register(commands)
    for name in ("inspect", "distance", "export", "simulate"):
        sub = commands.add_parser(name)
        source = sub.add_mutually_exclusive_group()
        source.add_argument("--code", choices=("steane", "shor", "five", "toric"), default="steane")
        source.add_argument("--input", help="versioned code JSON")
        sub.add_argument("--size", type=int, default=3, help="toric lattice side")
        if name in ("distance", "simulate"):
            sub.add_argument("--max-weight", type=int, default=3)
            sub.add_argument("--max-candidates", type=int, default=100_000)
        if name == "distance":
            sub.add_argument("--method", choices=("enumerate", "portfolio"), default="enumerate")
            sub.add_argument("--heuristic-seed", type=int, default=0)
            sub.add_argument("--heuristic-restarts", type=int, default=32)
            sub.add_argument("--heuristic-evaluations", type=int, default=10_000)
            sub.add_argument("--milp", action="store_true")
        if name == "export":
            sub.add_argument("output")
        if name == "simulate":
            sub.add_argument("--probability", type=float, default=0.01)
            sub.add_argument("--trials", type=int, default=1000)
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--decoder", choices=("exhaustive", "bp", "bp-osd0"),
                             default="exhaustive")
            sub.add_argument("--iterations", type=int, default=50)
            sub.add_argument("--min-sum-scale", type=float, default=0.8)
    search = commands.add_parser("search")
    search.add_argument("--orders", nargs="+", type=int, default=[3, 3])
    search.add_argument("--terms", type=int, default=3)
    search.add_argument("--trials", type=int, default=10)
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--max-weight", type=int, default=2)
    search.add_argument("--max-candidates", type=int, default=10_000)
    args = parser.parse_args(argv)
    if args.command == "run":
        return cli_workflow.run(args)
    try:
        if args.command == "circuit-memory":
            _json(cli_circuits.run(args))
            return 0
        if args.command in ("study", "study-results"):
            _json(cli_study.run(args))
            return 0
        if args.command in ("family", "construction"):
            _json(cli_structured.run(args))
            return 0
        if args.command == "search":
            for record in bicycle_candidates(args.orders, terms=args.terms, trials=args.trials,
                                              seed=args.seed, max_weight=args.max_weight,
                                              max_candidates=args.max_candidates):
                print(json.dumps(record, allow_nan=False))
            return 0
        code = _code(args)
        if args.command == "export":
            dump(code, args.output)
        elif args.command == "inspect":
            _json(asdict(css_profile(code)) if isinstance(code, CSSCode) else
                  {"n": code.n, "k": code.k, "name": code.name})
        elif args.command == "distance":
            limits = {"max_weight": args.max_weight, "max_candidates": args.max_candidates}
            if args.milp and args.method != "portfolio":
                raise ValueError("--milp requires --method portfolio")
            if args.method == "portfolio" and not isinstance(code, CSSCode):
                raise ValueError("distance portfolio requires a CSS code")
            if isinstance(code, CSSCode):
                method = css_distance
                if args.method == "portfolio":
                    from qeclang.analysis import css_distance_portfolio
                    method = css_distance_portfolio
                    limits.update(seed=args.heuristic_seed, restarts=args.heuristic_restarts,
                                  max_evaluations=args.heuristic_evaluations, use_milp=args.milp)
                _json({sector: asdict(method(code, sector, **limits)) for sector in ("x", "z")})
            else:
                from qeclang.codes import StabilizerCode
                if not isinstance(code, StabilizerCode):
                    raise ValueError("CLI distance currently supports CSS and stabilizer codes")
                _json(asdict(stabilizer_distance(code, **limits)))
        elif args.command == "simulate":
            if not isinstance(code, CSSCode):
                raise ValueError("simulation requires a CSS code")
            decoder = None
            if args.decoder != "exhaustive":
                from qeclang.decoding import CSSBPDecoder
                marginal = 2 * args.probability / 3
                decoder = CSSBPDecoder(code, marginal, marginal,
                                       max_iterations=args.iterations,
                                       min_sum_scale=args.min_sum_scale,
                                       fallback="osd0" if args.decoder == "bp-osd0" else "none")
            _json(asdict(simulate_css_memory(code, args.probability, trials=args.trials,
                                             seed=args.seed, max_weight=args.max_weight,
                                             max_candidates_per_sector=args.max_candidates,
                                             decoder=decoder)))
    except (ValueError, TypeError, OSError, sqlite3.Error, ImportError) as error:
        print(f"qeclang: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
