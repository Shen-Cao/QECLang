"""CLI entry points for atomic, resumable construction search studies."""
from dataclasses import asdict

from qeclang.experiments.study import StudyConfig, read_study, run_study
from qeclang.cli_structured import add_analysis_arguments, analysis_options


def register(commands):
    command = commands.add_parser("study", help="resume a bounded structured search to a total trial count")
    command.add_argument("database")
    command.add_argument("--trials", type=int, default=10)
    command.add_argument("--lift", type=int, default=7)
    command.add_argument("--terms", type=int, default=3)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--strategy", choices=("random", "mutate"), default="random")
    command.add_argument("--restart-interval", type=int, default=10)
    add_analysis_arguments(command, max_bits=10_000_000)
    command.add_argument("--max-weight", type=int, default=2)
    command.add_argument("--max-candidates", type=int, default=100_000)
    command.add_argument("--max-distance-n", type=int, default=128)
    inspect = commands.add_parser("study-results", help="read completed study results without mutation")
    inspect.add_argument("database")


def run(args):
    if args.command == "study-results":
        return asdict(read_study(args.database))
    config = StudyConfig(lift=args.lift, terms=args.terms, seed=args.seed,
                         strategy=args.strategy, restart_interval=args.restart_interval,
                         **analysis_options(args), max_weight=args.max_weight,
                         max_candidates=args.max_candidates, max_distance_n=args.max_distance_n)
    return asdict(run_study(args.database, config, target_trials=args.trials))
