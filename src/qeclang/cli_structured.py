"""CLI commands for replayable symbolic construction analysis."""
from dataclasses import asdict

from qeclang.analysis.structured import analyze_lifted
from qeclang.constructions.family import CyclicLiftFamily
from qeclang.constructions import records


def register(commands):
    for name in ("construction", "family"):
        command = commands.add_parser(name, help="analyze a structured manifest without implicit expansion")
        command.add_argument("manifest")
        if name == "family":
            command.add_argument("--lift", type=int, required=True)
        command.add_argument("--backend", choices=("crt", "polynomial", "packed"), default="crt")
        command.add_argument("--max-degree", type=int, default=512)
        command.add_argument("--max-steps", type=int, default=10_000_000)
        command.add_argument("--max-bits", type=int, default=100_000_000)
        command.add_argument("--max-polynomial-degree", type=int, default=4096)
        command.add_argument("--max-operations", type=int, default=100_000)
        command.add_argument("--certificate-trace", action="store_true")


def run(args):
    family = None
    if args.command == "family":
        family = CyclicLiftFamily.load(args.manifest)
        construction = family.instantiate(args.lift)
    else:
        construction = records.load(args.manifest)
    result = analyze_lifted(construction, backend=args.backend, max_degree=args.max_degree,
                            max_steps=args.max_steps, max_bits=args.max_bits,
                            max_polynomial_degree=args.max_polynomial_degree,
                            max_operations=args.max_operations,
                            record_operations=args.certificate_trace)
    data = {"parameters": asdict(result), "rate": result.rate,
            "locality": asdict(construction.locality())}
    if family is not None:
        data.update(family_fingerprint=family.fingerprint(), lift=args.lift,
                    uniform_locality_bound=asdict(family.locality_bound()))
    return data
