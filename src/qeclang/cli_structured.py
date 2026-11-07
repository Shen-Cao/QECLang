"""CLI commands for replayable symbolic construction analysis."""
from dataclasses import asdict

from qeclang.analysis.structured import BACKENDS, analyze_lifted
from qeclang.constructions.family import CyclicLiftFamily
from qeclang.constructions import records

LIMITS = {
    "max_degree": (512, "maximum cyclic modulus degree (crt/polynomial)"),
    "max_steps": (10_000_000, "total symbolic rank arithmetic steps across X and Z"),
    "max_entries": (1_000_000, "symbolic rank matrix axes and entries"),
    "max_blocks": (1_000_000, "symbolic check and identity block payload"),
    "max_sites": (1_000_000, "each expanded packed axis"),
    "max_bits": (100_000_000, "packed checks plus CSS validation payload"),
    "max_polynomial_degree": (4096, "polynomial intermediate degree"),
    "max_operations": (100_000, "total polynomial elementary operations"),
    "max_characters": (10_000, "odd character count for abelian backends"),
    "max_field_degree": (64, "splitting-field degree for abelian backends"),
    "max_field_candidates": (4096, "candidates per irreducible/root search"),
    "max_factors": (16, "cyclic factors for abelian backends"),
    "max_component_entries": (1_000_000, "mixed-abelian component entries"),
    "max_component_axis": (10_000, "each mixed-abelian component axis"),
}


def add_analysis_arguments(command, *, max_bits=100_000_000, trace=False):
    command.add_argument("--backend", choices=BACKENDS, default="crt")
    for name, (default, description) in LIMITS.items():
        command.add_argument("--" + name.replace("_", "-"), type=int,
                             default=max_bits if name == "max_bits" else default, help=description)
    if trace:
        command.add_argument("--certificate-trace", action="store_true",
                             help="retain polynomial operation traces (polynomial backend only)")


def analysis_options(args):
    return {"backend": args.backend, **{name: getattr(args, name) for name in LIMITS}}


def register(commands):
    for name in ("construction", "family"):
        command = commands.add_parser(name, help="analyze a structured manifest without implicit expansion")
        command.add_argument("manifest")
        if name == "family":
            command.add_argument("--lift", type=int, required=True)
        add_analysis_arguments(command, trace=True)


def run(args):
    family = None
    if args.command == "family":
        family = CyclicLiftFamily.load(args.manifest)
        construction = family.instantiate(args.lift)
    else:
        construction = records.load(args.manifest)
    result = analyze_lifted(construction, **analysis_options(args),
                            record_operations=args.certificate_trace)
    data = {"parameters": asdict(result), "rate": result.rate,
            "locality": asdict(construction.locality())}
    if family is not None:
        data.update(family_fingerprint=family.fingerprint(), lift=args.lift,
                    uniform_locality_bound=asdict(family.locality_bound()))
    return data
