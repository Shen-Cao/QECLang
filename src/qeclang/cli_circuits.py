"""CLI for explicitly modeled noisy CSS memory experiments."""
from pathlib import Path

from qeclang.circuits import CircuitLimits, CircuitNoise, css_memory
from qeclang.circuits.stim_backend import StimLimits, compile_stim
from qeclang.circuits.stim_sampling import simulate_circuit_memory
from qeclang.codes import CSSCode
from qeclang.constructions import shor, steane, toric
from qeclang.io import load


def register(commands):
    parser = commands.add_parser("circuit-memory", help="simulate a sequential CSS extraction circuit with Stim")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--code", choices=("steane", "shor", "toric"), default="steane")
    source.add_argument("--input", help="CSS code JSON")
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--basis", choices=("x", "z"), default="z")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-error", type=float, default=0)
    parser.add_argument("--reset-error", type=float, default=0)
    parser.add_argument("--one-qubit-error", type=float, default=0)
    parser.add_argument("--two-qubit-error", type=float, default=0)
    parser.add_argument("--measurement-error", type=float, default=0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--fallback", choices=("none", "osd0"), default="none")
    parser.add_argument("--max-operations", type=int, default=1_000_000)
    parser.add_argument("--max-tableau-bits", type=int, default=10_000_000)
    parser.add_argument("--max-dem-work", type=int, default=50_000_000)
    parser.add_argument("--max-batch-bytes", type=int, default=10_000_000)
    parser.add_argument("--stim-output", help="optional exact compiled circuit artifact")


def run(args):
    code = load(args.input) if args.input else {
        "steane": steane, "shor": shor, "toric": lambda: toric(args.size)}[args.code]()
    if not isinstance(code, CSSCode):
        raise ValueError("circuit memory requires a CSS code")
    noise = CircuitNoise(data_depolarization=args.data_error, reset_flip=args.reset_error,
                         one_qubit_depolarization=args.one_qubit_error,
                         two_qubit_depolarization=args.two_qubit_error,
                         measurement_flip=args.measurement_error)
    ir = css_memory(code, basis=args.basis, rounds=args.rounds, noise=noise,
                    limits=CircuitLimits(max_operations=args.max_operations))
    if not ir.observables:
        raise ValueError("memory experiment requires at least one logical observable")
    artifact = compile_stim(ir, limits=StimLimits(max_tableau_bits=args.max_tableau_bits,
                                                  max_detector_model_work=args.max_dem_work))
    result = simulate_circuit_memory(artifact, trials=args.trials, seed=args.seed,
                                     batch_size=args.batch_size, max_batch_bytes=args.max_batch_bytes,
                                     decoder_options={"max_iterations": args.iterations,
                                                      "fallback": args.fallback})
    if args.stim_output:
        Path(args.stim_output).write_text(artifact.text, encoding="utf-8")
    return result.summary()
