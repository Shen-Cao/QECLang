"""Sequential CSS memory schedules with explicit ideal-state preparation."""
from dataclasses import asdict, dataclass, fields

from qeclang.algebra.binary import nonnegative_integer, set_bits
from qeclang.analysis.logicals import css_logicals
from qeclang.codes import CSSCode, Pauli, StabilizerCode
from qeclang.constructions.records import digest
from qeclang.io import to_dict
from .ir import (CircuitIR, CircuitLimits, CircuitMetadata, Operation, _bounded_tuple,
                 _probability)


@dataclass(frozen=True)
class CircuitNoise:
    """Independent Pauli fault probabilities; preparation itself is perfect."""
    data_depolarization: float = 0.0
    reset_flip: float = 0.0
    one_qubit_depolarization: float = 0.0
    two_qubit_depolarization: float = 0.0
    measurement_flip: float = 0.0

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _probability(getattr(self, field.name)))


def _validate_growth(code, basis, rounds, noise, limits):
    n, mx, mz = code.n, code.hx.nrows, code.hz.nrows
    checks = mx + mz
    reset, single, pair, measurement, data = (
        int(getattr(noise, field) > 0) for field in
        ("reset_flip", "one_qubit_depolarization", "two_qubit_depolarization",
         "measurement_flip", "data_depolarization"))
    edges = code.hx.nnz + code.hz.nnz
    per_round = checks * (2 + reset + measurement) + 2 * mx * (1 + single)
    per_round += edges * (1 + pair) + n * data + 1
    operations = rounds * per_round + n * (1 + measurement)
    gate_targets = operations - rounds + rounds * edges * (1 + pair)
    measured = rounds * checks + n
    same = code.hz if basis == "z" else code.hx
    detectors = rounds * checks + same.nrows
    references = checks * (2 * rounds - 1) + same.nrows + same.nnz
    for count, limit, name in ((n + checks, limits.max_qubits, "qubits"),
                               (operations, limits.max_operations, "operations"),
                               (gate_targets, limits.max_gate_targets, "gate targets"),
                               (measured, limits.max_measurements, "measurements"),
                               (detectors, limits.max_detectors, "detectors"),
                               (references, limits.max_references, "references"),
                               (n, limits.max_preparation_generators, "preparation generators"),
                               (n * (2 * n + 2), limits.max_preparation_bits, "preparation bits")):
        if count > limit:
            raise ValueError(f"CSS memory schedule exceeds {name} limit")
    return references


def css_memory(code: CSSCode, *, basis="z", rounds=1, noise=CircuitNoise(),
               limits=CircuitLimits(), max_basis_bits=100_000_000,
               max_basis_dimension=4096, allowed_edges=None) -> CircuitIR:
    """Prepare all code checks and chosen logical X/Z with positive eigenvalues.

    This is a known stabilizer-state memory experiment, not an encoder for an
    arbitrary input state. One ancilla per retained check is reused each round.
    Checks run sequentially in X-then-Z order. allowed_edges, when supplied,
    is an undirected interaction graph; either CX direction uses the same edge.
    Zero-probability noise operations are omitted. Every round ends with TICK.
    """
    if not isinstance(code, CSSCode):
        raise TypeError("css_memory requires a packed CSSCode with explicit logical analysis")
    if basis not in ("x", "z"):
        raise ValueError("memory basis must be x or z")
    rounds = nonnegative_integer(rounds, "rounds")
    if not rounds:
        raise ValueError("rounds must be positive")
    if not isinstance(noise, CircuitNoise) or not isinstance(limits, CircuitLimits):
        raise TypeError("noise and limits must be CircuitNoise and CircuitLimits")
    bits = nonnegative_integer(max_basis_bits, "max_basis_bits")
    dimension = nonnegative_integer(max_basis_dimension, "max_basis_dimension")
    n, mx, mz = code.n, code.hx.nrows, code.hz.nrows
    if max(n, mx, mz) > dimension or 8 * n * n + 2 * (mx + mz) * (n + 1) > bits:
        raise ValueError("CSS memory logical basis exceeds its materialization budget")
    references = _validate_growth(code, basis, rounds, noise, limits)
    logicals = css_logicals(code)
    selected = logicals.z if basis == "z" else logicals.x
    if len(selected) > limits.max_observables or references + sum(v.bit_count() for v in selected) > limits.max_references:
        raise ValueError("CSS memory observables exceed reference limits")
    prep = tuple(Pauli(row, 0) for row in code.hx.row_basis().rows)
    prep += tuple(Pauli(0, row) for row in code.hz.row_basis().rows)
    prep += tuple(Pauli(0, row) if basis == "z" else Pauli(row, 0) for row in selected)
    preparation = StabilizerCode(n, prep, name=f"{code.name} logical {basis} eigenstate")
    n_qubits = n + mx + mz
    checks = tuple(("x", tuple(set_bits(row))) for row in code.hx.rows)
    checks += tuple(("z", tuple(set_bits(row))) for row in code.hz.rows)
    if allowed_edges is not None:
        graph = set()
        for edge in _bounded_tuple(allowed_edges, limits.max_gate_targets, "allowed edges"):
            values = _bounded_tuple(edge, 2, "interaction edge")
            values = tuple(nonnegative_integer(value, "interaction endpoint") for value in values)
            if len(values) != 2 or values[0] == values[1] or max(values) >= n_qubits:
                raise ValueError("interaction edges need two distinct in-range qubits")
            graph.add(tuple(sorted(values)))
        if any(tuple(sorted((site, n + check))) not in graph
               for check, (_, support) in enumerate(checks) for site in support):
            raise ValueError("check extraction requires an interaction absent from allowed_edges")
    operations, detectors = [], []
    measurement_count = 0
    previous = [None] * len(checks)

    def append(name, targets=(), probability=None):
        operations.append(Operation(name, targets, probability))

    def error(name, targets, probability):
        if probability:
            append(name, targets, probability)

    for _ in range(rounds):
        for site in range(n):
            error("DEPOLARIZE1", (site,), noise.data_depolarization)
        for check, (kind, support) in enumerate(checks):
            ancilla = n + check
            append("R", (ancilla,))
            error("X_ERROR", (ancilla,), noise.reset_flip)
            if kind == "x":
                append("H", (ancilla,))
                error("DEPOLARIZE1", (ancilla,), noise.one_qubit_depolarization)
            for site in support:
                targets = (ancilla, site) if kind == "x" else (site, ancilla)
                append("CX", targets)
                error("DEPOLARIZE2", targets, noise.two_qubit_depolarization)
            if kind == "x":
                append("H", (ancilla,))
                error("DEPOLARIZE1", (ancilla,), noise.one_qubit_depolarization)
            error("X_ERROR", (ancilla,), noise.measurement_flip)
            append("M", (ancilla,))
            current = measurement_count
            measurement_count += 1
            detectors.append((current,) if previous[check] is None else (previous[check], current))
            previous[check] = current
        append("TICK")
    data_start = measurement_count
    for site in range(n):
        error("X_ERROR" if basis == "z" else "Z_ERROR", (site,), noise.measurement_flip)
        append("M" if basis == "z" else "MX", (site,))
    for check, (kind, support) in enumerate(checks):
        if kind == basis:
            detectors.append((previous[check],) + tuple(data_start + site for site in support))
    observables = tuple(tuple(data_start + site for site in set_bits(logical)) for logical in selected)
    metadata = CircuitMetadata(basis, rounds, digest(to_dict(code)), code.name,
                               tuple(asdict(noise).items()))
    return CircuitIR(n_qubits, n, preparation, tuple(operations), tuple(detectors),
                     observables, metadata, limits=limits)


__all__ = ["CircuitNoise", "css_memory"]
