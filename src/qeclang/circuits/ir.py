"""Validated, dependency-free Clifford/Pauli-noise circuit records."""
from dataclasses import InitVar, asdict, dataclass, fields
from hashlib import sha256
import json
from math import isfinite
from numbers import Real

from qeclang.algebra.binary import nonnegative_integer
from qeclang.codes import StabilizerCode

ARITIES = {"R": 1, "H": 1, "CX": 2, "M": 1, "MX": 1, "X_ERROR": 1,
           "Z_ERROR": 1, "DEPOLARIZE1": 1, "DEPOLARIZE2": 2, "TICK": 0}
NOISE_OPERATIONS = frozenset(("X_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"))


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("noise probability must be finite and in [0,1]")
    return float(value)


def _bounded_tuple(values, maximum, name):
    if hasattr(values, "__len__") and len(values) > maximum:
        raise ValueError(f"{name} exceeds its configured limit")
    result = []
    for value in values:
        if len(result) >= maximum:
            raise ValueError(f"{name} exceeds its configured limit")
        result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class Operation:
    name: str
    targets: tuple[int, ...] = ()
    probability: float | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or self.name not in ARITIES:
            raise ValueError("unsupported circuit operation")
        targets = _bounded_tuple(self.targets, ARITIES[self.name], "operation arity")
        if len(targets) != ARITIES[self.name]:
            raise ValueError("operation has incorrect arity")
        targets = tuple(nonnegative_integer(t, "qubit target") for t in targets)
        if len(set(targets)) != len(targets):
            raise ValueError("operation targets must be distinct")
        object.__setattr__(self, "targets", targets)
        if self.name in NOISE_OPERATIONS:
            object.__setattr__(self, "probability", _probability(self.probability))
        elif self.probability is not None:
            raise ValueError("ideal operation cannot carry a noise probability")


@dataclass(frozen=True)
class CircuitLimits:
    max_qubits: int = 100_000
    max_operations: int = 1_000_000
    max_gate_targets: int = 2_000_000
    max_measurements: int = 100_000
    max_detectors: int = 100_000
    max_observables: int = 100_000
    max_references: int = 1_000_000
    max_preparation_generators: int = 100_000
    max_preparation_bits: int = 100_000_000

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name,
                               nonnegative_integer(getattr(self, field.name), field.name))


@dataclass(frozen=True)
class CircuitMetadata:
    basis: str
    rounds: int
    code_fingerprint: str
    code_name: str = ""
    noise: tuple[tuple[str, float], ...] = ()
    preparation_semantics: str = "ideal full stabilizer state; no encoder circuit"
    builder: str = "css_memory.v1"

    def __post_init__(self):
        if self.basis not in ("x", "z"):
            raise ValueError("memory basis must be x or z")
        rounds = nonnegative_integer(self.rounds, "rounds")
        if rounds < 1:
            raise ValueError("memory rounds must be positive")
        object.__setattr__(self, "rounds", rounds)
        if any(not isinstance(value, str) for value in
               (self.code_fingerprint, self.code_name, self.preparation_semantics, self.builder)):
            raise ValueError("circuit metadata labels must be strings")
        noise = tuple((name, _probability(value)) for name, value in self.noise)
        if any(not isinstance(name, str) for name, _ in noise) or len({n for n, _ in noise}) != len(noise):
            raise ValueError("noise metadata requires unique string names")
        object.__setattr__(self, "noise", noise)


@dataclass(frozen=True)
class CircuitIR:
    """Ideal data preparation followed by operations and measurement parities.

    Measurement IDs are absolute and zero-based in M/MX encounter order.
    Detectors and observables are canonical parity supports, with no repeated
    reference. Empty parities represent constant zero. Ancillas start in |0>.
    Limits bound record sizes, not simulator tableaux or compiler workspace.
    """
    n_qubits: int
    n_data: int
    preparation: StabilizerCode
    operations: tuple[Operation, ...]
    detectors: tuple[tuple[int, ...], ...] = ()
    observables: tuple[tuple[int, ...], ...] = ()
    metadata: CircuitMetadata | None = None
    limits: InitVar[CircuitLimits] = CircuitLimits()

    def __post_init__(self, limits):
        if not isinstance(limits, CircuitLimits):
            raise TypeError("limits must be CircuitLimits")
        nq = nonnegative_integer(self.n_qubits, "n_qubits")
        nd = nonnegative_integer(self.n_data, "n_data")
        if nd > nq or nq > limits.max_qubits:
            raise ValueError("circuit qubit dimensions exceed limits")
        if not isinstance(self.preparation, StabilizerCode) or self.preparation.n != nd:
            raise ValueError("preparation must be a stabilizer state on n_data qubits")
        if len(self.preparation.generators) > limits.max_preparation_generators:
            raise ValueError("preparation exceeds max_preparation_generators")
        if len(self.preparation.generators) * (2 * nd + 2) > limits.max_preparation_bits:
            raise ValueError("preparation exceeds max_preparation_bits")
        if self.preparation.k != 0:
            raise ValueError("preparation must specify a full-rank stabilizer state")
        if hasattr(self.operations, "__len__") and len(self.operations) > limits.max_operations:
            raise ValueError("operations exceeds its configured limit")
        operations = []
        measurements = targets = 0
        for operation in self.operations:
            if len(operations) >= limits.max_operations:
                raise ValueError("operations exceeds its configured limit")
            if not isinstance(operation, Operation):
                raise TypeError("operations must be Operation objects")
            targets += len(operation.targets)
            measurements += operation.name in ("M", "MX")
            if any(t >= nq for t in operation.targets):
                raise ValueError("operation target exceeds n_qubits")
            if targets > limits.max_gate_targets or measurements > limits.max_measurements:
                raise ValueError("gate targets or measurements exceed circuit limits")
            operations.append(operation)
        references = 0

        def parities(values, maximum, name):
            nonlocal references
            if hasattr(values, "__len__") and len(values) > maximum:
                raise ValueError(f"{name} exceeds its configured limit")
            result = []
            for record in values:
                if len(result) >= maximum:
                    raise ValueError(f"{name} exceeds its configured limit")
                row = _bounded_tuple(record, limits.max_references - references, "measurement references")
                row = tuple(nonnegative_integer(value, "measurement reference") for value in row)
                if len(set(row)) != len(row) or any(value >= measurements for value in row):
                    raise ValueError("measurement references must be distinct and in range")
                references += len(row)
                result.append(tuple(sorted(row)))
            return tuple(result)

        object.__setattr__(self, "n_qubits", nq)
        object.__setattr__(self, "n_data", nd)
        object.__setattr__(self, "operations", tuple(operations))
        object.__setattr__(self, "detectors", parities(self.detectors, limits.max_detectors, "detectors"))
        object.__setattr__(self, "observables", parities(self.observables, limits.max_observables, "observables"))
        if self.metadata is not None and not isinstance(self.metadata, CircuitMetadata):
            raise TypeError("metadata must be CircuitMetadata or None")

    @property
    def measurement_count(self):
        return sum(operation.name in ("M", "MX") for operation in self.operations)

    def fingerprint(self):
        record = {"schema": "qeclang.circuit.v1", "n_qubits": self.n_qubits, "n_data": self.n_data,
                  "preparation": [(p.x, p.z, p.phase) for p in self.preparation.generators],
                  "operations": [asdict(operation) for operation in self.operations],
                  "detectors": self.detectors, "observables": self.observables,
                  "metadata": None if self.metadata is None else asdict(self.metadata)}
        text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(text.encode("utf-8")).hexdigest()


__all__ = ["Operation", "CircuitLimits", "CircuitMetadata", "CircuitIR"]
