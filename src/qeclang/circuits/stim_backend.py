"""Optional Stim compilation of bounded circuit records, without model approximation.

The tested dependency is Stim 1.16.0. Ideal state preparation is a boundary
condition, not an arbitrary-input encoder or a fault-tolerant preparation.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from qeclang.algebra.binary import nonnegative_integer
from .ir import CircuitIR


@dataclass(frozen=True)
class StimLimits:
    max_preparation_qubits: int = 256
    max_tableau_bits: int = 10_000_000
    max_preparation_characters: int = 1_000_000
    max_compiled_targets: int = 2_000_000
    max_compiled_instructions: int = 1_000_000
    max_detector_model_work: int = 50_000_000

    def __post_init__(self):
        for name, value in asdict(self).items():
            object.__setattr__(self, name, nonnegative_integer(value, name))


def _stim():
    try:
        import stim
    except ImportError as error:
        raise ImportError("Stim compilation requires qeclang[circuits]") from error
    return stim


def _digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def _pauli(pauli, width, stim):
    letters = "".join("IXZY"[((pauli.x >> j) & 1) | (((pauli.z >> j) & 1) << 1)]
                      for j in range(width))
    phase = (pauli.phase - (pauli.x & pauli.z).bit_count()) % 4
    if phase not in (0, 2):
        raise ValueError("preparation generator is not Hermitian")
    return stim.PauliString(("-" if phase == 2 else "+") + letters)


@dataclass(frozen=True, init=False)
class CompiledCircuit:
    """Factory-only artifact; metadata and resource counts come from compile_stim."""
    text: str
    source_fingerprint: str
    stim_version: str
    n_qubits: int
    measurements: int
    detectors: int
    observables: int
    preparation_targets: int
    compiled_targets: int
    fault_terms_bound: int
    metadata_json: str
    limits: StimLimits

    def __new__(cls, *args, **kwargs):
        raise TypeError("construct a validated artifact with compile_stim(ir)")

    @property
    def fingerprint(self):
        return _digest(self.text)

    def to_stim(self):
        """Return a fresh mutable native circuit; retained artifact text is immutable."""
        stim = _stim()
        if stim.__version__ != self.stim_version:
            raise ValueError("compiled artifact Stim version differs from current runtime")
        circuit = stim.Circuit(self.text)
        if ((circuit.num_qubits, circuit.num_measurements, circuit.num_detectors, circuit.num_observables)
                != (self.n_qubits, self.measurements, self.detectors, self.observables)):
            raise ValueError("native circuit dimensions disagree with saved artifact metadata")
        return circuit

    def summary(self):
        return {"source_fingerprint": self.source_fingerprint,
                "circuit_sha256": self.fingerprint, "stim_version": self.stim_version,
                "n_qubits": self.n_qubits, "measurements": self.measurements,
                "detectors": self.detectors, "observables": self.observables,
                "preparation_targets": self.preparation_targets,
                "compiled_targets": self.compiled_targets,
                "fault_terms_bound": self.fault_terms_bound,
                "metadata": json.loads(self.metadata_json), "limits": asdict(self.limits),
                "preparation": "ideal known stabilizer state via elimination",
                "preparation_graph_constrained": False}


def compile_stim(ir: CircuitIR, *, limits=StimLimits()):
    """Compile ideal state preparation, operations, then absolute-record parities.

    Native preparation synthesis has its own qubit/tableau-size preflight.
    Generated target/instruction counts are checked immediately afterwards;
    these counts do not bound native synthesis workspace or elapsed time.
    Only extraction interactions have been checked against any supplied graph.
    """
    if not isinstance(ir, CircuitIR) or not isinstance(limits, StimLimits):
        raise TypeError("compile_stim requires CircuitIR and StimLimits")
    if ir.n_data > limits.max_preparation_qubits or 4*ir.n_data**2 > limits.max_tableau_bits:
        raise ValueError("ideal preparation exceeds tableau limits")
    if 4*ir.n_qubits**2 > limits.max_tableau_bits:
        raise ValueError("total-qubit native reference tableau exceeds max_tableau_bits")
    if ir.n_data * len(ir.preparation.generators) > limits.max_preparation_characters:
        raise ValueError("preparation Pauli strings exceed max_preparation_characters")
    source_targets = sum(len(op.targets) for op in ir.operations)
    references = sum(map(len, ir.detectors)) + sum(map(len, ir.observables))
    if source_targets + references + ir.n_qubits > limits.max_compiled_targets:
        raise ValueError("circuit source exceeds max_compiled_targets")
    if len(ir.operations) + len(ir.detectors) + len(ir.observables) + bool(ir.n_qubits) > limits.max_compiled_instructions:
        raise ValueError("circuit source exceeds max_compiled_instructions")
    stim = _stim()
    generators = [_pauli(p, ir.n_data, stim) for p in ir.preparation.basis]
    tableau = stim.Tableau.from_stabilizers(generators, allow_underconstrained=False)
    preparation = tableau.to_circuit(method="elimination")
    if ir.n_data and preparation.num_qubits < ir.n_data:
        preparation.append("I", [ir.n_data-1])
    if preparation.to_tableau() != tableau:
        raise ArithmeticError("synthesized ideal preparation disagrees with its exact tableau")
    preparation_targets = sum(len(op.targets_copy()) for op in preparation)
    total_targets = source_targets + references + ir.n_qubits + preparation_targets
    if total_targets > limits.max_compiled_targets:
        raise ValueError("synthesized preparation exceeds max_compiled_targets")
    if (len(preparation) + len(ir.operations) + len(ir.detectors) + len(ir.observables)
            + bool(ir.n_qubits) > limits.max_compiled_instructions):
        raise ValueError("synthesized preparation exceeds max_compiled_instructions")
    circuit = stim.Circuit()
    if ir.n_qubits:
        circuit.append("R", range(ir.n_qubits))
    circuit += preparation
    fault_terms = 0
    for operation in ir.operations:
        if operation.probability is None:
            circuit.append(operation.name, operation.targets)
        else:
            circuit.append(operation.name, operation.targets, operation.probability)
            if operation.probability:
                fault_terms += {"X_ERROR": 1, "Z_ERROR": 1,
                                "DEPOLARIZE1": 3, "DEPOLARIZE2": 15}[operation.name]
    count = ir.measurement_count
    for support in ir.detectors:
        circuit.append("DETECTOR", [stim.target_rec(index-count) for index in support])
    for logical, support in enumerate(ir.observables):
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(index-count) for index in support], logical)
    if (circuit.num_qubits != ir.n_qubits or circuit.num_measurements != count or circuit.num_detectors != len(ir.detectors)
            or circuit.num_observables != len(ir.observables)):
        raise ArithmeticError("compiled measurement/detector dimensions disagree with IR")
    artifact = object.__new__(CompiledCircuit)
    values = {"text": str(circuit), "source_fingerprint": ir.fingerprint(),
              "stim_version": stim.__version__, "n_qubits": ir.n_qubits,
              "measurements": count, "detectors": len(ir.detectors),
              "observables": len(ir.observables), "preparation_targets": preparation_targets,
              "compiled_targets": total_targets, "fault_terms_bound": fault_terms,
              "metadata_json": json.dumps(asdict(ir.metadata) if ir.metadata else None, sort_keys=True),
              "limits": limits}
    for name, value in values.items():
        object.__setattr__(artifact, name, value)
    return artifact


@dataclass(frozen=True)
class CompiledDetectorModel:
    model: object
    text: str
    circuit_fingerprint: str
    stim_version: str

    @property
    def fingerprint(self):
        return _digest(self.text)


def compile_detector_model(compiled: CompiledCircuit, **import_limits):
    """Request an undecomposed, gauge-free model with disjoint approximation off.

    A conservative fault-term × parity-count work proxy is checked before native
    error analysis. Import limits then bound actual model expansion. Unsupported
    exact conversions raise; there is no graphlike conversion or fallback.
    """
    if not isinstance(compiled, CompiledCircuit):
        raise TypeError("detector compilation requires CompiledCircuit")
    work = compiled.fault_terms_bound * (compiled.detectors + compiled.observables)
    if work > compiled.limits.max_detector_model_work:
        raise ValueError("native detector analysis exceeds max_detector_model_work")
    from .stim_dem import import_detector_model
    circuit = compiled.to_stim()
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=False,
                                       allow_gauge_detectors=False,
                                       approximate_disjoint_errors=False)
    model = import_detector_model(dem, gauge_free=True, num_detectors=compiled.detectors,
                                   num_observables=compiled.observables, **import_limits)
    if model.num_detectors != compiled.detectors or model.num_observables != compiled.observables:
        raise ArithmeticError("detector model dimensions disagree with compiled circuit")
    return CompiledDetectorModel(model, str(dem), compiled.fingerprint, compiled.stim_version)
