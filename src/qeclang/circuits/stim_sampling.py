"""Bounded circuit sampling and logical prediction using strict Stim detector models."""
from dataclasses import asdict
import platform

from qeclang.algebra.binary import nonnegative_integer
from qeclang.decoding import wilson_interval
from .detector_sampling import DetectorEvaluator
from .ir import CircuitIR
from .stim_backend import CompiledCircuit, StimLimits, compile_detector_model, compile_stim


def _positive(value, name):
    value = nonnegative_integer(value, name)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def simulate_circuit_memory(source, *, trials, seed, batch_size=256,
                            decoder_options=None, confidence=.95,
                            max_trials=1_000_000, max_batch_bytes=10_000_000,
                            max_record_frame_bits=100_000_000, max_cache_entries=0,
                            compile_limits=StimLimits(), detector_limits=None):
    """Sample actual circuit detector/logical flips, then decode their raw DEM.

    source is CircuitIR or a previously compiled immutable artifact. compile_limits
    applies when compiling an IR. Sampling uses packed uint8 output; batch-byte
    guards include padding. The record/frame-bit guard is a separate proxy for
    native working state, not a complete allocation or wall-time guarantee.

    A seed is reproducible only with matching circuit/version/platform/batching
    assumptions documented by Stim; no cross-version or cross-SIMD guarantee is
    made. Ideal preparation and one-basis logical readout remain explicit.
    """
    trials = _positive(trials, "trials")
    batch_size = _positive(batch_size, "batch_size")
    maximum = nonnegative_integer(max_trials, "max_trials")
    bytes_limit = nonnegative_integer(max_batch_bytes, "max_batch_bytes")
    frame_limit = nonnegative_integer(max_record_frame_bits, "max_record_frame_bits")
    if trials > maximum:
        raise ValueError("circuit sampling exceeds max_trials")
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise ValueError("Stim seed must be an integer in [0,2**64)")
    wilson_interval(0, trials, confidence)
    if not isinstance(compile_limits, StimLimits):
        raise TypeError("compile_limits must be StimLimits")
    if detector_limits is not None and not isinstance(detector_limits, dict):
        raise TypeError("detector_limits must be a dictionary")
    if isinstance(source, CircuitIR):
        if not source.observables:
            raise ValueError("memory experiment requires a declared logical observable")
        compiled = compile_stim(source, limits=compile_limits)
    elif isinstance(source, CompiledCircuit):
        compiled = source
    else:
        raise TypeError("memory source must be CircuitIR or CompiledCircuit")
    if compiled.observables == 0:
        raise ValueError("memory experiment requires a declared logical observable")
    batch = min(batch_size, trials)
    row_bytes = (compiled.detectors+7)//8 + (compiled.observables+7)//8
    if batch * row_bytes > bytes_limit:
        raise ValueError("packed circuit samples exceed max_batch_bytes")
    working_bits = batch * (compiled.measurements + 2*compiled.n_qubits)
    if working_bits > frame_limit:
        raise ValueError("circuit record/frame proxy exceeds max_record_frame_bits")
    dem = compile_detector_model(compiled, **(detector_limits or {}))
    evaluator = DetectorEvaluator(dem.model, decoder_options=decoder_options,
                                  max_samples=trials, max_cache_entries=max_cache_entries)
    circuit = compiled.to_stim()
    sampler = circuit.compile_detector_sampler(seed=seed)
    for start in range(0, trials, batch_size):
        detectors, observables = sampler.sample(min(batch_size, trials-start),
                                                separate_observables=True, bit_packed=True)
        evaluator.add((int.from_bytes(row.tobytes(), "little") for row in detectors),
                       (int.from_bytes(row.tobytes(), "little") for row in observables))
    context = {"circuit": compiled.summary(), "detector_model_sha256": dem.fingerprint,
               "stim_version": compiled.stim_version,
               "python_version": platform.python_version(), "architecture": platform.machine(),
               "batch_size": batch_size, "bit_packed": True,
               "max_trials": maximum, "max_batch_bytes": bytes_limit,
               "max_record_frame_bits": frame_limit,
               "detector_import_limits": detector_limits or {},
               "dem_settings": {"decompose_errors": False, "flatten_loops": False,
                                "allow_gauge_detectors": False, "approximate_disjoint_errors": False},
               "compiler_limits": asdict(compiled.limits)}
    return evaluator.result(confidence=confidence, seed=seed, sampling="stim_circuit", context=context)
