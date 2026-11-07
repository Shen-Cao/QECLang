"""Streaming independent-fault sampling and detector/logical prediction metrics."""
from dataclasses import dataclass
from itertools import zip_longest
import json
from numbers import Integral
from random import Random

from qeclang import __version__
from qeclang.algebra import BinaryMatrix, SparseBinaryMatrix
from qeclang.algebra.binary import nonnegative_integer, validate_vector
from qeclang.decoding import BPDecoder, wilson_interval
from .detector_model import DetectorModel


def _positive(value, name):
    value = nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _seed(value):
    if type(value) is not int:
        raise ValueError("seed must be an integer")
    return value


def _pack(row, width):
    if isinstance(row, Integral) and not isinstance(row, bool):
        return validate_vector(row, width, "sample")
    value, count = 0, 0
    for bit in row:
        if count == width:
            raise ValueError("sample row has too many bits")
        if not isinstance(bit, Integral):
            if getattr(getattr(bit, "dtype", None), "kind", None) != "b" or getattr(bit, "ndim", None) != 0:
                raise ValueError("sample coefficients must be binary integers or booleans")
        bit = int(bit)
        if bit not in (0, 1):
            raise ValueError("sample coefficients must be binary")
        value |= bit << count
        count += 1
    if count != width:
        raise ValueError("sample row has the wrong number of bits")
    return value


@dataclass(frozen=True)
class DetectorBatch:
    start: int
    detectors: tuple[int, ...]
    observables: tuple[int, ...]


def sample_detector_model(model: DetectorModel, *, trials, seed, batch_size=256,
                          max_batch_bits=10_000_000, max_draws=100_000_000):
    """Yield bounded packed batches from independent Bernoulli fault columns.

    The seeded stream is independent of batch size and draws once per mechanism
    per trial, including p=0/1. max_batch_bits bounds packed output payload, not
    Python object overhead. max_draws bounds trials*num_faults before sampling.
    This samples the native fault model, not an underlying physical circuit.
    """
    if not isinstance(model, DetectorModel):
        raise TypeError("sampling requires DetectorModel")
    trials = nonnegative_integer(trials, "trials")
    seed = _seed(seed)
    batch_size = _positive(batch_size, "batch_size")
    bits = nonnegative_integer(max_batch_bits, "max_batch_bits")
    draws = nonnegative_integer(max_draws, "max_draws")
    if model.num_faults * trials > draws:
        raise ValueError("detector model sampling exceeds max_draws")
    if min(batch_size, trials) * (model.num_detectors + model.num_observables) > bits:
        raise ValueError("detector model sampling exceeds max_batch_bits")
    rng = Random(seed)
    for start in range(0, trials, batch_size):
        ds, ls = [], []
        for _ in range(min(batch_size, trials-start)):
            fault = 0
            for j, probability in enumerate(model.probabilities):
                if rng.random() < probability:
                    fault |= 1 << j
            ds.append(model.syndrome(fault))
            ls.append(model.logical(fault))
        yield DetectorBatch(start, tuple(ds), tuple(ls))


@dataclass(frozen=True)
class DetectorEvaluation:
    trials: int
    failures: int
    decoder_failures: int
    logical_failures: int
    syndrome_mismatches: int
    failure_rate: float
    confidence: float
    confidence_interval: tuple[float, float]
    model_fingerprint: str
    seed: int | None
    provenance_json: str

    @property
    def provenance(self):
        """A fresh dictionary; modifying it cannot alter the saved evidence."""
        return json.loads(self.provenance_json)

    def summary(self):
        return {name: getattr(self, name) for name in
                ("trials", "failures", "decoder_failures", "logical_failures",
                 "syndrome_mismatches", "failure_rate", "confidence",
                 "confidence_interval", "model_fingerprint", "seed")} | {"provenance": self.provenance}


class DetectorEvaluator:
    """Accumulate streamed detector/observable rows using a reusable BP decoder.

    Original syndrome equations are checked independently of decoder status.
    Decoder failures (including accepted predictions with wrong syndrome) and
    observable prediction failures are disjoint. A failed add retains its valid
    prefix, but no final result is produced implicitly. Caching is opt-in and
    stores only scalar prediction outcomes, never BP message arrays.
    """
    def __init__(self, model: DetectorModel, *, decoder=None, decoder_options=None,
                 max_samples=1_000_000, max_cache_entries=0):
        if not isinstance(model, DetectorModel):
            raise TypeError("evaluation requires DetectorModel")
        if model.num_observables == 0:
            raise ValueError("memory evaluation requires at least one declared logical observable")
        self.model = model
        self.max_samples = nonnegative_integer(max_samples, "max_samples")
        self.max_cache_entries = nonnegative_integer(max_cache_entries, "max_cache_entries")
        if decoder is not None and decoder_options is not None:
            raise ValueError("provide decoder or decoder_options, not both")
        if decoder_options is not None and not isinstance(decoder_options, dict):
            raise TypeError("decoder_options must be a dictionary")
        if decoder is None:
            decoder = BPDecoder(model.checks, model.probabilities, **(decoder_options or {}))
        if not isinstance(decoder, BPDecoder):
            raise TypeError("detector evaluation currently requires BPDecoder")
        if (decoder.checks.shape != model.checks.shape
                or tuple(decoder.probabilities) != model.probabilities):
            raise ValueError("decoder must match the detector checks and fault probabilities")
        checks = (SparseBinaryMatrix.from_packed(decoder.checks)
                  if isinstance(decoder.checks, BinaryMatrix) else decoder.checks)
        if checks != model.checks:
            raise ValueError("decoder must match the detector checks and fault probabilities")
        self.decoder = decoder
        self._configuration = json.dumps(decoder.configuration, sort_keys=True, allow_nan=False)
        self._fingerprint = model.fingerprint()
        self._cache = {}
        self.trials = self.decoder_failures = self.logical_failures = self.syndrome_mismatches = 0

    def _prediction(self, syndrome):
        if syndrome in self._cache:
            return self._cache[syndrome]
        result = self.decoder.decode(syndrome)
        mismatch = False
        success, logical = False, None
        if result.success:
            try:
                fault = validate_vector(result.error, self.model.num_faults, "decoded fault vector")
            except ValueError:
                mismatch = True
            else:
                mismatch = self.model.syndrome(fault) != syndrome
                if not mismatch:
                    success, logical = True, self.model.logical(fault)
        value = success, logical, mismatch
        if len(self._cache) < self.max_cache_entries:
            self._cache[syndrome] = value
        return value

    def add(self, detector_rows, observable_rows):
        """Consume equally long iterables of packed integers or unpacked bit rows."""
        if json.dumps(self.decoder.configuration, sort_keys=True, allow_nan=False) != self._configuration:
            raise ValueError("decoder configuration changed during streamed evaluation")
        missing = object()
        for detector, observable in zip_longest(detector_rows, observable_rows, fillvalue=missing):
            if detector is missing or observable is missing:
                raise ValueError("detector and observable sample counts differ")
            if self.trials == self.max_samples:
                raise ValueError("detector evaluation exceeds max_samples")
            syndrome = _pack(detector, self.model.num_detectors)
            expected = _pack(observable, self.model.num_observables)
            success, predicted, mismatch = self._prediction(syndrome)
            self.trials += 1
            self.syndrome_mismatches += int(mismatch)
            if not success:
                self.decoder_failures += 1
            elif predicted != expected:
                self.logical_failures += 1
        return self

    def result(self, *, confidence=.95, seed=None, context=None, sampling="external_samples"):
        if self.trials == 0:
            raise ValueError("no samples have been evaluated")
        if sampling not in ("external_samples", "independent_binary_model", "stim_circuit"):
            raise ValueError("unsupported sampling provenance kind")
        if seed is not None:
            seed = _seed(seed)
        if context is not None and not isinstance(context, dict):
            raise TypeError("sample context must be a dictionary")
        failures = self.decoder_failures + self.logical_failures
        interval = wilson_interval(failures, self.trials, confidence)
        provenance = {"qeclang_version": __version__, "sampling": sampling,
                      "decoder": "normalized_min_sum", "decoder_configuration": json.loads(self._configuration),
                      "max_samples": self.max_samples, "max_cache_entries": self.max_cache_entries,
                      "cached_syndromes": len(self._cache), "num_faults": self.model.num_faults,
                      "num_detectors": self.model.num_detectors, "num_observables": self.model.num_observables,
                      "sample_context": {} if context is None else context}
        return DetectorEvaluation(self.trials, failures, self.decoder_failures, self.logical_failures,
                                   self.syndrome_mismatches, failures/self.trials, confidence, interval,
                                   self._fingerprint, seed,
                                   json.dumps(provenance, sort_keys=True, allow_nan=False))


def evaluate_predictions(model, detector_rows, observable_rows, *, decoder=None,
                         decoder_options=None, max_samples=1_000_000, max_cache_entries=0,
                         confidence=.95, seed=None, context=None, sampling="external_samples"):
    wilson_interval(0, 1, confidence)
    evaluator = DetectorEvaluator(model, decoder=decoder, decoder_options=decoder_options,
                                   max_samples=max_samples, max_cache_entries=max_cache_entries)
    evaluator.add(detector_rows, observable_rows)
    return evaluator.result(confidence=confidence, seed=seed, context=context, sampling=sampling)


def simulate_detector_memory(model, *, trials, seed, decoder=None, decoder_options=None,
                             batch_size=256, max_batch_bits=10_000_000,
                             max_draws=100_000_000, max_cache_entries=0, confidence=.95):
    """Native independent-mechanism memory experiment, with explicit provenance."""
    trials = _positive(trials, "trials")
    seed = _seed(seed)
    wilson_interval(0, trials, confidence)
    evaluator = DetectorEvaluator(model, decoder=decoder, decoder_options=decoder_options,
                                   max_samples=trials, max_cache_entries=max_cache_entries)
    for batch in sample_detector_model(model, trials=trials, seed=seed, batch_size=batch_size,
                                       max_batch_bits=max_batch_bits, max_draws=max_draws):
        evaluator.add(batch.detectors, batch.observables)
    return evaluator.result(confidence=confidence, seed=seed, sampling="independent_binary_model",
                            context={"kind": "independent_binary_model", "batch_size": batch_size,
                                     "max_batch_bits": max_batch_bits, "max_draws": max_draws})


__all__ = ["DetectorBatch", "DetectorEvaluation", "DetectorEvaluator", "sample_detector_model",
           "evaluate_predictions", "simulate_detector_memory"]
