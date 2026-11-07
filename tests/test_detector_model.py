from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace

import pytest

from qeclang.algebra import SparseBinaryMatrix as S
from qeclang.circuits.detector_model import DetectorModel
from qeclang.circuits.detector_sampling import (DetectorEvaluator, evaluate_predictions,
                                                sample_detector_model, simulate_detector_memory)
from qeclang.decoding import BPDecoder


def model():
    # Fault 0 is a three-detector hyperedge; fault 1 is logical-only.
    return DetectorModel.from_errors([(.1, [0, 1, 2], [0]), (.3, [], [0])],
                                      num_detectors=4, num_observables=2)


def test_fault_mapping_hyperedges_logical_only_and_repeated_target_parity():
    example = model()
    assert (example.num_faults, example.num_detectors, example.num_observables) == (2, 4, 2)
    assert [example.syndrome(f) for f in range(4)] == [0, 7, 0, 7]
    assert [example.logical(f) for f in range(4)] == [0, 1, 1, 0]
    repeated = DetectorModel.from_errors([(.5, [0, 0, 1], [0, 0, 1]), (0, [], [])],
                                         num_detectors=2, num_observables=2)
    assert repeated.syndrome(1) == repeated.logical(1) == 2
    assert repeated.num_faults == 2
    assert repeated.probabilities == (.5, 0)
    with pytest.raises(ValueError):
        example.syndrome(4)


def test_portable_model_hash_and_immutable_semantics():
    example = model()
    record = json.loads(json.dumps(example.to_dict()))
    assert DetectorModel.from_dict(record) == example
    assert DetectorModel.from_dict(record).fingerprint() == example.fingerprint()
    assert len(example.fingerprint()) == 64
    modified = deepcopy(record)
    modified['probabilities'][0] = .2
    assert DetectorModel.from_dict(modified).fingerprint() != example.fingerprint()
    record['checks']['rows'][0].clear()
    assert example.checks.rows[0] == (0,)
    with pytest.raises(FrozenInstanceError):
        example.probabilities = (0, 0)


def test_model_validation_and_preallocation_guards():
    with pytest.raises(ValueError, match='fault columns'):
        DetectorModel(S((), 1), S((), 2), (.1,))
    with pytest.raises(ValueError, match='probability count'):
        DetectorModel(S((), 1), S((), 1), ())
    for probability in (float('nan'), -1, 1.1, '0.1', True):
        with pytest.raises(ValueError):
            DetectorModel(S((), 1), S((), 1), (probability,))
    def forbidden():
        raise AssertionError('probabilities should not be consumed after axis failure')
        yield .1
    with pytest.raises(ValueError, match='axis'):
        DetectorModel(S((), 10**9), S((), 10**9), forbidden())
    with pytest.raises(ValueError, match='max_nonzeros'):
        DetectorModel.from_errors([(.1, [0, 0], [])], num_detectors=1,
                                  num_observables=0, max_nonzeros=1)
    record = model().to_dict()
    for options in ({'max_faults': 1}, {'max_detectors': 3},
                    {'max_observables': 1}, {'max_nonzeros': 1}):
        with pytest.raises(ValueError):
            DetectorModel.from_dict(record, **options)
    record['unknown'] = 3
    with pytest.raises(ValueError, match='schema'):
        DetectorModel.from_dict(record)


def test_native_sample_stream_is_batch_invariant_and_endpoint_exact():
    example = model()
    def rows(size):
        batches = list(sample_detector_model(example, trials=31, seed=82, batch_size=size))
        return tuple(x for b in batches for x in b.detectors), tuple(x for b in batches for x in b.observables)
    assert rows(1) == rows(7) == rows(31)
    exact = DetectorModel.from_errors([(0, [0], []), (1, [1, 2], [0])],
                                      num_detectors=3, num_observables=1)
    batch = next(sample_detector_model(exact, trials=5, seed=0))
    assert batch.detectors == (6,) * 5 and batch.observables == (1,) * 5
    assert simulate_detector_memory(exact, trials=5, seed=0).failures == 0
    with pytest.raises(ValueError, match='max_batch_bits'):
        list(sample_detector_model(exact, trials=5, seed=0, max_batch_bits=19))
    with pytest.raises(ValueError, match='max_draws'):
        list(sample_detector_model(exact, trials=5, seed=0, max_draws=9))
    assert list(sample_detector_model(exact, trials=0, seed=0, max_batch_bits=0, max_draws=0)) == []


def test_streaming_metrics_provenance_and_observable_failures():
    example = model()
    a = simulate_detector_memory(example, trials=200, seed=5, batch_size=7)
    b = simulate_detector_memory(example, trials=200, seed=5, batch_size=31)
    assert a.failures == b.failures and a.confidence_interval == b.confidence_interval
    assert a.decoder_failures == 0 and 0 < a.logical_failures < a.trials
    assert a.failures == a.logical_failures
    assert a.provenance['sampling'] == 'independent_binary_model'
    data = a.provenance
    data['decoder_configuration']['max_iterations'] = -1
    assert a.provenance['decoder_configuration']['max_iterations'] == 50
    assert a.model_fingerprint == example.fingerprint()
    json.dumps(a.summary(), allow_nan=False)
    evaluator = DetectorEvaluator(example, max_samples=200)
    for batch in sample_detector_model(example, trials=200, seed=5, batch_size=13):
        evaluator.add(batch.detectors, batch.observables)
    assert evaluator.result(seed=5).failures == a.failures


def test_decoder_failure_and_syndrome_consistency_are_distinct_from_logical_failure():
    example = DetectorModel(S(((0, 1),), 2), S(((0,),), 2), (.1, .1))
    result = evaluate_predictions(example, [1], [0], decoder_options={'max_iterations': 0})
    assert result.decoder_failures == 1 and result.logical_failures == 0
    decoder = BPDecoder(example.checks, example.probabilities)
    decoder.decode = lambda syndrome: SimpleNamespace(success=True, error=0)
    result = evaluate_predictions(example, [1], [0], decoder=decoder)
    assert result.decoder_failures == result.syndrome_mismatches == 1
    assert result.logical_failures == 0
    logical_only = DetectorModel(S((), 1), S(((0,),), 1), (.1,))
    result = evaluate_predictions(logical_only, [0], [1])
    assert result.decoder_failures == 0 and result.logical_failures == 1


def test_sample_shape_limits_no_observables_and_configuration_changes():
    example = model()
    evaluator = DetectorEvaluator(example, max_samples=1)
    with pytest.raises(ValueError, match='no samples'):
        evaluator.result()
    with pytest.raises(ValueError, match='counts differ'):
        evaluator.add([0], [])
    with pytest.raises(ValueError, match='wrong number'):
        evaluator.add([[0]], [[0, 0]])
    with pytest.raises(ValueError, match='binary'):
        evaluator.add([[0., 0., 0., 0.]], [[0, 0]])
    evaluator.add([0], [0])
    with pytest.raises(ValueError, match='max_samples'):
        evaluator.add([0], [0])
    changing = DetectorEvaluator(example)
    changing.decoder.max_iterations = 4
    with pytest.raises(ValueError, match='configuration changed'):
        changing.add([0], [0])
    no_logicals = DetectorModel(S(((),), 0), S((), 0), ())
    assert next(sample_detector_model(no_logicals, trials=1, seed=0)).observables == (0,)
    with pytest.raises(ValueError, match='logical observable'):
        DetectorEvaluator(no_logicals)
    with pytest.raises(ValueError, match='logical observable'):
        simulate_detector_memory(no_logicals, trials=1, seed=0)


def test_numpy_boolean_rows_and_bounded_cache():
    np = pytest.importorskip('numpy')
    example = model()
    detectors = np.array([[0, 0, 0, 0], [1, 1, 1, 0]], dtype=np.bool_)
    observables = np.array([[0, 0], [1, 0]], dtype=np.bool_)
    evaluator = DetectorEvaluator(example, max_cache_entries=1)
    evaluator.add(detectors, observables)
    result = evaluator.result(context={'circuit_hash': 'example'}, sampling='stim_circuit')
    assert result.failures == 0
    assert result.provenance['cached_syndromes'] == 1
    assert result.provenance['sampling'] == 'stim_circuit'


def test_stim_undecomposed_hyperedges_logicals_and_declared_dimensions():
    stim = pytest.importorskip('stim')
    from qeclang.circuits.stim_dem import import_detector_model
    dem = stim.DetectorErrorModel('''
        error(.1) D0 D1 D2 L0
        error(.2) L1
        error(0) D3
        error(1) D4 L0
        detector D7
        logical_observable L3
    ''')
    imported = import_detector_model(dem, gauge_free=True)
    assert (imported.num_faults, imported.num_detectors, imported.num_observables) == (4, 8, 4)
    assert imported.probabilities == pytest.approx((.1, .2, 0, 1))
    assert imported.syndrome(1) == 7 and imported.logical(2) == 2
    with pytest.raises(ValueError, match='gauge-free'):
        import_detector_model(dem)
    with pytest.raises(ValueError, match='separator'):
        import_detector_model(stim.DetectorErrorModel('error(.1) D0 ^ D1 L0'), gauge_free=True)


def test_stim_nested_repeat_offsets_and_raw_fault_replay():
    stim = pytest.importorskip('stim')
    np = pytest.importorskip('numpy')
    from qeclang.circuits.stim_dem import import_detector_model
    dem = stim.DetectorErrorModel('''
        detector D0
        repeat 2 {
            error(.2) D0 D1 L0
            repeat 2 {
                error(.3) D1 L1 L1
                shift_detectors 1
            }
            shift_detectors 2
        }
        detector D2
        logical_observable L2
    ''')
    imported = import_detector_model(dem, gauge_free=True)
    assert imported == import_detector_model(dem.flattened(), gauge_free=True)
    assert imported.num_detectors == dem.num_detectors == 11
    assert imported.num_observables == dem.num_observables == 3
    faults = np.array([[(word >> j) & 1 for j in range(imported.num_faults)]
                       for word in range(1 << imported.num_faults)], dtype=np.bool_)
    ds, ls, _ = dem.compile_sampler(seed=2).sample(len(faults), recorded_errors_to_replay=faults)
    for word, (detectors, logicals) in enumerate(zip(ds, ls)):
        assert imported.syndrome(word) == sum(int(bit) << j for j, bit in enumerate(detectors))
        assert imported.logical(word) == sum(int(bit) << j for j, bit in enumerate(logicals))


def test_stim_empty_trailing_shifts_repeated_ids_and_expansion_limits(monkeypatch):
    stim = pytest.importorskip('stim')
    from qeclang.circuits.stim_dem import import_detector_model
    for text in ('shift_detectors 5', 'detector D0\nshift_detectors 10',
                 'error(.5) D2 D2 L1 L1\nshift_detectors 10'):
        dem = stim.DetectorErrorModel(text)
        imported = import_detector_model(dem, gauge_free=True)
        assert imported.num_detectors == dem.num_detectors
        assert imported.num_observables == dem.num_observables
    empty = import_detector_model(stim.DetectorErrorModel(), gauge_free=True,
                                   num_detectors=5, num_observables=2)
    assert empty.checks.shape == (5, 0) and empty.observables.shape == (2, 0)
    def forbidden(*args, **kwargs):
        raise AssertionError('expansion should not reach native row construction')
    monkeypatch.setattr(DetectorModel, 'from_errors', forbidden)
    massive = stim.DetectorErrorModel('repeat 1000000000 {\nshift_detectors 0\n}')
    with pytest.raises(ValueError, match='max_expanded_instructions'):
        import_detector_model(massive, gauge_free=True)
    errors = stim.DetectorErrorModel('repeat 10 {\nerror(.1) D0\nshift_detectors 1\n}')
    with pytest.raises(ValueError, match='max_faults'):
        import_detector_model(errors, gauge_free=True, max_faults=9)
    with pytest.raises(ValueError, match='max_expanded_targets'):
        import_detector_model(errors, gauge_free=True, max_expanded_targets=3)
