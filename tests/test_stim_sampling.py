import json

import pytest

stim = pytest.importorskip("stim")
from qeclang.algebra import BinaryMatrix as M  # noqa: E402
from qeclang.codes import CSSCode  # noqa: E402
from qeclang.constructions import steane, toric  # noqa: E402
from qeclang.circuits import CircuitNoise, css_memory  # noqa: E402
from qeclang.circuits.stim_backend import compile_detector_model, compile_stim  # noqa: E402
from qeclang.circuits.stim_sampling import simulate_circuit_memory  # noqa: E402
from qeclang.cli import main  # noqa: E402


@pytest.mark.parametrize("basis", ["x", "z"])
@pytest.mark.parametrize("code", [steane(), toric(2)])
def test_noiseless_full_pipeline_preserves_zero_fault_model_widths(code, basis):
    source = css_memory(code, basis=basis, rounds=2)
    compiled = compile_stim(source)
    dem = compile_detector_model(compiled)
    assert dem.model.num_faults == 0
    assert dem.model.num_detectors == len(source.detectors)
    assert dem.model.num_observables == code.k
    result = simulate_circuit_memory(compiled, trials=25, seed=1, batch_size=7)
    assert result.trials == 25
    assert result.failures == result.decoder_failures == result.logical_failures == 0
    assert result.confidence_interval[1] > 0
    assert result.provenance["sampling"] == "stim_circuit"
    json.dumps(result.summary(), allow_nan=False)


def test_noisy_full_pipeline_reproducibility_and_exact_probability_endpoint():
    source = css_memory(steane(), rounds=2, noise=CircuitNoise(
        data_depolarization=.01, reset_flip=.01, one_qubit_depolarization=.01,
        two_qubit_depolarization=.01, measurement_flip=.01))
    options = dict(trials=80, seed=3, batch_size=16,
                   decoder_options={"max_iterations": 5, "fallback": "osd0"})
    first = simulate_circuit_memory(source, **options)
    second = simulate_circuit_memory(source, **options)
    assert first == second
    assert first.trials == 80 and first.syndrome_mismatches == 0
    assert first.failures == first.decoder_failures + first.logical_failures
    deterministic = css_memory(steane(), rounds=2, noise=CircuitNoise(measurement_flip=1))
    result = simulate_circuit_memory(deterministic, trials=8, seed=3,
                                     decoder_options={"fallback": "osd0"})
    assert result.failures == 0


def test_zero_logical_metric_and_sampler_guards(monkeypatch):
    state = CSSCode(M((1,), 1), M((), 1))
    with pytest.raises(ValueError, match="logical observable"):
        simulate_circuit_memory(css_memory(state, rounds=1), trials=1, seed=0)
    compiled = compile_stim(css_memory(steane(), rounds=1))
    import qeclang.circuits.stim_sampling as module
    def forbidden(*args, **kwargs):
        raise AssertionError("sampling guard must precede DEM compilation")
    monkeypatch.setattr(module, "compile_detector_model", forbidden)
    for kwargs in ({"max_batch_bytes": 0}, {"max_record_frame_bits": 0},
                   {"max_trials": 0}, {"batch_size": 0}, {"confidence": 1}):
        with pytest.raises(ValueError):
            simulate_circuit_memory(compiled, trials=2, seed=0, **kwargs)
    for seed in (-1, 1 << 64, True):
        with pytest.raises(ValueError):
            simulate_circuit_memory(compiled, trials=1, seed=seed)


def test_circuit_cli_and_exact_artifact(tmp_path, capsys):
    path = tmp_path / "memory.stim"
    assert main(["circuit-memory", "--rounds", "2", "--trials", "12", "--basis", "x",
                 "--stim-output", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["trials"] == 12 and result["failures"] == 0
    assert result["provenance"]["sampling"] == "stim_circuit"
    assert stim.Circuit(path.read_text()).num_observables == 1
    assert main(["circuit-memory", "--max-operations", "0"]) == 2
    error = capsys.readouterr().err
    assert "limit" in error or "max_operations" in error
