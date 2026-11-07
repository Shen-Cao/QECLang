from dataclasses import replace

import pytest

stim = pytest.importorskip("stim")
from qeclang.circuits import CircuitIR, Operation, CircuitNoise, css_memory  # noqa: E402
from qeclang.circuits.stim_backend import StimLimits, compile_detector_model, compile_stim  # noqa: E402
from qeclang.codes import Pauli, StabilizerCode  # noqa: E402
from qeclang.constructions import steane, toric  # noqa: E402


def packed(row):
    return sum(int(value) << i for i, value in enumerate(row))


def fault_signature(ir, after, site, label):
    """Independent binary Pauli propagation oracle for the ideal extraction gates."""
    x = z = 0
    outcomes = []
    for index, op in enumerate(ir.operations):
        if index == after + 1:
            x ^= (label & 1) << site
            z ^= (label >> 1) << site
        if op.name == "R":
            x &= ~(1 << op.targets[0])
            z &= ~(1 << op.targets[0])
        elif op.name == "H":
            q = op.targets[0]
            delta = ((x >> q) ^ (z >> q)) & 1
            x ^= delta << q
            z ^= delta << q
        elif op.name == "CX":
            control, target = op.targets
            x ^= ((x >> control) & 1) << target
            z ^= ((z >> target) & 1) << control
        elif op.name == "M":
            outcomes.append((x >> op.targets[0]) & 1)
        elif op.name == "MX":
            outcomes.append((z >> op.targets[0]) & 1)
        elif op.name != "TICK":
            raise AssertionError("oracle expects ideal operations")
    def parities(supports):
        return sum((sum(outcomes[j] for j in support) % 2) << i for i, support in enumerate(supports))
    return parities(ir.detectors), parities(ir.observables)


@pytest.mark.parametrize("factory", [steane, lambda: toric(2)])
@pytest.mark.parametrize("basis", ["x", "z"])
def test_noiseless_memories_and_immutable_artifact(factory, basis):
    ir = css_memory(factory(), basis=basis, rounds=3)
    artifact = compile_stim(ir)
    circuit = artifact.to_stim()
    detectors, observables = circuit.compile_detector_sampler(seed=12).sample(32, separate_observables=True)
    assert not detectors.any() and not observables.any()
    assert artifact.measurements == ir.measurement_count
    assert artifact.detectors == len(ir.detectors)
    assert artifact.observables == len(ir.observables)
    circuit.append("H", [0])
    assert str(artifact.to_stim()) == artifact.text
    assert artifact.fingerprint == artifact.summary()["circuit_sha256"]
    with pytest.raises(TypeError, match="compile_stim"):
        replace(artifact, n_qubits=0)


@pytest.mark.parametrize("basis", ["x", "z"])
def test_every_single_location_pauli_matches_independent_propagation(basis):
    ir = css_memory(steane(), basis=basis, rounds=1)
    locations = [(-1, site) for site in range(ir.n_data)]
    locations.extend((index, site) for index, op in enumerate(ir.operations) for site in op.targets)
    for after, site in locations:
        for label in (1, 2, 3):
            inserted = []
            if label & 1:
                inserted.append(Operation("X_ERROR", (site,), 1.0))
            if label & 2:
                inserted.append(Operation("Z_ERROR", (site,), 1.0))
            faulty = replace(ir, operations=ir.operations[:after+1] + tuple(inserted) + ir.operations[after+1:],
                             metadata=None)
            circuit = compile_stim(faulty).to_stim()
            detectors, observables = circuit.compile_detector_sampler(seed=0).sample(1, separate_observables=True)
            assert (packed(detectors[0]), packed(observables[0])) == fault_signature(ir, after, site, label)


@pytest.mark.parametrize("phase", [1, 3])
def test_signed_y_state_preparation(phase):
    ir = CircuitIR(1, 1, StabilizerCode(1, (Pauli(1, 1, phase),)), ())
    artifact = compile_stim(ir)
    simulator = stim.TableauSimulator()
    simulator.do(artifact.to_stim())
    expected = stim.PauliString("+Y" if phase == 1 else "-Y")
    assert simulator.peek_observable_expectation(expected) == 1


def test_signed_reference_is_not_forced_to_zero():
    ir = CircuitIR(1, 1, StabilizerCode(1, (Pauli(0, 1, 2),)),
                   (Operation("M", (0,)),), ((0,),), ((0,),))
    circuit = compile_stim(ir).to_stim()
    assert list(circuit.reference_sample()) == [True]
    detectors, observables = circuit.compile_detector_sampler(seed=0).sample(4, separate_observables=True)
    assert not detectors.any() and not observables.any()


def test_compiler_preflight_and_native_detector_model_limits(monkeypatch):
    ir = css_memory(steane(), rounds=2, noise=CircuitNoise(measurement_flip=.1))
    artifact = compile_stim(ir, limits=StimLimits(max_detector_model_work=0))
    with pytest.raises(ValueError, match="max_detector_model_work"):
        compile_detector_model(artifact)
    import qeclang.circuits.stim_backend as backend
    def forbidden():
        raise AssertionError("native compiler must follow preflight guards")
    monkeypatch.setattr(backend, "_stim", forbidden)
    for limits in (StimLimits(max_preparation_qubits=1), StimLimits(max_tableau_bits=1),
                   StimLimits(max_preparation_characters=1), StimLimits(max_compiled_targets=1),
                   StimLimits(max_compiled_instructions=1)):
        with pytest.raises(ValueError):
            compile_stim(ir, limits=limits)
    many_ancillas = replace(ir, n_qubits=1000)
    with pytest.raises(ValueError, match="total-qubit"):
        compile_stim(many_ancillas, limits=StimLimits(max_tableau_bits=100_000))


def test_random_detector_is_rejected_without_gauge_approximation():
    ir = css_memory(steane(), rounds=1)
    invalid = replace(ir, detectors=ir.detectors + ((ir.measurement_count-1,),))
    compiled = compile_stim(invalid)
    with pytest.raises(ValueError, match="non-deterministic"):
        compiled.to_stim().detector_error_model(decompose_errors=False,
            allow_gauge_detectors=False, approximate_disjoint_errors=False)
