from dataclasses import replace
from math import sqrt
from random import Random

import pytest

from qeclang.algebra import BinaryMatrix as B
from qeclang.codes import CSSCode, Pauli, StabilizerCode
from qeclang.circuits import CircuitIR, CircuitLimits, CircuitNoise, Operation, css_memory


def repetition():
    return CSSCode(B((), 3), B((3, 6), 3), "repetition")


def test_manual_two_round_reference_and_logical_preparation():
    circuit = css_memory(repetition(), rounds=2, basis="z")
    assert (circuit.n_data, circuit.n_qubits, circuit.measurement_count) == (3, 5, 7)
    assert circuit.detectors == ((0,), (1,), (0, 2), (1, 3), (2, 4, 5), (3, 5, 6))
    assert circuit.observables == ((4,),)
    one_round = (Operation("R", (3,)), Operation("CX", (0, 3)), Operation("CX", (1, 3)),
                 Operation("M", (3,)), Operation("R", (4,)), Operation("CX", (1, 4)),
                 Operation("CX", (2, 4)), Operation("M", (4,)), Operation("TICK"))
    assert circuit.operations == one_round * 2 + tuple(Operation("M", (i,)) for i in range(3))
    assert circuit.preparation.n == 3 and circuit.preparation.k == 0
    assert len(circuit.preparation.generators) == 3
    x = css_memory(repetition(), rounds=2, basis="x")
    assert x.detectors == circuit.detectors[:4]
    assert x.observables == ((4, 5, 6),)
    assert all(operation.name == "MX" for operation in x.operations[-3:])
    assert x.fingerprint() != circuit.fingerprint()
    assert circuit.fingerprint() == css_memory(repetition(), rounds=2).fingerprint()


def test_x_check_gate_order_and_redundant_measurements():
    code = CSSCode(B((3, 3), 2), B((), 2))
    circuit = css_memory(code, basis="x")
    assert circuit.operations[:6] == (Operation("R", (2,)), Operation("H", (2,)),
                                     Operation("CX", (2, 0)), Operation("CX", (2, 1)),
                                     Operation("H", (2,)), Operation("M", (2,)))
    assert circuit.detectors == ((0,), (1,), (0, 2, 3), (1, 2, 3))
    assert circuit.preparation.rank == 2
    assert circuit.measurement_count == 4


@pytest.mark.parametrize("name,targets,p", [("S", (0,), None), ("H", (), None),
    ("CX", (0,), None), ("CX", (0, 0), None), ("TICK", (0,), None),
    ("M", (0,), .1), ("X_ERROR", (0,), None), ("Z_ERROR", (0,), -1),
    ("DEPOLARIZE1", (0,), float("nan")), ("DEPOLARIZE2", (0, 1), 1.1),
    ("H", (True,), None), ("H", (-1,), None)])
def test_invalid_operations(name, targets, p):
    with pytest.raises(ValueError):
        Operation(name, targets, p)


def test_ir_state_targets_and_absolute_references():
    state = StabilizerCode(1, (Pauli.from_label("Z"),))
    valid = CircuitIR(1, 1, state, (Operation("M", (0,)),), ((0,),), ((0,),))
    assert valid.measurement_count == 1
    with pytest.raises(ValueError, match="target"):
        CircuitIR(1, 1, state, (Operation("H", (1,)),))
    with pytest.raises(ValueError, match="full-rank"):
        CircuitIR(1, 1, StabilizerCode(1, ()), ())
    with pytest.raises(ValueError, match="commute"):
        StabilizerCode(1, (Pauli.from_label("X"), Pauli.from_label("Z")))
    for refs in (((1,),), ((-1,),), ((0, 0),)):
        with pytest.raises(ValueError, match="reference"):
            CircuitIR(1, 1, state, valid.operations, refs)
    with pytest.raises(ValueError, match="qubit"):
        CircuitIR(0, 1, state, ())
    assert replace(valid, limits=CircuitLimits(max_qubits=1)).fingerprint() == valid.fingerprint()


def test_streaming_limits_stop_before_consuming_remaining_operations():
    state = StabilizerCode(1, (Pauli.from_label("Z"),))

    def operations():
        yield Operation("M", (0,))
        raise AssertionError("measurement cap must stop before consuming this entry")

    with pytest.raises(ValueError, match="measurements"):
        CircuitIR(1, 1, state, operations(), limits=CircuitLimits(max_measurements=0))


def test_noise_locations_endpoints_and_growth_preflight():
    code = CSSCode(B((3,), 2), B((3,), 2))
    noise = CircuitNoise(1, 1, 1, 1, 1)
    circuit = css_memory(code, rounds=2, basis="x", noise=noise)
    assert circuit.operations[:2] == (Operation("DEPOLARIZE1", (0,), 1),
                                     Operation("DEPOLARIZE1", (1,), 1))
    for i, operation in enumerate(circuit.operations):
        if operation.name == "R":
            assert circuit.operations[i + 1] == Operation("X_ERROR", operation.targets, 1)
        if operation.name == "H":
            assert circuit.operations[i + 1] == Operation("DEPOLARIZE1", operation.targets, 1)
        if operation.name == "CX":
            assert circuit.operations[i + 1] == Operation("DEPOLARIZE2", operation.targets, 1)
        if operation.name in ("M", "MX"):
            assert circuit.operations[i - 1] == Operation("X_ERROR" if operation.name == "M" else "Z_ERROR",
                                                         operation.targets, 1)
    limits = CircuitLimits(max_operations=len(circuit.operations),
                           max_gate_targets=sum(len(o.targets) for o in circuit.operations),
                           max_measurements=circuit.measurement_count,
                           max_detectors=len(circuit.detectors),
                           max_references=sum(map(len, circuit.detectors + circuit.observables)))
    assert css_memory(code, rounds=2, basis="x", noise=noise, limits=limits) == circuit
    for field in ("max_operations", "max_gate_targets", "max_measurements", "max_detectors", "max_references"):
        with pytest.raises(ValueError):
            css_memory(code, rounds=2, basis="x", noise=noise,
                       limits=replace(limits, **{field: getattr(limits, field) - 1}))
    assert Operation("X_ERROR", (0,), 0).probability == 0


def test_guards_run_before_basis_or_schedule_materialization(monkeypatch):
    import qeclang.circuits.schedules as module

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must reject before extracting logicals")

    monkeypatch.setattr(module, "css_logicals", forbidden)
    with pytest.raises(ValueError, match="operations"):
        css_memory(repetition(), rounds=10**12)
    with pytest.raises(ValueError, match="materialization"):
        css_memory(repetition(), max_basis_bits=0)
    with pytest.raises(ValueError, match="preparation"):
        css_memory(repetition(), limits=CircuitLimits(max_preparation_bits=0))


def test_allowed_interaction_graph():
    allowed = ((0, 3), (1, 3), (1, 4), (2, 4))
    assert css_memory(repetition(), allowed_edges=allowed).n_qubits == 5
    with pytest.raises(ValueError, match="absent"):
        css_memory(repetition(), allowed_edges=allowed[:-1])
    with pytest.raises(ValueError, match="interaction"):
        css_memory(repetition(), allowed_edges=((0, 0),))


def sample_ideal(circuit, seed):
    """Independent small state-vector reference for noiseless schedule checks."""
    state = [0j] * (1 << circuit.n_qubits)
    state[0] = 1
    for generator in circuit.preparation.generators:
        transformed = [0j] * len(state)
        for index, amplitude in enumerate(state):
            transformed[index ^ generator.x] = ((1j ** generator.phase) *
                (-1) ** ((index & generator.z).bit_count()) * amplitude)
        state = [a + b for a, b in zip(state, transformed)]
        norm = sqrt(sum(abs(a) ** 2 for a in state))
        state = [a / norm for a in state]
    rng, measurements = Random(seed), []

    def hadamard(target):
        for i in range(len(state)):
            if not (i >> target) & 1:
                j = i | (1 << target)
                a, b = state[i], state[j]
                state[i], state[j] = (a + b) / sqrt(2), (a - b) / sqrt(2)

    def measure(target):
        probability = sum(abs(a) ** 2 for i, a in enumerate(state) if (i >> target) & 1)
        bit = int(rng.random() < probability)
        norm = sqrt(probability if bit else 1 - probability)
        for i in range(len(state)):
            state[i] = state[i] / norm if ((i >> target) & 1) == bit else 0j
        return bit

    for operation in circuit.operations:
        if operation.name == "TICK":
            continue
        target = operation.targets[0]
        if operation.name == "H":
            hadamard(target)
        elif operation.name == "CX":
            control, target = operation.targets
            for i in range(len(state)):
                if ((i >> control) & 1) and not ((i >> target) & 1):
                    j = i | (1 << target)
                    state[i], state[j] = state[j], state[i]
        elif operation.name in ("M", "MX"):
            if operation.name == "MX":
                hadamard(target)
            measurements.append(measure(target))
        elif operation.name == "R":
            if measure(target):
                for i in range(len(state)):
                    if not ((i >> target) & 1):
                        j = i | (1 << target)
                        state[i], state[j] = state[j], state[i]
        else:
            raise AssertionError("reference expects an ideal circuit")
    return measurements


def test_all_noiseless_detector_and_observable_parities():
    codes = (repetition(), CSSCode(B((3,), 2), B((3,), 2)), CSSCode(B((15,), 4), B((15,), 4)))
    for code in codes:
        for basis in ("x", "z"):
            circuit = css_memory(code, basis=basis, rounds=3)
            for seed in range(8):
                outcomes = sample_ideal(circuit, seed)
                assert all(sum(outcomes[i] for i in support) % 2 == 0
                           for support in circuit.detectors + circuit.observables)


def test_empty_code_and_invalid_schedule_options():
    empty = CSSCode(B((), 0), B((), 0))
    circuit = css_memory(empty)
    assert circuit.n_qubits == circuit.n_data == circuit.measurement_count == 0
    assert circuit.operations == (Operation("TICK"),)
    assert circuit.detectors == circuit.observables == ()
    for kwargs in ({"rounds": 0}, {"rounds": True}, {"basis": "y"}):
        with pytest.raises(ValueError):
            css_memory(repetition(), **kwargs)
    with pytest.raises(ValueError):
        CircuitNoise(measurement_flip=float("inf"))
