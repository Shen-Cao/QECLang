"""Run a declared circuit-noise experiment; requires qeclang[circuits]."""
import json

from qeclang.circuits import CircuitNoise, css_memory
from qeclang.circuits.stim_sampling import simulate_circuit_memory
from qeclang.constructions import steane


def main():
    noise = CircuitNoise(data_depolarization=.001, reset_flip=.001,
                         one_qubit_depolarization=.001, two_qubit_depolarization=.001,
                         measurement_flip=.001)
    for basis in ("x", "z"):
        source = css_memory(steane(), basis=basis, rounds=3, noise=noise)
        result = simulate_circuit_memory(source, trials=100, seed=7, batch_size=32,
                                          decoder_options={"max_iterations": 10, "fallback": "osd0"})
        print(json.dumps(result.summary(), allow_nan=False))


if __name__ == "__main__":
    main()
