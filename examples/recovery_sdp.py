"""Compare small fixed-encoding recovery objectives under explicitly named noise."""
import argparse
from dataclasses import asdict
import json

import numpy as np

from qeclang.recovery import optimize_recovery
from qeclang.semantic import (IsometryCode, KrausChannel, amplitude_damping,
                              dephasing, depolarizing, entanglement_fidelity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probabilities", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    parser.add_argument("--encoding", choices=("identity", "repetition"), default="identity")
    parser.add_argument("--noise", choices=("phase_flip", "amplitude_damping", "depolarizing"),
                        default="phase_flip")
    parser.add_argument("--solver", default="SCS")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    sites = 1 if args.encoding == "identity" else 3
    v = np.zeros((2**sites, 2))
    v[0, 0] = v[-1, 1] = 1
    code = IsometryCode((2,)*sites, v)
    factory = {"phase_flip": dephasing, "amplitude_damping": amplitude_damping,
               "depolarizing": depolarizing}[args.noise]
    for probability in args.probabilities:
        single = factory(probability)
        noise = single
        for _ in range(sites-1):
            noise = noise.tensor(single)
        result = optimize_recovery(code, noise, solver=args.solver)
        record = {"encoding": args.encoding, "noise": args.noise,
                  "probability": probability, "recovery": result.summary()}
        if args.extract and result.choi is not None:
            recovered = result.as_channel(repair_tolerance=1e-6, tolerance=1e-6)
            effective = KrausChannel((v,)).then(noise).then(recovered)
            record.update(extracted_channel_fidelity=entanglement_fidelity(effective),
                          conversion=asdict(recovered.conversion_report))
        print(json.dumps(record, allow_nan=False))


if __name__ == "__main__":
    main()
