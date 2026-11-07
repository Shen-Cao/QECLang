# CSS memory circuits and detector models

A code's parity checks do not specify a measurement circuit. The circuit layer adds an ordered extraction schedule, an explicit initialization/readout boundary, and a fault model. The initial memory builder is a sequential reference construction. It does not claim an arbitrary-input encoder, a fault-tolerant schedule, preservation of code distance, or a noise threshold.

## Native circuit contract

`CircuitIR` contains an ideal full stabilizer-state preparation on the data qubits, a bounded tuple of typed `Operation` records, and detector/observable measurement parities. Ancillas occupy the remaining qubit indices and initially have state `|0>`. Its operation vocabulary is `R`, `H`, directed `CX`, `M`, `MX`, `X_ERROR`, `Z_ERROR`, `DEPOLARIZE1`, `DEPOLARIZE2`, and `TICK`. Ideal operations and explicit stochastic faults are separate records.

`M` and `MX` produce one bit each, indexed from zero in encounter order. Detector and observable records reference those absolute measurement indices. Supports are sorted, unique, and in range; an empty parity is constant zero. All parity annotations can therefore be emitted after the operations without changing their meaning. This layer validates dimensions, supported operations, target arities, preparation completeness, and record bounds; arbitrary manually supplied detector parities are not automatically proven deterministic.

```python
from qeclang.circuits import CircuitNoise, css_memory
from qeclang.constructions import steane

circuit = css_memory(steane(), basis="z", rounds=3,
                     noise=CircuitNoise(data_depolarization=0.001,
                                        measurement_flip=0.001))
print(circuit.measurement_count, len(circuit.detectors), circuit.fingerprint())
```

A circuit fingerprint identifies the exact description, including operation order, preparation, measurement references, and metadata. It is not a physical-equivalence test. The builder's code-description fingerprint includes the descriptive code name as well as the checks.

## Initialization and extraction

`css_memory` fixes all stabilizer checks and the selected complete logical X or Z frame to positive eigenvalues. Independent stabilizer generators plus the selected logical generators define a unique data state. This is a known-state memory experiment; completing stabilizers to one state does not specify how an arbitrary unknown logical input should be encoded.

The builder retains every input check, including redundant and zero-weight rows, and assigns one ancilla per check. Each round measures X checks followed by Z checks, sequentially:

| Check | Ordered ancilla operations |
|---|---|
| Z support | Reset to zero; `CX(data, ancilla)` for each site; measure ancilla in Z |
| X support | Reset to zero; H; `CX(ancilla, data)` for each site; H; measure ancilla in Z |

Ancillas are reset before reuse in the next round. Gate direction matters even though `allowed_edges`, when supplied, describes undirected permitted interactions. That graph constrains extraction CX operations only. Ideal state preparation is an abstract boundary; a synthesized preparation circuit may use other edges and is outside this hardware constraint.

Each round ends with `TICK`. Here ticks mark rounds, not calibrated durations or optimized parallel gate layers. The builder's sequential order and support order can affect fault propagation. Bounded check weight alone gives no fault-tolerance guarantee for this circuit.

## Detector and logical boundaries

Let `m[r,j]` be the measurement of retained check `j` in round `r`. The initialization fixes its ideal eigenvalue to +1, so the first detector is `m[0,j]`. Subsequent detectors are `m[r,j] XOR m[r-1,j]`.

At the end, all data qubits are destructively measured in the requested memory basis. Only checks of that same basis receive final boundary detectors: XOR their last ancilla record with the final data records on their supports. Checks of the other basis cannot be closed using these data measurements. Logical observables are final-data parities of the chosen same-basis logical representatives. X- and Z-memory experiments are separate; incompatible logical observables are not measured simultaneously.

For `m` retained checks, `n` data qubits, `T` rounds, and `m_b` checks matching the readout basis, there are `Tm+n` measurements, `Tm+m_b` detectors, and `k` logical observables. Redundant check rows remain distinct records. A zero-logical code can define a valid circuit, but an empty observable vector does not establish protected-memory performance.

## Fault placement

`CircuitNoise` defines independent stochastic locations with these exact placements:

| Parameter | Placement and convention |
|---|---|
| `data_depolarization` | DEPOLARIZE1 on every data qubit at the start of each round |
| `reset_flip` | X_ERROR immediately after each ancilla R |
| `one_qubit_depolarization` | DEPOLARIZE1 immediately after each extraction H |
| `two_qubit_depolarization` | DEPOLARIZE2 immediately after each extraction CX |
| `measurement_flip` | X_ERROR immediately before M; Z_ERROR immediately before final MX |

Single-qubit depolarization has total nonidentity Pauli probability `p`, with X/Y/Z each `p/3`. Two-qubit depolarization assigns total probability `p` uniformly to the 15 nonidentity two-qubit Paulis. These are the [Stim gate conventions](https://github.com/quantumlib/Stim/blob/v1.16.0/doc/gates.md).

Premeasurement Pauli faults are equivalent to classical readout flips for this builder because measured ancillas are reset before reuse, final data are discarded, and there is no feedback. A general manually authored circuit retains the literal Pauli-before-measurement semantics. The storage parameter is per round; there is no per-idle-wire timing model. Ideal data initialization carries no faults in this experiment. Zero-probability fault operations are omitted.

## Independent detector-model contract

`DetectorModel` stores two sparse binary maps with a shared fault-column count. For an independent Bernoulli fault vector `f`, detector events are `H f` and logical flips are `L f`. Fault columns represent mechanisms, not physical qubits. A mechanism may touch more than two detectors, only logical observables, or neither. No graphlike decomposition or merging is implied.

`from_errors` accepts `(probability, detector_ids, observable_ids)` records with explicit detector/observable counts. Repeated target IDs cancel over GF(2). Its incidence budget counts raw target visits, so duplicates cannot bypass input limits. Explicit dimensions preserve isolated detectors/observables and zero-error models. The versioned `qeclang.detector_model.v1` format and fingerprint retain the complete ordered `H,L,p` presentation.

A decoder predicts a fault representative consistent with the detector record, then predicts logical flips through `L`. Agreement with `H` alone does not establish logical success. This model assumes independent mechanisms; arbitrary correlated physical noise requires an explicit conversion or a richer model. Native model samples and direct circuit samples must retain different provenance.

## Resource boundaries

`CircuitLimits` guards qubits, operations, targets, measurements, detector/observable counts, measurement references, and preparation-generator payload. The memory builder preflights round-dependent growth before allocating the schedule, and bounds its packed logical-basis analysis separately. Detector models have independent fault/axis/incidence budgets. These are representation counts and payload proxies, not peak-memory or elapsed-time guarantees.

The optional compiler, detector conversion, and sampling interfaces have additional backend-specific limits. A successful bounded computation does not establish asymptotic scaling. Circuit error rates are tied to the declared locations and boundaries; they must not be equated directly with the perfect-syndrome simulator's per-site error probability.

## Optional Stim boundary

`compile_stim(ir)` produces an immutable `CompiledCircuit` text snapshot with the source fingerprint, Stim version, dimensions, and preparation/resource metadata. `CompiledCircuit` is factory-created; `to_stim()` returns a fresh mutable native circuit and rechecks the runtime version and saved circuit dimensions. The compiler prepares the ideal data state by synthesizing a complete signed stabilizer tableau and comparing the synthesized circuit's tableau exactly with that target. It then appends the extraction operations and measurement-record annotations. This does not use Stim's randomized signed-flow checker as an exact proof.

Install the optional backend with `python -m pip install -e '.[circuits]'` from the repository root. The tested backend is Stim 1.16.0. Its [tagged API reference](https://raw.githubusercontent.com/quantumlib/Stim/v1.16.0/doc/python_api_reference_vDev.md) distinguishes state-compatible tableaux, Clifford synthesis, detector sampling, and detector error models. Synthesized circuit text and seeded samples are not promised stable across Stim versions. Sampling can also depend on SIMD capabilities and the sequence/batching of calls; archive the artifact and sampling configuration rather than claiming cross-machine bitwise reproducibility.

`compile_detector_model` requests `decompose_errors=False`, `allow_gauge_detectors=False`, and `approximate_disjoint_errors=False`. Nondeterministic detector declarations or unsupported exact model conversions raise errors. No approximation or matching conversion is substituted. The native model importer preserves whole independent fault mechanisms, supports bounded repeat/offset expansion, and rejects separator decompositions and unsupported instructions. Raw-model imports require an explicit `gauge_free=True` provenance assertion: an isolated probability of one half cannot distinguish a physical fault from random gauge information. The [DEM format](https://github.com/quantumlib/Stim/blob/v1.16.0/doc/file_format_dem_detector_error_model.md) defines those mechanism semantics.

Explicit compiled detector/observable counts preserve trailing isolated nodes and error-free models. In a generic signed-preparation IR, a deterministic noiseless measurement parity can be one. Stim's detector and logical outputs are flips relative to its reference, not necessarily raw measurement XORs. The compiler does not force an all-zero reference. The CSS memory builder's positive initialization makes its stated boundary parities zero in the noiseless experiment.

Native preparation has qubit/tableau and Pauli-string preflight limits; total qubits include ancillas. Synthesized instruction/target counts are checked after synthesis. Detector analysis has an additional conservative fault-term times parity-count work proxy. Import bounds expanded instructions, targets, faults, nesting, and dimensions before repeat expansion and sparse-row construction. Existing native objects and per-instruction target/argument copies are outside a hard memory cap; foreign synthesis, analysis workspace, and elapsed time are not bounded by these count limits.

## Logical evaluation

`DetectorEvaluator` accepts streamed detector/observable samples and a BP decoder matched to the exact model and its probabilities. It independently checks the returned fault vector against the original detector equations before computing predicted logical flips. Decoder failures and wrong logical predictions are disjoint failure categories; their sum defines the reported failure rate and Wilson interval. Optional caching stores scalar predictions, not BP message arrays.

`simulate_detector_memory` samples the native independent-mechanism model. This is distinct from direct Stim circuit sampling; evaluation provenance distinguishes `independent_binary_model`, `stim_circuit`, and external records. Native sampling consumes one RNG draw per mechanism per trial, including deterministic endpoints, so its seeded stream does not depend on batch size. A native-model experiment is not automatically evidence about an unprovided physical circuit.

Memory evaluation requires at least one declared logical observable and at least one evaluated sample. This prevents vacuous success for zero-logical circuits. Streamed input widths and lengths, sample counts, cache size, RNG draw count, and output batch size have explicit limits. An interrupted or malformed input retains only the successfully evaluated prefix; the caller must request any resulting partial statistics explicitly.

Direct circuit sampling is available through the optional backend:

```python
from qeclang.circuits.stim_sampling import simulate_circuit_memory

result = simulate_circuit_memory(circuit, trials=100, seed=17, batch_size=25,
                                  decoder_options={"fallback": "osd0"})
print(result.summary())
```

This compiles the circuit when necessary, obtains its strict detector model, samples actual circuit detector/logical flips, and evaluates predictions against those observed flips. `max_batch_bytes` bounds packed uint8 outputs including padding; `max_record_frame_bits` separately bounds a conservative native record/frame proxy. These do not bound all foreign workspace. The result retains the artifact/model hashes, version, architecture, decoder settings, noise/boundary metadata, batch size, and relevant limits. With a precompiled artifact, its saved compiler limits apply; the `compile_limits` argument is used only when the source is a native IR.

A trial covers the entire requested memory experiment. Its failure rate is not automatically a per-round rate and should not be divided by the round count without a stated statistical model. A nonzero upper confidence limit remains appropriate after zero observed failures. Arbitrary Kraus noise, measurement feedback, gauge protocols, calibrated idle timing, and an optimized extraction schedule are outside this initial circuit model.

## Command-line experiment

```sh
mkdir -p output
qeclang circuit-memory --code steane --basis z --rounds 3 --trials 100 \
  --seed 7 --batch-size 25 --data-error 0.001 --measurement-error 0.001 \
  --fallback osd0 --stim-output output/memory.stim
```

The command prints the experiment summary and optionally writes the exact compiled circuit whose hash appears in that summary. `--input` accepts a CSS code JSON instead of a named fixture. Separate options specify reset, one-qubit, two-qubit, storage, and measurement error probabilities; all default to zero. `--max-operations`, `--max-tableau-bits`, `--max-dem-work`, and `--max-batch-bytes` expose the principal CLI budgets. `examples/circuit_memory.py` runs both memory bases as separate experiments. Generated circuit files and results belong in ignored output directories.
