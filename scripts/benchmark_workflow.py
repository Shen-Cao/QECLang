"""Emit controlled workflow timing observations and a relational-corruption fixture.

Outputs are JSON Lines on stdout. This is a warm, single-process microbenchmark,
not a speedup assertion. Computation, materialization, serialization, strict
replay, and standalone inspection are deliberately separate timed scopes.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from random import Random
from statistics import median
from time import get_clock_info, perf_counter_ns
from unittest.mock import patch

from qeclang.algebra import BinaryMatrix
from qeclang.analysis import css_distance
from qeclang.analysis.certificate_io import certificate_record, presentation_fingerprint
from qeclang.analysis.certificates import certify_css_logical
from qeclang.codes import CSSCode, SparseCSSCode
from qeclang.constructions import steane, toric
from qeclang.constructions.records import canonical_json
from qeclang.io import from_dict, to_dict
from qeclang.provenance import runtime_provenance
from qeclang.transforms import permute_qubits
from qeclang.workflow import ExecutionLimits, Program, execute, replay
from qeclang.workflow.artifacts import describe
from qeclang.workflow.inspect_run import InspectionLimits, inspect_run
from qeclang.workflow.relations import transport_witness

SCHEMA = "qeclang.workflow.benchmark.v1"
CASES = ("steane-exact", "steane-bounded", "toric2")
OPERATIONS = ("css.literal", "css.chain", "chain.css", "css.distance", "css.witness",
              "css.permute", "css.sparse", "sparse.css", "css.transport")


def _node(name, op, inputs=(), **params):
    return {"id": name, "op": op, "inputs": list(inputs), "params": params}


def make_program(code, *, max_weight, max_candidates):
    permutation = list(range(1, code.n)) + [0]
    return Program.from_dict({"schema": "qeclang.program.v1", "nodes": [
        _node("base", "css.literal", code=to_dict(code)),
        _node("complex", "css.chain", ("base",)),
        _node("original", "chain.css", ("complex",)),
        _node("distance", "css.distance", ("original",), sector="x",
              max_weight=max_weight, max_candidates=max_candidates),
        _node("logical", "css.witness", ("original", "distance")),
        _node("relabeled", "css.permute", ("original",), permutation=permutation),
        _node("sparse", "css.sparse", ("relabeled",)),
        _node("final", "sparse.css", ("sparse",)),
        _node("transported", "css.transport", ("original", "final", "logical"),
              permutation=permutation),
    ], "outputs": ["distance", "final", "transported"]})


def make_case(name):
    if name == "steane-exact":
        code, weight, candidates = steane(), 2, 28
    elif name == "steane-bounded":
        code, weight, candidates = steane(), 0, 0
    elif name == "toric2":
        code, weight, candidates = toric(2), 1, 8
    else:
        raise ValueError("unknown benchmark case")
    return code, make_program(code, max_weight=weight, max_candidates=candidates)


def direct_kernels(program, limits):
    """Manual baseline for this exact DAG, without workflow execution machinery."""
    if tuple(node.op for node in program.nodes) != OPERATIONS:
        raise ValueError("direct baseline requires the benchmark's fixed transport DAG")
    nodes = program.nodes
    base = from_dict(nodes[0].params["code"], max_qubits=limits.max_axis,
                     max_checks=limits.max_axis, max_bits=limits.max_matrix_bits)
    chain = base.to_chain_complex()
    original = chain.to_css()
    options = nodes[3].params
    result = css_distance(original, options["sector"], max_weight=options["max_weight"],
                          max_candidates=min(options["max_candidates"], limits.max_candidates))
    distance = {**asdict(result), "source": presentation_fingerprint(original),
                "sector": options["sector"]}
    cert = certify_css_logical(original, options["sector"], result.witness)
    witness = certificate_record(original, cert)
    permuted = permute_qubits(original, nodes[5].params["permutation"])
    sparse = SparseCSSCode.from_packed(permuted, max_validation_work=limits.max_validation_work)
    final = sparse.to_packed(max_bits=limits.max_matrix_bits)
    transported = transport_witness(original, final, nodes[8].params["permutation"], witness)
    return dict(zip((node.id for node in nodes),
                    (base, chain, original, distance, witness, permuted, sparse, final, transported)))


def materialize(program, values):
    artifacts, bindings = {}, {}
    for node in program.nodes:
        artifact = describe(node.result_type, values[node.id])
        artifacts[artifact["digest"]] = artifact
        bindings[node.id] = artifact["digest"]
    return {"artifacts": artifacts, "outputs": {node: bindings[node] for node in program.outputs}}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def benchmark_case(name, *, repeats=7, warmups=1, order_seed=17,
                   limits=ExecutionLimits(), inspection_limits=InspectionLimits()):
    if type(repeats) is not int or repeats < 1 or type(warmups) is not int or warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    if type(order_seed) is not int:
        raise ValueError("order_seed must be an integer")
    provenance = runtime_provenance()
    code, program = make_case(name)
    declaration = program.to_dict()
    expected_checked = execute(program, limits=limits)
    expected_unchecked = execute(program, limits=limits, verify_relations=False)
    for record in (expected_checked, expected_unchecked):
        _require(record["status"] == "completed", "benchmark policy refused the reference program")
    expected_materialized = materialize(program, direct_kernels(program, limits))
    for record in (expected_checked, expected_unchecked):
        _require(all(record[key] == expected_materialized[key] for key in expected_materialized),
                 "direct and workflow artifacts differ")
    expected_inspection = inspect_run(program, expected_checked, limits=inspection_limits)
    _require(expected_inspection["status"] == "checked", "reference inspection failed")
    expected_json = canonical_json(expected_checked)
    calls = {
        "program_parse": lambda: Program.from_dict(declaration),
        "direct_kernels": lambda: direct_kernels(program, limits),
        "direct_kernels_and_artifacts": lambda: materialize(program, direct_kernels(program, limits)),
        "program_unchecked": lambda: execute(program, limits=limits, verify_relations=False),
        "program_checked": lambda: execute(program, limits=limits),
        "record_serialization": lambda: canonical_json(expected_checked),
        "strict_replay": lambda: replay(program, expected_checked, limits=limits),
        "standalone_inspection": lambda: inspect_run(program, expected_checked, limits=inspection_limits),
    }

    def validate(mode, value):
        if mode == "program_parse":
            valid = value.fingerprint() == program.fingerprint()
        elif mode == "direct_kernels":
            valid = materialize(program, value) == expected_materialized
        elif mode == "direct_kernels_and_artifacts":
            valid = value == expected_materialized
        elif mode == "program_checked":
            valid = value == expected_checked
        elif mode == "program_unchecked":
            valid = value == expected_unchecked
        elif mode == "record_serialization":
            valid = value == expected_json
        elif mode == "strict_replay":
            valid = value is True
        else:
            valid = value == expected_inspection
        _require(valid, f"correctness comparison failed in {mode}")

    for _ in range(warmups):
        for mode, call in calls.items():
            validate(mode, call())
    observations = []
    rng = Random(order_seed)
    for repetition in range(repeats):
        order = list(calls)
        rng.shuffle(order)
        for mode in order:
            start = perf_counter_ns()
            value = calls[mode]()
            seconds = (perf_counter_ns() - start) / 1_000_000_000
            validate(mode, value)  # Correctness checks are outside every timed interval.
            observations.append({"repetition": repetition, "mode": mode, "seconds": seconds})
    _require(runtime_provenance() == provenance, "source/runtime changed during benchmark")
    timing = {}
    for mode in calls:
        values = [row["seconds"] for row in observations if row["mode"] == mode]
        timing[mode] = {"median_seconds": median(values), "minimum_seconds": min(values),
                        "maximum_seconds": max(values), "observations": len(values)}
    clock = get_clock_info("perf_counter")
    return {"schema": SCHEMA, "kind": "timing", "case": name,
            "n": code.n, "k": code.k, "program": declaration,
            "program_fingerprint": program.fingerprint(), "runtime": provenance,
            "benchmark_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "execution_limits": asdict(limits), "inspection_limits": asdict(inspection_limits),
            "repeats": repeats, "warmups": warmups, "order_seed": order_seed,
            "clock": {"name": "perf_counter_ns", "resolution_seconds": clock.resolution},
            "timings": timing, "observations": observations,
            "checked_ledger": expected_checked["ledger"],
            "unchecked_ledger": expected_unchecked["ledger"],
            "reference_run_sha256": sha256(expected_json.encode()).hexdigest(),
            "reference_run_bytes": len(expected_json.encode()),
            "distance": expected_checked["artifacts"][expected_checked["outputs"]["distance"]]["payload"],
            "inspection": expected_inspection}


def corruption_fixture(*, limits=ExecutionLimits(), inspection_limits=InspectionLimits()):
    """Inject a structurally valid wrong producer; execute computes all hashes itself."""
    import qeclang.workflow.executor as executor
    original = CSSCode(BinaryMatrix((3,), 4), BinaryMatrix((12,), 4))
    wrong = CSSCode(BinaryMatrix((5,), 4), BinaryMatrix((10,), 4))
    identity = list(range(4))
    declaration = {"schema": "qeclang.program.v1", "nodes": [
        _node("base", "css.literal", code=to_dict(original)),
        _node("target", "css.permute", ("base",), permutation=identity)],
        "outputs": ["base", "target"]}
    bare = Program.from_dict(declaration)
    producer = executor.evaluate
    def faulty(op, args, params, ledger):
        return wrong if op == "css.permute" else producer(op, args, params, ledger)
    with patch.object(executor, "evaluate", faulty):
        checked = execute(bare, limits=limits)
        unchecked = execute(bare, limits=limits, verify_relations=False)
    inspected = inspect_run(bare, unchecked, limits=inspection_limits)
    _require((original.n, original.k) == (wrong.n, wrong.k), "fixture changed scalar parameters")
    _require(checked["status"] == "verification_failed", "optional relation gate did not reject")
    _require(unchecked["status"] == "completed", "unchecked control failed unexpectedly")
    _require(inspected["status"] == "invalid_record" and "relationship" in inspected["reason"],
             "standalone inspection did not reject the wrong relationship")
    for name, code in (("base", original), ("target", wrong)):
        artifact = describe("css", code)
        _require(unchecked["outputs"][name] == artifact["digest"] and
                 unchecked["artifacts"][artifact["digest"]] == artifact,
                 "corrupted-control artifact hashes were not recomputed correctly")
    with_witness = deepcopy(declaration)
    with_witness["nodes"].extend([
        _node("distance", "css.distance", ("base",), sector="x", max_weight=0, max_candidates=0),
        _node("logical", "css.witness", ("base", "distance")),
        _node("transported", "css.transport", ("base", "target", "logical"), permutation=identity)])
    with_witness["outputs"] = ["transported"]
    mandatory_program = Program.from_dict(with_witness)
    with patch.object(executor, "evaluate", faulty):
        mandatory = execute(mandatory_program, limits=limits, verify_relations=False)
    _require(mandatory["status"] == "verification_failed" and
             mandatory["trace"][-1]["op"] == "css.transport", "mandatory transport gate did not reject")
    return {"schema": SCHEMA, "kind": "corruption", "runtime": runtime_provenance(),
            "injection": "wrong css.permute producer inside process-local mock context",
            "source": to_dict(original), "wrong_target": to_dict(wrong),
            "source_parameters": {"n": original.n, "k": original.k},
            "target_parameters": {"n": wrong.n, "k": wrong.k},
            "declared_permutation": identity, "artifact_hashes_recomputed": True,
            "execution_limits": asdict(limits), "inspection_limits": asdict(inspection_limits),
            "checked_status": checked["status"], "unchecked_status": unchecked["status"],
            "inspection": inspected, "mandatory_transport_status": mandatory["status"],
            "mandatory_transport_step": mandatory["trace"][-1],
            "unchecked_control_record": unchecked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--order-seed", type=int, default=17)
    parser.add_argument("--max-candidates", type=int, default=100_000)
    args = parser.parse_args()
    limits = ExecutionLimits(max_candidates=args.max_candidates)
    for case in args.cases:
        print(json.dumps(benchmark_case(case, repeats=args.repeats, warmups=args.warmups,
                                         order_seed=args.order_seed, limits=limits),
                         sort_keys=True, allow_nan=False))
    print(json.dumps(corruption_fixture(limits=limits), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
