"""Assemble explicit cross-run witness reuse after fresh archive inspection.

Only a checked finite CSS upper witness is extracted. The assembled program
verifies the supplied bidirectional generator maps when executed; assembly does
not search for equivalent presentations, solve equations, or write storage.
"""
from dataclasses import fields

from qeclang.analysis.certificate_io import presentation_fingerprint, verify_certificate_record
from qeclang.codes import CSSCode
from qeclang.workflow import Program
from qeclang.workflow.artifacts import describe
from qeclang.workflow.import_artifact import ImportBudget, import_artifact
from qeclang.workflow.json_input import JSONLimits, validate_json_data
from .bundles import ArchiveLimits, retrieve
from .protocol import EvidenceStoreLimitExceeded, validate_key

_MAPS = frozenset(("x_forward", "x_reverse", "z_forward", "z_reverse"))


def _json_policy(limits):
    return JSONLimits(**{item.name: getattr(limits.program, item.name) for item in fields(JSONLimits)})


def _map_admission(maps, limits):
    validate_json_data(maps, limits=_json_policy(limits))
    if type(maps) is not dict or set(maps) != _MAPS:
        raise ValueError("reuse requires all four explicit forward/reverse X/Z maps")
    for matrix in maps.values():
        if type(matrix) is not list:
            raise ValueError("generator maps must be arrays of support rows")
        for row in matrix:
            if type(row) is not list:
                raise ValueError("generator map rows must be arrays of indices")
            previous = -1
            for index in row:
                if type(index) is not int or index <= previous:
                    raise ValueError("generator indices must be sorted, distinct nonnegative integers")
                previous = index


def _target_payload(target, limits):
    if not isinstance(target, CSSCode):
        raise TypeError("reuse target must be CSSCode; load support records explicitly")
    n, mx, mz = target.n, target.hx.nrows, target.hz.nrows
    admission = limits.inspection.artifacts
    if (max(n, mx, mz) > admission.max_axis or n > limits.program.max_qubits
            or mx + mz > limits.program.max_checks):
        raise EvidenceStoreLimitExceeded("reuse target exceeds the reader/program dimension limits")
    if (mx + mz)*n + mx*mz > admission.max_matrix_bits:
        raise EvidenceStoreLimitExceeded("reuse target exceeds the reader matrix payload limit")
    if target.hx.nnz + target.hz.nnz + mx + mz > limits.program.max_payload_items:
        raise EvidenceStoreLimitExceeded("reuse target exceeds the program item budget")
    return describe("css", target, limits=_json_policy(limits))["payload"]


def _select_checked_witness(received, witness_node, limits):
    inspection = received["inspection"]
    if inspection["status"] not in ("checked", "checked_prefix"):
        raise ValueError("reuse requires a freshly checked archive result")
    claims = [claim for claim in inspection["claims"]
              if claim.get("node") == witness_node and claim.get("scope") == "instance"
              and claim.get("predicate") == "distance_upper_bound"]
    if len(claims) != 1:
        raise ValueError("selected node has no unique independently checked logical upper witness")
    claim = claims[0]
    program = Program.from_dict(received["program"], limits=limits.program)
    nodes = {node.id: node for node in program.nodes}
    node = nodes.get(witness_node)
    if node is None or node.result_type != "witness" or node.code_id is None:
        raise ValueError("selected node is not a source-bound witness")
    source_node = nodes.get(node.code_id)
    if source_node is None or source_node.result_type != "css":
        raise ValueError("witness source is not an archived CSS presentation")
    run = received["run"]
    completed = {step["node"]: step for step in run["trace"] if step["status"] == "completed"}
    if node.id not in completed or source_node.id not in completed:
        raise ValueError("selected witness or source is outside the completed archived prefix")
    witness_key = completed[node.id]["output"]
    if claim["evidence"] != witness_key:
        raise ValueError("checked claim is not bound to the selected witness artifact")
    source_key = completed[source_node.id]["output"]
    policy = limits.inspection.artifacts
    budget = ImportBudget(max_artifact_bytes=policy.max_bytes,
                          max_validation_work=policy.max_validation_work)
    source = import_artifact(run["artifacts"][source_key], "css", limits=policy, budget=budget)
    certificate = import_artifact(run["artifacts"][witness_key], "witness", limits=policy, budget=budget)
    if (certificate["kind"] != "css_logical"
            or certificate["source"] != presentation_fingerprint(source)
            or certificate["source"] != claim["source"]
            or certificate["claim"]["sector"] != claim["sector"]
            or certificate["claim"]["sector"] != node.sector
            or certificate["claim"]["distance_upper_bound"] != claim["bound"]
            or not verify_certificate_record(source, certificate, max_axis=source.n, max_bits=2*source.n)):
        raise ValueError("archived certificate does not verify against its bound CSS source")
    return source, certificate


def make_reuse_program(store, key, witness_node, target, maps, *, allow_prefix=False,
                       limits=ArchiveLimits()) -> Program:
    """Read and recheck a witness, then assemble a four-node data-only program.

    target must be a CSSCode. Each map is an array of sorted generator-index
    supports. Forward maps express target rows in source rows; reverse maps
    express source rows in target rows. Physical coordinates remain fixed.

    The returned program validates map shapes statically and checks their actual
    equations at mandatory css.rebase_witness execution/inspection. It contains
    no distance operation. Origin bundle/node fields are provenance hints, not
    signatures or verification shortcuts. No store.save call is made.
    """
    if not isinstance(limits, ArchiveLimits) or type(allow_prefix) is not bool:
        raise TypeError("reuse requires ArchiveLimits and a boolean prefix policy")
    validate_key(key)
    if type(witness_node) is not str or not witness_node or len(witness_node) > limits.program.max_id_length:
        raise ValueError("witness node must be a bounded nonempty identifier")
    _map_admission(maps, limits)
    target_record = _target_payload(target, limits)
    received = retrieve(store, key, limits=limits, allow_prefix=allow_prefix)
    source, certificate = _select_checked_witness(received, witness_node, limits)
    if target.n != source.n:
        raise ValueError("generator rebasing requires identical physical coordinate width")
    source_record = describe("css", source, limits=_json_policy(limits))["payload"]
    return Program.from_dict({"schema": "qeclang.program.v1", "nodes": [
        {"id": "source", "op": "css.literal", "inputs": [], "params": {"code": source_record}},
        {"id": "imported_witness", "op": "css.import_witness", "inputs": ["source"],
         "params": {"certificate": certificate, "origin": {"bundle_id": key, "node": witness_node}}},
        {"id": "target", "op": "css.literal", "inputs": [], "params": {"code": target_record}},
        {"id": "rebased_witness", "op": "css.rebase_witness",
         "inputs": ["source", "target", "imported_witness"], "params": maps},
    ], "outputs": ["target", "rebased_witness"]}, limits=limits.program)


__all__ = ["make_reuse_program"]
