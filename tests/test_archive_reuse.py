from copy import deepcopy
from dataclasses import replace

import pytest

from qeclang.algebra import BinaryMatrix as B
from qeclang.analysis.certificate_io import presentation_fingerprint, verify_certificate_record
from qeclang.archive import ArchiveLimits, ArchiveRefused, InMemoryEvidenceStore, bundle_id, publish
from qeclang.archive.protocol import EvidenceStoreLimitExceeded
from qeclang.archive.reuse import make_reuse_program
from qeclang.codes import CSSCode
from qeclang.constructions.records import canonical_json
from qeclang.io import from_dict, to_dict
from qeclang.transforms import permute_qubits
from qeclang.workflow import ExecutionLimits, Program, execute, inspect_run
from qeclang.workflow.artifacts import describe


def source_code():
    return CSSCode(B((3, 12), 4), B((15,), 4), "asymmetric checks")


def target_and_maps(source=None):
    source = source_code() if source is None else source
    a, b = source.hx.rows
    z, = source.hz.rows
    target = CSSCode(B((b, a, a ^ b, 0), 4), B((z, z, 0), 4), "redundant target")
    maps = {"x_forward": [[1], [0], [0, 1], []], "x_reverse": [[1], [0]],
            "z_forward": [[0], [0], []], "z_reverse": [[0]]}
    return target, maps


def archived(*, sector="x", prefix=False, witness=True, permuted=False, output_witness=True):
    code = source_code()
    nodes = [{"id": "base", "op": "css.literal", "inputs": [], "params": {"code": to_dict(code)}}]
    source_id = "base"
    if permuted:
        permutation = [1, 2, 3, 0]
        nodes.append({"id": "permuted", "op": "css.permute", "inputs": ["base"],
                      "params": {"permutation": permutation}})
        source_id, code = "permuted", permute_qubits(code, permutation)
    nodes.append({"id": "distance", "op": "css.distance", "inputs": [source_id],
                  "params": {"sector": sector, "max_weight": 1, "max_candidates": 4}})
    if witness:
        nodes.append({"id": "logical", "op": "css.witness", "inputs": [source_id, "distance"], "params": {}})
    if prefix:
        nodes.extend([
            {"id": "stop", "op": "css.sparse", "inputs": [source_id], "params": {}},
            {"id": "later", "op": "css.witness", "inputs": [source_id, "distance"], "params": {}}])
    outputs = ["logical"] if witness and output_witness else [source_id]
    program = Program.from_dict({"schema": "qeclang.program.v1", "nodes": nodes, "outputs": outputs})
    policy = ExecutionLimits(max_validation_work=0) if prefix else ExecutionLimits()
    run = execute(program, limits=policy)
    store = InMemoryEvidenceStore()
    receipt = publish(program, run, store, allow_prefix=prefix)
    return code, program, run, store, receipt


class ReadOnlyView:
    def __init__(self, store):
        self.store = store
        self.reads = 0
    def fetch(self, key):
        self.reads += 1
        return self.store.fetch(key)
    def save(self, *args):
        raise AssertionError("assembling reuse must not write to the archive")


@pytest.mark.parametrize("sector", ["x", "z"])
def test_cross_run_rebinding_with_asymmetric_redundant_checks_and_no_search(sector, monkeypatch):
    source, _, _, store, receipt = archived(sector=sector)
    target, maps = target_and_maps(source)
    view = ReadOnlyView(store)
    def forbidden(*args, **kwargs):
        raise AssertionError("reuse must not run distance search, basis computation, or map discovery")
    import qeclang.workflow.operations as operations
    import qeclang.workflow.row_equivalence as equivalence
    monkeypatch.setattr(operations, "css_distance", forbidden)
    monkeypatch.setattr(operations, "certify_css_logical", forbidden)
    monkeypatch.setattr(equivalence, "row_maps", forbidden)
    for name in ("rank", "solve", "nullspace", "row_basis"):
        monkeypatch.setattr(B, name, forbidden)
    program = make_reuse_program(view, receipt["bundle_id"], "logical", target, maps)
    assert view.reads == 1 and store.bundle_count == 1
    assert [node.op for node in program.nodes] == ["css.literal", "css.import_witness", "css.literal", "css.rebase_witness"]
    assert program.outputs == ("target", "rebased_witness")
    assert program.nodes[1].params["origin"] == {"bundle_id": receipt["bundle_id"], "node": "logical"}
    run = execute(program, limits=ExecutionLimits(max_candidates=0, max_basis_bits=0), verify_relations=False)
    assert run["status"] == "completed"
    assert run["ledger"]["used"]["candidates"] == 0
    assert run["ledger"]["peak_requested"]["basis_bits"] == 0
    certificate = run["artifacts"][run["outputs"]["rebased_witness"]]["payload"]
    assert certificate["source"] == presentation_fingerprint(target)
    assert certificate["source"] != presentation_fingerprint(source)
    assert verify_certificate_record(target, certificate)
    inspected = inspect_run(program, run)
    assert inspected["status"] == "checked"
    assert any(claim["node"] == "rebased_witness" and claim["bound"] == 2 for claim in inspected["claims"])
    assert view.reads == 1  # Execution/inspection uses the embedded certificate, not hidden retrieval.


def test_source_is_selected_through_nominal_binding_not_first_literal_or_outputs():
    source, _, _, store, receipt = archived(permuted=True, output_witness=False)
    target, maps = target_and_maps(source)
    program = make_reuse_program(store, receipt["bundle_id"], "logical", target, maps)
    embedded = from_dict(program.nodes[0].params["code"])
    assert embedded.hx == source.hx and embedded.hz == source.hz
    assert embedded.hx != source_code().hx
    assert execute(program)["status"] == "completed"


def test_checked_prefix_requires_opt_in_and_cannot_supply_unexecuted_witness():
    source, _, _, store, receipt = archived(prefix=True)
    assert receipt["inspection"]["status"] == "checked_prefix"
    target, maps = target_and_maps(source)
    with pytest.raises(ArchiveRefused):
        make_reuse_program(store, receipt["bundle_id"], "logical", target, maps)
    program = make_reuse_program(store, receipt["bundle_id"], "logical", target, maps, allow_prefix=True)
    assert execute(program)["status"] == "completed"
    with pytest.raises(ValueError, match="checked logical upper witness"):
        make_reuse_program(store, receipt["bundle_id"], "later", target, maps, allow_prefix=True)


def test_receipt_does_not_override_fresh_reader_policy():
    source, _, _, store, receipt = archived()
    assert receipt["inspection"]["status"] == "checked"
    target, maps = target_and_maps(source)
    defaults = ArchiveLimits()
    restricted = replace(defaults, inspection=replace(defaults.inspection, max_certificates=0))
    with pytest.raises(ArchiveRefused) as error:
        make_reuse_program(store, receipt["bundle_id"], "logical", target, maps, limits=restricted)
    assert error.value.inspection["status"] == "budget_exhausted"


def test_producer_assurance_cannot_turn_distance_report_into_reusable_witness():
    source, program, run, _, _ = archived(witness=False)
    run["trace"][-1]["evidence"]["assurance"] = "independently_checked"
    store = InMemoryEvidenceStore()
    receipt = publish(program, run, store)
    assert receipt["inspection"]["claims"] == []
    target, maps = target_and_maps(source)
    for node in ("distance", "base", "missing"):
        with pytest.raises(ValueError, match="checked logical upper witness"):
            make_reuse_program(store, receipt["bundle_id"], node, target, maps)


def test_rehashed_invalid_certificate_is_reinspected_instead_of_trusting_old_receipt():
    source, program, run, store, receipt = archived()
    assert receipt["inspection"]["status"] == "checked"
    corrupt = deepcopy(run)
    old_key = corrupt["outputs"]["logical"]
    payload = corrupt["artifacts"][old_key]["payload"]
    payload["certificate"]["dual"] = "0x0"
    altered = describe("witness", payload)
    del corrupt["artifacts"][old_key]
    corrupt["artifacts"][altered["digest"]] = altered
    corrupt["trace"][-1]["output"] = altered["digest"]
    corrupt["outputs"]["logical"] = altered["digest"]
    document = {"schema": "qeclang.evidence_bundle.v1", "program": program.to_dict(), "run": corrupt}
    raw = canonical_json(document).encode("ascii")
    key = bundle_id(raw)
    store.save(key, raw)
    target, maps = target_and_maps(source)
    with pytest.raises(ArchiveRefused) as error:
        make_reuse_program(store, key, "logical", target, maps)
    assert error.value.inspection["status"] == "invalid_record"
    assert store.bundle_count == 2


def test_map_equations_are_deferred_but_mandatory_even_when_optional_checks_are_off():
    source, _, _, store, receipt = archived()
    target, maps = target_and_maps(source)
    maps["x_forward"][0] = []  # Well-shaped, but cannot reproduce the nonzero target row.
    program = make_reuse_program(store, receipt["bundle_id"], "logical", target, maps)
    run = execute(program, verify_relations=False)
    assert run["status"] == "verification_failed"
    assert run["trace"][-1]["node"] == "rebased_witness"
    assert "rebased_witness" not in run["outputs"]


def test_origin_is_a_hint_and_program_inputs_are_detached():
    source, _, _, store, receipt = archived()
    target, maps = target_and_maps(source)
    program = make_reuse_program(store, receipt["bundle_id"], "logical", target, maps)
    maps["x_forward"][0].clear()
    assert program.nodes[-1].params["x_forward"][0] == [1]
    altered = program.to_dict()
    altered["nodes"][1]["params"]["origin"] = {"bundle_id": "f"*64, "node": "unrelated_hint"}
    independent = Program.from_dict(altered)
    run = execute(independent, limits=ExecutionLimits(max_candidates=0, max_basis_bits=0))
    assert run["status"] == "completed"
    assert inspect_run(independent, run)["status"] == "checked"


def test_bad_inputs_and_target_limits_are_rejected_without_fetching():
    target, maps = target_and_maps()
    class NoReader:
        def fetch(self, key):
            raise AssertionError("input admission must precede fetch")
    for invalid in (None, [], {**maps, "extra": []}, {**maps, "x_forward": [[1, 1], [0], [0, 1], []]}):
        with pytest.raises((ValueError, TypeError)):
            make_reuse_program(NoReader(), "0"*64, "logical", target, invalid)
    defaults = ArchiveLimits()
    constrained = replace(defaults, inspection=replace(defaults.inspection,
        artifacts=replace(defaults.inspection.artifacts, max_axis=1)))
    with pytest.raises(EvidenceStoreLimitExceeded):
        make_reuse_program(NoReader(), "0"*64, "logical", target, maps, limits=constrained)
    with pytest.raises(TypeError):
        make_reuse_program(NoReader(), "0"*64, "logical", to_dict(target), maps)


def test_target_physical_coordinates_cannot_change():
    source, _, _, store, receipt = archived()
    target, maps = target_and_maps(source)
    wider = CSSCode(B(target.hx.rows, 5), B(target.hz.rows, 5))
    with pytest.raises(ValueError, match="physical coordinate width"):
        make_reuse_program(store, receipt["bundle_id"], "logical", wider, maps)
