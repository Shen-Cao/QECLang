from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from qeclang.algebra import BinaryMatrix as B
from qeclang.analysis.certificate_io import certificate_record, presentation_fingerprint
from qeclang.analysis.certificates import certify_css_logical
from qeclang.analysis.low_weight import low_weight_id
from qeclang.archive import ArchiveLimits, ArchiveRefused, InMemoryEvidenceStore, bundle_id, publish
from qeclang.archive.distance_reuse import make_distance_reuse_program
from qeclang.codes import CSSCode
from qeclang.constructions.records import canonical_json
from qeclang.io import to_dict
from qeclang.transforms import permute_qubits
from qeclang.workflow import ExecutionLimits, Program, execute, inspect_run
from qeclang.workflow.artifacts import describe

EXAMPLE = Path(__file__).parents[1] / "examples/programs/exact_steane.json"


def source_code():
    # Steane plus a Bell pair, |+>, and |0>: both sectors need zero/star memberships.
    return CSSCode(B((85, 102, 120, 384, 512), 11), B((85, 102, 120, 384, 1024), 11))


def archived(*, prefix=False, bounded=False):
    code = source_code()
    data = Program.load(EXAMPLE).to_dict()
    data["nodes"][0]["params"]["code"] = to_dict(code)
    if bounded:
        upper = next(node for node in data["nodes"] if node["id"] == "x_upper")
        upper.update(op="css.import_witness", inputs=["code"],
                     params={"certificate": certificate_record(code, certify_css_logical(code, "x", 127))})
    if prefix:
        data["nodes"].extend([
            {"id": "stop", "op": "css.sparse", "inputs": ["code"], "params": {}},
            {"id": "later", "op": "css.full_distance",
             "inputs": ["code", "x_distance", "z_distance"], "params": {}}])
    data["outputs"] = ["code"]  # Reusable evidence need not have been a requested output.
    program = Program.from_dict(data)
    policy = ExecutionLimits(max_validation_work=0) if prefix else ExecutionLimits()
    run = execute(program, limits=policy)
    store = InMemoryEvidenceStore()
    key = publish(program, run, store, allow_prefix=prefix)["bundle_id"]
    return code, program, run, store, key


def target_relation(source, kind):
    if kind == "permutation":
        permutation = list(reversed(range(source.n)))
        return permute_qubits(source, permutation), {"relation": kind, "permutation": permutation}
    def change(matrix):
        return B(tuple(reversed(matrix.rows)) + (matrix.rows[0] ^ matrix.rows[1], 0, matrix.rows[0]), source.n)
    target = CSSCode(change(source.hx), change(source.hz))
    forward = [[4], [3], [2], [1], [0], [0, 1], [], [0]]
    reverse = [[4], [3], [2], [1], [0]]
    return target, {"relation": kind, "maps": {
        "x_forward": forward, "x_reverse": reverse,
        "z_forward": deepcopy(forward), "z_reverse": deepcopy(reverse)}}


class ReadOnlyView:
    def __init__(self, store):
        self.store, self.reads = store, 0
    def fetch(self, key):
        self.reads += 1
        return self.store.fetch(key)
    def save(self, *args):
        raise AssertionError("distance reuse must not write storage")


@pytest.mark.parametrize("kind", ["permutation", "generator_spaces"])
@pytest.mark.parametrize("selected,sector", [("x_distance", "x"), ("z_distance", "z"), ("distance", "pauli")])
def test_actual_premises_reused_without_any_producer_or_elimination(kind, selected, sector, monkeypatch):
    source, _, _, store, key = archived()
    target, relation = target_relation(source, kind)
    view = ReadOnlyView(store)
    def forbidden(*args, **kwargs):
        raise AssertionError("interval reuse must not recompute proofs, search, solve, or discover maps")
    import qeclang.analysis.low_weight as low_weight
    import qeclang.workflow.operations as operations
    import qeclang.workflow.row_equivalence as row_equivalence
    monkeypatch.setattr(low_weight, "certify_low_weight", forbidden)
    monkeypatch.setattr(operations, "css_distance", forbidden)
    monkeypatch.setattr(operations, "certify_css_logical", forbidden)
    monkeypatch.setattr(row_equivalence, "row_maps", forbidden)
    for name in ("rank", "solve", "rref", "row_basis", "nullspace", "contains"):
        monkeypatch.setattr(B, name, forbidden)
    program = make_distance_reuse_program(view, key, selected, target, relation)
    assert program.outputs == ("target", "transported_distance")
    exclusions = [node for node in program.nodes if node.op == "css.import_exclusion"]
    assert len(exclusions) == (2 if sector == "pauli" else 1)
    assert all(node.params["record"]["memberships"] == [[4], [3]] for node in exclusions)
    assert all(node.params["origin"]["bundle_id"] == key for node in exclusions)
    run = execute(program, verify_relations=False,
                  limits=ExecutionLimits(max_candidates=0, max_exclusion_steps=0, max_basis_bits=0))
    assert run["status"] == "completed", run["trace"][-1]
    assert run["ledger"]["used"]["candidates"] == run["ledger"]["used"]["exclusion_steps"] == 0
    assert run["ledger"]["peak_requested"]["basis_bits"] == 0
    inspection = inspect_run(program, run)
    assert inspection["status"] == "checked", inspection
    claim, = [claim for claim in inspection["claims"] if claim["node"] == "transported_distance"]
    assert claim["source"] == presentation_fingerprint(target) != presentation_fingerprint(source)
    assert claim["sector"] == sector and claim["status"] == "exact" and claim["value"] == 3
    assert view.reads == store.bundle_count == 1


def test_bounded_interval_is_reused_without_promotion_to_exact():
    source, _, _, store, key = archived(bounded=True)
    target, relation = target_relation(source, "generator_spaces")
    program = make_distance_reuse_program(store, key, "x_distance", target, relation)
    inspection = inspect_run(program, execute(program))
    claim = inspection["claims"][-1]
    assert inspection["status"] == "checked"
    assert (claim["lower_bound"], claim["upper_bound"], claim["status"], claim["value"]) == (3, 7, "bounded", None)


def test_completed_prefix_requires_opt_in_and_unexecuted_join_is_refused():
    source, _, _, store, key = archived(prefix=True)
    target, relation = target_relation(source, "permutation")
    with pytest.raises(ArchiveRefused):
        make_distance_reuse_program(store, key, "distance", target, relation)
    program = make_distance_reuse_program(store, key, "distance", target, relation, allow_prefix=True)
    assert inspect_run(program, execute(program))["status"] == "checked"
    with pytest.raises(ValueError, match="completed checked archive prefix"):
        make_distance_reuse_program(store, key, "later", target, relation, allow_prefix=True)


def test_current_reader_policy_overrides_prior_publication():
    source, _, _, store, key = archived()
    target, relation = target_relation(source, "permutation")
    defaults = ArchiveLimits()
    strict = replace(defaults, inspection=replace(defaults.inspection, max_exclusion_verify_steps=0))
    with pytest.raises(ArchiveRefused) as error:
        make_distance_reuse_program(store, key, "distance", target, relation, limits=strict)
    assert error.value.inspection["status"] == "budget_exhausted"


def test_rehashed_invalid_archived_membership_is_not_reused():
    source, program, run, store, _ = archived()
    corrupted = deepcopy(run)
    step = next(item for item in corrupted["trace"] if item["node"] == "x_exclusion")
    old = step["output"]
    record = corrupted["artifacts"][old]["payload"]
    record["memberships"][0] = []
    record["record_id"] = low_weight_id(record)
    description = describe("exclusion", record)
    corrupted["artifacts"].pop(old)
    corrupted["artifacts"][description["digest"]] = description
    step["output"] = description["digest"]
    for item in corrupted["trace"]:
        item["inputs"] = [description["digest"] if value == old else value for value in item["inputs"]]
    raw = canonical_json({"schema": "qeclang.evidence_bundle.v1", "program": program.to_dict(),
                          "run": corrupted}).encode("ascii")
    key = bundle_id(raw)
    store.save(key, raw)
    target, relation = target_relation(source, "permutation")
    with pytest.raises(ArchiveRefused) as error:
        make_distance_reuse_program(store, key, "distance", target, relation)
    assert error.value.inspection["status"] == "invalid_record"


@pytest.mark.parametrize("kind", ["permutation", "generator_spaces"])
def test_well_shaped_false_relation_is_deferred_but_mandatory(kind):
    source, _, _, store, key = archived()
    target, relation = target_relation(source, kind)
    if kind == "permutation":
        relation["permutation"] = list(range(source.n))
    else:
        relation["maps"]["x_forward"][0] = []
    program = make_distance_reuse_program(store, key, "distance", target, relation)
    run = execute(program, verify_relations=False)
    assert run["status"] == "verification_failed"
    assert "transported_distance" not in run["outputs"]


def test_native_reconstruction_from_imports_is_allowed_but_transport_ancestry_is_explicitly_refused():
    source, _, _, store, key = archived()
    target, relation = target_relation(source, "permutation")
    first = make_distance_reuse_program(store, key, "distance", target, relation)
    key2 = publish(first, execute(first), store)["bundle_id"]
    again = make_distance_reuse_program(store, key2, "source_distance", target, relation)
    assert inspect_run(again, execute(again))["status"] == "checked"
    with pytest.raises(ValueError, match="transported interval ancestry is not reconstructed"):
        make_distance_reuse_program(store, key2, "transported_distance", target,
                                    {"relation": "permutation", "permutation": list(range(target.n))})
    for selected in ("x_report", "x_upper", "code"):
        with pytest.raises(ValueError, match="supports native"):
            make_distance_reuse_program(store, key, selected, target, relation)


def test_input_admission_precedes_reads_and_program_detaches_relation_data():
    class NoReader:
        def fetch(self, key):
            raise AssertionError("invalid input should be rejected before archive I/O")
    source = source_code()
    for relation in (None, {}, {"relation": "permutation", "permutation": [0] * source.n},
                     {"relation": "generator_spaces", "maps": {}}):
        with pytest.raises(ValueError):
            make_distance_reuse_program(NoReader(), "0" * 64, "distance", source, relation)
    source, _, _, store, key = archived()
    target, relation = target_relation(source, "permutation")
    program = make_distance_reuse_program(store, key, "distance", target, relation)
    relation["permutation"].reverse()
    assert program.nodes[-1].params["permutation"] != relation["permutation"]
    assert execute(program)["status"] == "completed"


def test_origin_is_strict_data_but_never_an_authentication_shortcut():
    source, _, _, store, key = archived()
    target, relation = target_relation(source, "permutation")
    data = make_distance_reuse_program(store, key, "distance", target, relation).to_dict()
    for node in data["nodes"]:
        if node["op"] in ("css.import_exclusion", "css.import_witness"):
            node["params"]["origin"] = {"bundle_id": "f" * 64, "node": "untrusted_hint"}
    program = Program.from_dict(data)
    review = inspect_run(program, execute(program))
    assert review["status"] == "checked" and review["claims"][-1]["value"] == 3
    next(node for node in data["nodes"] if node["op"] == "css.import_exclusion")["params"]["origin"] = {
        "bundle_id": "../../other", "node": "untrusted_hint"}
    with pytest.raises(ValueError, match="bundle_id"):
        Program.from_dict(data)


def test_physical_width_cannot_change_and_source_is_selected_by_bound_node():
    source, program, run, store, key = archived()
    wider = CSSCode(B(source.hx.rows, source.n + 1), B(source.hz.rows, source.n + 1))
    with pytest.raises(ValueError, match="physical coordinate width"):
        make_distance_reuse_program(store, key, "distance", wider,
                                    {"relation": "permutation", "permutation": list(range(wider.n))})
    data = program.to_dict()
    data["nodes"].insert(0, {"id": "unrelated", "op": "css.literal", "inputs": [],
                             "params": {"code": to_dict(CSSCode(B((), 0), B((), 0)))}})
    distinct = Program.from_dict(data)
    key = publish(distinct, execute(distinct), store)["bundle_id"]
    target, relation = target_relation(source, "permutation")
    reused = make_distance_reuse_program(store, key, "distance", target, relation)
    assert reused.nodes[0].params["code"]["n"] == source.n
    assert inspect_run(reused, execute(reused))["status"] == "checked"
