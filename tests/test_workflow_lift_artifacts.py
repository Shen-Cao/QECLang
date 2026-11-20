from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json

import pytest

from qeclang.analysis.structured import analyze_lifted
from qeclang.constructions import LiftedProduct
from qeclang.constructions import records
from qeclang.symbolic import GroupAlgebraElement as G, GroupAlgebraMatrix as M
from qeclang.workflow.artifacts import ARTIFACT_SCHEMA, describe
from qeclang.workflow.import_artifact import (ArtifactImportError, ImportBudget,
                                              ImportLimitExceeded, ImportLimits, import_artifact)
from qeclang.workflow.json_input import JSONLimitExceeded, JSONLimits, canonical_json
from qeclang.workflow.lift_artifacts import PARAMETERS_SCHEMA


def scalar(lift=3):
    a = G.from_terms((lift,), [(0,), (1,)])
    b = G.from_terms((lift,), [(0,), (2,)])
    return LiftedProduct(M(((a,),), 1, (lift,)), M(((b,),), 1, (lift,)))


def seeds():
    one = G.from_terms((3, 5), [(0, 0)])
    x = G.from_terms((3, 5), [(1, 0)])
    return LiftedProduct(M(((one+x, x), (x, one)), 2, (3, 5)),
                         M(((one,),), 1, (3, 5)))


def parameters(construction=None, backend="polynomial", **options):
    construction = scalar() if construction is None else construction
    if backend == "polynomial":
        options.setdefault("record_operations", True)
    result = analyze_lifted(construction, backend=backend, **options)
    return json.loads(json.dumps(asdict(result)))


def rehash(record):
    record["digest"] = sha256(canonical_json({"schema": ARTIFACT_SCHEMA,
        "kind": record["kind"], "payload": record["payload"]}).encode()).hexdigest()
    return record


def test_lift_uses_existing_canonical_construction_record():
    construction = seeds()
    artifact = describe("lift", construction)
    assert artifact["payload"] == records.to_dict(construction)
    restored = import_artifact(artifact, "lift")
    assert restored == construction
    assert describe("lift", restored) == artifact
    assert restored.a.orders == (3, 5)
    assert restored.a.shape == (2, 2) and restored.b.shape == (1, 1)


def test_symbolic_sites_are_separate_from_packed_matrix_axis(monkeypatch):
    construction = scalar(3001)
    artifact = describe("lift", construction)
    def forbidden(*args, **kwargs):
        raise AssertionError("structured import must not lower or construct checks")
    monkeypatch.setattr(M, "lower", forbidden)
    monkeypatch.setattr(M, "lower_sparse", forbidden)
    monkeypatch.setattr(LiftedProduct, "lower", forbidden)
    monkeypatch.setattr(LiftedProduct, "symbolic_checks", forbidden)
    restored = import_artifact(artifact, "lift", limits=ImportLimits(max_axis=1))
    assert restored.n == 6002
    with pytest.raises(ImportLimitExceeded, match="expanded"):
        import_artifact(artifact, "lift", limits=ImportLimits(max_lift_sites=6001))


@pytest.mark.parametrize("change", [
    {"max_lift_sites": 59}, {"max_lift_seed_axis": 1}, {"max_lift_seed_entries": 4},
    {"max_lift_terms": 5}, {"max_lift_group_size": 14}, {"max_lift_factors": 1},
])
def test_lift_caps_precede_ring_object_allocation(change, monkeypatch):
    artifact = describe("lift", seeds())
    def forbidden(*args, **kwargs):
        raise AssertionError("resource refusal must precede ring construction")
    monkeypatch.setattr(G, "from_terms", forbidden)
    with pytest.raises(ImportLimitExceeded):
        import_artifact(artifact, "lift", limits=ImportLimits(**change))


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(action="different action"),
    lambda p: p.update(extra=True),
    lambda p: p["a"]["entries"][0][0].reverse(),
    lambda p: p["a"]["entries"][0][0].append([0, 0]),
    lambda p: p["a"]["entries"][0][0].append([3, 0]),
])
def test_rehashed_noncanonical_or_invalid_lift_payloads_fail(mutate):
    artifact = describe("lift", seeds())
    mutate(artifact["payload"])
    with pytest.raises(ArtifactImportError):
        import_artifact(rehash(artifact), "lift")


def test_empty_seed_dimensions_survive_roundtrip():
    construction = LiftedProduct(M.zeros(0, 3, (5,)), M.zeros(2, 0, (5,)))
    artifact = describe("lift", construction)
    restored = import_artifact(artifact, "lift")
    assert restored.n == 0
    assert restored.a.shape == (0, 3) and restored.b.shape == (2, 0)
    assert artifact["payload"]["b"]["entries"] == [[], []]


@pytest.mark.parametrize("backend", ["crt", "polynomial", "abelian", "mixed-abelian", "packed"])
def test_parameter_roundtrip_keeps_current_backend_evidence(backend):
    value = parameters(backend=backend)
    artifact = describe("parameters", value)
    assert artifact["payload"] == {"schema": PARAMETERS_SCHEMA, **value}
    assert import_artifact(artifact, "parameters") == value
    assert artifact["payload"]["status"] == "exact"
    assert "assurance" not in artifact and "verified" not in artifact["payload"]


def test_refused_parameters_keep_no_partial_rank_claim():
    unsupported = parameters(scalar(2), backend="crt")
    exhausted = parameters(max_steps=0)
    for value, status in ((unsupported, "unsupported"), (exhausted, "budget_exhausted")):
        assert value["status"] == status
        assert import_artifact(describe("parameters", value), "parameters") == value
        assert value["k"] is value["rank_x"] is value["rank_z"] is None
        assert value["x_evidence"] is value["z_evidence"] is None


def test_import_does_not_certify_even_a_consistent_exact_parameter_claim(monkeypatch):
    import qeclang.workflow.rank_proof as proof
    value = parameters()
    assert proof.verify_dimension_proof(scalar(), value).status == "verified"
    assert value["rank_x"] > 0
    forged = deepcopy(value)
    forged["rank_x"] -= 1
    forged["k"] += 1
    forged["x_evidence"]["rank"] -= 1
    assert proof.verify_dimension_proof(scalar(), forged).status != "verified"
    def forbidden(*args, **kwargs):
        raise AssertionError("artifact reader must not run the scientific verifier")
    monkeypatch.setattr(proof, "verify_dimension_proof", forbidden)
    imported = import_artifact(describe("parameters", forged), "parameters")
    assert imported == forged and imported["status"] == "exact"


def test_parameter_proof_arrays_and_depth_are_bounded_before_copying():
    artifact = describe("parameters", parameters())
    with pytest.raises(ImportLimitExceeded, match="items"):
        import_artifact(artifact, "parameters", limits=ImportLimits(max_parameter_entries=10))
    with pytest.raises(ImportLimitExceeded, match="depth"):
        import_artifact(artifact, "parameters", limits=ImportLimits(max_payload_depth=3))
    with pytest.raises(ImportLimitExceeded, match="physical size"):
        import_artifact(artifact, "parameters", limits=ImportLimits(max_lift_sites=5))


def test_large_words_refuse_before_whole_json_decimal_conversion(monkeypatch):
    import qeclang.workflow.artifacts as artifacts
    value = parameters()
    value["x_evidence"]["diagonal"][0] = 1 << 20000
    def forbidden(*args, **kwargs):
        raise AssertionError("whole payload serialization must follow JSON admission")
    monkeypatch.setattr(artifacts, "_canonical", forbidden)
    with pytest.raises(JSONLimitExceeded, match="integer"):
        describe("parameters", value)
    record = {"kind": "parameters", "digest": "0"*64,
              "payload": {"schema": PARAMETERS_SCHEMA, **value}}
    with pytest.raises(ImportLimitExceeded, match="integer"):
        import_artifact(record, "parameters")


def test_admitted_word_boundary_and_stricter_caller_policy():
    value = parameters()
    value["x_evidence"]["diagonal"][0] = 1 << 4095
    artifact = describe("parameters", value)
    assert import_artifact(artifact, "parameters") == value
    with pytest.raises(JSONLimitExceeded, match="integer"):
        describe("parameters", value, limits=JSONLimits(max_integer_bits=4095))
    with pytest.raises(ImportLimitExceeded, match="integer"):
        import_artifact(artifact, "parameters", limits=ImportLimits(max_integer_bits=4095))


def test_generic_description_also_admits_integer_words_before_serializing(monkeypatch):
    import qeclang.workflow.artifacts as artifacts
    value = {"source": "0"*64, "sector": "x", "lower_bound": 1, "upper_bound": 1,
             "witness": 1 << 20000, "searched_through": 0, "candidates": 0, "status": "exact"}
    monkeypatch.setattr(artifacts, "_canonical", lambda value: pytest.fail("serialized before admission"))
    with pytest.raises(JSONLimitExceeded, match="integer"):
        describe("distance", value)


@pytest.mark.parametrize("change", [
    {"construction_fingerprint": "invalid"}, {"method": "unknown"}, {"n": True},
    {"status": "solver_optimal"}, {"steps": -1}, {"reason": 5}, {"assurance": "verified"},
])
def test_malformed_parameter_headers_are_not_claims(change):
    value = parameters()
    with pytest.raises(ValueError):
        describe("parameters", {**value, **change})


def test_parameter_import_is_detached_and_shares_cumulative_byte_budget():
    value = parameters()
    artifact = describe("parameters", value)
    budget = ImportBudget(max_artifact_bytes=100000)
    imported = import_artifact(artifact, "parameters", budget=budget)
    assert budget.used["artifact_bytes"] == len(canonical_json(artifact).encode())
    imported["x_evidence"]["diagonal"][0] = 999
    assert artifact["payload"]["x_evidence"]["diagonal"][0] != 999
    with pytest.raises(ImportLimitExceeded):
        import_artifact(artifact, "parameters", budget=ImportBudget(max_artifact_bytes=1))
    assert describe("parameters", value)["digest"] != describe(
        "parameters", {**value, "construction_fingerprint": "f"*64})["digest"]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), "\ud800"])
def test_structured_payload_uses_shared_finite_unicode_admission(invalid):
    value = parameters()
    value["x_evidence"]["extra_metadata"] = invalid
    with pytest.raises(ValueError):
        describe("parameters", value)
