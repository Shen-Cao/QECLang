from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json

import pytest

from qeclang.algebra import BinaryMatrix, SparseBinaryMatrix
from qeclang.analysis.certificate_io import (certificate_record, presentation_fingerprint,
                                             verify_certificate_record)
from qeclang.analysis.certificates import certify_css_logical
from qeclang.analysis.distance import css_distance
from qeclang.codes import CSSCode, ChainComplex, SparseCSSCode
from qeclang.constructions import steane
from qeclang.workflow.artifacts import ARTIFACT_SCHEMA, CHAIN_SCHEMA, DISTANCE_SCHEMA, describe


def distance_value(code=None, *, max_weight=3, max_candidates=100_000, sector="x"):
    code = steane() if code is None else code
    result = css_distance(code, sector, max_weight=max_weight, max_candidates=max_candidates)
    return {"source": presentation_fingerprint(code), "sector": sector, **asdict(result)}


def logical_record():
    code = steane()
    return code, certificate_record(code, certify_css_logical(code, "x", 7))


def test_digest_contract_and_json_compatibility():
    record = describe("css", steane())
    encoded = json.dumps({"schema": ARTIFACT_SCHEMA, "kind": record["kind"],
                          "payload": record["payload"]}, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    assert record["digest"] == sha256(encoded.encode("utf-8")).hexdigest()
    assert set(record) == {"kind", "digest", "payload"}
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert len(record["digest"]) == 64
    other_schema = encoded.replace(ARTIFACT_SCHEMA, "qeclang.workflow.artifact.v2")
    assert sha256(other_schema.encode()).hexdigest() != record["digest"]


def test_names_are_removed_not_merely_ignored_by_hash():
    code = steane()
    assert describe("css", code) == describe("css", replace(code, name="another label"))
    chain = code.to_chain_complex()
    assert describe("chain", chain) == describe("chain", replace(chain, name="renamed"))
    sparse = SparseCSSCode.from_packed(code)
    renamed = SparseCSSCode(sparse.hx, sparse.hz, "different", max_validation_work=100000)
    assert describe("sparse_css", sparse) == describe("sparse_css", renamed)
    assert "name" not in describe("css", code)["payload"]


def test_representation_kind_separates_equal_code_payloads():
    code = steane()
    packed = describe("css", code)
    sparse = describe("sparse_css", SparseCSSCode.from_packed(code))
    assert packed["payload"] == sparse["payload"]
    assert packed["kind"] != sparse["kind"]
    assert packed["digest"] != sparse["digest"]


def test_generator_order_redundancy_and_qubit_order_are_significant():
    code = steane()
    reordered = CSSCode(BinaryMatrix(tuple(reversed(code.hx.rows)), code.n), code.hz)
    redundant = CSSCode(BinaryMatrix(code.hx.rows + (code.hx.rows[0], 0), code.n), code.hz)
    assert len({describe("css", candidate)["digest"] for candidate in
                (code, reordered, redundant)}) == 3
    first = CSSCode(BinaryMatrix((1,), 3), BinaryMatrix((), 3))
    permuted = CSSCode(BinaryMatrix((2,), 3), BinaryMatrix((), 3))
    assert describe("css", first)["digest"] != describe("css", permuted)["digest"]
    assert describe("css", redundant)["payload"]["hx"][-1] == []


def test_chain_shapes_degree_order_and_empty_boundaries():
    first = ChainComplex(BinaryMatrix.zeros(0, 3), BinaryMatrix.zeros(3, 2))
    second = ChainComplex(BinaryMatrix.zeros(0, 3), BinaryMatrix.zeros(3, 0))
    payload = describe("chain", first)["payload"]
    assert payload == {"schema": CHAIN_SCHEMA, "coefficient_field": "GF(2)",
                       "degree_order": [0, 1, 2], "matrix_action": "column_vectors",
                       "dimensions": [0, 3, 2],
                       "d1": {"shape": [0, 3], "rows": []},
                       "d2": {"shape": [3, 2], "rows": [[], [], []]}}
    assert describe("chain", first)["digest"] != describe("chain", second)["digest"]
    code = steane()
    ordinary = describe("chain", code.to_chain_complex())["payload"]
    assert ordinary["d1"]["rows"] == describe("css", code)["payload"]["hx"]
    assert ordinary["d2"]["shape"] == [code.n, code.hz.nrows]
    assert ordinary["d2"]["rows"][0] == [0]


def test_sparse_description_never_lowers_or_analyzes(monkeypatch):
    sparse = SparseCSSCode(SparseBinaryMatrix(((0, 999999), ()), 1000000),
                           SparseBinaryMatrix((), 1000000))
    def forbidden(*args, **kwargs):
        raise AssertionError("artifact serialization must not lower or analyze")
    monkeypatch.setattr(SparseCSSCode, "to_packed", forbidden)
    monkeypatch.setattr(SparseBinaryMatrix, "to_packed", forbidden)
    monkeypatch.setattr(BinaryMatrix, "rank", forbidden)
    payload = describe("sparse_css", sparse)["payload"]
    assert payload["n"] == 1000000
    assert payload["hx"] == [[0, 999999], []]
    assert payload["hz"] == []
    assert "validation_work" not in payload


def test_exact_bounded_and_empty_distance_payloads():
    exact = describe("distance", distance_value())["payload"]
    bounded = describe("distance", distance_value(max_candidates=0))["payload"]
    weight_limited = describe("distance", distance_value(max_weight=1))["payload"]
    assert exact["schema"] == DISTANCE_SCHEMA
    assert exact["status"] == "exact" and exact["lower_bound"] == exact["upper_bound"] == 3
    assert bounded["status"] == "budget_exhausted" and bounded["lower_bound"] == 1
    assert weight_limited["status"] == "weight_limit"
    state = CSSCode(BinaryMatrix((1,), 1), BinaryMatrix((), 1))
    empty = describe("distance", distance_value(state))["payload"]
    assert empty["status"] == "no_logicals"
    assert empty["lower_bound"] is empty["upper_bound"] is empty["witness"] is None
    for payload in (exact, bounded, empty):
        assert "assurance" not in payload and "completion" not in payload


def test_source_sector_and_execution_evidence_are_digest_bound():
    first = distance_value(max_candidates=0)
    second = {**first, "source": "a"*64}
    third = {**first, "sector": "z"}
    fourth = {**first, "candidates": 2}
    assert len({describe("distance", data)["digest"] for data in
                (first, second, third, fourth)}) == 4


@pytest.mark.parametrize("change", [
    {"source": "not a hash"}, {"sector": "pauli"}, {"status": "solver_optimal"},
    {"candidates": True}, {"searched_through": -1}, {"lower_bound": 4},
    {"status": "weight_limit"}, {"witness": 1}, {"upper_bound": float("inf")},
    {"assurance": "formally_verified"},
])
def test_malformed_or_inconsistent_distance_records_rejected(change):
    with pytest.raises(ValueError):
        describe("distance", {**distance_value(), **change})


def test_witness_keeps_existing_schema_without_claiming_verification():
    code, record = logical_record()
    described = describe("witness", record)
    assert described["payload"] == record
    assert described["payload"]["schema"] == "qeclang.certificate.v1"
    changed = deepcopy(record)
    changed["certificate"]["dual"] = "0x0"
    assert not verify_certificate_record(code, changed)
    unverified = describe("witness", changed)
    assert unverified["digest"] != described["digest"]
    assert "assurance" not in unverified and "verified" not in unverified["payload"]


def test_descriptions_detach_mutable_nested_inputs_and_outputs():
    _, record = logical_record()
    described = describe("witness", record)
    original_digest = described["digest"]
    record["certificate"]["dual"] = "0x0"
    assert described["payload"]["certificate"]["dual"] != "0x0"
    described["payload"]["claim"]["distance_upper_bound"] = 99
    assert record["claim"]["distance_upper_bound"] == 3
    assert original_digest != describe("witness", record)["digest"]


@pytest.mark.parametrize("change", [
    {"schema": "qeclang.certificate.v2"}, {"source": "B"*64},
    {"kind": "binary_rank"}, {"certificate": {"sector": "x"}},
    {"claim": {"distance_upper_bound": True, "sector": "x"}},
    {"assurance": "checked"},
])
def test_invalid_witness_record_shape_rejected(change):
    _, record = logical_record()
    with pytest.raises(ValueError):
        describe("witness", {**record, **change})


def test_canonical_key_order_and_typed_value_requirements():
    _, record = logical_record()
    reversed_keys = dict(reversed(list(record.items())))
    reversed_keys["certificate"] = dict(reversed(list(record["certificate"].items())))
    assert describe("witness", reversed_keys) == describe("witness", record)
    for kind in ("css", "chain", "sparse_css"):
        with pytest.raises(TypeError):
            describe(kind, {})
    for kind in ("rank", "CSS", None, []):
        with pytest.raises(ValueError):
            describe(kind, steane())
