from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256

import pytest

from qeclang.algebra import BinaryMatrix as B, SparseBinaryMatrix as S
from qeclang.analysis.certificate_io import certificate_record, presentation_fingerprint, verify_certificate_record
from qeclang.analysis.certificates import certify_css_logical
from qeclang.analysis.distance import css_distance
from qeclang.codes import CSSCode, SparseCSSCode
from qeclang.constructions import steane
from qeclang.workflow.artifacts import ARTIFACT_SCHEMA, describe
from qeclang.workflow.import_artifact import (ArtifactImportError, ImportBudget, ImportLimitExceeded,
                                             ImportLimits, import_artifact)
from qeclang.workflow.json_input import canonical_json, measure_json_data


def resign(record):
    result = deepcopy(record)
    body = {"schema": ARTIFACT_SCHEMA, "kind": result["kind"], "payload": result["payload"]}
    result["digest"] = sha256(canonical_json(body).encode()).hexdigest()
    return result


def small_code():
    return CSSCode(B((3,), 4), B((12,), 4))


def distance_record(code, sector="x"):
    return describe("distance", {**asdict(css_distance(code, sector)),
                                  "source": presentation_fingerprint(code), "sector": sector})


def test_math_artifact_roundtrips_empty_redundant_and_named_presentations():
    cases = (CSSCode(B((), 0), B((), 0)), CSSCode(B((0, 0), 0), B((0,), 0)),
             CSSCode(B((), 4), B((), 4)),
             CSSCode(B((3, 12, 3, 0), 5), B((15, 0, 15), 5), "decorative"))
    for code in cases:
        for kind, value in (("css", code), ("chain", code.to_chain_complex()),
                            ("sparse_css", SparseCSSCode.from_packed(code))):
            record = describe(kind, value)
            before = deepcopy(record)
            imported = import_artifact(record, kind)
            assert describe(kind, imported) == record
            assert record == before
            assert imported.name == ""


def test_import_does_not_compute_rank_or_distance_or_lower_sparse(monkeypatch):
    import qeclang.analysis.distance as distance

    code = small_code()
    records = [(kind, describe(kind, value)) for kind, value in
               (("css", code), ("chain", code.to_chain_complex()),
                ("sparse_css", SparseCSSCode.from_packed(code)))]

    def forbidden(*args, **kwargs):
        raise AssertionError("import must not perform rank, search, or sparse lowering")

    monkeypatch.setattr(B, "rank", forbidden)
    monkeypatch.setattr(B, "nullspace", forbidden)
    monkeypatch.setattr(S, "to_packed", forbidden)
    monkeypatch.setattr(SparseCSSCode, "to_packed", forbidden)
    monkeypatch.setattr(distance, "css_distance", forbidden)
    for kind, record in records:
        assert describe(kind, import_artifact(record, kind)) == record


def test_digest_domain_kind_and_unknown_fields():
    record = describe("css", small_code())
    with pytest.raises(ArtifactImportError, match="kind"):
        import_artifact(record, "sparse_css")
    for change in (dict(digest="0" * 64), dict(kind="sparse_css"), dict(extra=1)):
        mutated = {**record, **change}
        with pytest.raises(ArtifactImportError):
            import_artifact(mutated, "css")
    mutated = deepcopy(record)
    mutated["payload"]["name"] = "not canonical"
    with pytest.raises(ArtifactImportError):
        import_artifact(resign(mutated), "css")
    # Identical CSS payloads still have representation-domain-separated digests.
    sparse = describe("sparse_css", SparseCSSCode.from_packed(small_code()))
    assert sparse["payload"] == record["payload"] and sparse["digest"] != record["digest"]


@pytest.mark.parametrize("support", [[0, 0], [1, 0], [True, 1], [0.0, 1], [0, 4]])
def test_noncanonical_or_noninteger_supports_rejected(support):
    for kind in ("css", "sparse_css"):
        record = describe("css", small_code())
        record["kind"] = kind
        record["payload"]["hx"][0] = support
        with pytest.raises(ArtifactImportError):
            import_artifact(resign(record), kind)


def test_commutation_and_chain_boundary_are_revalidated():
    for kind in ("css", "sparse_css"):
        record = describe("css", small_code())
        record["kind"] = kind
        record["payload"]["hz"] = [[0]]
        with pytest.raises(ArtifactImportError, match="commute"):
            import_artifact(resign(record), kind)
    chain = describe("chain", small_code().to_chain_complex())
    chain["payload"]["d2"]["rows"][0] = [0]
    with pytest.raises(ArtifactImportError, match="boundary"):
        import_artifact(resign(chain), "chain")


def test_chain_shapes_conventions_and_boolean_dimensions():
    original = describe("chain", small_code().to_chain_complex())
    for key, value in (("dimensions", [1, 4, 2]), ("degree_order", [False, 1, 2]),
                       ("matrix_action", "row_vectors"), ("coefficient_field", "GF(3)")):
        record = deepcopy(original)
        record["payload"][key] = value
        with pytest.raises(ArtifactImportError):
            import_artifact(resign(record), "chain")
    record = deepcopy(original)
    record["payload"]["d1"]["shape"] = [True, 4]
    with pytest.raises(ArtifactImportError):
        import_artifact(resign(record), "chain")
    record = deepcopy(original)
    record["payload"]["d2"]["rows"].pop()
    with pytest.raises(ArtifactImportError):
        import_artifact(resign(record), "chain")


def test_resource_refusals_are_typed_and_precede_construction(monkeypatch):
    import qeclang.workflow.import_artifact as module

    record = describe("css", small_code())

    def forbidden(*args, **kwargs):
        raise AssertionError("constructor ran before import resource preflight")

    monkeypatch.setattr(module, "CSSCode", forbidden)
    for options in ({"max_bytes": 0}, {"max_axis": 3}, {"max_matrix_bits": 8},
                    {"max_validation_work": 5}, {"max_payload_items": 1},
                    {"max_payload_depth": 1}, {"max_string_length": 1}):
        with pytest.raises(ImportLimitExceeded):
            import_artifact(record, "css", limits=replace(ImportLimits(), **options))


def test_sparse_import_needs_no_packed_budget_or_construction(monkeypatch):
    code = SparseCSSCode.from_packed(small_code())
    record = describe("sparse_css", code)
    import qeclang.workflow.import_artifact as module

    def forbidden(*args, **kwargs):
        raise AssertionError("sparse import must not construct a packed matrix")

    monkeypatch.setattr(module, "BinaryMatrix", forbidden)
    imported = import_artifact(record, "sparse_css", limits=ImportLimits(max_matrix_bits=0))
    assert imported.hx == code.hx and imported.hz == code.hz


def test_shared_import_byte_and_work_budgets():
    record = describe("css", small_code())
    size = len(canonical_json(record).encode())
    assert measure_json_data(record) == size
    budget = ImportBudget(max_artifact_bytes=2 * size, max_validation_work=12)
    import_artifact(record, "css", budget=budget)
    import_artifact(record, "css", budget=budget)
    assert budget.used == {"artifact_bytes": 2 * size, "validation_work": 12}
    with pytest.raises(ImportLimitExceeded):
        import_artifact(record, "css", budget=budget)
    budget = ImportBudget(max_validation_work=6)
    import_artifact(record, "css", budget=budget)
    with pytest.raises(ImportLimitExceeded, match="validation"):
        import_artifact(record, "css", budget=budget)
    assert budget.used["validation_work"] == 6
    assert budget.used["artifact_bytes"] == 2 * size
    with pytest.raises(ArtifactImportError):
        budget.charge("validation_work", True)


def test_distance_records_are_structural_claims_not_certified_lower_bounds():
    code = steane()
    record = distance_record(code)
    value = import_artifact(record, "distance")
    assert "schema" not in value and type(value["witness"]) is int
    # Weight-seven all-X is logical, but the actual Steane distance is three.
    spoofed = deepcopy(record)
    spoofed["payload"].update(lower_bound=7, upper_bound=7, witness=127,
                              searched_through=6, candidates=0, status="exact")
    imported = import_artifact(resign(spoofed), "distance")
    assert imported["lower_bound"] == 7  # Import makes no truth claim about this number.
    empty_claim = deepcopy(record)
    empty_claim["payload"].update(lower_bound=None, upper_bound=None, witness=None,
                                  searched_through=0, candidates=0, status="no_logicals")
    assert import_artifact(resign(empty_claim), "distance")["status"] == "no_logicals"
    inconsistent = deepcopy(record)
    inconsistent["payload"]["lower_bound"] = 4
    with pytest.raises(ArtifactImportError):
        import_artifact(resign(inconsistent), "distance")
    boolean = deepcopy(record)
    boolean["payload"]["candidates"] = False
    with pytest.raises(ArtifactImportError):
        import_artifact(resign(boolean), "distance")


def test_witness_import_is_syntax_only_and_rejects_boolean_weights():
    code = steane()
    witness = certify_css_logical(code, "x", 127)
    record = describe("witness", certificate_record(code, witness))
    imported = import_artifact(record, "witness")
    assert verify_certificate_record(code, imported)
    invalid_proof = deepcopy(record)
    invalid_proof["payload"]["certificate"]["dual"] = "0x0"
    imported = import_artifact(resign(invalid_proof), "witness")
    assert not verify_certificate_record(code, imported)
    for section, key in (("certificate", "weight"), ("claim", "distance_upper_bound")):
        boolean = deepcopy(record)
        boolean["payload"][section][key] = True
        with pytest.raises(ArtifactImportError):
            import_artifact(resign(boolean), "witness")
    noncanonical = deepcopy(record)
    noncanonical["payload"]["certificate"]["operator"] = "0x07f"
    with pytest.raises(ArtifactImportError):
        import_artifact(resign(noncanonical), "witness")


def test_oversized_words_empty_axes_and_malformed_containers():
    record = describe("css", CSSCode(B((), 5), B((), 5)))
    with pytest.raises(ImportLimitExceeded, match="count"):
        import_artifact(record, "css", limits=ImportLimits(max_axis=4))
    distance = distance_record(small_code())
    distance["payload"].update(witness=1 << 2048)
    with pytest.raises(ImportLimitExceeded, match="axis"):
        import_artifact(resign(distance), "distance")
    cycle = {}
    cycle["self"] = cycle
    with pytest.raises(ArtifactImportError, match="cyclic"):
        import_artifact(cycle, "css")
    for malformed in (None, [], {"kind": "css", "digest": 1, "payload": {}},
                      {"kind": "css", "digest": "a", "payload": float("nan")}):
        with pytest.raises(ArtifactImportError):
            import_artifact(malformed, "css")


def test_imported_claim_dicts_are_detached():
    code = steane()
    record = describe("witness", certificate_record(code, certify_css_logical(code, "x", 127)))
    imported = import_artifact(record, "witness")
    imported["certificate"]["dual"] = "0x0"
    assert record["payload"]["certificate"]["dual"] != "0x0"
