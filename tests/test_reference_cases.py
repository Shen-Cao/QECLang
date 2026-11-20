from dataclasses import FrozenInstanceError
import json

import pytest

from qeclang.algebra import BinaryMatrix
from qeclang.analysis import css_distance
from qeclang.analysis.certificate_io import certificate_record, presentation_fingerprint, verify_certificate_record
from qeclang.analysis.certificates import certify_css_logical, certify_rank
from qeclang.analysis.low_weight import certify_low_weight, check_low_weight
from qeclang.catalog.references import REFERENCE_SCHEMA, build_reference, reference_cases
from qeclang.workflow.row_equivalence import row_maps, verify_row_equivalence


def evidence(case):
    code = build_reference(case.id)
    ranks = [certify_rank(matrix, max_bits=100_000) for matrix in (code.hx, code.hz)]
    rank_records = [certificate_record(matrix, proof) for matrix, proof in zip((code.hx, code.hz), ranks)]
    sectors = []
    if case.expected_k:
        for sector, expected in (("x", case.expected_dx), ("z", case.expected_dz)):
            result = css_distance(code, sector, max_weight=expected, max_candidates=10_000)
            assert result.upper_bound == expected
            upper = certificate_record(code, certify_css_logical(code, sector, result.witness))
            lower = (certify_low_weight(code, sector, through=expected - 1, max_steps=100_000,
                                       max_axis=64, max_bits=100_000, max_indices=10_000)
                     if expected > 1 else None)
            sectors.append((expected, lower, upper))
    return code, ranks, rank_records, sectors


@pytest.mark.parametrize("case", reference_cases(), ids=lambda case: case.id)
def test_all_parameter_expectations_have_bounded_native_rank_and_distance_evidence(case):
    code, ranks, rank_records, sectors = evidence(case)
    assert code.n == case.expected_n
    assert (code.hx.nrows, code.hz.nrows) == (case.x_checks, case.z_checks)
    assert code.n - sum(proof.rank for proof in ranks) == case.expected_k
    for matrix, record in zip((code.hx, code.hz), rank_records):
        assert verify_certificate_record(matrix, record, max_axis=64, max_bits=100_000)
    for expected, lower, upper in sectors:
        assert verify_certificate_record(code, upper, max_axis=64, max_bits=100_000)
        assert upper["claim"]["distance_upper_bound"] == expected
        if lower is not None:
            checked = check_low_weight(code, lower, max_steps=100_000, max_axis=64,
                                       max_bits=100_000, max_indices=10_000)
            assert checked.status == "checked_exclusion" and checked.excluded_through == expected - 1
    if not case.expected_k:
        assert case.expected_dx is case.expected_dz is case.expected_distance is None
        assert sum(proof.rank for proof in ranks) == code.n
    else:
        assert case.expected_distance == min(case.expected_dx, case.expected_dz)


def test_all_proofs_remain_checkable_without_producers_or_elimination(monkeypatch):
    generated = [evidence(case) for case in reference_cases()]
    def forbidden(*args, **kwargs):
        raise AssertionError("certificate inspection must not regenerate an analysis")
    import qeclang.analysis.low_weight as low_weight
    import qeclang.analysis.distance as distance
    import qeclang.analysis.certificates as certificates
    for name in ("rank", "solve", "rref", "row_basis", "nullspace", "contains"):
        monkeypatch.setattr(BinaryMatrix, name, forbidden)
    monkeypatch.setattr(low_weight, "certify_low_weight", forbidden)
    monkeypatch.setattr(distance, "css_distance", forbidden)
    monkeypatch.setattr(certificates, "certify_rank", forbidden)
    monkeypatch.setattr(certificates, "certify_css_logical", forbidden)
    for code, _, rank_records, sectors in generated:
        for matrix, record in zip((code.hx, code.hz), rank_records):
            assert verify_certificate_record(matrix, record, max_axis=64, max_bits=100_000)
        for _, lower, upper in sectors:
            assert verify_certificate_record(code, upper, max_axis=64, max_bits=100_000)
            if lower is not None:
                assert check_low_weight(code, lower, max_steps=100_000,
                                        max_axis=64, max_bits=100_000).status == "checked_exclusion"


def test_catalog_is_immutable_ordered_and_metadata_is_not_a_verification_receipt():
    cases = reference_cases()
    assert type(cases) is tuple and len(cases) == 14
    assert [case.id for case in cases] == sorted({case.id for case in cases})
    assert all(case.id == case.case_id and case.version == 1 for case in cases)
    with pytest.raises(FrozenInstanceError):
        cases[0].expected_k = 999
    for case in cases:
        first = case.to_dict()
        assert json.loads(json.dumps(first, allow_nan=False)) == first
        assert first["schema"] == REFERENCE_SCHEMA
        assert first["parameter_status"] == "reference_expectations"
        assert first["scope"] == "positive_binary_css_finite_instance"
        first["expected"]["k"] = 999
        first["provenance"]["citations"].append("not a reference")
        assert case.to_dict()["expected"]["k"] != 999
        assert "not a reference" not in case.citation_urls
        assert case.source_type in ("analytic_literature", "algebraic_fixture")
        if case.source_type == "analytic_literature":
            assert case.citation_urls
        assert all(url.startswith(("https://arxiv.org/abs/", "https://doi.org/")) for url in case.citation_urls)


def test_builds_and_descriptions_do_not_run_parameter_analysis(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("reference loading must not silently certify parameters")
    for name in ("rank", "solve", "rref", "row_basis", "nullspace", "contains"):
        monkeypatch.setattr(BinaryMatrix, name, forbidden)
    for case in reference_cases():
        assert build_reference(case.id).n == case.to_dict()["expected"]["n"]


def test_regeneration_retains_exact_order_and_zero_redundant_rows():
    for case in reference_cases():
        first, second = build_reference(case.id), build_reference(case.id)
        assert first is not second
        assert (first.hx, first.hz) == (second.hx, second.hz)
        assert presentation_fingerprint(first) == presentation_fingerprint(second)
    redundant = build_reference("steane_redundant")
    assert redundant.hx.rows == redundant.hz.rows == (85, 102, 120, 51, 0, 85)
    bell = build_reference("bell_redundant")
    assert bell.hx.rows == (3, 3, 0) and bell.hz.rows == (3, 0, 3)
    assert build_reference("repetition_z_checks_3").hx.shape == (0, 3)
    assert build_reference("empty").hx.shape == (0, 0)


def test_redundant_equivalence_and_equal_parameter_metric_foil_are_distinct():
    source, redundant = build_reference("steane"), build_reference("steane_redundant")
    assert presentation_fingerprint(source) != presentation_fingerprint(redundant)
    assert verify_row_equivalence(source, redundant, row_maps(source, redundant))
    cases = {case.id: case for case in reference_cases()}
    first, second = cases["css_4_2_2"], cases["disjoint_checks_4_2_1"]
    assert (first.expected_n, first.expected_k) == (second.expected_n, second.expected_k) == (4, 2)
    assert (first.expected_distance, second.expected_distance) == (2, 1)
    assert build_reference(first.id).hx.row_basis() != build_reference(second.id).hx.row_basis()


def test_guard_refusals_precede_constructor_calls(monkeypatch):
    import qeclang.catalog.references as references
    def forbidden(*args, **kwargs):
        raise AssertionError("constructor should not run after preflight refusal")
    monkeypatch.setattr(references, "_construct", forbidden)
    for case in reference_cases():
        checks = case.x_checks + case.z_checks
        bits = checks * case.expected_n + case.x_checks * case.z_checks
        for limits in ([{"max_qubits": case.expected_n - 1}] if case.expected_n else []):
            with pytest.raises(ValueError, match="size limits"):
                build_reference(case.id, **limits)
        if checks:
            with pytest.raises(ValueError, match="size limits"):
                build_reference(case.id, max_checks=checks - 1)
        if bits:
            with pytest.raises(ValueError, match="size limits"):
                build_reference(case.id, max_bits=bits - 1)


def test_invalid_ids_limits_and_exact_empty_boundary():
    with pytest.raises(KeyError):
        build_reference("../external")
    with pytest.raises(TypeError):
        build_reference(None)
    for value in (True, -1, 1.5):
        with pytest.raises(ValueError):
            build_reference("steane", max_bits=value)
    code = build_reference("empty", max_qubits=0, max_checks=0, max_bits=0)
    assert code.n == code.hx.nrows == code.hz.nrows == 0
