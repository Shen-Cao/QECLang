from copy import deepcopy
from hashlib import sha256
import json
from random import Random

import pytest

from qeclang.algebra import BinaryMatrix as B
from qeclang.analysis.certificate_io import presentation_fingerprint, verify_certificate_record
from qeclang.analysis.low_weight import (LOW_WEIGHT_DOMAIN, certify_low_weight,
                                         check_low_weight, low_weight_id)
from qeclang.codes import CSSCode, Pauli, StabilizerCode
from qeclang.constructions import steane
from qeclang.io import to_dict


def reseal(record):
    result = deepcopy(record)
    result["record_id"] = low_weight_id(result)
    return result


def degenerate():
    return CSSCode(B((3, 6, 12), 4), B((15,), 4))


def test_record_id_and_streaming_presentation_identity_are_byte_compatible():
    rng = Random(722)
    for n in range(9):
        for _ in range(12):
            # The empty opposite sector permits arbitrary, redundant X rows.
            rows = tuple(rng.randrange(1 << n) for _ in range(4))
            code = CSSCode(B(rows, n), B((), n), "ignored display name")
            old = to_dict(code)
            old.pop("name")
            expected = sha256(json.dumps(old, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            assert presentation_fingerprint(code) == expected
    record = certify_low_weight(steane(), "x")
    body = {key: value for key, value in record.items() if key != "record_id"}
    expected = sha256(LOW_WEIGHT_DOMAIN + json.dumps(body, sort_keys=True,
                        separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    assert record["record_id"] == expected


def test_steane_excludes_two_layers_without_membership_or_rank_work():
    code = steane()
    for sector in ("x", "z"):
        record = certify_low_weight(code, sector)
        assert record["status"] == "excluded" and record["memberships"] == []
        assert record["claim"] == {"predicate": "nontrivial_weight_exclusion", "through": 2}
        checked = check_low_weight(code, record)
        assert checked.status == "checked_exclusion" and checked.excluded_through == 2
        assert checked.upper_bound is None


def test_degeneracy_star_memberships_and_zero_column_obligations():
    code = degenerate()
    record = certify_low_weight(code, "x")
    assert record["status"] == "excluded"
    assert record["memberships"] == [[0], [0, 1], [0, 1, 2]]
    assert check_low_weight(code, record).status == "checked_exclusion"
    assert code.k == 0
    assert "lower_bound" not in record and "distance" not in record
    interleaved = CSSCode(B((1, 4, 10), 4), B((10,), 4))
    record = certify_low_weight(interleaved, "x")
    assert record["memberships"] == [[0], [1], [2]]
    assert check_low_weight(interleaved, record).status == "checked_exclusion"
    first = certify_low_weight(interleaved, "x", through=1)
    assert first["memberships"] == [[0], [1]]


def test_counterexamples_have_source_bound_odd_dual_certificates():
    cases = ((CSSCode(B((), 3), B((3, 6), 3)), "z", 1),
             (CSSCode(B((), 3), B((7,), 3)), "x", 2))
    for code, sector, weight in cases:
        record = certify_low_weight(code, sector)
        assert record["status"] == "found" and record["memberships"] is None
        assert record["claim"] == {"predicate": "distance_upper_bound", "upper_bound": weight}
        assert verify_certificate_record(code, record["witness"])
        checked = check_low_weight(code, record)
        assert checked.status == "checked_witness" and checked.upper_bound == weight
        assert checked.excluded_through is None
    through_one = certify_low_weight(cases[1][0], "x", through=1)
    assert through_one["status"] == "excluded"


def test_producer_and_verifier_use_distinct_work_and_quota_statuses():
    code = degenerate()
    complete = certify_low_weight(code, "x")
    assert certify_low_weight(code, "x", max_steps=complete["steps"]) == complete
    refused = certify_low_weight(code, "x", max_steps=complete["steps"] - 1)
    assert refused["status"] == "budget_exhausted" and 0 < refused["steps"] < complete["steps"]
    assert refused["memberships"] is refused["witness"] is refused["claim"] is None
    assert check_low_weight(code, refused).status == "no_evidence"
    verified = check_low_weight(code, complete)
    assert check_low_weight(code, complete, max_steps=verified.steps).status == "checked_exclusion"
    stopped = check_low_weight(code, complete, max_steps=verified.steps - 1)
    assert stopped.status == "budget_exhausted" and 0 < stopped.steps < verified.steps
    assert stopped.excluded_through is stopped.upper_bound is stopped.source is None
    reported = deepcopy(complete)
    reported["steps"] += 10**20
    reported = reseal(reported)
    assert check_low_weight(code, reported).steps == verified.steps


def test_memory_axis_and_reference_limits_refuse_without_partial_claims():
    code = degenerate()
    complete = certify_low_weight(code, "x")
    for limits in ({"max_axis": 3}, {"max_bits": 0}, {"max_indices": 0}, {"max_steps": 0}):
        produced = certify_low_weight(code, "x", **limits)
        assert produced["status"] == "budget_exhausted" and produced["claim"] is None
        checked = check_low_weight(code, complete, **limits)
        assert checked.status == "budget_exhausted" and checked.excluded_through is None
    empty = CSSCode(B((), 0), B((), 0))
    record = certify_low_weight(empty, "x", max_axis=0, max_bits=0, max_indices=0, max_steps=0)
    assert record["status"] == "excluded"
    assert check_low_weight(empty, record, max_axis=0, max_bits=0,
                            max_indices=0, max_steps=0).status == "checked_exclusion"


def test_verifier_never_calls_elimination_producer_or_packed_transpose(monkeypatch):
    import qeclang.analysis.low_weight as module
    import qeclang.analysis.low_weight_support as support

    code = degenerate()
    excluded = certify_low_weight(code, "x")
    free = CSSCode(B((), 1), B((), 1))
    found = certify_low_weight(free, "x")

    def forbidden(*args, **kwargs):
        raise AssertionError("independent reader invoked a producing algorithm")

    for method in ("rank", "rref", "solve", "nullspace", "transpose"):
        monkeypatch.setattr(B, method, forbidden)
    monkeypatch.setattr(module, "certify_low_weight", forbidden)
    monkeypatch.setattr(module, "_basis", forbidden)
    monkeypatch.setattr(module, "producer_targets", forbidden)
    monkeypatch.setattr(support, "producer_targets", forbidden)
    assert check_low_weight(code, excluded).status == "checked_exclusion"
    assert check_low_weight(free, found).status == "checked_witness"


def test_rehashed_missing_reordered_or_wrong_memberships_fail():
    code = degenerate()
    record = certify_low_weight(code, "x")
    attacks = (record["memberships"][:-1], list(reversed(record["memberships"])),
               [[1], [0, 1], [0, 1, 2]], [[0, 0], [0, 1], [0, 1, 2]],
               [[True], [0, 1], [0, 1, 2]])
    for memberships in attacks:
        altered = deepcopy(record)
        altered["memberships"] = memberships
        checked = check_low_weight(code, reseal(altered))
        assert checked.status == "invalid_record"
        assert checked.excluded_through is None


def test_subject_sector_status_and_typed_claim_tampering_fail():
    code = degenerate()
    record = certify_low_weight(code, "x")
    for key, value in (("source", "0" * 64), ("sector", "z"), ("through", True),
                       ("status", "exact"), ("steps", False), ("reason", "extra")):
        altered = deepcopy(record)
        altered[key] = value
        assert check_low_weight(code, reseal(altered)).status == "invalid_record"
    altered = deepcopy(record)
    altered["claim"]["through"] = True
    assert check_low_weight(code, reseal(altered)).status == "invalid_record"
    record["record_id"] = "0" * 64
    assert check_low_weight(code, record).status == "invalid_record"


def test_nested_witness_source_is_bounded_before_record_hashing(monkeypatch):
    import qeclang.analysis.low_weight as module

    code = CSSCode(B((), 1), B((), 1))
    record = certify_low_weight(code, "x")

    def forbidden(*args, **kwargs):
        raise AssertionError("malformed witness source reached hash serialization")

    monkeypatch.setattr(module, "low_weight_id", forbidden)
    for value in ([0] * 1000, "a" * 1000, "0" * 64):
        changed = deepcopy(record)
        changed["witness"]["source"] = value
        assert check_low_weight(code, changed).status == "invalid_record"


def test_witness_parity_and_word_size_are_verified():
    code = CSSCode(B((), 3), B((7,), 3))
    record = certify_low_weight(code, "x")
    for key, value in (("dual", "0x0"), ("operator", "0x8"), ("weight", True)):
        changed = deepcopy(record)
        changed["witness"]["certificate"][key] = value
        assert check_low_weight(code, reseal(changed)).status == "invalid_record"


def test_no_unmetered_generic_solve_in_counterexample_production(monkeypatch):
    code = CSSCode(B((), 3), B((3, 6), 3))

    def forbidden(*args, **kwargs):
        raise AssertionError("counterexample must use charged tracked elimination")

    monkeypatch.setattr(B, "solve", forbidden)
    result = certify_low_weight(code, "z")
    assert result["status"] == "found" and verify_certificate_record(code, result["witness"])


def test_bad_requests_are_errors_and_bad_records_are_diagnostics():
    code = steane()
    for kwargs in ({"through": 0}, {"through": 3}, {"through": True}, {"max_steps": -1},
                   {"max_bits": True}, {"max_indices": float("nan")}):
        with pytest.raises(ValueError):
            certify_low_weight(code, "x", **kwargs)
    with pytest.raises(TypeError):
        certify_low_weight(StabilizerCode(1, (Pauli.from_label("-Z"),)), "x")
    for malformed in (None, [], {}, {"schema": "other"}):
        assert check_low_weight(code, malformed).status == "invalid_record"
    oversized = CSSCode(B((), 1 << 5000), B((), 1 << 5000))
    with pytest.raises(ValueError, match="portable"):
        certify_low_weight(oversized, "x")
