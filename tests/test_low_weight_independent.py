"""Small independent support enumeration checks the column-class certificate theorem."""
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from random import Random

import pytest

from qeclang.algebra import BinaryMatrix as B
from qeclang.analysis.low_weight import certify_low_weight, check_low_weight
from qeclang.codes import CSSCode
from qeclang.io import to_dict
from qeclang.workflow import ExecutionLimits, Program, execute, inspect_run


def scalar_parity(left, right, n):
    return sum(((left >> i) & 1) * ((right >> i) & 1) for i in range(n)) % 2


def scalar_span(rows):
    values = {0}
    for row in rows:
        values |= {value ^ row for value in values}
    return values


def brute_low_logicals(code, sector, through):
    same, opposite = (code.hx, code.hz) if sector == "x" else (code.hz, code.hx)
    span = scalar_span(same.rows)
    return {sum(1 << i for i in support)
            for weight in range(1, min(through, code.n) + 1)
            for support in combinations(range(code.n), weight)
            if (candidate := sum(1 << i for i in support)) not in span
            and all(scalar_parity(candidate, row, code.n) == 0 for row in opposite.rows)}


def random_css(n, rng):
    x = tuple(rng.randrange(1 << n) for _ in range(rng.randrange(5)))
    allowed = [value for value in range(1 << n)
               if all(scalar_parity(value, row, n) == 0 for row in x)]
    z = tuple(rng.choice(allowed) for _ in range(rng.randrange(5)))
    return CSSCode(B(x, n), B(z, n))


@pytest.mark.parametrize("sector", ["x", "z"])
@pytest.mark.parametrize("through", [1, 2])
def test_random_css_matches_independent_support_enumeration(sector, through):
    rng = Random(76291)
    for n in range(8):
        for _ in range(16):
            code = random_css(n, rng)
            logicals = brute_low_logicals(code, sector, through)
            record = certify_low_weight(code, sector, through)
            checked = check_low_weight(code, record)
            if logicals:
                assert record["status"] == "found"
                assert checked.status == "checked_witness"
                assert checked.upper_bound in {value.bit_count() for value in logicals}
                assert int(record["witness"]["certificate"]["operator"], 16) in logicals
            else:
                assert record["status"] == "excluded"
                assert checked.status == "checked_exclusion"
                assert checked.excluded_through == through
                assert checked.upper_bound is None


@pytest.mark.parametrize("code", [
    CSSCode(B((), 0), B((), 0)),
    CSSCode(B((0, 0), 0), B((0,), 0)),
    CSSCode(B((), 3), B((), 3)),
    CSSCode(B.identity(3), B((0, 0), 3)),
    CSSCode(B((3, 3, 0), 2), B((3, 0, 3), 2)),
    CSSCode(B((3, 6, 12), 4), B((15,), 4)),
])
def test_zero_empty_redundant_and_degenerate_boundaries(code):
    for sector in ("x", "z"):
        for through in (1, 2):
            checked = check_low_weight(code, certify_low_weight(code, sector, through))
            expected = "checked_witness" if brute_low_logicals(code, sector, through) else "checked_exclusion"
            assert checked.status == expected


def test_independent_checker_never_solves_or_searches(monkeypatch):
    codes = [CSSCode(B((3, 6, 12), 4), B((15,), 4)), CSSCode(B((), 3), B((), 3))]
    records = [(code, certify_low_weight(code, sector, through))
               for code in codes for sector in ("x", "z") for through in (1, 2)]
    def forbidden(*args, **kwargs):
        raise AssertionError("independent checking must not solve or enumerate supports")
    import qeclang.analysis.distance as distance
    for name in ("rank", "solve", "rref", "row_basis", "nullspace", "contains"):
        monkeypatch.setattr(B, name, forbidden)
    monkeypatch.setattr(distance, "css_distance", forbidden)
    monkeypatch.setattr(distance, "binary_words", forbidden)
    for code, record in records:
        assert check_low_weight(code, record).status in ("checked_exclusion", "checked_witness")


def test_exact_steane_workflow_has_no_producer_or_verifier_enumeration(monkeypatch):
    program = Program.load(Path(__file__).parents[1] / "examples/programs/exact_steane.json")
    run = execute(program, limits=ExecutionLimits(max_candidates=0))
    assert run["status"] == "completed"
    assert run["ledger"]["used"]["candidates"] == 0
    def forbidden(*args, **kwargs):
        raise AssertionError("independent exact-distance inspection must not search or eliminate")
    import qeclang.analysis.distance as distance
    import qeclang.workflow.operations as operations
    monkeypatch.setattr(distance, "css_distance", forbidden)
    monkeypatch.setattr(distance, "binary_words", forbidden)
    monkeypatch.setattr(operations, "css_distance", forbidden)
    for name in ("rank", "solve", "rref", "row_basis", "nullspace", "contains"):
        monkeypatch.setattr(B, name, forbidden)
    inspection = inspect_run(program, run)
    assert inspection["status"] == "checked"
    exact = [claim for claim in inspection["claims"] if claim["predicate"] == "sector_distance"]
    assert len(exact) == 2 and {claim["sector"] for claim in exact} == {"x", "z"}
    for claim in exact:
        assert claim["scope"] == "instance" and claim["status"] == "exact"
        assert claim["lower_bound"] == claim["upper_bound"] == claim["value"] == 3
    full = [claim for claim in inspection["claims"] if claim["predicate"] == "logical_distance"]
    assert len(full) == 1 and full[0]["sector"] == "pauli"
    assert full[0]["scope"] == "instance" and full[0]["status"] == "exact"
    assert full[0]["lower_bound"] == full[0]["upper_bound"] == full[0]["value"] == 3


def test_zero_logical_exclusion_cannot_create_a_finite_distance_claim():
    data = Program.load(Path(__file__).parents[1] / "examples/programs/exact_steane.json").to_dict()
    data["nodes"][0]["params"]["code"] = to_dict(CSSCode(B((3,), 2), B((3,), 2)))
    program = Program.from_dict(data)
    run = execute(program, limits=ExecutionLimits(max_candidates=0))
    assert run["status"] == "evidence_unavailable"
    inspection = inspect_run(program, run, allow_prefix=True)
    assert inspection["status"] == "checked_prefix"
    assert any(claim["predicate"] == "nontrivial_weight_exclusion" for claim in inspection["claims"])
    assert not any(claim["predicate"] in ("sector_distance", "logical_distance", "distance_upper_bound")
                   for claim in inspection["claims"])


def rehash(record):
    body = {key: value for key, value in record.items() if key != "record_id"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    record["record_id"] = sha256(b"qeclang.low_weight.v1\0" + encoded).hexdigest()
    return record


@pytest.mark.parametrize("mutation", [
    lambda record: record["memberships"].pop(),
    lambda record: record["memberships"].append([]),
    lambda record: record["memberships"].__setitem__(1, []),
    lambda record: record["memberships"].reverse(),
    lambda record: record["memberships"].__setitem__(0, [False]),
    lambda record: record["memberships"].__setitem__(0, [0, 0]),
    lambda record: record["memberships"].__setitem__(0, [3]),
    lambda record: record.__setitem__("source", "0" * 64),
    lambda record: record.__setitem__("through", True),
    lambda record: record["claim"].__setitem__("through", 1),
])
def test_rehashed_incomplete_wrong_or_malformed_star_evidence_is_rejected(mutation):
    code = CSSCode(B((3, 6, 12), 4), B((15,), 4))
    record = certify_low_weight(code, "x", 2)
    assert record["status"] == "excluded" and len(record["memberships"]) == 3
    invalid = deepcopy(record)
    mutation(invalid)
    result = check_low_weight(code, rehash(invalid))
    assert result.status == "invalid_record"
    assert result.excluded_through is None and result.upper_bound is None


def test_rehash_cannot_upgrade_weight_one_exclusion_to_weight_two():
    code = CSSCode(B((), 2), B((3,), 2))
    record = certify_low_weight(code, "x", 1)
    assert record["status"] == "excluded" and record["memberships"] == []
    assert brute_low_logicals(code, "x", 2) == {3}
    record["through"] = record["claim"]["through"] = 2
    assert check_low_weight(code, rehash(record)).status == "invalid_record"


def test_rehashed_false_upper_pairing_and_producer_refusal_do_not_create_claims():
    code = CSSCode(B((), 3), B((), 3))
    witness = certify_low_weight(code, "x", 2)
    witness["witness"]["certificate"]["dual"] = "0x0"
    assert check_low_weight(code, rehash(witness)).status == "invalid_record"
    refusal = certify_low_weight(code, "x", 2, max_steps=0)
    assert refusal["status"] == "budget_exhausted"
    review = check_low_weight(code, refusal)
    assert review.status == "no_evidence"
    assert review.excluded_through is None and review.upper_bound is None
    refusal.update(status="excluded", claim={"predicate": "nontrivial_weight_exclusion", "through": 2},
                   memberships=[], reason=None)
    assert check_low_weight(code, rehash(refusal)).status == "invalid_record"
