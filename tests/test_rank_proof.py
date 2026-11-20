from copy import deepcopy
from dataclasses import asdict
from itertools import product
import json
from random import Random

import pytest

from qeclang.analysis.structured import analyze_lifted
from qeclang.constructions.lifted import LiftedProduct
from qeclang.constructions.records import fingerprint
from qeclang.symbolic import GroupAlgebraElement as G, GroupAlgebraMatrix as M
from qeclang.workflow.rank_proof import verify_dimension_proof


def fixture(lift=4):
    a = G.from_terms((lift,), [(0,), (1,)])
    b = G.from_terms((lift,), [(0,), (2,)])
    return LiftedProduct(M(((a, b), (b, a)), 2, (lift,)), M(((a,),), 1, (lift,)))


def proof(construction):
    result = analyze_lifted(construction, backend="polynomial", record_operations=True)
    assert result.status == "exact"
    return json.loads(json.dumps(asdict(result)))


def test_json_proofs_for_random_rectangular_even_odd_and_empty_seeds():
    rng = Random(192)
    for lift in (1, 2, 3, 4, 6):
        for ma, na, mb, nb in product(range(3), repeat=4):
            def matrix(m, n):
                return M(tuple(tuple(G.from_terms((lift,), [(j,) for j in range(lift)
                                                            if rng.getrandbits(1)])
                                     for _ in range(n)) for _ in range(m)), n, (lift,))
            construction = LiftedProduct(matrix(ma, na), matrix(mb, nb))
            payload = proof(construction)
            result = verify_dimension_proof(construction, payload)
            assert result.status == "verified", result.reason
            assert (result.n, result.k, result.rank_x, result.rank_z) == (
                payload["n"], payload["k"], payload["rank_x"], payload["rank_z"])


def test_reader_does_not_call_rank_producers_or_binary_expansion(monkeypatch):
    import qeclang.analysis.structured as structured
    import qeclang.symbolic.repeated_root as polynomial
    from qeclang.algebra import BinaryMatrix

    construction = fixture()
    payload = proof(construction)

    def forbidden(*args, **kwargs):
        raise AssertionError("proof verification must not repeat rank production or expansion")

    monkeypatch.setattr(structured, "analyze_lifted", forbidden)
    monkeypatch.setattr(polynomial, "cyclic_rank_general", forbidden)
    monkeypatch.setattr(BinaryMatrix, "rank", forbidden)
    monkeypatch.setattr(M, "lower", forbidden)
    monkeypatch.setattr(M, "lower_sparse", forbidden)
    assert verify_dimension_proof(construction, payload).status == "verified"


@pytest.mark.parametrize("field", ["n", "k", "rank_x", "rank_z", "steps"])
def test_boolean_parameter_claims_are_not_integers(field):
    construction = fixture()
    payload = proof(construction)
    payload[field] = True
    result = verify_dimension_proof(construction, payload)
    assert result.status == "invalid_proof" and result.steps == 0


def test_source_binding_shape_indices_and_trace_required():
    construction = fixture()
    original = proof(construction)
    for field, value in (("construction_fingerprint", "0" * 64), ("method", "crt"), ("reason", "pretend")):
        changed = deepcopy(original)
        changed[field] = value
        assert verify_dimension_proof(construction, changed).status == "invalid_proof"
    for field, value in (("shape", [2, 10**100]), ("lift", 10**100), ("rank", True),
                         ("operation_count", True), ("steps", False), ("operations", None)):
        changed = deepcopy(original)
        changed["x_evidence"][field] = value
        result = verify_dimension_proof(construction, changed)
        assert result.status == "invalid_proof" and result.steps == 0
    changed = deepcopy(original)
    changed["x_evidence"]["operations"][0]["source"] = True
    assert verify_dimension_proof(construction, changed).status == "invalid_proof"


def test_independent_replay_rejects_altered_diagonal_after_spending_work():
    construction = fixture()
    payload = proof(construction)
    payload["z_evidence"]["diagonal"][0] ^= 1
    result = verify_dimension_proof(construction, payload)
    assert result.status == "invalid_proof" and result.steps > 0
    assert result.n is result.k is result.rank_x is result.rank_z is None


def test_reader_budget_is_shared_and_retains_consumed_work_on_refusal():
    construction = fixture()
    payload = proof(construction)
    complete = verify_dimension_proof(construction, payload)
    assert complete.steps > 1
    assert verify_dimension_proof(construction, payload, max_steps=complete.steps).status == "verified"
    stopped = verify_dimension_proof(construction, payload, max_steps=complete.steps - 1)
    assert stopped.status == "budget_exhausted" and 0 < stopped.steps <= complete.steps - 1
    assert stopped.k is None
    for options in ({"max_blocks": 0}, {"max_entries": 0}, {"max_degree": 0},
                    {"max_polynomial_degree": 0}, {"max_operations": 0}, {"max_steps": 0}):
        stopped = verify_dimension_proof(construction, payload, **options)
        assert stopped.status == "budget_exhausted" and stopped.steps == 0


def test_oversized_source_and_proof_polynomials_refuse_before_shifts(monkeypatch):
    construction = fixture()
    payload = proof(construction)
    payload["x_evidence"]["diagonal"][0] = 1 << 10_000
    assert verify_dimension_proof(construction, payload).status == "budget_exhausted"
    empty = LiftedProduct(M.zeros(0, 0, (1,)), M.zeros(0, 0, (1,)))
    payload = proof(empty)
    huge = LiftedProduct(M.zeros(0, 0, (10**100,)), M.zeros(0, 0, (10**100,)))
    payload["construction_fingerprint"] = fingerprint(huge)

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized source must be rejected before check construction")

    monkeypatch.setattr(LiftedProduct, "symbolic_checks", forbidden)
    result = verify_dimension_proof(huge, payload)
    assert result.status == "budget_exhausted" and result.steps == 0


def test_symbolic_monomial_quota_precedes_check_materialization(monkeypatch):
    element = G.from_terms((8,), [(j,) for j in range(8)])
    construction = LiftedProduct(M(((element,),), 1, (8,)), M(((element,),), 1, (8,)))
    payload = proof(construction)
    assert verify_dimension_proof(construction, payload, max_symbolic_terms=32).status == "verified"

    def forbidden(*args, **kwargs):
        raise AssertionError("monomial quota must be checked before symbolic materialization")

    monkeypatch.setattr(LiftedProduct, "symbolic_checks", forbidden)
    result = verify_dimension_proof(construction, payload, max_symbolic_terms=31)
    assert result.status == "budget_exhausted" and result.steps == 0
    assert "symbolic_terms" in result.reason


def test_producer_step_counts_are_not_relabelled_as_verified_consumption():
    construction = fixture()
    payload = proof(construction)
    expected = verify_dimension_proof(construction, payload)
    payload["x_evidence"]["steps"] += 10**20
    payload["steps"] += 10**20
    checked = verify_dimension_proof(construction, payload)
    assert checked.status == "verified" and checked.steps == expected.steps


def test_invalid_reader_limits_remain_caller_errors():
    construction = fixture()
    payload = proof(construction)
    for invalid in (-1, True, float("nan")):
        with pytest.raises(ValueError):
            verify_dimension_proof(construction, payload, max_steps=invalid)
