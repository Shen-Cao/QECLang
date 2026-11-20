from copy import deepcopy

import pytest

from qeclang.algebra import BinaryMatrix as B
from qeclang.analysis.certificate_io import presentation_fingerprint
from qeclang.analysis.distance import css_distance
from qeclang.codes import CSSCode, Pauli, StabilizerCode
from qeclang.constructions import steane
from qeclang.transforms import hadamard_dual, permute_qubits
from qeclang.workflow.distance_transport import (check_transported_interval,
                                                 transport_interval_conclusion,
                                                 transport_interval_work)
from qeclang.workflow.relations import RelationError
from qeclang.workflow.row_equivalence import row_equivalence_work, row_maps


def interval(code, sector="x", lower=3, upper=3):
    return {"schema": "qeclang.workflow.distance_claim.v1", "source": presentation_fingerprint(code),
            "sector": sector, "lower_bound": lower, "upper_bound": upper,
            "status": "exact" if lower == upper else "bounded"}


def equivalent_presentations():
    source = steane()
    def changed(matrix):
        return B(tuple(reversed(matrix.rows)) + (matrix.rows[0] ^ matrix.rows[1], 0), matrix.ncols)
    return source, CSSCode(changed(source.hx), changed(source.hz))


@pytest.mark.parametrize("sector", ["x", "z", "pauli"])
@pytest.mark.parametrize("upper", [3, 5])
def test_generator_space_transport_keeps_interval_and_rebinds_subject(sector, upper):
    source, target = equivalent_presentations()
    claim = interval(source, sector, 3, upper)
    before = deepcopy(claim)
    params = {"relation": "generator_spaces", "maps": row_maps(source, target)}
    result = transport_interval_conclusion(source, target, claim, params)
    assert claim == before and result is not claim
    assert result == {**claim, "source": presentation_fingerprint(target)}
    assert result["source"] != claim["source"]
    assert check_transported_interval(source, target, claim, result, params) is None
    assert transport_interval_work(source, target, params) == row_equivalence_work(source, target, params["maps"])


@pytest.mark.parametrize("sector", ["x", "z", "pauli"])
def test_permutation_transport_preserves_sector_and_exact_status(sector):
    source = steane()
    permutation = [1, 0, 3, 2, 6, 4, 5]
    target = permute_qubits(source, permutation)
    params = {"relation": "permutation", "permutation": permutation}
    claim = interval(source, sector)
    result = transport_interval_conclusion(source, target, claim, params)
    assert result["sector"] == sector and result["status"] == "exact"
    assert check_transported_interval(source, target, claim, result, params) is None
    inverse = [permutation.index(j) for j in range(source.n)]
    assert transport_interval_conclusion(target, source, result,
        {"relation": "permutation", "permutation": inverse}) == claim


def test_hadamard_sector_swap_is_not_an_allowed_fixed_sector_relation():
    source = CSSCode(B((), 3), B((3, 6), 3))
    target = hadamard_dual(source)
    assert source.n == target.n and source.k == target.k
    assert css_distance(source, "x").distance == 3
    assert css_distance(target, "x").distance == 1
    with pytest.raises(RelationError):
        transport_interval_conclusion(source, target, interval(source),
                                      {"relation": "permutation", "permutation": [0, 1, 2]})
    maps = {"x_forward": [[], []], "x_reverse": [], "z_forward": [], "z_reverse": [[], []]}
    with pytest.raises(RelationError):
        transport_interval_conclusion(source, target, interval(source),
                                      {"relation": "generator_spaces", "maps": maps})


def test_equal_n_k_or_one_way_inclusion_cannot_transport():
    source = CSSCode(B((3,), 4), B((12,), 4))
    wrong = CSSCode(B((5,), 4), B((10,), 4))
    assert (source.n, source.k) == (wrong.n, wrong.k)
    with pytest.raises(RelationError):
        transport_interval_conclusion(source, wrong, interval(source, lower=1, upper=1),
                                      {"relation": "permutation", "permutation": [0, 1, 2, 3]})
    free = CSSCode(B((), 4), B((), 4))
    maps = {"x_forward": [], "x_reverse": [[]], "z_forward": [], "z_reverse": [[]]}
    with pytest.raises(RelationError):
        transport_interval_conclusion(source, free, interval(source, lower=1, upper=1),
                                      {"relation": "generator_spaces", "maps": maps})


def test_independent_reader_never_invokes_producers_or_distance(monkeypatch):
    import qeclang.workflow.distance_transport as module
    import qeclang.workflow.row_equivalence as row_module

    source, target = equivalent_presentations()
    params = {"relation": "generator_spaces", "maps": row_maps(source, target)}
    claim = interval(source)
    result = transport_interval_conclusion(source, target, claim, params)

    def forbidden(*args, **kwargs):
        raise AssertionError("conditional reader must not compute or trust producer outputs")

    monkeypatch.setattr(module, "transport_interval_conclusion", forbidden)
    monkeypatch.setattr(row_module, "row_maps", forbidden)
    for name in ("rank", "solve", "nullspace", "transpose"):
        monkeypatch.setattr(B, name, forbidden)
    assert check_transported_interval(source, target, claim, result, params) is None


def test_raw_source_claim_is_not_authenticated_by_transport_relation():
    source = steane()
    fake = interval(source, lower=1, upper=1)
    assert css_distance(source, "x").distance == 3
    params = {"relation": "permutation", "permutation": list(range(source.n))}
    conditional = transport_interval_conclusion(source, source, fake, params)
    assert conditional == fake
    assert not {"verified", "assurance", "certificate"}.intersection(conditional)
    # This checks the implication edge only; the workflow must reject fake as an unchecked premise.
    assert check_transported_interval(source, source, fake, conditional, params) is None


@pytest.mark.parametrize("change", [{"sector": "z"}, {"lower_bound": 2, "status": "bounded"},
                                    {"upper_bound": 4, "status": "bounded"},
                                    {"source": "0" * 64}, {"scope": "family"},
                                    {"lower_bound": True}, {"status": "bounded"}])
def test_target_interval_fields_cannot_change(change):
    source, target = equivalent_presentations()
    params = {"relation": "generator_spaces", "maps": row_maps(source, target)}
    claim = interval(source)
    result = transport_interval_conclusion(source, target, claim, params)
    with pytest.raises(RelationError):
        check_transported_interval(source, target, claim, {**result, **change}, params)


def test_zero_qubits_signs_invalid_maps_and_size_limits():
    empty = CSSCode(B((), 0), B((), 0))
    params = {"relation": "permutation", "permutation": []}
    assert transport_interval_work(empty, empty, params) > 0
    with pytest.raises(RelationError):
        transport_interval_conclusion(empty, empty, interval(empty, lower=1, upper=1), params)
    positive = CSSCode(B((), 1), B((1,), 1))
    negative = StabilizerCode(1, (Pauli.from_label("-Z"),))
    with pytest.raises(RelationError):
        transport_interval_conclusion(positive, negative, interval(positive, lower=1, upper=1),
                                      {"relation": "permutation", "permutation": [0]})
    source = steane()
    for permutation in ([0] * 7, list(range(6)), [0, 1, 2, 3, 4, 5, True]):
        with pytest.raises(RelationError):
            transport_interval_conclusion(source, source, interval(source),
                                          {"relation": "permutation", "permutation": permutation})


def test_host_work_cap_precedes_mandatory_relation_scan(monkeypatch):
    import qeclang.workflow.distance_transport as module

    source = steane()
    params = {"relation": "permutation", "permutation": list(range(source.n))}
    claim = interval(source)
    units = transport_interval_work(source, source, params)
    assert transport_interval_conclusion(source, source, claim, params, max_work=units) == claim

    def forbidden(*args, **kwargs):
        raise AssertionError("work cap must precede mathematical relation scan")

    monkeypatch.setattr(module, "check_relation", forbidden)
    with pytest.raises(RelationError, match="work"):
        transport_interval_conclusion(source, source, claim, params, max_work=units - 1)
    with pytest.raises(RelationError, match="axis"):
        transport_interval_conclusion(source, source, claim, params, max_axis=6)


@pytest.mark.parametrize("params", [{}, {"relation": "hadamard"},
                                    {"relation": "permutation", "permutation": [], "maps": {}},
                                    {"relation": "generator_spaces", "x_forward": []}])
def test_only_explicit_registered_metric_relations(params):
    source = steane()
    with pytest.raises(RelationError):
        transport_interval_conclusion(source, source, interval(source), params)
