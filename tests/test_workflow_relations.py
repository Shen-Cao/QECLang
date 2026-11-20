from copy import deepcopy
from dataclasses import replace

import pytest

from qeclang.algebra import BinaryMatrix as B, SparseBinaryMatrix as S
from qeclang.analysis.certificate_io import (certificate_record, presentation_fingerprint,
                                            verify_certificate_record)
from qeclang.analysis.certificates import certify_css_logical
from qeclang.analysis.logicals import css_logicals
from qeclang.codes import CSSCode, ChainComplex, Pauli, SparseCSSCode, StabilizerCode
from qeclang.transforms import permute_qubits
from qeclang.workflow.relations import RelationError, check_relation, transport_witness


def code():
    return CSSCode(B((3,), 4), B((12,), 4), "source label")


def test_based_edges_preserve_empty_redundant_and_zero_checks():
    cases = (
        CSSCode(B((), 0), B((), 0)),
        CSSCode(B((0, 0), 0), B((0,), 0)),
        CSSCode(B((), 3), B((), 3)),
        CSSCode(B((3, 12, 3, 0), 5), B((15, 0, 15), 5)),
    )
    for source in cases:
        chain = source.to_chain_complex()
        sparse = SparseCSSCode.from_packed(source)
        for op, left, right in (("css.chain", source, chain), ("chain.css", chain, source),
                                ("css.sparse", source, sparse), ("sparse.css", sparse, source)):
            guarantees = check_relation(op, left, right, {})
            assert "physical_basis_identity" in guarantees
            assert "check_order" in guarantees and "check_labels" in guarantees
            assert "hamming_metric" in guarantees and "logical_quotient" in guarantees
            assert "name" not in guarantees
        permuted = check_relation("css.permute", source, source,
                                   {"permutation": list(range(source.n))})
        assert "qubit_bijection" in permuted and "physical_basis_identity" not in permuted


def test_decorative_names_do_not_change_relations():
    source = code()
    assert check_relation("css.chain", source, replace(source.to_chain_complex(), name="renamed"), {})
    assert check_relation("css.sparse", source, replace(SparseCSSCode.from_packed(source), name="renamed"), {})
    assert check_relation("css.permute", source, replace(source, name="renamed"),
                          {"permutation": list(range(source.n))})


def test_checks_are_independent_of_producers_and_rank(monkeypatch):
    source = code()
    chain, sparse = source.to_chain_complex(), SparseCSSCode.from_packed(source)
    permutation = [2, 0, 3, 1]
    target = permute_qubits(source, permutation)
    logical = css_logicals(source).x[0]
    record = certificate_record(source, certify_css_logical(source, "x", logical))

    def forbidden(*args, **kwargs):
        raise AssertionError("relational checker must not call producers, transpose, or elimination")

    monkeypatch.setattr(CSSCode, "to_chain_complex", forbidden)
    monkeypatch.setattr(ChainComplex, "to_css", forbidden)
    monkeypatch.setattr(SparseCSSCode, "from_packed", forbidden)
    monkeypatch.setattr(SparseCSSCode, "to_packed", forbidden)
    monkeypatch.setattr(B, "transpose", forbidden)
    monkeypatch.setattr(B, "rank", forbidden)
    monkeypatch.setattr(S, "to_packed", forbidden)
    monkeypatch.setattr(S, "from_packed", forbidden)
    import qeclang.transforms as transforms
    monkeypatch.setattr(transforms, "permute_qubits", forbidden)
    for op, left, right in (("css.chain", source, chain), ("chain.css", chain, source),
                            ("css.sparse", source, sparse), ("sparse.css", sparse, source)):
        assert check_relation(op, left, right, {})
    transported = transport_witness(source, target, permutation, record)
    assert verify_certificate_record(target, transported)


def test_n_k_and_weight_preservation_alone_cannot_certify_the_relation():
    source = code()
    switched_labels = CSSCode(source.hz, source.hx)
    changed_support = CSSCode(B((5,), 4), B((10,), 4))
    for wrong in (switched_labels, changed_support):
        assert (wrong.n, wrong.k) == (source.n, source.k)
        assert wrong.hx.row_basis().rows[0].bit_count() == source.hx.rows[0].bit_count()
        with pytest.raises(RelationError, match="relationship"):
            check_relation("css.chain", source, wrong.to_chain_complex(), {})
        with pytest.raises(RelationError, match="relationship"):
            check_relation("css.sparse", source, SparseCSSCode.from_packed(wrong), {})
        with pytest.raises(RelationError, match="relationship"):
            check_relation("css.permute", source, wrong, {"permutation": [0, 1, 2, 3]})


def test_order_redundancy_and_explicit_shapes_are_preserved():
    source = CSSCode(B((3, 12), 4), B((15,), 4))
    reordered = CSSCode(B((12, 3), 4), source.hz)
    redundant = CSSCode(B((3, 12, 0), 4), source.hz)
    assert source.k == reordered.k == redundant.k
    for wrong in (reordered, redundant):
        with pytest.raises(RelationError):
            check_relation("chain.css", source.to_chain_complex(), wrong, {})
        with pytest.raises(RelationError):
            check_relation("sparse.css", SparseCSSCode.from_packed(source), wrong, {})
    empty_checks = CSSCode(B((), 2), B((), 2))
    wider = CSSCode(B((1,), 3), B((), 3))
    assert empty_checks.k == wider.k
    with pytest.raises(RelationError):
        check_relation("css.chain", empty_checks, wider.to_chain_complex(), {})
    # A zero boundary column still represents a retained check coordinate.
    wrong_chain = ChainComplex(empty_checks.hx, B((0, 0), 1))
    assert wrong_chain.betti_numbers[1] == empty_checks.k
    with pytest.raises(RelationError):
        check_relation("css.chain", empty_checks, wrong_chain, {})


def test_signed_pauli_sectors_are_not_silently_treated_as_positive_css():
    positive = CSSCode(B((), 1), B((1,), 1))
    negative = StabilizerCode(1, (Pauli.from_label("-Z"),))
    assert (positive.n, positive.k) == (negative.n, negative.k)
    with pytest.raises(RelationError, match="CSS dialect"):
        check_relation("chain.css", positive.to_chain_complex(), negative, {})


@pytest.mark.parametrize("permutation", [[0, 0, 2, 3], [0, 1, 2], [0, 1, 2, 4],
                                         [0, 1, 2, -1], [0, 1, 2, True], [0, 1, 2, 3.0], None])
def test_invalid_qubit_maps(permutation):
    with pytest.raises(RelationError):
        check_relation("css.permute", code(), code(), {"permutation": permutation})


def test_permutation_direction_is_old_site_to_new_site():
    source = code()
    permutation = [2, 0, 3, 1]
    target = CSSCode(B((5,), 4), B((10,), 4))
    assert check_relation("css.permute", source, target, {"permutation": permutation})
    inverse = [permutation.index(j) for j in range(4)]
    with pytest.raises(RelationError):
        check_relation("css.permute", source, target, {"permutation": inverse})


@pytest.mark.parametrize("sector", ["x", "z"])
def test_logical_witness_transport_is_source_bound_and_reversible(sector):
    source = code()
    permutation = [2, 0, 3, 1]
    target = permute_qubits(source, permutation)
    logicals = css_logicals(source)
    logical = (logicals.x if sector == "x" else logicals.z)[0]
    record = certificate_record(source, certify_css_logical(source, sector, logical))
    before = deepcopy(record)
    transported = transport_witness(source, target, permutation, record)
    assert record == before and transported is not record
    assert transported["source"] == presentation_fingerprint(target)
    assert transported["claim"] == record["claim"]
    assert verify_certificate_record(target, transported)
    for name in ("operator", "dual"):
        original = int(record["certificate"][name], 16)
        expected = sum(((original >> i) & 1) << permutation[i] for i in range(source.n))
        assert int(transported["certificate"][name], 16) == expected
    inverse = [permutation.index(j) for j in range(source.n)]
    assert transport_witness(target, source, inverse, transported) == record


def test_witness_transport_rejects_tampering_and_wrong_target():
    source = code()
    permutation = [2, 0, 3, 1]
    target = permute_qubits(source, permutation)
    logical = css_logicals(source).x[0]
    record = certificate_record(source, certify_css_logical(source, "x", logical))
    mutations = []
    for key, value in (("dual", "0x0"), ("operator", "0x10"), ("weight", 999), ("phase", 2)):
        altered = deepcopy(record)
        altered["certificate"][key] = value
        mutations.append(altered)
    wrong_source = deepcopy(record)
    wrong_source["source"] = presentation_fingerprint(target)
    mutations.append(wrong_source)
    for altered in mutations:
        with pytest.raises(RelationError, match="certificate"):
            transport_witness(source, target, permutation, altered)
    with pytest.raises(RelationError, match="relationship"):
        transport_witness(source, source, permutation, record)


@pytest.mark.parametrize("op,params", [("chain.map", {}), ("css.chain", {"unused": 0}),
                                     ("css.permute", {}), ("css.permute", {"permutation": [0, 1, 2, 3], "phase": 2}),
                                     ("css.sparse", None)])
def test_no_unregistered_relation_or_parameter_claims(op, params):
    with pytest.raises(RelationError):
        check_relation(op, code(), code(), params)


def test_malformed_matrix_rows_cannot_hide_width_or_support_changes():
    source = code()
    forged = CSSCode(source.hx, source.hz)
    malformed = B((1,), 4)
    object.__setattr__(malformed, "rows", (1 << 4,))
    object.__setattr__(forged, "hx", malformed)
    with pytest.raises(RelationError, match="width"):
        check_relation("css.permute", source, forged, {"permutation": [0, 1, 2, 3]})
    sparse = SparseCSSCode.from_packed(source)
    malformed_sparse = S(((0, 1),), 4)
    object.__setattr__(malformed_sparse, "rows", ((0, 0),))
    object.__setattr__(sparse, "hx", malformed_sparse)
    with pytest.raises(RelationError, match="canonical"):
        check_relation("css.sparse", source, sparse, {})
