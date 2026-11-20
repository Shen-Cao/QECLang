from copy import deepcopy

import pytest

from qeclang.algebra import BinaryMatrix as B
from qeclang.analysis.certificate_io import certificate_record, presentation_fingerprint, verify_certificate_record
from qeclang.analysis.certificates import certify_css_logical
from qeclang.analysis.logicals import css_logicals
from qeclang.codes import CSSCode, Pauli, StabilizerCode
from qeclang.decoding import CSSBPDecoder
from qeclang.transforms import independent_checks
from qeclang.workflow.relations import RelationError
from qeclang.workflow.row_equivalence import (check_rebound, rebind_witness, row_equivalence_work,
                                               row_maps, validate_map_supports, verify_row_equivalence)


def presentations():
    source = CSSCode(B((3, 12), 4), B((15,), 4))
    target = CSSCode(B((15, 3, 0), 4), B((15, 15), 4))
    return source, target


def map_syndrome(rows, syndrome):
    return sum((sum((syndrome >> i) & 1 for i in support) % 2) << j
               for j, support in enumerate(rows))


def test_bidirectional_generator_and_physical_syndrome_identities():
    source, target = presentations()
    maps = row_maps(source, target)
    assert maps == {"x_forward": [[0, 1], [0], []], "x_reverse": [[1], [0, 1]],
                    "z_forward": [[0], [0]], "z_reverse": [[0]]}
    guarantees = verify_row_equivalence(source, target, maps)
    assert "positive_stabilizer_subspace" in guarantees and "sector_distances" in guarantees
    assert "syndrome_maps_on_physical_errors" in guarantees
    assert all(name not in guarantees for name in ("check_weights", "check_order", "tanner_graph",
                                                    "logical_success_probability", "decoder_performance"))
    for word in range(1 << source.n):
        for sector in ("x", "z"):
            old = getattr(source, "h" + sector).matvec(word)
            new = getattr(target, "h" + sector).matvec(word)
            assert map_syndrome(maps[sector + "_forward"], old) == new
            assert map_syndrome(maps[sector + "_reverse"], new) == old
    assert map_syndrome(maps["z_forward"], map_syndrome(maps["z_reverse"], 1)) != 1
    assert source.hx.rows[0].bit_count() != target.hx.rows[0].bit_count()
    assert source.hz.T.rows != target.hz.T.rows


def test_row_reduction_redundant_and_zero_coordinate_cases():
    source, target = presentations()
    for old, new in ((target, independent_checks(target)),
                     (CSSCode(B((), 0), B((), 0)), CSSCode(B((0, 0), 0), B((0,), 0))),
                     (CSSCode(B((), 3), B((), 3)), CSSCode(B((0,), 3), B((0, 0), 3)))):
        maps = row_maps(old, new)
        assert verify_row_equivalence(old, new, maps)
        assert old.n == new.n and old.k == new.k
    assert row_maps(source, target) == row_maps(source, target)


def test_one_sided_inclusion_and_equal_parameters_are_insufficient():
    source = CSSCode(B((3,), 3), B((), 3))
    target = CSSCode(B((), 3), B((), 3))
    maps = {"x_forward": [], "x_reverse": [[]], "z_forward": [], "z_reverse": []}
    with pytest.raises(RelationError, match="x_reverse"):
        verify_row_equivalence(source, target, maps)
    old, good = presentations()
    wrong = CSSCode(B((5, 10, 0), 4), good.hz)
    assert (old.n, old.k) == (wrong.n, wrong.k)
    with pytest.raises(RelationError):
        verify_row_equivalence(old, wrong, row_maps(old, good))
    with pytest.raises(RelationError, match="not equivalent"):
        row_maps(old, wrong)


def test_noisy_syndrome_graph_and_decoder_outcome_do_not_transfer():
    # Same repetition code, different central variable in its check graph.
    source = CSSCode(B((), 3), B((3, 6), 3))
    target = CSSCode(B((), 3), B((3, 5), 3))
    assert verify_row_equivalence(source, target, row_maps(source, target))
    options = dict(max_iterations=1, fallback="none")
    old_decoder = CSSBPDecoder(source, 0.1, 0.1, **options)
    new_decoder = CSSBPDecoder(target, 0.1, 0.1, **options)
    old = old_decoder.decode(0, source.hz.matvec(1))
    new = new_decoder.decode(0, target.hz.matvec(1))
    assert not old.success and new.success
    assert new.x.error == 1


@pytest.mark.parametrize("sector", ["x", "z"])
def test_witness_rebinding_preserves_words_and_rebinds_presentation(sector):
    source, target = presentations()
    maps = row_maps(source, target)
    logicals = css_logicals(source)
    operator = (logicals.x if sector == "x" else logicals.z)[0]
    original = certificate_record(source, certify_css_logical(source, sector, operator))
    snapshot = deepcopy(original)
    rebound = rebind_witness(source, target, maps, original)
    assert original == snapshot
    assert rebound["certificate"] == original["certificate"] and rebound["claim"] == original["claim"]
    assert rebound["source"] == presentation_fingerprint(target) != original["source"]
    assert verify_certificate_record(target, rebound)
    assert check_rebound(source, target, maps, original, rebound) is None
    assert rebind_witness(target, source, row_maps(target, source), rebound) == original


def test_reader_is_independent_of_producers_solve_and_rank(monkeypatch):
    import qeclang.workflow.row_equivalence as module

    source, target = presentations()
    maps = row_maps(source, target)
    original = certificate_record(source, certify_css_logical(source, "x", 5))
    rebound = rebind_witness(source, target, maps, original)

    def forbidden(*args, **kwargs):
        raise AssertionError("row-equivalence reader must not invoke producer or elimination")

    monkeypatch.setattr(module, "row_maps", forbidden)
    monkeypatch.setattr(module, "rebind_witness", forbidden)
    for method in ("solve", "rank", "nullspace", "transpose"):
        monkeypatch.setattr(B, method, forbidden)
    assert verify_row_equivalence(source, target, maps)
    assert check_rebound(source, target, maps, original, rebound) is None


def test_changed_but_valid_target_certificate_is_not_the_declared_rebinding():
    source, target = presentations()
    maps = row_maps(source, target)
    original = certificate_record(source, certify_css_logical(source, "x", 5))
    rebound = rebind_witness(source, target, maps, original)
    different_operator = certificate_record(target, certify_css_logical(target, "x", 10))
    assert verify_certificate_record(target, different_operator)
    with pytest.raises(RelationError, match="operator"):
        check_rebound(source, target, maps, original, different_operator)
    different_dual = deepcopy(rebound)
    different_dual["certificate"]["dual"] = hex(int(rebound["certificate"]["dual"], 16) ^ 15)
    assert verify_certificate_record(target, different_dual)
    with pytest.raises(RelationError, match="dual"):
        check_rebound(source, target, maps, original, different_dual)
    with pytest.raises(RelationError, match="source-bound"):
        check_rebound(source, target, maps, original, original)


@pytest.mark.parametrize("support", [[0, 0], [1, 0], [True], [0.0], [-1], [2], (0,)])
def test_row_map_index_syntax_is_canonical(support):
    source, target = presentations()
    maps = row_maps(source, target)
    maps["x_forward"][0] = support
    with pytest.raises(RelationError):
        verify_row_equivalence(source, target, maps)


def test_map_keys_shapes_count_bounds_and_work_proxy():
    source, target = presentations()
    maps = row_maps(source, target)
    assert validate_map_supports(maps, (2, 1), (3, 2)) == (8, 9)
    assert row_equivalence_work(source, target, maps) == 8 + 5 * 8 + 2 * 9
    for limits in ({"max_rows": 7}, {"max_indices": 8}):
        with pytest.raises(RelationError):
            validate_map_supports(maps, (2, 1), (3, 2), **limits)
    for changed in ({key: value for key, value in maps.items() if key != "x_reverse"},
                    {**maps, "phase": 2}, {**maps, "x_reverse": []}):
        with pytest.raises(RelationError):
            verify_row_equivalence(source, target, changed)


def test_signs_width_changes_and_certificate_tampering_rejected():
    positive = CSSCode(B((), 1), B((1,), 1))
    negative = StabilizerCode(1, (Pauli.from_label("-Z"),))
    with pytest.raises(RelationError, match="positive"):
        verify_row_equivalence(positive, negative, {})
    with pytest.raises(RelationError, match="physical"):
        verify_row_equivalence(positive, CSSCode(B((), 2), B((1,), 2)), {})
    source, target = presentations()
    maps = row_maps(source, target)
    original = certificate_record(source, certify_css_logical(source, "x", 5))
    for field, value in (("operator", "0x100"), ("dual", "0x0"), ("weight", True), ("phase", 2)):
        altered = deepcopy(original)
        altered["certificate"][field] = value
        with pytest.raises(RelationError):
            rebind_witness(source, target, maps, altered)


def test_producer_and_witness_guards_precede_materialization(monkeypatch):
    import qeclang.workflow.row_equivalence as module

    source, target = presentations()
    maps = row_maps(source, target)
    original = certificate_record(source, certify_css_logical(source, "x", 5))

    def forbidden(*args, **kwargs):
        raise AssertionError("resource preflight must precede production/checking")

    monkeypatch.setattr(B, "transpose", forbidden)
    for limits in ({"max_bits": 0}, {"max_axis": 3}):
        with pytest.raises(RelationError):
            row_maps(source, target, **limits)
    monkeypatch.setattr(module, "verify_row_equivalence", forbidden)
    for limits in ({"max_bits": 15}, {"max_axis": 3}):
        with pytest.raises(RelationError):
            rebind_witness(source, target, maps, original, **limits)
        with pytest.raises(RelationError):
            check_rebound(source, target, maps, original, original, **limits)
