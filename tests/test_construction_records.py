from copy import deepcopy
import json

import pytest

from qeclang.constructions.lifted import LiftedProduct
from qeclang.constructions.records import (ACTION, ConstructionLimits, dump, fingerprint,
                                          from_dict, load, to_dict)
from qeclang.symbolic import GroupAlgebraElement as E, GroupAlgebraMatrix as M


def construction():
    a = E.from_terms((3, 5), [(0, 0), (1, 2)])
    b = E.from_terms((3, 5), [(2, 1)])
    return LiftedProduct(M(((a, b),), 2, (3, 5)), M(((b,), (a,)), 1, (3, 5)))


def test_roundtrip_preserves_lift_action_and_fingerprint(tmp_path):
    original = construction()
    data = to_dict(original)
    restored = from_dict(json.loads(json.dumps(data)))
    assert restored == original
    assert restored.lower().hx == original.lower().hx
    assert restored.lower().hz == original.lower().hz
    assert fingerprint(restored) == fingerprint(original)
    assert len(fingerprint(original)) == 64
    path = tmp_path / "construction.json"
    dump(original, path)
    assert load(path) == original
    reordered = deepcopy(data)
    reordered["a"]["entries"][0][0].reverse()
    assert fingerprint(from_dict(reordered)) == fingerprint(original)


def test_fingerprint_preserves_meaningful_conventions():
    original = construction()
    assert fingerprint(original) != fingerprint(LiftedProduct(original.b, original.a))
    data = to_dict(original)
    data["action"] = ACTION + "-different"
    with pytest.raises(ValueError, match="semantics"):
        from_dict(data)


@pytest.mark.parametrize("field,value", [("schema", "future"), ("kind", "eval"),
    ("coefficient_field", "GF(3)"), ("convention", "transpose-only"),
    ("physical_ordering", "group-first"), ("orders", [True]), ("orders", [0])])
def test_unsupported_records_rejected(field, value):
    data = to_dict(construction())
    data[field] = value
    with pytest.raises(ValueError):
        from_dict(data)


def test_malformed_entries_and_unknown_fields():
    for update in (lambda d: d.update(command="__import__('os')"),
                   lambda d: d["a"].update(extra=1),
                   lambda d: d["a"].update(shape=[100, 2]),
                   lambda d: d["a"]["entries"][0][0].append([0, 0]),
                   lambda d: d["a"]["entries"][0][0].append([-1, 0]),
                   lambda d: d["a"]["entries"][0][0].append([3, 0]),
                   lambda d: d["a"]["entries"][0][0].append([1.0, 0])):
        data = to_dict(construction())
        update(data)
        with pytest.raises(ValueError):
            from_dict(data)


@pytest.mark.parametrize("limits", [ConstructionLimits(max_factors=1),
    ConstructionLimits(max_order=4), ConstructionLimits(max_group_size=14),
    ConstructionLimits(max_rows=1), ConstructionLimits(max_columns=1),
    ConstructionLimits(max_entries=3), ConstructionLimits(max_terms=5),
    ConstructionLimits(max_sites=59)])
def test_limits_checked_before_algebra_allocation(limits, monkeypatch):
    data = to_dict(construction())
    def forbid(*args, **kwargs):
        raise AssertionError("algebra objects were allocated before validation")
    monkeypatch.setattr(E, "from_terms", forbid)
    with pytest.raises(ValueError):
        from_dict(data, limits=limits)


def test_bounded_file_and_duplicate_object_keys(tmp_path):
    path = tmp_path / "data.json"
    dump(construction(), path)
    with pytest.raises(ValueError, match="max_bytes"):
        load(path, limits=ConstructionLimits(max_bytes=5))
    path.write_text('{"schema":"one","schema":"two"}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        load(path)


def test_explicit_empty_dimensions():
    original = LiftedProduct(M.zeros(0, 2, (3,)), M.zeros(1, 0, (3,)))
    restored = from_dict(to_dict(original))
    assert restored == original
    assert restored.a.shape == (0, 2)
    assert restored.b.shape == (1, 0)
    assert restored.lower().n == 0


@pytest.mark.parametrize("value", [True, -1, 1.0])
def test_invalid_budget_types(value):
    with pytest.raises(ValueError):
        ConstructionLimits(max_sites=value)
