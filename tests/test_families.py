from copy import deepcopy
from itertools import product
import json

import pytest

from qeclang.constructions.family import CyclicLiftFamily, LaurentMatrix, LiftDomain
from qeclang.constructions.records import ConstructionLimits, fingerprint
from qeclang.symbolic import GroupAlgebraElement


def family(domain=LiftDomain()):
    return CyclicLiftFamily(LaurentMatrix((((-3, 0, 3), (1,)),), 2),
                           LaurentMatrix((((0, 2),), ((-1,),)), 1), domain)


def test_laurent_normalization_and_specialization():
    assert LaurentMatrix((((-1, 3, -1, 2),),), 1).rows == (((2, 3),),)
    original = family()
    instance = original.instantiate(3)
    assert instance.a.rows[0][0].terms == frozenset({(0,)})
    assert instance.b.rows[1][0].terms == frozenset({(2,)})
    assert instance.n == 12
    assert (instance.lower().hx @ instance.lower().hz.T).is_zero
    assert original.a.rows[0][0] == (-3, 0, 3)


def test_uniform_locality_bounds_including_collisions():
    original = family(LiftDomain(1, 40, False))
    bound = original.locality_bound()
    strict_reduction = False
    for lift in range(1, 41):
        exact = original.instantiate(lift).locality()
        assert exact.max_check_weight <= bound.max_check_weight
        assert exact.max_qubit_degree <= bound.max_qubit_degree
        assert exact.x_nonzeros <= lift*bound.x_nonzeros_per_lift
        assert exact.z_nonzeros <= lift*bound.z_nonzeros_per_lift
        strict_reduction |= exact.max_check_weight < bound.max_check_weight
    assert strict_reduction


def test_manifest_roundtrip_canonical_order_and_file(tmp_path):
    original = family()
    data = json.loads(json.dumps(original.to_dict()))
    restored = CyclicLiftFamily.from_dict(data)
    assert restored == original
    assert restored.fingerprint() == original.fingerprint()
    for lift in (1, 3, 5, 7, 9):
        assert fingerprint(restored.instantiate(lift)) == fingerprint(original.instantiate(lift))
    data["a"]["entries"][0][0].reverse()
    assert CyclicLiftFamily.from_dict(data).fingerprint() == original.fingerprint()
    path = tmp_path / "family.json"
    original.dump(path)
    assert CyclicLiftFamily.load(path) == original
    with pytest.raises(ValueError, match="max_bytes"):
        CyclicLiftFamily.load(path, limits=ConstructionLimits(max_bytes=2))


def test_parameter_domain_is_enforced():
    original = family()
    for invalid in (0, -1, 2, True, 3.0):
        with pytest.raises(ValueError):
            original.instantiate(invalid)
    bounded = family(LiftDomain(3, 7))
    for invalid in (1, 8, 9):
        with pytest.raises(ValueError):
            bounded.instantiate(invalid)
    for allowed in (3, 5, 7):
        bounded.instantiate(allowed)
    assert family(LiftDomain(2, 2, False)).instantiate(2).a.orders == (2,)
    for args in ((2, 2), (3, 1), (0, None), (1, None, 1)):
        with pytest.raises(ValueError):
            LiftDomain(*args)


def test_manifest_budgets_and_malformed_data():
    for limits in (ConstructionLimits(max_rows=1), ConstructionLimits(max_columns=1),
                   ConstructionLimits(max_entries=3), ConstructionLimits(max_terms=6),
                   ConstructionLimits(max_exponent=2)):
        with pytest.raises(ValueError):
            CyclicLiftFamily.from_dict(family().to_dict(), limits=limits)
    for update in (lambda d: d.update(extra=1),
                   lambda d: d.update(template_ring="eval"),
                   lambda d: d["a"]["entries"][0][0].append(3),
                   lambda d: d["a"]["entries"][0][0].append(True),
                   lambda d: d["domain"].update(odd_only="yes")):
        data = deepcopy(family().to_dict())
        update(data)
        with pytest.raises(ValueError):
            CyclicLiftFamily.from_dict(data)


def test_instantiation_checks_budgets_before_allocation(monkeypatch):
    original = family()
    def forbid(*args, **kwargs):
        raise AssertionError("specialization occurred before budgets were checked")
    monkeypatch.setattr(GroupAlgebraElement, "from_terms", forbid)
    for limits in (ConstructionLimits(max_sites=11), ConstructionLimits(max_terms=6),
                   ConstructionLimits(max_order=2), ConstructionLimits(max_group_size=2),
                   ConstructionLimits(max_entries=3), ConstructionLimits(max_exponent=2)):
        with pytest.raises(ValueError):
            original.instantiate(3, limits=limits)


def test_empty_shapes_and_zero_family():
    for m, n in product(range(3), repeat=2):
        seed = LaurentMatrix(tuple(((),)*n for _ in range(m)), n)
        original = CyclicLiftFamily(seed, seed)
        restored = CyclicLiftFamily.from_dict(original.to_dict())
        assert restored.a.shape == (m, n)
        assert original.locality_bound().max_check_weight == 0
        assert original.instantiate(3).n == 3*(m*m+n*n)


def test_invalid_seed_types():
    for bad in (True, 1.0, "x"):
        with pytest.raises(ValueError):
            LaurentMatrix((((bad,),),), 1)
    with pytest.raises(ValueError):
        LaurentMatrix((((), ()),), 1)
    with pytest.raises(TypeError):
        CyclicLiftFamily(None, LaurentMatrix((), 0))
