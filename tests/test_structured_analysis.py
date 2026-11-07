import json
import pytest

from qeclang.analysis.structured import analyze_lifted
from qeclang.cli import main
from qeclang.constructions.family import CyclicLiftFamily, LaurentMatrix, LiftDomain
from qeclang.constructions import records
from qeclang.constructions.lifted import LiftedProduct
from qeclang.symbolic import GroupAlgebraElement as G, GroupAlgebraMatrix as M


def fixture():
    a = LaurentMatrix((((0,), (1,), ()), ((), (0,), (-1,))), 3)
    b = LaurentMatrix((((0,), (2,)),), 2)
    return CyclicLiftFamily(a, b, LiftDomain(odd_only=False))


def test_symbolic_parameters_and_explicit_fallback():
    family = fixture()
    for lift in (1, 3, 5, 7, 9, 15):
        instance = family.instantiate(lift)
        symbolic = analyze_lifted(instance)
        packed = analyze_lifted(instance, backend="packed")
        assert symbolic.status == packed.status == "exact"
        assert (symbolic.n, symbolic.k, symbolic.rank_x, symbolic.rank_z) == (
            packed.n, packed.k, packed.rank_x, packed.rank_z)
    even = family.instantiate(4)
    assert analyze_lifted(even).status == "unsupported"
    poly = analyze_lifted(even, backend="polynomial", record_operations=True)
    assert poly.status == "exact"
    assert poly.k == analyze_lifted(even, backend="packed").k
    from qeclang.symbolic.repeated_root import verify_rank_certificate
    hx, hz = even.symbolic_checks()
    assert verify_rank_certificate(hx, poly.x_evidence)
    assert verify_rank_certificate(hz, poly.z_evidence)
    assert analyze_lifted(even, backend="polynomial", max_steps=0).status == "budget_exhausted"
    assert analyze_lifted(even, backend="packed").status == "exact"
    assert analyze_lifted(even, backend="packed", max_bits=1).status == "budget_exhausted"
    assert analyze_lifted(family.instantiate(3), max_steps=1).status == "budget_exhausted"
    assert analyze_lifted(family.instantiate(3), max_blocks=0).status == "budget_exhausted"


def test_cli_family_and_construction(tmp_path, capsys):
    family = fixture()
    path = tmp_path / "family.json"
    family.dump(path)
    assert main(["family", str(path), "--lift", "7"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["parameters"]["status"] == "exact"
    assert data["family_fingerprint"] == family.fingerprint()
    path = tmp_path / "instance.json"
    records.dump(family.instantiate(4), path)
    assert main(["construction", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["parameters"]["status"] == "unsupported"
    assert main(["construction", str(path), "--backend", "packed"]) == 0
    assert json.loads(capsys.readouterr().out)["parameters"]["status"] == "exact"


def abelian_fixture(orders):
    a = G.from_terms(orders, [(0,) * len(orders), (1,) * len(orders)])
    b = G.from_terms(orders, [(0,) * len(orders), tuple(2 for _ in orders)])
    return LiftedProduct(M(((a,),), 1, orders), M(((b,),), 1, orders))


@pytest.mark.parametrize("orders", [(3, 3), (3, 5), (2, 3), (4, 2), (6, 5)])
def test_explicit_abelian_dispatch_matches_packed(orders):
    construction = abelian_fixture(orders)
    packed = analyze_lifted(construction, backend="packed")
    mixed = analyze_lifted(construction, backend="mixed-abelian")
    assert packed.status == mixed.status == "exact"
    assert (mixed.k, mixed.rank_x, mixed.rank_z) == (packed.k, packed.rank_x, packed.rank_z)
    assert mixed.method == "mixed-abelian"
    assert mixed.x_evidence.orders == orders
    odd = analyze_lifted(construction, backend="abelian")
    if all(order % 2 for order in orders):
        assert odd.status == "exact" and odd.k == packed.k
    else:
        assert odd.status == "unsupported" and odd.k is None
    assert analyze_lifted(construction, backend="polynomial").status == "unsupported"


@pytest.mark.parametrize("backend,orders", [("crt", (7,)), ("polynomial", (6,)),
                                           ("abelian", (3, 5)), ("mixed-abelian", (2, 3))])
def test_symbolic_rank_work_budget_is_shared_across_checks(backend, orders):
    construction = abelian_fixture(orders)
    result = analyze_lifted(construction, backend=backend)
    assert result.status == "exact"
    assert result.steps == result.x_evidence.steps + result.z_evidence.steps
    assert analyze_lifted(construction, backend=backend, max_steps=result.steps).status == "exact"
    stopped = analyze_lifted(construction, backend=backend, max_steps=result.steps - 1)
    assert stopped.status == "budget_exhausted"
    assert stopped.method == backend and stopped.k is None


def test_abelian_limits_trace_applicability_and_no_expansion(monkeypatch):
    construction = abelian_fixture((2, 3))

    def forbidden(*args, **kwargs):
        raise AssertionError("symbolic analysis must not call packed lower")

    monkeypatch.setattr(LiftedProduct, "lower", forbidden)
    assert analyze_lifted(construction, backend="mixed-abelian").status == "exact"
    for options in ({"max_component_axis": 1}, {"max_component_entries": 1},
                    {"max_characters": 1}, {"max_field_degree": 1},
                    {"max_field_candidates": 0}, {"max_factors": 1}, {"max_entries": 0}):
        stopped = analyze_lifted(construction, backend="mixed-abelian", **options)
        assert stopped.status == "budget_exhausted" and stopped.method == "mixed-abelian"
    with pytest.raises(ValueError, match="polynomial"):
        analyze_lifted(construction, backend="mixed-abelian", record_operations=True)
    with pytest.raises(ValueError):
        analyze_lifted(construction, backend="mixed-abelian", max_component_axis=float("nan"))


def test_packed_axis_and_validation_payload_are_explicit():
    construction = abelian_fixture((3,))
    assert construction.n == 6 and construction.check_counts == (3, 3)
    with pytest.raises(ValueError, match="bits"):
        construction.lower(max_bits=44)
    with pytest.raises(ValueError, match="sites"):
        construction.lower(max_sites=5)
    assert construction.lower(max_bits=45, max_sites=6).n == 6
    assert analyze_lifted(construction, backend="packed", max_sites=5).status == "budget_exhausted"
    assert analyze_lifted(construction, backend="packed", max_sites=6, max_bits=45).status == "exact"


def test_cli_abelian_choices_and_limits(tmp_path, capsys):
    path = tmp_path / "mixed.json"
    records.dump(abelian_fixture((2, 3)), path)
    args = ["construction", str(path), "--backend", "mixed-abelian"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["parameters"]["status"] == "exact"
    assert main(args + ["--max-component-axis", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["parameters"]["status"] == "budget_exhausted"
    assert main(args + ["--certificate-trace"]) == 2
    assert "polynomial" in capsys.readouterr().err
