import json

from qeclang.analysis.structured import analyze_lifted
from qeclang.cli import main
from qeclang.constructions.family import CyclicLiftFamily, LaurentMatrix, LiftDomain
from qeclang.constructions import records


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
