import json

import pytest

from qeclang.constructions import bacon_shor, five_qubit, steane
from qeclang.codes import Pauli, SubsystemCode
from qeclang.experiments import bicycle_candidates
from qeclang.io import dump, from_dict, load, to_dict


def test_roundtrip(tmp_path):
    signed = SubsystemCode(2, (Pauli(0, 1),), sector=(Pauli(0, 1, 2),))
    for code in (steane(), five_qubit(), bacon_shor(), signed):
        data = to_dict(code)
        assert to_dict(from_dict(json.loads(json.dumps(data)))) == data
        path = tmp_path / "code.json"
        dump(code, path)
        assert to_dict(load(path)) == data
        with pytest.raises(ValueError):
            load(path, max_bytes=1)


def test_invalid_data():
    data = to_dict(steane())
    for modified in (dict(data, n=True), dict(data, kind="exec"), dict(data, extra=0),
                     dict(data, hx=[[1, 1]]), dict(data, hz=[[7]])):
        with pytest.raises(ValueError):
            from_dict(modified)
    with pytest.raises(ValueError):
        from_dict(data, max_bits=1)


def test_search_reproducible_and_checkable():
    a = list(bicycle_candidates((2, 3), terms=2, trials=4, seed=17))
    assert a == list(bicycle_candidates((2, 3), terms=2, trials=4, seed=17))
    assert len(a) == 4
    for record in a:
        code = from_dict(record["code"])
        assert code.n == 12
        for sector in ("x", "z"):
            result = record["distance_" + sector]
            if result["witness"] is not None:
                v = result["witness"]
                opposite, same = (code.hz, code.hx) if sector == "x" else (code.hx, code.hz)
                assert opposite.matvec(v) == 0
                assert not same.contains(v)
                assert v.bit_count() == result["upper_bound"]


def test_invalid_limits(tmp_path):
    with pytest.raises(ValueError):
        load(tmp_path / "missing", max_bytes=-2)
    with pytest.raises(TypeError):
        to_dict(object())
    for options in ({"seed": None}, {"max_candidates": -1}, {"max_weight": -1}):
        with pytest.raises(ValueError):
            list(bicycle_candidates(trials=0, **options))
