"""Versioned, data-only interchange for executable binary code representations.

No pickle, dynamic imports, eval, or executable construction expressions are
accepted. Serialized check rows are supports, preserving generator locality.
"""
import json
from pathlib import Path

from qeclang.algebra import BinaryMatrix
from qeclang.codes import CSSCode, Pauli, StabilizerCode, SubsystemCode

SCHEMA = "qeclang.code.v1"


def _support(word):
    return [i for i in range(word.bit_length()) if (word >> i) & 1]


def _pauli_record(p):
    return {"x": _support(p.x), "z": _support(p.z), "phase": p.phase}


def to_dict(code):
    if not isinstance(code, (CSSCode, StabilizerCode, SubsystemCode)):
        raise TypeError("unsupported code representation")
    data = {"schema": SCHEMA, "n": code.n, "name": code.name}
    if isinstance(code, CSSCode):
        data.update(kind="css", hx=[_support(v) for v in code.hx.rows],
                    hz=[_support(v) for v in code.hz.rows])
    elif isinstance(code, (StabilizerCode, SubsystemCode)):
        data.update(kind="subsystem" if isinstance(code, SubsystemCode) else "stabilizer",
                    generators=[_pauli_record(p) for p in code.generators])
        if isinstance(code, SubsystemCode) and code.sector is not None:
            data["sector"] = [_pauli_record(p) for p in code.sector]
    else:
        raise TypeError("unsupported code representation")
    return data


def _integer(value, name, lower=0, upper=None):
    if type(value) is not int or value < lower or (upper is not None and value > upper):
        raise ValueError(f"invalid {name}")
    return value


def from_dict(data, *, max_qubits=100_000, max_checks=100_000, max_bits=100_000_000):
    for key, value in (("max_qubits", max_qubits), ("max_checks", max_checks),
                       ("max_bits", max_bits)):
        _integer(value, key)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("unsupported code schema")
    n = _integer(data.get("n"), "n", upper=max_qubits)
    name = data.get("name", "")
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    kind = data.get("kind")
    fields = {"schema", "n", "name", "kind"}
    fields |= {"hx", "hz"} if kind == "css" else {"generators"}
    if kind == "subsystem":
        fields.add("sector")
    if data.keys() - fields:
        raise ValueError("unknown code fields")

    def word(support):
        if not isinstance(support, list) or len(support) > n:
            raise ValueError("support must be a list of distinct site indices")
        result = 0
        for i in support:
            _integer(i, "site index", upper=n-1)
            if (result >> i) & 1:
                raise ValueError("duplicate site index")
            result |= 1 << i
        return result

    def rows(key):
        value = data.get(key)
        if not isinstance(value, list) or len(value) > max_checks:
            raise ValueError("invalid or oversized check list")
        width = n if kind == "css" else 2*n
        if len(value) * width > max_bits:
            raise ValueError("check matrix exceeds max_bits")
        return value

    if kind == "css":
        return CSSCode(BinaryMatrix(tuple(map(word, rows("hx"))), n),
                       BinaryMatrix(tuple(map(word, rows("hz"))), n), name=name)
    if kind not in ("stabilizer", "subsystem"):
        raise ValueError("unsupported code kind")
    def pauli(p):
        if not isinstance(p, dict) or set(p) != {"x", "z", "phase"}:
            raise ValueError("invalid Pauli record")
        return Pauli(word(p["x"]), word(p["z"]),
                     _integer(p["phase"], "phase", upper=3))
    generators = tuple(map(pauli, rows("generators")))
    if kind == "subsystem":
        sector = tuple(map(pauli, rows("sector"))) if "sector" in data else None
        return SubsystemCode(n, generators, name=name, sector=sector)
    return StabilizerCode(n, generators, name=name)


def load(path, *, max_bytes=10_000_000, **limits):
    _integer(max_bytes, "max_bytes")
    with Path(path).open("rb") as stream:
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError("code file exceeds max_bytes")
    return from_dict(json.loads(content), **limits)


def dump(code, path):
    Path(path).write_text(json.dumps(to_dict(code), indent=2) + "\n", encoding="utf-8")
