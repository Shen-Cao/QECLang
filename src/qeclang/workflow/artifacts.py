"""Canonical descriptions of workflow values, without executing or certifying them.

Names are omitted from mathematical presentations. Generator order, redundant
rows, zero dimensions, and physical coordinate order remain significant.
Describing a result or certificate does not establish its mathematical claims;
the workflow runner records verification and assurance separately.
"""
from hashlib import sha256
import json
import re

from qeclang.algebra import BinaryMatrix
from qeclang.algebra.binary import set_bits
from qeclang.analysis.certificate_io import SCHEMA as CERTIFICATE_SCHEMA
from qeclang.codes import CSSCode, ChainComplex, SparseCSSCode
from qeclang.io import SCHEMA as CODE_SCHEMA, to_dict
from .json_input import JSONLimits, validate_json_data
from .lift_artifacts import PARAMETERS_SCHEMA, lift_payload, parameter_payload

ARTIFACT_SCHEMA = "qeclang.workflow.artifact.v1"
CHAIN_SCHEMA = "qeclang.workflow.chain.v1"
DISTANCE_SCHEMA = "qeclang.workflow.distance.v1"
KINDS = frozenset(("css", "chain", "sparse_css", "distance", "witness", "lift", "parameters"))
_SERIALIZATION_LIMITS = JSONLimits(max_integer_bits=4096)
_DISTANCE_FIELDS = frozenset(("source", "sector", "lower_bound", "upper_bound", "witness",
                              "searched_through", "candidates", "status"))
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_HEX_WORD = re.compile(r"0x(?:0|[1-9a-f][0-9a-f]*)\Z")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _source(value):
    if not isinstance(value, str) or _HEX_DIGEST.fullmatch(value) is None:
        raise ValueError("source must be a lowercase SHA-256 presentation fingerprint")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _matrix(matrix: BinaryMatrix):
    return {"shape": [matrix.nrows, matrix.ncols],
            "rows": [list(set_bits(row)) for row in matrix.rows]}


def _distance(value):
    if not isinstance(value, dict) or set(value) != _DISTANCE_FIELDS:
        raise ValueError("distance requires source, sector, and every DistanceResult field")
    _source(value["source"])
    if value["sector"] not in ("x", "z"):
        raise ValueError("workflow distance artifacts currently require CSS sector x or z")
    searched = _integer(value["searched_through"], "searched_through")
    candidates = _integer(value["candidates"], "candidates")
    status = value["status"]
    lower, upper, witness = (value[key] for key in ("lower_bound", "upper_bound", "witness"))
    if status == "no_logicals":
        if (lower, upper, witness, searched, candidates) != (None, None, None, 0, 0):
            raise ValueError("no_logicals cannot carry finite bounds or search evidence")
    elif status in ("exact", "weight_limit", "budget_exhausted"):
        _integer(lower, "lower_bound", 1)
        _integer(upper, "upper_bound", 1)
        _integer(witness, "witness", 1)
        if lower > upper or lower != searched + 1:
            raise ValueError("distance bounds disagree with completed weight layers")
        if witness.bit_count() != upper:
            raise ValueError("CSS upper bound disagrees with witness Hamming weight")
        if (status == "exact") != (lower == upper):
            raise ValueError("exact status requires equal bounds; stopped statuses require a gap")
    else:
        raise ValueError("unsupported distance status")
    return {"schema": DISTANCE_SCHEMA, **value}


def _witness(value):
    required = {"schema", "source", "kind", "claim", "certificate"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("witness must be an existing certificate record")
    if value["schema"] != CERTIFICATE_SCHEMA:
        raise ValueError("unsupported witness certificate schema")
    _source(value["source"])
    sectors = {"css_logical": ("x", "z"), "stabilizer_logical": ("pauli",)}
    if not isinstance(value["kind"], str) or value["kind"] not in sectors:
        raise ValueError("witness artifacts require a logical certificate")
    certificate, claim = value["certificate"], value["claim"]
    if (not isinstance(certificate, dict)
            or set(certificate) != {"sector", "operator", "dual", "weight"}):
        raise ValueError("invalid logical witness fields")
    if certificate["sector"] not in sectors[value["kind"]]:
        raise ValueError("logical witness sector disagrees with certificate kind")
    weight = _integer(certificate["weight"], "witness weight", 1)
    for key in ("operator", "dual"):
        if not isinstance(certificate[key], str) or _HEX_WORD.fullmatch(certificate[key]) is None:
            raise ValueError("witness words must use canonical lowercase hexadecimal")
    if (not isinstance(claim, dict) or set(claim) != {"distance_upper_bound", "sector"}
            or type(claim["distance_upper_bound"]) is not int
            or claim != {"distance_upper_bound": weight, "sector": certificate["sector"]}):
        raise ValueError("logical witness claim disagrees with certificate fields")
    return value


def describe(kind, value, *, limits=_SERIALIZATION_LIMITS):
    """Return detached JSON-compatible kind/digest/payload for a workflow value.

    The digest hashes canonical JSON containing ARTIFACT_SCHEMA, kind, and the
    normalized payload. It identifies a representation/result record, not a
    code-equivalence class, proof, authenticity signature, or resource policy.
    Code names are discarded so equal digests have equal canonical payloads.
    Caller execution limits must bound representation construction. Bounded JSON
    admission checks all payload integers before decimal conversion; this helper
    neither lowers sparse data nor performs rank, distance, or certificate checks.
    """
    if not isinstance(limits, JSONLimits):
        raise TypeError("artifact serialization limits must be JSONLimits")
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError("unsupported workflow artifact kind")
    if kind == "css":
        if not isinstance(value, CSSCode):
            raise TypeError("css artifact requires CSSCode")
        payload = to_dict(value)
        payload.pop("name")
    elif kind == "sparse_css":
        if not isinstance(value, SparseCSSCode):
            raise TypeError("sparse_css artifact requires SparseCSSCode")
        payload = {"schema": CODE_SCHEMA, "kind": "css", "n": value.n,
                   "hx": [list(row) for row in value.hx.rows],
                   "hz": [list(row) for row in value.hz.rows]}
    elif kind == "chain":
        if not isinstance(value, ChainComplex):
            raise TypeError("chain artifact requires ChainComplex")
        payload = {"schema": CHAIN_SCHEMA, "coefficient_field": "GF(2)",
                   "degree_order": [0, 1, 2], "matrix_action": "column_vectors",
                   "dimensions": list(value.dimensions),
                   "d1": _matrix(value.d1), "d2": _matrix(value.d2)}
    elif kind == "lift":
        payload = lift_payload(value)
    elif kind == "parameters":
        payload = parameter_payload(value, limits=limits)
    elif kind == "distance":
        validate_json_data(value, limits=limits)
        payload = _distance(value)
    else:
        validate_json_data(value, limits=limits)
        payload = _witness(value)
    validate_json_data(payload, limits=limits)
    # JSON round-trip detaches caller-owned nested data and normalizes containers.
    payload = json.loads(_canonical(payload))
    encoded = _canonical({"schema": ARTIFACT_SCHEMA, "kind": kind, "payload": payload})
    return {"kind": kind, "digest": sha256(encoded.encode("utf-8")).hexdigest(),
            "payload": payload}


__all__ = ["ARTIFACT_SCHEMA", "CHAIN_SCHEMA", "DISTANCE_SCHEMA", "PARAMETERS_SCHEMA", "KINDS", "describe"]
