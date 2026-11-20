"""Bounded artifact decoding, with mathematical admission but no claim certification.

CSS/chain/sparse constructors recheck commutation. Distance, witness, low-weight
evidence and combined claims receive structural validation only: source bindings,
logical operators, exclusions and exactness need a separate scientific checker.
"""
from dataclasses import dataclass, field, fields
from hashlib import sha256
import json
import re

from qeclang.algebra import BinaryMatrix, SparseBinaryMatrix
from qeclang.codes import CSSCode, ChainComplex, SparseCSSCode
from .artifacts import ARTIFACT_SCHEMA, CHAIN_SCHEMA, DISTANCE_SCHEMA, KINDS, describe
from .json_input import JSONLimitExceeded, JSONLimits, canonical_json, measure_json_data
from .lift_artifacts import (LiftArtifactLimitExceeded, import_lift_payload,
                             import_parameter_payload)
from .low_weight_payload import (LowWeightPayloadLimitExceeded, import_distance_claim_payload,
                                 import_exclusion_payload)


class ArtifactImportError(ValueError):
    """Malformed, noncanonical, digest-mismatched, or mathematically invalid artifact."""


class ImportLimitExceeded(ValueError):
    """An artifact admission, representation, or cumulative import limit was exceeded."""


def _integer(value, name, maximum=None):
    if type(value) is not int or value < 0:
        raise ArtifactImportError(f"{name} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ImportLimitExceeded(f"{name} exceeds its configured limit")
    return value


@dataclass(frozen=True)
class ImportLimits:
    max_bytes: int = 8_000_000
    max_axis: int = 1024
    max_matrix_bits: int = 8_000_000
    max_validation_work: int = 1_000_000
    max_payload_items: int = 1_000_000
    max_payload_depth: int = 32
    max_string_length: int = 8192
    max_integer_bits: int = 4096
    max_lift_sites: int = 100_000
    max_lift_seed_axis: int = 1024
    max_lift_seed_entries: int = 100_000
    max_lift_terms: int = 100_000
    max_lift_group_size: int = 100_000
    max_lift_factors: int = 16
    max_parameter_entries: int = 100_000
    max_low_weight_indices: int = 1_000_000

    def __post_init__(self):
        for item in fields(self):
            _integer(getattr(self, item.name), item.name)
        if self.max_payload_depth > 128 or self.max_integer_bits > 4096:
            raise ValueError("import depth/integer limits must not exceed 128/4096")


@dataclass
class ImportBudget:
    """Shared admitted-record bytes and reserved incidence-validation work.

    Counters include admitted bytes even when later digest/math checks fail.
    Axis/matrix ceilings are per artifact. Neither limit measures peak memory.
    """
    max_artifact_bytes: int = 8_000_000
    max_validation_work: int = 1_000_000
    used: dict[str, int] = field(default_factory=lambda: {"artifact_bytes": 0, "validation_work": 0}, init=False)

    def __post_init__(self):
        _integer(self.max_artifact_bytes, "max_artifact_bytes")
        _integer(self.max_validation_work, "max_validation_work")

    def remaining(self, resource):
        return getattr(self, "max_" + resource) - self.used[resource]

    def charge(self, resource, amount):
        _integer(amount, "import charge")
        if amount > self.remaining(resource):
            raise ImportLimitExceeded(f"cumulative import {resource} budget exhausted")
        self.used[resource] += amount


def _json_admission(data, limits, byte_limit):
    """Check caller-owned containers before serialization or runtime allocation."""
    admission = {item.name: getattr(limits, item.name) for item in fields(JSONLimits)}
    admission["max_bytes"] = byte_limit
    try:
        return measure_json_data(data, limits=JSONLimits(**admission))
    except JSONLimitExceeded as error:
        raise ImportLimitExceeded(str(error)) from error
    except (ValueError, TypeError, RecursionError) as error:
        raise ArtifactImportError(str(error)) from error


def _keys(data, names):
    if type(data) is not dict or set(data) != set(names):
        raise ArtifactImportError("artifact fields do not match the declared schema")


def _supports(rows, width, limits, expected_rows=None):
    if type(rows) is not list:
        raise ArtifactImportError("matrix rows must be arrays of supports")
    _integer(len(rows), "matrix row count", limits.max_axis)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ArtifactImportError("matrix shape disagrees with row count")
    for support in rows:
        if type(support) is not list or len(support) > width:
            raise ArtifactImportError("support is not a canonical array")
        previous = -1
        for site in support:
            _integer(site, "support site")
            if not previous < site < width:
                raise ArtifactImportError("support sites must be strictly increasing and in range")
            previous = site
    return rows


def _packed(rows, width):
    return BinaryMatrix(tuple(sum(1 << site for site in support) for support in rows), width)


def _validation_work(left, right, width, cap, *, chain=False):
    base = len(left) + (width if chain else len(right)) + sum(map(len, left)) + sum(map(len, right))
    if chain:
        base += len(right)
    if base > cap:
        raise ImportLimitExceeded("incidence validation budget exhausted")
    degrees = {}
    if not chain:
        for support in right:
            for site in support:
                degrees[site] = degrees.get(site, 0) + 1
    for support in left:
        for site in support:
            base += len(right[site]) if chain else degrees.get(site, 0)
            if base > cap:
                raise ImportLimitExceeded("incidence validation budget exhausted")
    return base


def _code_value(payload, kind, limits, budget):
    if kind == "chain":
        _keys(payload, ("schema", "coefficient_field", "degree_order", "matrix_action", "dimensions", "d1", "d2"))
        if (payload["schema"] != CHAIN_SCHEMA or payload["coefficient_field"] != "GF(2)"
                or payload["matrix_action"] != "column_vectors" or payload["degree_order"] != [0, 1, 2]
                or any(type(value) is not int for value in payload["degree_order"])):
            raise ArtifactImportError("unsupported chain conventions")
        dimensions = payload["dimensions"]
        if type(dimensions) is not list or len(dimensions) != 3:
            raise ArtifactImportError("chain dimensions require degrees 0, 1, 2")
        m, n, z = (_integer(v, "chain dimension", limits.max_axis) for v in dimensions)
        matrices = []
        for key, shape in (("d1", [m, n]), ("d2", [n, z])):
            block = payload[key]
            _keys(block, ("shape", "rows"))
            if block["shape"] != shape or any(type(v) is not int for v in block["shape"]):
                raise ArtifactImportError("boundary shape disagrees with chain dimensions")
            matrices.append(_supports(block["rows"], shape[1], limits, shape[0]))
        left, right = matrices
    else:
        _keys(payload, ("schema", "kind", "n", "hx", "hz"))
        if payload["schema"] != "qeclang.code.v1" or payload["kind"] != "css":
            raise ArtifactImportError("unsupported CSS payload schema")
        n = _integer(payload["n"], "qubit count", limits.max_axis)
        left, right = (_supports(payload[key], n, limits) for key in ("hx", "hz"))
        m, z = len(left), len(right)
    if kind != "sparse_css" and (m + z) * n + m * z > limits.max_matrix_bits:
        raise ImportLimitExceeded("packed matrices and validation product exceed max_matrix_bits")
    available = limits.max_validation_work if budget is None else min(limits.max_validation_work, budget.remaining("validation_work"))
    work = _validation_work(left, right, z if kind == "chain" else n, available, chain=kind == "chain")
    if budget is not None:
        budget.charge("validation_work", work)
    if kind == "sparse_css":
        return SparseCSSCode(SparseBinaryMatrix.from_supports(left, n),
                             SparseBinaryMatrix.from_supports(right, n), max_validation_work=work)
    if kind == "chain":
        return ChainComplex(_packed(left, n), _packed(right, z))
    return CSSCode(_packed(left, n), _packed(right, n))


def _claim_value(payload, kind, limits):
    if kind == "distance":
        expected = {"schema", "source", "sector", "lower_bound", "upper_bound", "witness",
                    "searched_through", "candidates", "status"}
        _keys(payload, expected)
        if payload["schema"] != DISTANCE_SCHEMA:
            raise ArtifactImportError("unsupported distance payload schema")
        for key in ("lower_bound", "upper_bound", "searched_through"):
            if payload[key] is not None:
                _integer(payload[key], key, limits.max_axis)
        if payload["witness"] is not None:
            word = _integer(payload["witness"], "distance witness")
            if word.bit_length() > limits.max_axis:
                raise ImportLimitExceeded("distance witness exceeds max_axis")
        return {key: value for key, value in payload.items() if key != "schema"}
    _keys(payload, ("schema", "source", "kind", "claim", "certificate"))
    certificate = payload["certificate"]
    _keys(certificate, ("sector", "operator", "dual", "weight"))
    _integer(certificate["weight"], "certificate weight", limits.max_axis)
    width = 2 * limits.max_axis if payload["kind"] == "stabilizer_logical" else limits.max_axis
    for key in ("operator", "dual"):
        word = certificate[key]
        if type(word) is not str or re.fullmatch(r"0x(?:0|[1-9a-f][0-9a-f]*)", word) is None:
            raise ArtifactImportError("certificate words must be canonical hexadecimal")
        if len(word) > max(1, (width + 3) // 4) + 2 or int(word, 16).bit_length() > width:
            raise ImportLimitExceeded("certificate word exceeds max_axis")
    return json.loads(canonical_json(payload))


def import_artifact(record, expected_kind, *, limits=ImportLimits(), budget=None):
    """Decode a canonical artifact; result/evidence claims are not certified here.

    Input is caller-owned JSON data. Packed payload caps include the composition
    product; sparse admission uses incidence-work limits without dense lowering.
    An optional ImportBudget accumulates bytes/work across records. Constructors
    verify only dialect mathematics; returned result dictionaries remain claims.
    """
    if not isinstance(limits, ImportLimits) or (budget is not None and not isinstance(budget, ImportBudget)):
        raise TypeError("limits/budget must be ImportLimits/ImportBudget")
    if type(expected_kind) is not str or expected_kind not in KINDS:
        raise ArtifactImportError("unsupported expected artifact kind")
    byte_limit = limits.max_bytes if budget is None else min(limits.max_bytes, budget.remaining("artifact_bytes"))
    size = _json_admission(record, limits, byte_limit)
    if budget is not None:
        budget.charge("artifact_bytes", size)
    _keys(record, ("kind", "digest", "payload"))
    if record["kind"] != expected_kind or type(record["digest"]) is not str:
        raise ArtifactImportError("artifact kind or digest type does not match")
    encoded = canonical_json({"schema": ARTIFACT_SCHEMA, "kind": expected_kind, "payload": record["payload"]})
    if record["digest"] != sha256(encoded.encode("utf-8")).hexdigest():
        raise ArtifactImportError("artifact digest mismatch")
    try:
        if expected_kind in ("css", "chain", "sparse_css"):
            value = _code_value(record["payload"], expected_kind, limits, budget)
        elif expected_kind == "lift":
            value = import_lift_payload(record["payload"], limits=limits)
        elif expected_kind == "parameters":
            value = import_parameter_payload(record["payload"], limits=limits)
        elif expected_kind == "exclusion":
            value = import_exclusion_payload(record["payload"], limits=limits)
        elif expected_kind == "distance_claim":
            value = import_distance_claim_payload(record["payload"], limits=limits)
        else:
            value = _claim_value(record["payload"], expected_kind, limits)
        json_limits = JSONLimits(**{item.name: getattr(limits, item.name) for item in fields(JSONLimits)})
        if canonical_json(describe(expected_kind, value, limits=json_limits)) != canonical_json(record):
            raise ArtifactImportError("artifact payload is not canonical")
    except ImportLimitExceeded:
        raise
    except (JSONLimitExceeded, LiftArtifactLimitExceeded, LowWeightPayloadLimitExceeded) as error:
        raise ImportLimitExceeded(str(error)) from error
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise ArtifactImportError(str(error)) from error
    return value


__all__ = ["ArtifactImportError", "ImportLimitExceeded", "ImportLimits", "ImportBudget", "import_artifact"]
