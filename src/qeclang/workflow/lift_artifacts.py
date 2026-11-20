"""Bounded structured-construction and reported-parameter artifact payloads.

Import reconstructs ring seeds but never expands checks or runs rank/proof
algorithms. Parameter dictionaries remain unverified claims. Workflow integers
are limited to 4096 bits before decimal JSON serialization.
"""
import json
import re

from qeclang.constructions import records
from qeclang.constructions.lifted import LiftedProduct
from .json_input import JSONLimits, canonical_json, validate_json_data

PARAMETERS_SCHEMA = "qeclang.workflow.parameters.v1"
PARAMETER_FIELDS = frozenset(("construction_fingerprint", "n", "status", "method", "k",
                              "rank_x", "rank_z", "steps", "reason", "x_evidence", "z_evidence"))
PARAMETER_BACKENDS = ("crt", "polynomial", "abelian", "mixed-abelian", "packed")
SERIALIZATION_LIMITS = JSONLimits(max_integer_bits=4096)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class LiftArtifactLimitExceeded(ValueError):
    """A structured artifact's seed, site, or parameter-specific cap was exceeded."""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _bounded(value, limit, name):
    _integer(value, name)
    if value > limit:
        raise LiftArtifactLimitExceeded(f"{name} exceeds its configured limit")
    return value


def lift_payload(value):
    """Use the existing physical-order-aware construction schema unchanged."""
    if not isinstance(value, LiftedProduct):
        raise TypeError("lift artifact requires LiftedProduct")
    return records.to_dict(value)


def _parameter_record(value):
    if type(value) is not dict or set(value) != PARAMETER_FIELDS:
        raise ValueError("parameters require every StructuredParameters field")
    source = value["construction_fingerprint"]
    if type(source) is not str or _DIGEST.fullmatch(source) is None:
        raise ValueError("parameters require a lowercase construction fingerprint")
    n = _integer(value["n"], "parameter n")
    if value["method"] not in PARAMETER_BACKENDS:
        raise ValueError("unsupported structured parameter backend")
    if value["status"] not in ("exact", "unsupported", "budget_exhausted"):
        raise ValueError("unsupported structured parameter status")
    if value["steps"] is not None:
        _integer(value["steps"], "parameter steps")
    if value["reason"] is not None and type(value["reason"]) is not str:
        raise ValueError("parameter reason must be a string or null")
    if value["status"] != "exact":
        if any(value[key] is not None for key in ("k", "rank_x", "rank_z", "x_evidence", "z_evidence")):
            raise ValueError("refused parameter analysis cannot report partial ranks or proofs")
        return
    k, rx, rz = (_integer(value[key], key) for key in ("k", "rank_x", "rank_z"))
    if k + rx + rz != n:
        raise ValueError("reported n, k, and CSS ranks are inconsistent")
    for key, rank in (("x_evidence", rx), ("z_evidence", rz)):
        evidence = value[key]
        if evidence is None:
            continue
        if type(evidence) is not dict:
            raise ValueError("rank evidence must be an object or null")
        if not {"shape", "rank", "steps", "method", "evidence"} <= evidence.keys():
            raise ValueError("rank evidence lacks its common structural fields")
        shape = evidence["shape"]
        if type(shape) is not list or len(shape) != 2:
            raise ValueError("rank evidence shape requires two dimensions")
        for dimension in shape:
            _integer(dimension, "rank evidence dimension")
        if _integer(evidence["rank"], "component rank") != rank:
            raise ValueError("rank evidence disagrees with reported sector rank")
        _integer(evidence["steps"], "rank evidence steps")
        if type(evidence["method"]) is not str or type(evidence["evidence"]) is not str:
            raise ValueError("rank evidence descriptions must be strings")


def parameter_payload(value, *, limits=SERIALIZATION_LIMITS):
    """Tag a bounded, list-normalized asdict(StructuredParameters) dictionary.

    Structural consistency is checked; no factorization, rank, elementary trace,
    source relationship, or exactness claim is verified by this function.
    """
    validate_json_data(value, limits=limits)
    _parameter_record(value)
    return {"schema": PARAMETERS_SCHEMA, **value}


def construction_limits(limits):
    """Map workflow import policy to the standalone construction reader's caps."""
    return records.ConstructionLimits(
        max_bytes=limits.max_bytes, max_factors=limits.max_lift_factors,
        max_order=limits.max_lift_group_size, max_group_size=limits.max_lift_group_size,
        max_rows=limits.max_lift_seed_axis, max_columns=limits.max_lift_seed_axis,
        max_entries=limits.max_lift_seed_entries, max_terms=limits.max_lift_terms,
        max_exponent=limits.max_lift_group_size, max_sites=limits.max_lift_sites)


def _preflight_lift(payload, limits):
    """Distinguish resource refusal before the constructor creates any ring data."""
    if type(payload) is not dict:
        raise ValueError("lift payload must be a construction object")
    orders = payload.get("orders")
    if type(orders) is not list or not orders:
        raise ValueError("lift orders must be a nonempty array")
    _bounded(len(orders), limits.max_lift_factors, "lift factors")
    group_size = 1
    for order in orders:
        _integer(order, "cyclic order", 1)
        if group_size > limits.max_lift_group_size // order:
            raise LiftArtifactLimitExceeded("lift group size exceeds its configured limit")
        group_size *= order
    shapes, entries, terms = [], 0, 0
    for name in ("a", "b"):
        seed = payload.get(name)
        if type(seed) is not dict or set(seed) != {"shape", "entries"}:
            raise ValueError("invalid lift seed fields")
        shape, rows = seed["shape"], seed["entries"]
        if type(shape) is not list or len(shape) != 2:
            raise ValueError("lift seed shape requires two dimensions")
        m, n = (_bounded(d, limits.max_lift_seed_axis, "lift seed axis") for d in shape)
        entries += m*n
        _bounded(entries, limits.max_lift_seed_entries, "lift seed entries")
        if type(rows) is not list or len(rows) != m:
            raise ValueError("lift seed row count differs from its shape")
        for row in rows:
            if type(row) is not list or len(row) != n:
                raise ValueError("lift seed column count differs from its shape")
            for support in row:
                if type(support) is not list:
                    raise ValueError("lift monomial support must be an array")
                terms += len(support)
                _bounded(terms, limits.max_lift_terms, "lift monomial terms")
        shapes.append((m, n))
    (ma, na), (mb, nb) = shapes
    _bounded(group_size * max(na*nb + ma*mb, ma*nb, na*mb), limits.max_lift_sites,
             "lift expanded axes")


def import_lift_payload(payload, *, limits):
    """Reconstruct validated symbolic seeds under separate lift-size bounds.

    The enclosing artifact importer must first perform bounded JSON admission
    and check the digest. This helper performs no binary or sparse lowering.
    """
    _preflight_lift(payload, limits)
    return records.from_dict(payload, limits=construction_limits(limits))


def import_parameter_payload(payload, *, limits):
    """Return an untagged parameter dictionary after bounded structural admission."""
    parameter_limits = JSONLimits(max_bytes=limits.max_bytes,
        max_payload_items=min(limits.max_payload_items, limits.max_parameter_entries),
        max_payload_depth=limits.max_payload_depth, max_string_length=limits.max_string_length,
        max_integer_bits=limits.max_integer_bits)
    validate_json_data(payload, limits=parameter_limits)
    if type(payload) is not dict or set(payload) != PARAMETER_FIELDS | {"schema"}:
        raise ValueError("parameter artifact fields do not match the schema")
    if payload["schema"] != PARAMETERS_SCHEMA:
        raise ValueError("unsupported parameter artifact schema")
    value = {key: item for key, item in payload.items() if key != "schema"}
    _parameter_record(value)
    _bounded(value["n"], limits.max_lift_sites, "parameter physical size")
    return json.loads(canonical_json(value))


__all__ = ["PARAMETERS_SCHEMA", "PARAMETER_FIELDS", "PARAMETER_BACKENDS",
           "SERIALIZATION_LIMITS", "LiftArtifactLimitExceeded", "lift_payload",
           "parameter_payload", "construction_limits", "import_lift_payload", "import_parameter_payload"]
