"""Bounded syntax for low-weight evidence and combined distance-claim artifacts.

Admission does not check source bindings, membership identities, obligation
completeness, parity, record_id recomputation, or distance claims. Those belong
exclusively to the scientific checker. In particular, excluded weights alone
never establish existence of a protected logical operator.
"""
import json
import re

from .json_input import JSONLimits, canonical_json, validate_json_data

EXCLUSION_SCHEMA = "qeclang.low_weight.v1"
DISTANCE_CLAIM_SCHEMA = "qeclang.workflow.distance_claim.v1"
_SERIALIZATION_LIMITS = JSONLimits(max_integer_bits=4096)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_WORD = re.compile(r"0x(?:0|[1-9a-f][0-9a-f]*)\Z")
_FIELDS = {"schema", "record_id", "source", "sector", "through", "status", "steps",
           "claim", "memberships", "witness", "reason"}


class LowWeightPayloadLimitExceeded(ValueError):
    """Low-weight evidence exceeded its configured input-size bounds."""


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("low-weight artifact fields do not match their schema")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _source(value):
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError("source/record IDs must be lowercase SHA-256 digests")


def _sector(value):
    if type(value) is not str or value not in ("x", "z"):
        raise ValueError("sector must be x or z")


def _cap(value, maximum, name):
    if value > maximum:
        raise LowWeightPayloadLimitExceeded(f"{name} exceeds its configured limit")


def _witness_shape(value, source, sector, bound):
    _keys(value, ("schema", "source", "kind", "claim", "certificate"))
    if (value["schema"] != "qeclang.certificate.v1" or value["kind"] != "css_logical"
            or value["source"] != source):
        raise ValueError("found witness must use the same CSS presentation")
    certificate, claim = value["certificate"], value["claim"]
    _keys(certificate, ("sector", "operator", "dual", "weight"))
    _keys(claim, ("sector", "distance_upper_bound"))
    if certificate["sector"] != sector or claim["sector"] != sector:
        raise ValueError("found witness sector differs from its enclosing record")
    if (_integer(certificate["weight"], "witness weight", 1) != bound
            or _integer(claim["distance_upper_bound"], "witness upper bound", 1) != bound):
        raise ValueError("found witness bound differs from its enclosing claim")
    for key in ("operator", "dual"):
        if type(certificate[key]) is not str or _WORD.fullmatch(certificate[key]) is None:
            raise ValueError("found witness words must be canonical hexadecimal")


def _exclusion_shape(value, *, max_axis=None, max_indices=None):
    _keys(value, _FIELDS)
    if value["schema"] != EXCLUSION_SCHEMA:
        raise ValueError("unsupported low-weight evidence schema")
    _source(value["source"])
    _source(value["record_id"])
    _sector(value["sector"])
    if type(value["through"]) is not int or value["through"] not in (1, 2):
        raise ValueError("through must be exactly one or two")
    _integer(value["steps"], "steps")
    status, claim = value["status"], value["claim"]
    if status == "budget_exhausted":
        if (claim is not None or value["memberships"] is not None or value["witness"] is not None
                or type(value["reason"]) is not str or not value["reason"]):
            raise ValueError("budget exhaustion cannot carry a claim or partial evidence")
        return
    if status not in ("excluded", "found") or value["reason"] is not None:
        raise ValueError("unsupported low-weight status or success reason")
    if status == "excluded":
        _keys(claim, ("predicate", "through"))
        if (claim["predicate"] != "nontrivial_weight_exclusion"
                or type(claim["through"]) is not int or claim["through"] != value["through"]):
            raise ValueError("exclusion claim differs from its requested weight range")
        rows = value["memberships"]
        if type(rows) is not list or value["witness"] is not None:
            raise ValueError("excluded record requires membership lists and no witness")
        if max_axis is not None:
            _cap(len(rows), max_axis, "membership row count")
        indices = 0
        for support in rows:
            if type(support) is not list:
                raise ValueError("membership supports must be lists")
            indices += len(support)
            if max_indices is not None:
                _cap(indices, max_indices, "membership indices")
            previous = -1
            for index in support:
                _integer(index, "generator index")
                if index <= previous:
                    raise ValueError("membership supports must be strictly increasing")
                if max_axis is not None:
                    _cap(index + 1, max_axis, "generator index width")
                previous = index
    else:
        _keys(claim, ("predicate", "upper_bound"))
        bound = _integer(claim["upper_bound"], "upper_bound", 1)
        if (claim["predicate"] != "distance_upper_bound" or bound > value["through"]
                or value["memberships"] is not None):
            raise ValueError("found claim must be a witness within the requested weight range")
        _witness_shape(value["witness"], value["source"], value["sector"], bound)
        if max_axis is not None:
            _cap(bound, max_axis, "witness weight")
            for key in ("operator", "dual"):
                word = value["witness"]["certificate"][key]
                _cap(len(word), max(1, (max_axis+3)//4) + 2, "witness word length")
                _cap(int(word, 16).bit_length(), max_axis, "witness word width")


def _distance_shape(value, *, max_axis=None):
    _keys(value, ("schema", "source", "sector", "lower_bound", "upper_bound", "status"))
    if value["schema"] != DISTANCE_CLAIM_SCHEMA:
        raise ValueError("unsupported combined distance-claim schema")
    _source(value["source"])
    if type(value["sector"]) is not str or value["sector"] not in ("x", "z", "pauli"):
        raise ValueError("distance-claim sector must be x, z, or pauli")
    lower = _integer(value["lower_bound"], "lower_bound", 1)
    upper = _integer(value["upper_bound"], "upper_bound", 1)
    if (lower > upper or value["status"] not in ("exact", "bounded")
            or (value["status"] == "exact") != (lower == upper)):
        raise ValueError("distance-claim interval and status disagree")
    if max_axis is not None:
        _cap(upper, max_axis, "distance upper bound")


def exclusion_payload(value, *, limits=_SERIALIZATION_LIMITS):
    validate_json_data(value, limits=limits)
    _exclusion_shape(value)
    return value


def distance_claim_payload(value, *, limits=_SERIALIZATION_LIMITS):
    validate_json_data(value, limits=limits)
    _distance_shape(value)
    return value


def import_exclusion_payload(value, *, limits):
    """Import already JSON-admitted exclusion syntax without checking its mathematics."""
    _exclusion_shape(value, max_axis=limits.max_axis, max_indices=limits.max_low_weight_indices)
    return json.loads(canonical_json(value))


def import_distance_claim_payload(value, *, limits):
    """Import already JSON-admitted interval syntax without inferring assurance."""
    _distance_shape(value, max_axis=limits.max_axis)
    return json.loads(canonical_json(value))


__all__ = ["EXCLUSION_SCHEMA", "DISTANCE_CLAIM_SCHEMA", "LowWeightPayloadLimitExceeded",
           "exclusion_payload", "distance_claim_payload", "import_exclusion_payload",
           "import_distance_claim_payload"]
