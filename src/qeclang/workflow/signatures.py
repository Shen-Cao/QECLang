"""Closed operation signatures and static parameter/type refinements.

This module checks finite data and nominal provenance only. Code admission,
row identities, witness parity and exclusion mathematics remain runtime duties.
"""
from dataclasses import dataclass
import re
from types import MappingProxyType

from .json_input import integer as _integer
from .lift_types import construction_limits, validate_lift_literal, validate_parameter_request

_SIGNATURES = MappingProxyType({
    "lift.literal": ((), "lift"),
    "lift.parameters": (("lift",), "parameters"),
    "lift.css": (("lift",), "css"),
    "css.literal": ((), "css"),
    "css.chain": (("css",), "chain"),
    "chain.css": (("chain",), "css"),
    "css.sparse": (("css",), "sparse_css"),
    "sparse.css": (("sparse_css",), "css"),
    "css.permute": (("css",), "css"),
    "css.distance": (("css",), "distance"),
    "css.witness": (("css", "distance"), "witness"),
    "css.transport": (("css", "css", "witness"), "witness"),
    "css.import_witness": (("css",), "witness"),
    "css.rebase_witness": (("css", "css", "witness"), "witness"),
    "css.exclude": (("css",), "exclusion"),
    "css.combine_distance": (("css", "exclusion", "witness"), "distance_claim"),
    "css.full_distance": (("css", "distance_claim", "distance_claim"), "distance_claim"),
})
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_WORD = re.compile(r"0x(?:0|[1-9a-f][0-9a-f]*)\Z")


def operation_signature(op):
    if type(op) is not str or op not in _SIGNATURES:
        raise ValueError("unsupported workflow operation")
    return _SIGNATURES[op]


def require_fields(value, required, optional=()):
    if type(value) is not dict or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError(f"expected fields {tuple(required)} with optional {tuple(optional)}")


def identifier(value, limits):
    if type(value) is not str or len(value) > limits.max_id_length or _ID.fullmatch(value) is None:
        raise ValueError("node references require bounded ASCII identifiers")
    return value


def _literal(data, limits):
    require_fields(data, ("schema", "kind", "n", "hx", "hz"), ("name",))
    if data["schema"] != "qeclang.code.v1" or data["kind"] != "css":
        raise ValueError("css.literal requires a qeclang.code.v1 CSS record")
    n = _integer(data["n"], "literal n", maximum=limits.max_qubits)
    if "name" in data and type(data["name"]) is not str:
        raise ValueError("literal name must be a string")
    checks = 0
    for key in ("hx", "hz"):
        rows = data[key]
        if type(rows) is not list:
            raise ValueError("literal checks must be lists of supports")
        checks += len(rows)
        if checks > limits.max_checks:
            raise ValueError("literal check count exceeds max_checks")
        for support in rows:
            if type(support) is not list or len(support) > n:
                raise ValueError("literal support must be a list of distinct sites")
            seen = set()
            for site in support:
                _integer(site, "literal site")
                if site >= n or site in seen:
                    raise ValueError("literal support contains an out-of-range or repeated site")
                seen.add(site)
    return n


def _permutation(value, n):
    if type(value) is not list or len(value) != n:
        raise ValueError("permutation must contain every physical site")
    seen = set()
    for site in value:
        _integer(site, "permutation site")
        if site >= n or site in seen:
            raise ValueError("permutation must be a bijection")
        seen.add(site)


def _imported_witness(params, n, limits):
    """Check certificate/origin syntax without source hashing or Pauli parity."""
    require_fields(params, ("certificate",), ("origin",))
    record = params["certificate"]
    require_fields(record, ("schema", "source", "kind", "claim", "certificate"))
    if record["schema"] != "qeclang.certificate.v1" or record["kind"] != "css_logical":
        raise ValueError("imported witness must be a qeclang.certificate.v1 CSS logical record")
    if type(record["source"]) is not str or _DIGEST.fullmatch(record["source"]) is None:
        raise ValueError("certificate source must be a lowercase SHA-256 digest")
    certificate, claim = record["certificate"], record["claim"]
    require_fields(certificate, ("sector", "operator", "dual", "weight"))
    require_fields(claim, ("distance_upper_bound", "sector"))
    sector = certificate["sector"]
    if sector not in ("x", "z") or claim["sector"] != sector:
        raise ValueError("certificate and claim must have the same CSS sector")
    weight = _integer(certificate["weight"], "certificate weight", maximum=n)
    upper = _integer(claim["distance_upper_bound"], "distance_upper_bound", maximum=n)
    if weight == 0 or upper != weight:
        raise ValueError("certificate and claim must have the same positive weight")
    for key in ("operator", "dual"):
        value = certificate[key]
        if (type(value) is not str or len(value) > max(1, (n+3)//4) + 2
                or _WORD.fullmatch(value) is None or int(value, 16).bit_length() > n):
            raise ValueError("certificate words must be canonical hexadecimal within physical width")
    if "origin" in params:
        origin = params["origin"]
        require_fields(origin, ("bundle_id", "node"))
        if type(origin["bundle_id"]) is not str or _DIGEST.fullmatch(origin["bundle_id"]) is None:
            raise ValueError("origin bundle_id must be a lowercase SHA-256 digest")
        identifier(origin["node"], limits)
    return sector


@dataclass(frozen=True)
class NodeType:
    result_type: str
    n: int
    mx: int
    mz: int
    code_id: str | None
    sector: str | None


def derive_node_type(op, args, params, limits):
    """Refine a closed operation's type from prior typed nodes and finite params."""
    expected, result_type = operation_signature(op)
    if tuple(arg.result_type for arg in args) != expected:
        raise ValueError(f"input types do not match {op}: expected {expected}")
    code_id = sector = None
    n = args[0].n if args else 0
    mx = args[0].mx if args else 0
    mz = args[0].mz if args else 0
    if op == "lift.literal":
        require_fields(params, ("construction",))
        shape = validate_lift_literal(params["construction"], limits=construction_limits(limits))
        n, mx, mz = shape.n, shape.x_checks, shape.z_checks
    elif op == "lift.parameters":
        validate_parameter_request(params)
        code_id = args[0].id
    elif op == "css.literal":
        require_fields(params, ("code",))
        n = _literal(params["code"], limits)
        mx, mz = len(params["code"]["hx"]), len(params["code"]["hz"])
    elif op in ("css.permute", "css.transport"):
        require_fields(params, ("permutation",))
        _permutation(params["permutation"], n)
        if op == "css.transport":
            if args[1].n != n or args[2].code_id != args[0].id:
                raise ValueError("transport requires equal dimensions and a witness bound to its source CSS node")
            code_id, sector = args[1].id, args[2].sector
            mx, mz = args[1].mx, args[1].mz
    elif op == "css.import_witness":
        sector = _imported_witness(params, n, limits)
        code_id = args[0].id
    elif op == "css.rebase_witness":
        from .row_equivalence import validate_map_supports
        target, witness = args[1], args[2]
        if target.n != n or witness.code_id != args[0].id:
            raise ValueError("rebasing requires equal physical widths and a source-bound witness")
        validate_map_supports(params, (mx, mz), (target.mx, target.mz),
                              max_rows=limits.max_payload_items, max_indices=limits.max_map_entries)
        code_id, sector, mx, mz = args[1].id, witness.sector, target.mx, target.mz
    elif op == "css.exclude":
        require_fields(params, ("sector", "through", "max_steps"))
        if params["sector"] not in ("x", "z"):
            raise ValueError("exclusion sector must be x or z")
        if type(params["through"]) is not int or params["through"] not in (1, 2):
            raise ValueError("through must be exactly one or two")
        _integer(params["max_steps"], "max_steps")
        code_id, sector = args[0].id, params["sector"]
    elif op == "css.combine_distance":
        require_fields(params, ())
        exclusion, witness = args[1], args[2]
        if exclusion.code_id != args[0].id or witness.code_id != args[0].id:
            raise ValueError("exclusion and witness must be bound to the supplied CSS node")
        if exclusion.sector != witness.sector:
            raise ValueError("exclusion and witness sectors must agree")
        code_id, sector = args[0].id, exclusion.sector
    elif op == "css.full_distance":
        require_fields(params, ())
        first, second = args[1], args[2]
        if first.code_id != args[0].id or second.code_id != args[0].id:
            raise ValueError("sector distance claims must be bound to the supplied CSS node")
        if {first.sector, second.sector} != {"x", "z"}:
            raise ValueError("full distance requires one x and one z claim")
        code_id, sector = args[0].id, "pauli"
    elif op == "css.distance":
        require_fields(params, ("sector", "max_weight", "max_candidates"))
        if params["sector"] not in ("x", "z"):
            raise ValueError("distance sector must be x or z")
        _integer(params["max_weight"], "max_weight")
        _integer(params["max_candidates"], "max_candidates")
        code_id, sector = args[0].id, params["sector"]
    elif op == "css.witness":
        require_fields(params, ())
        if args[1].code_id != args[0].id:
            raise ValueError("distance evidence is bound to a different CSS node")
        code_id, sector = args[0].id, args[1].sector
    else:
        require_fields(params, ())
    return NodeType(result_type, n, mx, mz, code_id, sector)


__all__ = ["NodeType", "operation_signature", "derive_node_type", "require_fields", "identifier"]
