"""Validated, immutable, data-only programs for the closed QEC workflow dialect.

Parsing performs schema, shape, nominal type, provenance and resource checks;
it does not construct codes, execute algorithms, import plugins, or write state.
Mathematical admission checks remain the executor/core's responsibility.
"""
from dataclasses import dataclass, fields
from hashlib import sha256
import json
import re
from types import MappingProxyType

from .json_input import (JSONLimits, canonical_json as _canonical, load_json,
                         loads_json, validate_json_data)
from .lift_types import construction_limits, validate_lift_literal, validate_parameter_request

SCHEMA = "qeclang.program.v1"
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
})
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_WORD = re.compile(r"0x(?:0|[1-9a-f][0-9a-f]*)\Z")


def _integer(value, name, *, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be a nonnegative integer within its limit")
    return value


@dataclass(frozen=True)
class ProgramLimits:
    """Parsing/count bounds, excluding caller data and Python object overhead.

    max_bytes bounds both supplied UTF-8 and canonical JSON; max_payload_items
    counts containers, scalars and object keys. max_depth is dependency depth;
    max_payload_depth bounds JSON container nesting. max_edges counts all input
    references; max_fanout counts input-reference uses of each prior node.
    Seed entry/term bounds are shared by both seeds within each lifted literal.
    max_symbolic_sites bounds its computed expanded axes without allocating them;
    it is separate from CSS-literal max_qubits and executor packed-axis limits.
    max_map_entries bounds the sum of coefficient indices in four rebase maps;
    empty map rows remain subject to the shared JSON payload-item budget.
    """
    max_bytes: int = 10_000_000
    max_nodes: int = 1_000
    max_depth: int = 64
    max_payload_items: int = 1_000_000
    max_payload_depth: int = 64
    max_string_length: int = 8_192
    max_integer_bits: int = 64
    max_id_length: int = 64
    max_inputs: int = 8
    max_outputs: int = 64
    max_edges: int = 4_000
    max_fanout: int = 1_000
    max_qubits: int = 100_000
    max_checks: int = 100_000
    max_seed_axis: int = 10_000
    max_seed_entries: int = 100_000
    max_seed_terms: int = 1_000_000
    max_group_factors: int = 16
    max_cyclic_order: int = 1_000_000
    max_group_size: int = 1_000_000
    max_symbolic_sites: int = 1_000_000_000
    max_map_entries: int = 1_000_000

    def __post_init__(self):
        for item in fields(self):
            _integer(getattr(self, item.name), item.name)
        if self.max_payload_depth > 128:
            raise ValueError("max_payload_depth must not exceed 128")
        if self.max_integer_bits > 4_096:
            raise ValueError("max_integer_bits must not exceed 4096")


def _check_limits(limits):
    if not isinstance(limits, ProgramLimits):
        raise TypeError("limits must be ProgramLimits")


def _admission(limits):
    return JSONLimits(**{item.name: getattr(limits, item.name) for item in fields(JSONLimits)})


def _keys(value, required, optional=()):
    if type(value) is not dict or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError(f"expected fields {tuple(required)} with optional {tuple(optional)}")


def _identifier(value, limits):
    if type(value) is not str or len(value) > limits.max_id_length or _ID.fullmatch(value) is None:
        raise ValueError("node references require bounded ASCII identifiers")
    return value


def _literal(data, limits):
    _keys(data, ("schema", "kind", "n", "hx", "hz"), ("name",))
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
    _keys(params, ("certificate",), ("origin",))
    record = params["certificate"]
    _keys(record, ("schema", "source", "kind", "claim", "certificate"))
    if record["schema"] != "qeclang.certificate.v1" or record["kind"] != "css_logical":
        raise ValueError("imported witness must be a qeclang.certificate.v1 CSS logical record")
    if type(record["source"]) is not str or _DIGEST.fullmatch(record["source"]) is None:
        raise ValueError("certificate source must be a lowercase SHA-256 digest")
    certificate, claim = record["certificate"], record["claim"]
    _keys(certificate, ("sector", "operator", "dual", "weight"))
    _keys(claim, ("distance_upper_bound", "sector"))
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
        _keys(origin, ("bundle_id", "node"))
        if type(origin["bundle_id"]) is not str or _DIGEST.fullmatch(origin["bundle_id"]) is None:
            raise ValueError("origin bundle_id must be a lowercase SHA-256 digest")
        _identifier(origin["node"], limits)
    return sector


@dataclass(frozen=True, init=False)
class ProgramNode:
    id: str
    op: str
    inputs: tuple[str, ...]
    result_type: str
    n: int
    mx: int
    mz: int
    depth: int
    code_id: str | None
    sector: str | None
    _params_json: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("ProgramNode is constructed by Program.from_dict")

    @property
    def params(self):
        return json.loads(self._params_json)

    def to_dict(self):
        return {"id": self.id, "op": self.op, "inputs": list(self.inputs), "params": self.params}


def _node(data, prior, limits):
    _keys(data, ("id", "op", "inputs", "params"))
    name = _identifier(data["id"], limits)
    if name in prior:
        raise ValueError(f"duplicate node id: {name}")
    op, refs, params = data["op"], data["inputs"], data["params"]
    if type(op) is not str or op not in _SIGNATURES:
        raise ValueError("unsupported workflow operation")
    if type(refs) is not list or len(refs) > limits.max_inputs:
        raise ValueError("node inputs exceed max_inputs or are not a list")
    inputs = tuple(_identifier(ref, limits) for ref in refs)
    if any(ref not in prior for ref in inputs):
        raise ValueError("node inputs must reference prior nodes; forward references and cycles are forbidden")
    expected, result_type = _SIGNATURES[op]
    if tuple(prior[ref].result_type for ref in inputs) != expected:
        raise ValueError(f"input types do not match {op}: expected {expected}")
    depth = 1 + max((prior[ref].depth for ref in inputs), default=0)
    if depth > limits.max_depth:
        raise ValueError("program dependency depth exceeds max_depth")
    code_id = sector = None
    n = prior[inputs[0]].n if inputs else 0
    mx = prior[inputs[0]].mx if inputs else 0
    mz = prior[inputs[0]].mz if inputs else 0
    if op == "lift.literal":
        _keys(params, ("construction",))
        shape = validate_lift_literal(params["construction"], limits=construction_limits(limits))
        n, mx, mz = shape.n, shape.x_checks, shape.z_checks
    elif op == "lift.parameters":
        validate_parameter_request(params)
        code_id = inputs[0]
    elif op == "css.literal":
        _keys(params, ("code",))
        n = _literal(params["code"], limits)
        mx, mz = len(params["code"]["hx"]), len(params["code"]["hz"])
    elif op in ("css.permute", "css.transport"):
        _keys(params, ("permutation",))
        _permutation(params["permutation"], n)
        if op == "css.transport":
            if prior[inputs[1]].n != n or prior[inputs[2]].code_id != inputs[0]:
                raise ValueError("transport requires equal dimensions and a witness bound to its source CSS node")
            code_id, sector = inputs[1], prior[inputs[2]].sector
            mx, mz = prior[inputs[1]].mx, prior[inputs[1]].mz
    elif op == "css.import_witness":
        sector = _imported_witness(params, n, limits)
        code_id = inputs[0]
    elif op == "css.rebase_witness":
        from .row_equivalence import validate_map_supports
        target, witness = prior[inputs[1]], prior[inputs[2]]
        if target.n != n or witness.code_id != inputs[0]:
            raise ValueError("rebasing requires equal physical widths and a source-bound witness")
        validate_map_supports(params, (mx, mz), (target.mx, target.mz),
                              max_rows=limits.max_payload_items, max_indices=limits.max_map_entries)
        code_id, sector, mx, mz = inputs[1], witness.sector, target.mx, target.mz
    elif op == "css.distance":
        _keys(params, ("sector", "max_weight", "max_candidates"))
        if params["sector"] not in ("x", "z"):
            raise ValueError("distance sector must be x or z")
        _integer(params["max_weight"], "max_weight")
        _integer(params["max_candidates"], "max_candidates")
        code_id, sector = inputs[0], params["sector"]
    elif op == "css.witness":
        _keys(params, ())
        if prior[inputs[1]].code_id != inputs[0]:
            raise ValueError("distance evidence is bound to a different CSS node")
        code_id, sector = inputs[0], prior[inputs[1]].sector
    else:
        _keys(params, ())
    node = object.__new__(ProgramNode)
    values = {"id": name, "op": op, "inputs": inputs, "result_type": result_type,
              "n": n, "mx": mx, "mz": mz, "depth": depth, "code_id": code_id, "sector": sector,
              "_params_json": _canonical(params)}
    for key, value in values.items():
        object.__setattr__(node, key, value)
    return node


@dataclass(frozen=True, init=False)
class Program:
    """A topologically ordered, closed-operation DAG with nominal evidence types.

    Parameter results bind to their originating lift node ID through code_id.
    Distance/witness values bind to their originating CSS node ID. Even an
    identity conversion creates a different nominal CSS value; reuse requires
    explicit witness transport or rebasing. Check counts mx/mz are static metadata
    for source/target row-map dimensions. Imported certificate source/parity and
    origin claims are not authenticated by parsing. Parsing does not prove literal commutation or
    the source/target permutation relation; the executor must check those.
    """
    nodes: tuple[ProgramNode, ...]
    outputs: tuple[str, ...]
    limits: ProgramLimits
    _json: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("construct a validated Program using from_dict, loads, or load")

    @classmethod
    def from_dict(cls, data, *, limits=ProgramLimits()):
        _check_limits(limits)
        validate_json_data(data, limits=_admission(limits))
        _keys(data, ("schema", "nodes", "outputs"))
        if data["schema"] != SCHEMA:
            raise ValueError("unsupported program schema")
        rows, outputs = data["nodes"], data["outputs"]
        if type(rows) is not list or len(rows) > limits.max_nodes:
            raise ValueError("program exceeds max_nodes or nodes is not a list")
        if type(outputs) is not list or len(outputs) > limits.max_outputs:
            raise ValueError("program exceeds max_outputs or outputs is not a list")
        prior, fanout, edges = {}, {}, 0
        for row in rows:
            node = _node(row, prior, limits)
            edges += len(node.inputs)
            if edges > limits.max_edges:
                raise ValueError("program exceeds max_edges")
            for ref in node.inputs:
                fanout[ref] = fanout.get(ref, 0) + 1
                if fanout[ref] > limits.max_fanout:
                    raise ValueError("program exceeds max_fanout")
            prior[node.id] = node
        result_ids = tuple(_identifier(ref, limits) for ref in outputs)
        if len(set(result_ids)) != len(result_ids) or any(ref not in prior for ref in result_ids):
            raise ValueError("outputs must be unique references to existing nodes")
        result = object.__new__(cls)
        object.__setattr__(result, "nodes", tuple(prior.values()))
        object.__setattr__(result, "outputs", result_ids)
        object.__setattr__(result, "limits", limits)
        object.__setattr__(result, "_json", _canonical(data))
        return result

    @classmethod
    def loads(cls, content, *, limits=ProgramLimits()):
        _check_limits(limits)
        return cls.from_dict(loads_json(content, limits=_admission(limits)), limits=limits)

    @classmethod
    def load(cls, path, *, limits=ProgramLimits()):
        _check_limits(limits)
        return cls.from_dict(load_json(path, limits=_admission(limits)), limits=limits)

    def to_dict(self):
        return json.loads(self._json)

    def fingerprint(self):
        """Canonical program identity, not physical-code or alpha equivalence."""
        return sha256(self._json.encode("ascii")).hexdigest()


__all__ = ["Program", "ProgramNode", "ProgramLimits", "SCHEMA"]
