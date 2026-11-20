"""Validated, immutable, data-only programs for the closed QEC workflow dialect.

Parsing performs schema, shape, nominal type, provenance and resource checks;
it does not construct codes, execute algorithms, import plugins, or write state.
Mathematical admission checks remain the executor/core's responsibility.
"""
from dataclasses import dataclass, fields
from hashlib import sha256
import json

from .json_input import (JSONLimits, canonical_json as _canonical, load_json,
                         loads_json, validate_json_data)
from .signatures import (derive_node_type, identifier as _identifier, operation_signature,
                          require_fields as _keys)

SCHEMA = "qeclang.program.v1"


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
    expected, _ = operation_signature(op)
    if type(refs) is not list or len(refs) > limits.max_inputs:
        raise ValueError("node inputs exceed max_inputs or are not a list")
    inputs = tuple(_identifier(ref, limits) for ref in refs)
    if any(ref not in prior for ref in inputs):
        raise ValueError("node inputs must reference prior nodes; forward references and cycles are forbidden")
    if tuple(prior[ref].result_type for ref in inputs) != expected:
        raise ValueError(f"input types do not match {op}: expected {expected}")
    depth = 1 + max((prior[ref].depth for ref in inputs), default=0)
    if depth > limits.max_depth:
        raise ValueError("program dependency depth exceeds max_depth")
    derived = derive_node_type(op, tuple(prior[ref] for ref in inputs), params, limits)
    node = object.__new__(ProgramNode)
    values = {"id": name, "op": op, "inputs": inputs, "depth": depth,
              **vars(derived), "_params_json": _canonical(params)}
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
