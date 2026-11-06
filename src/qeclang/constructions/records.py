"""Bounded, data-only interchange for commutative lifted-product constructions."""
from dataclasses import dataclass, fields
from hashlib import sha256
import json
from pathlib import Path

from qeclang.symbolic import GroupAlgebraElement, GroupAlgebraMatrix
from .lifted import LiftedProduct

SCHEMA = "qeclang.construction.v1"
CONVENTION = "hx=[A*kron*I,I*kron*B.star];hz=[I*kron*B,A.star*kron*I]"
ORDERING = "VV,CC;lexicographic-base-pairs;group-site-inside-block"
ACTION = "row-site-to-column-site-plus-exponent"


@dataclass(frozen=True)
class ConstructionLimits:
    """Input budgets, excluding caller-owned data and Python object overhead.

    max_sites limits each expanded code/check axis without materializing it.
    Term/entry limits are shared by both seed matrices.
    """
    max_bytes: int = 10_000_000
    max_factors: int = 16
    max_order: int = 1_000_000
    max_group_size: int = 1_000_000
    max_rows: int = 10_000
    max_columns: int = 10_000
    max_entries: int = 100_000
    max_terms: int = 1_000_000
    max_exponent: int = 1_000_000_000
    max_sites: int = 1_000_000

    def __post_init__(self):
        for field in fields(self):
            integer(getattr(self, field.name), field.name)


def integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}")
    return value


def keys(data, expected, name):
    if not isinstance(data, dict) or set(data) != set(expected):
        raise ValueError(f"invalid {name} fields")


def canonical_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def digest(data):
    """SHA256 of the canonical structured description, not code equivalence."""
    return sha256(canonical_json(data).encode("utf-8")).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def read_json(path, limits):
    if not isinstance(limits, ConstructionLimits):
        raise TypeError("limits must be ConstructionLimits")
    with Path(path).open("rb") as stream:
        content = stream.read(limits.max_bytes + 1)
    if len(content) > limits.max_bytes:
        raise ValueError("construction file exceeds max_bytes")
    return json.loads(content, object_pairs_hook=_unique_object)


def write_json(data, path):
    Path(path).write_text(canonical_json(data) + "\n", encoding="utf-8")


def check_orders(orders, limits):
    if not isinstance(orders, (list, tuple)) or not 1 <= len(orders) <= limits.max_factors:
        raise ValueError("invalid or oversized group factors")
    size = 1
    for order in orders:
        integer(order, "cyclic order", 1, limits.max_order)
        if size > limits.max_group_size // order:
            raise ValueError("group order exceeds max_group_size")
        size *= order
    return size


def check_sites(ashape, bshape, size, limits):
    ma, na = ashape
    mb, nb = bshape
    if size * max(na*nb + ma*mb, ma*nb, na*mb) > limits.max_sites:
        raise ValueError("construction axes exceed max_sites")


def matrix_data(matrix):
    return {"shape": [matrix.nrows, matrix.ncols],
            "entries": [[[list(term) for term in sorted(e.terms)] for e in row]
                        for row in matrix.rows]}


def scan_matrices(matrices, limits, term_validator):
    """Validate both nested matrices before constructing algebra objects."""
    shapes, total_entries, total_terms = [], 0, 0
    for matrix in matrices:
        keys(matrix, ("shape", "entries"), "seed matrix")
        shape, rows = matrix["shape"], matrix["entries"]
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("matrix shape must have two dimensions")
        m = integer(shape[0], "row count", maximum=limits.max_rows)
        n = integer(shape[1], "column count", maximum=limits.max_columns)
        total_entries += m*n
        if total_entries > limits.max_entries:
            raise ValueError("seed matrices exceed max_entries")
        if not isinstance(rows, list) or len(rows) != m:
            raise ValueError("seed row count disagrees with shape")
        for row in rows:
            if not isinstance(row, list) or len(row) != n:
                raise ValueError("seed column count disagrees with shape")
            for terms in row:
                if not isinstance(terms, list):
                    raise ValueError("entry terms must be a list")
                total_terms += len(terms)
                if total_terms > limits.max_terms:
                    raise ValueError("seed matrices exceed max_terms")
                term_validator(terms)
        shapes.append((m, n))
    return shapes


def to_dict(construction):
    if not isinstance(construction, LiftedProduct):
        raise TypeError("construction must be LiftedProduct")
    return {"schema": SCHEMA, "kind": "abelian_lifted_product", "coefficient_field": "GF(2)",
            "orders": list(construction.a.orders), "action": ACTION,
            "convention": CONVENTION, "physical_ordering": ORDERING,
            "a": matrix_data(construction.a), "b": matrix_data(construction.b)}


def from_dict(data, *, limits=ConstructionLimits()):
    if not isinstance(limits, ConstructionLimits):
        raise TypeError("limits must be ConstructionLimits")
    keys(data, ("schema", "kind", "coefficient_field", "orders", "action", "convention",
                "physical_ordering", "a", "b"), "construction")
    expected = {"schema": SCHEMA, "kind": "abelian_lifted_product", "coefficient_field": "GF(2)",
                "action": ACTION, "convention": CONVENTION, "physical_ordering": ORDERING}
    if any(data[key] != value for key, value in expected.items()):
        raise ValueError("unsupported construction semantics")
    orders = data["orders"]
    size = check_orders(orders, limits)

    def validate(terms):
        seen = set()
        for term in terms:
            if not isinstance(term, list) or len(term) != len(orders):
                raise ValueError("monomial dimension disagrees with group")
            for exponent, order in zip(term, orders):
                integer(exponent, "canonical exponent", maximum=order-1)
            value = tuple(term)
            if value in seen:
                raise ValueError("duplicate monomial in canonical ring entry")
            seen.add(value)
    ashape, bshape = scan_matrices((data["a"], data["b"]), limits, validate)
    check_sites(ashape, bshape, size, limits)

    def decode(matrix):
        rows = tuple(tuple(GroupAlgebraElement.from_terms(orders, terms) for terms in row)
                     for row in matrix["entries"])
        return GroupAlgebraMatrix(rows, matrix["shape"][1], tuple(orders))
    return LiftedProduct(decode(data["a"]), decode(data["b"]))


def fingerprint(construction):
    return digest(to_dict(construction))


def dump(construction, path):
    write_json(to_dict(construction), path)


def load(path, *, limits=ConstructionLimits()):
    return from_dict(read_json(path, limits), limits=limits)
