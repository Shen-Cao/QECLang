"""Cyclic lifted-product families with Laurent seeds and explicit parameter domains."""
from dataclasses import dataclass

from qeclang.symbolic import GroupAlgebraElement, GroupAlgebraMatrix
from .lifted import LiftedProduct
from .records import (ACTION, CONVENTION, ORDERING, ConstructionLimits, check_orders,
                      check_sites, digest, integer, keys, read_json, scan_matrices, write_json)

SCHEMA = "qeclang.family.v1"


@dataclass(frozen=True)
class LaurentMatrix:
    """GF(2)[x,x^-1] matrix; entries are finite tuples of integer exponents.

    Equal integer exponents cancel before any lift is chosen. Row/column shapes
    are explicit, including matrices with zero rows or zero columns.
    """
    rows: tuple[tuple[tuple[int, ...], ...], ...]
    ncols: int

    def __post_init__(self):
        width = integer(self.ncols, "column count")
        normalized = []
        for row in self.rows:
            if len(row) != width:
                raise ValueError("Laurent matrix row width disagrees with ncols")
            entries = []
            for terms in row:
                support = set()
                for exponent in terms:
                    if type(exponent) is not int:
                        raise ValueError("Laurent exponents must be integers")
                    if exponent in support:
                        support.remove(exponent)
                    else:
                        support.add(exponent)
                entries.append(tuple(sorted(support)))
            normalized.append(tuple(entries))
        object.__setattr__(self, "rows", tuple(normalized))

    @property
    def shape(self):
        return len(self.rows), self.ncols

    @property
    def row_weights(self):
        return tuple(sum(map(len, row)) for row in self.rows)

    @property
    def column_weights(self):
        return tuple(sum(len(row[j]) for row in self.rows) for j in range(self.ncols))

    def to_dict(self):
        return {"shape": list(self.shape),
                "entries": [[list(terms) for terms in row] for row in self.rows]}


@dataclass(frozen=True)
class LiftDomain:
    """Positive lift orders; odd-only by default, with optional finite maximum."""
    minimum: int = 1
    maximum: int | None = None
    odd_only: bool = True

    def __post_init__(self):
        integer(self.minimum, "minimum lift", 1)
        if self.maximum is not None:
            integer(self.maximum, "maximum lift", self.minimum)
        if type(self.odd_only) is not bool:
            raise ValueError("odd_only must be a boolean")
        first = self.minimum + (self.odd_only and self.minimum % 2 == 0)
        if self.maximum is not None and first > self.maximum:
            raise ValueError("lift domain is empty")

    def check(self, lift):
        integer(lift, "lift", self.minimum, self.maximum)
        if self.odd_only and lift % 2 == 0:
            raise ValueError("lift is outside the odd-only domain")

    def to_dict(self):
        return {"minimum": self.minimum, "maximum": self.maximum, "odd_only": self.odd_only}


@dataclass(frozen=True)
class FamilyLocalityBound:
    max_check_weight: int
    max_qubit_degree: int
    x_nonzeros_per_lift: int
    z_nonzeros_per_lift: int


@dataclass(frozen=True)
class CyclicLiftFamily:
    """Fixed Laurent seeds specialized modulo x^L-1 for each allowed L.

    Locality is uniformly bounded by pre-specialization support counts. No
    dimension, distance, rate or asymptotic goodness is asserted by this type.
    """
    a: LaurentMatrix
    b: LaurentMatrix
    domain: LiftDomain = LiftDomain()

    def __post_init__(self):
        if not isinstance(self.a, LaurentMatrix) or not isinstance(self.b, LaurentMatrix):
            raise TypeError("family seeds must be LaurentMatrix")
        if not isinstance(self.domain, LiftDomain):
            raise TypeError("domain must be LiftDomain")

    def instantiate(self, lift, *, limits=ConstructionLimits()):
        """Return symbolic seeds; resource budgets are checked before specialization."""
        if not isinstance(limits, ConstructionLimits):
            raise TypeError("limits must be ConstructionLimits")
        self.domain.check(lift)
        size = check_orders([lift], limits)
        check_sites(self.a.shape, self.b.shape, size, limits)
        self._check_templates(limits)

        def specialize(matrix):
            rows = tuple(tuple(GroupAlgebraElement.from_terms((lift,), ((e,) for e in terms))
                               for terms in row) for row in matrix.rows)
            return GroupAlgebraMatrix(rows, matrix.ncols, (lift,))
        return LiftedProduct(specialize(self.a), specialize(self.b))

    def _check_templates(self, limits):
        entries = terms = 0
        for matrix in (self.a, self.b):
            m, n = matrix.shape
            if m > limits.max_rows or n > limits.max_columns:
                raise ValueError("template dimensions exceed limits")
            entries += m*n
            if entries > limits.max_entries:
                raise ValueError("templates exceed max_entries")
            for row in matrix.rows:
                for entry in row:
                    terms += len(entry)
                    if terms > limits.max_terms:
                        raise ValueError("templates exceed max_terms")
                    if any(abs(e) > limits.max_exponent for e in entry):
                        raise ValueError("template exponent exceeds max_exponent")

    def locality_bound(self):
        ar, ac = self.a.row_weights, self.a.column_weights
        br, bc = self.b.row_weights, self.b.column_weights
        ma, na = self.a.shape
        mb, nb = self.b.shape

        def pair(left, right):
            return max(left) + max(right) if left and right else 0
        return FamilyLocalityBound(max(pair(ar, bc), pair(ac, br)),
                                   max(pair(ac, bc), pair(ar, br)),
                                   nb*sum(ar) + ma*sum(br), na*sum(br) + mb*sum(ar))

    def to_dict(self):
        return {"schema": SCHEMA, "kind": "cyclic_lift_family", "coefficient_field": "GF(2)",
                "template_ring": "GF(2)[x,x^-1]", "action": ACTION, "convention": CONVENTION,
                "physical_ordering": ORDERING, "domain": self.domain.to_dict(),
                "a": self.a.to_dict(), "b": self.b.to_dict()}

    @classmethod
    def from_dict(cls, data, *, limits=ConstructionLimits()):
        if not isinstance(limits, ConstructionLimits):
            raise TypeError("limits must be ConstructionLimits")
        keys(data, ("schema", "kind", "coefficient_field", "template_ring", "action",
                    "convention", "physical_ordering", "domain", "a", "b"), "family")
        expected = {"schema": SCHEMA, "kind": "cyclic_lift_family", "coefficient_field": "GF(2)",
                    "template_ring": "GF(2)[x,x^-1]", "action": ACTION,
                    "convention": CONVENTION, "physical_ordering": ORDERING}
        if any(data[key] != value for key, value in expected.items()):
            raise ValueError("unsupported family semantics")
        keys(data["domain"], ("minimum", "maximum", "odd_only"), "domain")
        domain = LiftDomain(**data["domain"])

        def validate(terms):
            seen = set()
            for e in terms:
                if type(e) is not int or abs(e) > limits.max_exponent:
                    raise ValueError("invalid or oversized Laurent exponent")
                if e in seen:
                    raise ValueError("duplicate exponent in canonical Laurent entry")
                seen.add(e)
        scan_matrices((data["a"], data["b"]), limits, validate)

        def decode(matrix):
            return LaurentMatrix(tuple(tuple(tuple(e) for e in row)
                                       for row in matrix["entries"]), matrix["shape"][1])
        return cls(decode(data["a"]), decode(data["b"]), domain)

    def fingerprint(self):
        return digest(self.to_dict())

    def dump(self, path):
        write_json(self.to_dict(), path)

    @classmethod
    def load(cls, path, *, limits=ConstructionLimits()):
        return cls.from_dict(read_json(path, limits), limits=limits)
