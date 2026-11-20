"""Static, bounded types for finite abelian lifted-product program literals.

Validation reuses the construction-record scanner but never instantiates ring
objects, performs polynomial arithmetic, expands matrices, or computes ranks.
"""
from dataclasses import dataclass

from qeclang.constructions import records

PARAMETER_BACKENDS = ("crt", "polynomial", "abelian", "mixed-abelian")


@dataclass(frozen=True)
class LiftType:
    orders: tuple[int, ...]
    a_shape: tuple[int, int]
    b_shape: tuple[int, int]
    group_size: int
    n: int
    x_checks: int
    z_checks: int


def construction_limits(program_limits):
    """Translate program admission bounds to the matching core record caps.

    The caller supplies ProgramLimits; ConstructionLimits independently validates
    all translated counts. Expanded symbolic axes are bounds on integer shape
    metadata here, not permission to allocate packed matrices during execution.
    """
    return records.ConstructionLimits(
        max_bytes=program_limits.max_bytes,
        max_factors=program_limits.max_group_factors,
        max_order=program_limits.max_cyclic_order,
        max_group_size=program_limits.max_group_size,
        max_rows=program_limits.max_seed_axis,
        max_columns=program_limits.max_seed_axis,
        max_entries=program_limits.max_seed_entries,
        max_terms=program_limits.max_seed_terms,
        max_exponent=program_limits.max_cyclic_order,
        max_sites=program_limits.max_symbolic_sites,
    )


def validate_lift_literal(data, *, limits=records.ConstructionLimits()) -> LiftType:
    """Validate canonical construction data and derive its unexpanded shape.

    Each entry's exponent tuples must be unique, reduced modulo the declared
    cyclic orders, and lexicographically ordered. Normalize an existing instance
    with constructions.records.to_dict before placing it into a program. Empty
    matrix axes and zero ring entries retain their explicit shape information.
    """
    if not isinstance(limits, records.ConstructionLimits):
        raise TypeError("lift literal validation requires ConstructionLimits")
    records.keys(data, ("schema", "kind", "coefficient_field", "orders", "action", "convention",
                        "physical_ordering", "a", "b"), "construction")
    expected = {"schema": records.SCHEMA, "kind": "abelian_lifted_product",
                "coefficient_field": "GF(2)", "action": records.ACTION,
                "convention": records.CONVENTION, "physical_ordering": records.ORDERING}
    if any(data[key] != value for key, value in expected.items()):
        raise ValueError("unsupported lifted construction semantics")
    orders = data["orders"]
    if type(orders) is not list:
        raise ValueError("cyclic orders must be a JSON list")
    size = records.check_orders(orders, limits)

    def terms(values):
        previous = None
        for term in values:
            if type(term) is not list or len(term) != len(orders):
                raise ValueError("monomial dimension disagrees with the group factors")
            for exponent, order in zip(term, orders):
                records.integer(exponent, "canonical exponent", maximum=order-1)
            current = tuple(term)
            if previous is not None and current <= previous:
                raise ValueError("monomials must be unique and in canonical lexicographic order")
            previous = current

    a, b = records.scan_matrices((data["a"], data["b"]), limits, terms)
    records.check_sites(a, b, size, limits)
    ma, na = a
    mb, nb = b
    return LiftType(tuple(orders), a, b, size, size*(na*nb + ma*mb),
                    size*ma*nb, size*na*mb)


def validate_parameter_request(params):
    """Validate an explicit backend request, not its eventual mathematical support.

    An unsupported ring/backend combination remains a valid typed request; the
    execution result must report unsupported instead of silently substituting a
    backend. Operation traces are defined only for the polynomial backend.
    """
    records.keys(params, ("backend", "max_steps", "record_operations"), "lift.parameters")
    if type(params["backend"]) is not str or params["backend"] not in PARAMETER_BACKENDS:
        raise ValueError("lift.parameters requires a supported explicit symbolic backend")
    records.integer(params["max_steps"], "max_steps")
    if type(params["record_operations"]) is not bool:
        raise ValueError("record_operations must be boolean")
    if params["record_operations"] and params["backend"] != "polynomial":
        raise ValueError("record_operations requires the polynomial backend")


__all__ = ["LiftType", "PARAMETER_BACKENDS", "construction_limits",
           "validate_lift_literal", "validate_parameter_request"]
