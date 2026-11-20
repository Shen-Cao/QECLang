"""Bidirectional positive-CSS generator equivalence and witness rebinding.

Changing generators preserves the code and physical logical classes, but can
change generator weights, Tanner graphs, syndrome coordinates and decoder or
measurement performance. Reader checks use XOR identities, never elimination.
"""
from qeclang.algebra.binary import nonnegative_integer, set_bits
from qeclang.analysis.certificate_io import certificate_record, verify_certificate_record
from qeclang.analysis.certificates import LogicalWitness, verify_css_logical
from qeclang.codes import CSSCode
from .relations import RelationError, _css

_KEYS = ("x_forward", "x_reverse", "z_forward", "z_reverse")
_GUARANTEES = ("physical_qubit_count", "physical_basis_identity", "x_and_z_sector_labels",
               "positive_stabilizer_subspace", "logical_quotient", "hamming_metric",
               "sector_distances", "syndrome_maps_on_physical_errors")


def _counts(values):
    if (not isinstance(values, (tuple, list)) or len(values) != 2 or
            any(type(value) is not int or value < 0 for value in values)):
        raise RelationError("sector row counts must be two nonnegative integers")
    return tuple(values)


def _layout(maps, source_counts, target_counts):
    sx, sz = _counts(source_counts)
    tx, tz = _counts(target_counts)
    if type(maps) is not dict or len(maps) != 4 or set(maps) != set(_KEYS):
        raise RelationError("row maps require x/z forward and reverse supports")
    layout = (("x_forward", tx, sx), ("x_reverse", sx, tx),
              ("z_forward", tz, sz), ("z_reverse", sz, tz))
    for key, rows, _ in layout:
        if type(maps[key]) is not list or len(maps[key]) != rows:
            raise RelationError(f"{key} row count does not match its destination generators")
    return layout


def validate_map_supports(maps, source_counts, target_counts, *, max_rows=None, max_indices=None):
    """Validate JSON support maps without constructing matrices or checking identities.

    Counts are (X rows, Z rows). Forward map rows index source generators;
    reverse rows index target generators. Supports must be sorted, distinct
    plain integers. Optional limits apply jointly to all four maps.
    """
    layout = _layout(maps, source_counts, target_counts)
    rows = sum(count for _, count, _ in layout)
    if max_rows is not None and rows > nonnegative_integer(max_rows, "max_rows"):
        raise RelationError("row maps exceed max_rows")
    maximum = None if max_indices is None else nonnegative_integer(max_indices, "max_indices")
    indices = 0
    for key, _, width in layout:
        for support in maps[key]:
            if type(support) is not list or len(support) > width:
                raise RelationError("row-map supports must be canonical index arrays")
            indices += len(support)
            if maximum is not None and indices > maximum:
                raise RelationError("row maps exceed max_indices")
            previous = -1
            for index in support:
                if type(index) is not int or not previous < index < width:
                    raise RelationError("row-map indices must be strictly increasing and in range")
                previous = index
    return rows, indices


def _shapes(source, target):
    if not isinstance(source, CSSCode) or not isinstance(target, CSSCode):
        raise RelationError("row equivalence requires positive packed CSSCode presentations")
    if source.n != target.n:
        raise RelationError("row-equivalent codes must use the same physical sites")
    return ((source.hx.nrows, source.hz.nrows), (target.hx.nrows, target.hz.nrows))


def row_equivalence_work(source, target, maps):
    """Deterministic pre-charge proxy for estimation and XOR-identity verification.

    Five units per source/target generator plus two per referenced map index,
    and eight fixed units, cover row visits, index checks and XORs in this
    path. Bigint width, hashing, witness checks and map production are excluded.
    Shape/support-length inspection occurs here; index correctness and the
    mathematical equations are checked only by verify_row_equivalence.
    """
    counts = _shapes(source, target)
    _css(source)
    _css(target)
    layout = _layout(maps, *counts)
    indices = 0
    for key, _, width in layout:
        for support in maps[key]:
            if type(support) is not list or len(support) > width:
                raise RelationError("row-map support length is invalid")
            indices += len(support)
    return 8 + 5 * sum(rows for _, rows, _ in layout) + 2 * indices


def verify_row_equivalence(source, target, maps) -> tuple[str, ...]:
    """Check H_target=A H_source and H_source=B H_target separately for X/Z.

    Two-way inclusion is required. The maps need not be inverse on arbitrary
    syndrome strings: redundant checks restrict the set of physical syndromes.
    No graph, generator-weight, extraction, or decoder-performance claim follows.
    """
    counts = _shapes(source, target)
    sx, sz = _css(source)
    tx, tz = _css(target)
    validate_map_supports(maps, *counts)
    for key, old, new in (("x_forward", sx, tx), ("x_reverse", tx, sx),
                          ("z_forward", sz, tz), ("z_reverse", tz, sz)):
        for support, expected in zip(maps[key], new.rows):
            value = 0
            for index in support:
                value ^= old.rows[index]
            if value != expected:
                raise RelationError(f"{key} does not reproduce the declared generator")
    return _GUARANTEES


def row_maps(source, target, *, max_bits=100_000_000, max_axis=100_000):
    """Opt-in host producer for bidirectional maps using exact linear solves.

    The payload proxy includes source/target rows, transposes, augmented solves
    and worst-case packed map storage. It excludes returned index-object overhead
    and does not bound runtime. Reader verification does not invoke this producer.
    """
    source_counts, target_counts = _shapes(source, target)
    bits = nonnegative_integer(max_bits, "max_bits")
    axis = nonnegative_integer(max_axis, "max_axis")
    if max(source.n, *source_counts, *target_counts) > axis:
        raise RelationError("row-map production exceeds max_axis")
    payload = sum(3 * source.n * (s + t + 2) + 2 * s * t
                  for s, t in zip(source_counts, target_counts))
    if payload > bits:
        raise RelationError("row-map production exceeds max_bits")
    sx, sz = _css(source)
    tx, tz = _css(target)
    result = {}
    try:
        for key, old, new in (("x_forward", sx, tx), ("x_reverse", tx, sx),
                              ("z_forward", sz, tz), ("z_reverse", tz, sz)):
            equations = old.T
            result[key] = [list(set_bits(equations.solve(row))) for row in new.rows]
    except ValueError as error:
        raise RelationError("source and target generator spaces are not equivalent") from error
    verify_row_equivalence(source, target, result)
    return result


def _witness_preflight(source, target, max_bits, max_axis):
    source_counts, target_counts = _shapes(source, target)
    bits = nonnegative_integer(max_bits, "max_bits")
    axis = nonnegative_integer(max_axis, "max_axis")
    if max(source.n, *source_counts, *target_counts) > axis:
        raise RelationError("witness rebinding exceeds max_axis")
    if 2 * (source.n + target.n) > bits:
        raise RelationError("witness rebinding exceeds max_bits")


def _checked_witness(code, record):
    if (type(record) is not dict or record.get("kind") != "css_logical" or
            not verify_certificate_record(code, record, max_axis=code.n, max_bits=2 * code.n)):
        raise RelationError("invalid or differently source-bound CSS logical certificate")
    return record["certificate"]


def rebind_witness(source, target, maps, cert, *, max_bits=100_000_000, max_axis=100_000):
    """Keep operator/dual coordinates fixed and bind a verified upper witness to target."""
    _witness_preflight(source, target, max_bits, max_axis)
    verify_row_equivalence(source, target, maps)
    payload = _checked_witness(source, cert)
    witness = LogicalWitness(payload["sector"], int(payload["operator"], 16),
                             int(payload["dual"], 16), payload["weight"])
    if not verify_css_logical(target, witness):
        raise RelationError("unchanged witness failed on the row-equivalent target")
    rebound = certificate_record(target, witness)
    _checked_witness(target, rebound)
    return rebound


def check_rebound(source, target, maps, original, rebased, *, max_bits=100_000_000, max_axis=100_000):
    """Independently verify both certificates, row maps and unchanged logical words.

    This never calls row_maps, rebind_witness, elimination, or distance analysis.
    Valid certificates for different representatives do not satisfy this edge.
    """
    _witness_preflight(source, target, max_bits, max_axis)
    verify_row_equivalence(source, target, maps)
    old, new = _checked_witness(source, original), _checked_witness(target, rebased)
    if old["sector"] != new["sector"] or old["weight"] != new["weight"]:
        raise RelationError("rebinding changed witness sector or weight")
    if any(int(old[key], 16) != int(new[key], 16) for key in ("operator", "dual")):
        raise RelationError("rebinding changed the logical operator or its dual")


__all__ = ["validate_map_supports", "row_equivalence_work", "verify_row_equivalence",
           "row_maps", "rebind_witness", "check_rebound"]
