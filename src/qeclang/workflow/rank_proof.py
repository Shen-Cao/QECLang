"""Reader-bounded polynomial dimension proof checking, without rank production.

Stored producer step counters receive syntax/consistency checks only. Returned
steps count this verifier's accepted arithmetic-proxy charges, including work
before a refusal. Polynomial traces prove finite-instance ranks, not distance,
family rates, elapsed work, or mathematical novelty.
"""
from dataclasses import dataclass

from qeclang.algebra.binary import nonnegative_integer
from qeclang.algebra.polynomial import ArithmeticBudget, WorkLimitExceeded
from qeclang.constructions.lifted import LiftedProduct
from qeclang.constructions.records import fingerprint
from qeclang.symbolic.repeated_root import (CyclicPolynomialRankResult, PolynomialOperation,
                                           verify_rank_certificate)


@dataclass(frozen=True)
class DimensionProofResult:
    status: str
    steps: int
    reason: str | None = None
    n: int | None = None
    k: int | None = None
    rank_x: int | None = None
    rank_z: int | None = None


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _keys(value, names):
    if type(value) is not dict or len(value) != len(names) or set(value) != set(names):
        raise ValueError("polynomial proof fields do not match the required schema")


def _word(value, degree):
    value = _integer(value, "polynomial word")
    if value.bit_length() - 1 > degree:
        raise WorkLimitExceeded("proof polynomial exceeds max_polynomial_degree")
    return value


def _trace_size(proof):
    names = ("lift", "shape", "diagonal", "gcds", "rank", "steps", "operation_count",
             "operations", "method", "evidence")
    _keys(proof, names)
    if type(proof["operations"]) is not list:
        raise ValueError("polynomial proof requires an explicit JSON operation trace")
    count = _integer(proof["operation_count"], "operation_count")
    if count != len(proof["operations"]):
        raise ValueError("operation_count disagrees with trace length")
    return count


def _decode(proof, shape, lift, declared_rank, degree):
    if _integer(proof["lift"], "proof lift") != lift:
        raise ValueError("proof lift differs from construction")
    if (type(proof["shape"]) is not list or len(proof["shape"]) != 2 or
            any(type(value) is not int for value in proof["shape"]) or proof["shape"] != list(shape)):
        raise ValueError("proof shape differs from the symbolic check matrix")
    if _integer(proof["rank"], "proof rank") != declared_rank:
        raise ValueError("component rank differs from parameter claim")
    count = min(shape)
    words = []
    for key in ("diagonal", "gcds"):
        values = proof[key]
        if type(values) is not list or len(values) != count:
            raise ValueError("diagonal/gcd length differs from check shape")
        words.append(tuple(_word(value, degree) for value in values))
    if any(gcd == 0 or gcd.bit_length() - 1 > lift for gcd in words[1]):
        raise ValueError("claimed gcd cannot divide the cyclic modulus")
    for key in ("method", "evidence"):
        if proof[key] != CyclicPolynomialRankResult.__dataclass_fields__[key].default:
            raise ValueError("unsupported polynomial proof method/evidence label")
    operations = []
    for record in proof["operations"]:
        _keys(record, ("axis", "kind", "target", "source", "multiplier"))
        if record["axis"] not in ("row", "column") or record["kind"] not in ("swap", "add"):
            raise ValueError("invalid elementary operation kind")
        target, source = (_integer(record[key], "operation index") for key in ("target", "source"))
        axis = shape[0] if record["axis"] == "row" else shape[1]
        if target == source or target >= axis or source >= axis:
            raise ValueError("elementary operation indices must be distinct and in range")
        multiplier = _word(record["multiplier"], degree)
        if record["kind"] == "swap" and multiplier:
            raise ValueError("swap multiplier must be zero")
        operations.append(PolynomialOperation(record["axis"], record["kind"], target, source, multiplier))
    return CyclicPolynomialRankResult(lift, tuple(shape), words[0], words[1], declared_rank,
                                      _integer(proof["steps"], "reported steps"), len(operations), tuple(operations))


def _preflight(construction, payload, max_blocks, max_degree, max_polynomial_degree,
               max_operations, max_entries, max_symbolic_terms):
    if not isinstance(construction, LiftedProduct):
        raise ValueError("dimension proof requires its source LiftedProduct")
    _keys(payload, ("construction_fingerprint", "n", "status", "method", "k", "rank_x",
                    "rank_z", "steps", "reason", "x_evidence", "z_evidence"))
    if payload["status"] != "exact" or payload["method"] != "polynomial" or payload["reason"] is not None:
        raise ValueError("only exact polynomial parameter records with traces are supported")
    a, b = construction.a, construction.b
    if len(a.orders) != 1:
        raise ValueError("polynomial dimension proofs require one cyclic factor")
    lift = a.orders[0]
    if lift > max_degree or lift > max_polynomial_degree:
        raise WorkLimitExceeded("source cyclic degree exceeds reader limits")
    columns = a.ncols * b.ncols + a.nrows * b.nrows
    rows_x, rows_z = a.nrows * b.ncols, a.ncols * b.nrows
    if ((rows_x + rows_z) * columns > max_blocks or
            max(a.nrows, a.ncols, b.nrows, b.ncols) ** 2 > max_blocks):
        raise WorkLimitExceeded("source check construction exceeds max_blocks")
    if max(columns, rows_x, rows_z, rows_x * columns, rows_z * columns) > max_entries:
        raise WorkLimitExceeded("source check matrices exceed max_entries")
    locality = construction.locality()
    if (locality.x_nonzeros + locality.z_nonzeros) // lift > max_symbolic_terms:
        raise WorkLimitExceeded("source checks exceed max_symbolic_terms")
    if payload["construction_fingerprint"] != fingerprint(construction):
        raise ValueError("dimension proof is not bound to the supplied construction")
    n, k, rx, rz = (_integer(payload[key], key) for key in ("n", "k", "rank_x", "rank_z"))
    if (n != lift * columns or rx > lift * min(rows_x, columns) or
            rz > lift * min(rows_z, columns) or k != n - rx - rz):
        raise ValueError("parameter dimensions or rank claims are inconsistent")
    xp, zp = payload["x_evidence"], payload["z_evidence"]
    if _trace_size(xp) + _trace_size(zp) > max_operations:
        raise WorkLimitExceeded("combined polynomial traces exceed max_operations")
    x = _decode(xp, (rows_x, columns), lift, rx, max_polynomial_degree)
    z = _decode(zp, (rows_z, columns), lift, rz, max_polynomial_degree)
    if _integer(payload["steps"], "reported total steps") != x.steps + z.steps:
        raise ValueError("reported producer step counters are inconsistent")
    return n, k, rx, rz, x, z


def verify_dimension_proof(construction, parameters_payload, *, max_steps=10_000_000,
                           max_blocks=1_000_000, max_degree=512,
                           max_polynomial_degree=4096, max_operations=100_000,
                           max_entries=1_000_000, max_symbolic_terms=100_000) -> DimensionProofResult:
    """Check source-bound JSON polynomial rank evidence without search/elimination.

    Reader limits apply independently of producer claims. Both traces share one
    arithmetic ceiling and one operation-count ceiling. Limits precede shifts,
    block/monomial construction and polynomial replay. On failure, dimensions are absent
    and steps retains accepted verifier charges. Symbolic-check materialization,
    source hashing, and JSON admission are outside the arithmetic proxy; callers
    must bound JSON bytes/depth before calling this reader on external records.
    """
    max_steps = nonnegative_integer(max_steps, "max_steps")
    max_blocks = nonnegative_integer(max_blocks, "max_blocks")
    max_degree = nonnegative_integer(max_degree, "max_degree")
    max_polynomial_degree = nonnegative_integer(max_polynomial_degree, "max_polynomial_degree")
    max_operations = nonnegative_integer(max_operations, "max_operations")
    max_entries = nonnegative_integer(max_entries, "max_entries")
    max_symbolic_terms = nonnegative_integer(max_symbolic_terms, "max_symbolic_terms")
    meter = ArithmeticBudget(max_steps)
    try:
        n, k, rx, rz, x, z = _preflight(construction, parameters_payload, max_blocks, max_degree,
                                        max_polynomial_degree, max_operations, max_entries,
                                        max_symbolic_terms)
        hx, hz = construction.symbolic_checks(max_blocks=max_blocks)
        options = dict(max_lift=max_degree, max_polynomial_degree=max_polynomial_degree,
                       max_steps=max_steps, max_entries=max_entries, _meter=meter)
        if not verify_rank_certificate(hx, x, max_operations=max_operations, **options):
            return DimensionProofResult("invalid_proof", meter.used, "X rank trace failed verification")
        if not verify_rank_certificate(hz, z, max_operations=max_operations - x.operation_count, **options):
            return DimensionProofResult("invalid_proof", meter.used, "Z rank trace failed verification")
        return DimensionProofResult("verified", meter.used, n=n, k=k, rank_x=rx, rank_z=rz)
    except WorkLimitExceeded as error:
        return DimensionProofResult("budget_exhausted", meter.used, str(error))
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return DimensionProofResult("invalid_proof", meter.used, str(error))


__all__ = ["DimensionProofResult", "verify_dimension_proof"]
