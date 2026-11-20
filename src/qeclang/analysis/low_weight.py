"""Exact bounded certificates excluding nontrivial CSS weights one and two.

Exclusion alone says nothing about existence of logicals. In particular k=0
codes receive no finite distance. A separately checked logical upper witness
can combine with exclusion into a finite interval.
"""
from dataclasses import dataclass
import re

from qeclang.algebra.binary import set_bits
from qeclang.algebra.polynomial import WorkLimitExceeded
from qeclang.analysis.certificates import LogicalWitness, verify_css_logical
from .low_weight_support import (FIELDS, LOW_WEIGHT_DOMAIN, LOW_WEIGHT_SCHEMA, assemble,
    base_budget, columns, low_weight_id, options, producer_targets, reader_targets, request, source_id)


def _basis(rows, meter):
    basis = {}
    for index, original in enumerate(rows):
        meter.charge()
        row, coefficients = original, 1 << index
        while row:
            meter.charge()
            pivot = (row & -row).bit_length() - 1
            if pivot not in basis:
                basis[pivot] = row, coefficients
                break
            row ^= basis[pivot][0]
            coefficients ^= basis[pivot][1]
    return basis


def _reduce(word, basis, meter):
    coefficients = 0
    while word:
        meter.charge()
        pivot = (word & -word).bit_length() - 1
        if pivot not in basis:
            return None, pivot
        word ^= basis[pivot][0]
        coefficients ^= basis[pivot][1]
    return coefficients, None


def _counterexample(code, sector, word, free, basis, subject, meter):
    dual = 1 << free
    meter.charge(len(basis) * max(1, len(basis).bit_length()))
    for pivot in sorted(basis, reverse=True):
        meter.charge()
        if (basis[pivot][0] & dual).bit_count() % 2:
            dual ^= 1 << pivot
    certificate = LogicalWitness(sector, word, dual, word.bit_count())
    meter.charge(code.hx.nrows + code.hz.nrows + 2)
    if not verify_css_logical(code, certificate):
        raise ArithmeticError("constructed low-weight counterexample failed independent parity checks")
    return {"schema": "qeclang.certificate.v1", "source": subject, "kind": "css_logical",
            "claim": {"distance_upper_bound": certificate.weight, "sector": sector},
            "certificate": {"sector": sector, "operator": hex(word), "dual": hex(dual),
                            "weight": certificate.weight}}


def certify_low_weight(code, sector, through=2, *, max_steps=1_000_000, max_axis=4096,
                       max_bits=100_000_000, max_indices=1_000_000):
    """Produce complete exclusion memberships or a verified weight≤through witness.

    Column incidences, tracked elimination, reductions, dual back-substitution
    and coefficient output are charged. Source/content hashing is streaming
    bookkeeping outside this arithmetic proxy. max_bits bounds packed/incidence/
    coefficient payload proxies, excluding Python overhead and caller-owned data.
    """
    meter, limits = options(code, max_steps, max_axis, max_bits, max_indices)
    request(sector, through)
    subject = source_id(code)
    same, opposite = (code.hx, code.hz) if sector == "x" else (code.hz, code.hx)
    try:
        incidence, cost = columns(code, opposite, limits, meter)
        targets = iter(producer_targets(incidence, through, meter))
        first = next(targets, None)
        if first is None:
            return assemble(subject, sector, through, "excluded", meter.used, memberships=[])
        cost += 2 * min(code.n, same.nrows) * (code.n + same.nrows) + 2 * (code.n + same.nrows)
        if cost > limits["max_bits"]:
            raise WorkLimitExceeded("tracked row-space workspace exceeds max_bits")
        basis = _basis(same.rows, meter)
        memberships, indices = [], 0
        word = first
        while word is not None:
            coefficients, free = _reduce(word, basis, meter)
            if coefficients is None:
                witness = _counterexample(code, sector, word, free, basis, subject, meter)
                return assemble(subject, sector, through, "found", meter.used, witness=witness)
            count = coefficients.bit_count()
            if count > limits["max_indices"] - indices:
                raise WorkLimitExceeded("membership proof exceeds max_indices")
            indices += count
            if cost + indices * max(1, same.nrows.bit_length()) > limits["max_bits"]:
                raise WorkLimitExceeded("membership coefficient payload exceeds max_bits")
            meter.charge(1 + count)
            memberships.append(list(set_bits(coefficients)))
            word = next(targets, None)
        return assemble(subject, sector, through, "excluded", meter.used, memberships=memberships)
    except WorkLimitExceeded as error:
        return assemble(subject, sector, through, "budget_exhausted", meter.used, reason=str(error))


@dataclass(frozen=True)
class LowWeightCheckResult:
    status: str
    steps: int
    reason: str | None = None
    source: str | None = None
    sector: str | None = None
    excluded_through: int | None = None
    upper_bound: int | None = None


def _keys(value, names):
    if type(value) is not dict or len(value) != len(names) or set(value) != set(names):
        raise ValueError("invalid low-weight record fields")


def _integer(value):
    if type(value) is not int or value < 0 or value.bit_length() > 4096:
        raise ValueError("low-weight counters/indices must be bounded nonnegative integers")
    return value


def _witness(record, code, sector, through):
    _keys(record, ("schema", "source", "kind", "claim", "certificate"))
    if record["schema"] != "qeclang.certificate.v1" or record["kind"] != "css_logical":
        raise ValueError("counterexample requires a CSS logical certificate")
    if type(record["source"]) is not str or re.fullmatch(r"[0-9a-f]{64}", record["source"]) is None:
        raise ValueError("counterexample source must be a bounded presentation fingerprint")
    payload = record["certificate"]
    _keys(payload, ("sector", "operator", "dual", "weight"))
    weight = _integer(payload["weight"])
    if payload["sector"] != sector or not 1 <= weight <= through:
        raise ValueError("counterexample sector/weight differs from requested layers")
    _keys(record["claim"], ("sector", "distance_upper_bound"))
    if (_integer(record["claim"]["distance_upper_bound"]) != weight or
            record["claim"]["sector"] != sector):
        raise ValueError("counterexample claim differs from its certificate")
    words = []
    for name in ("operator", "dual"):
        word = payload[name]
        if (type(word) is not str or len(word) > max(1, (code.n + 3) // 4) + 2 or
                re.fullmatch(r"0x(?:0|[1-9a-f][0-9a-f]*)", word) is None):
            raise ValueError("invalid or oversized counterexample word")
        value = int(word, 16)
        if value.bit_length() > code.n:
            raise ValueError("counterexample word exceeds physical width")
        words.append(value)
    return LogicalWitness(sector, *words, weight)


def _record(record, code, limits, meter):
    _keys(record, FIELDS)
    if record["schema"] != LOW_WEIGHT_SCHEMA:
        raise ValueError("unsupported low-weight schema")
    for key in ("source", "record_id"):
        if type(record[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", record[key]) is None:
            raise ValueError("invalid low-weight source or record identifier")
    request(record["sector"], record["through"])
    _integer(record["steps"])
    status, claim = record["status"], record["claim"]
    same = code.hx if record["sector"] == "x" else code.hz
    indices, witness = 0, None
    if status == "excluded":
        _keys(claim, ("predicate", "through"))
        if (claim["predicate"] != "nontrivial_weight_exclusion" or type(claim["through"]) is not int
                or claim["through"] != record["through"] or record["witness"] is not None
                or record["reason"] is not None or type(record["memberships"]) is not list):
            raise ValueError("invalid exclusion claim")
        if len(record["memberships"]) > code.n:
            raise ValueError("too many low-weight membership obligations")
        for support in record["memberships"]:
            meter.charge()
            if type(support) is not list or len(support) > same.nrows:
                raise ValueError("membership coefficients require canonical support arrays")
            indices += len(support)
            if indices > limits["max_indices"]:
                raise WorkLimitExceeded("membership proof exceeds max_indices")
            previous = -1
            for index in support:
                meter.charge()
                if not previous < _integer(index) < same.nrows:
                    raise ValueError("membership indices must be distinct, increasing and in range")
                previous = index
    elif status == "found":
        _keys(claim, ("predicate", "upper_bound"))
        witness = _witness(record["witness"], code, record["sector"], record["through"])
        if record["witness"]["source"] != record["source"]:
            raise ValueError("counterexample has a different source binding")
        if (claim["predicate"] != "distance_upper_bound" or _integer(claim["upper_bound"]) != witness.weight
                or record["memberships"] is not None or record["reason"] is not None):
            raise ValueError("invalid counterexample claim")
    elif status == "budget_exhausted":
        if (claim is not None or record["memberships"] is not None or record["witness"] is not None
                or type(record["reason"]) is not str or not 0 < len(record["reason"]) <= 8192):
            raise ValueError("refusal must not carry a scientific claim")
    else:
        raise ValueError("invalid low-weight status")
    return indices * max(1, same.nrows.bit_length()), witness


def check_low_weight(code, result, *, max_steps=1_000_000, max_axis=4096,
                     max_bits=100_000_000, max_indices=1_000_000):
    """Regenerate obligations and check XOR memberships or a physical counterexample.

    No rank/solve, producer invocation, or enumeration of all weight-two pairs
    occurs. A valid producer refusal returns no_evidence. A reader quota refusal
    returns budget_exhausted. Returned steps are this checker's own proxy charges.
    """
    meter, limits = options(code, max_steps, max_axis, max_bits, max_indices)
    try:
        base = base_budget(code, limits)
        coefficient_bits, witness = _record(result, code, limits, meter)
        if base + coefficient_bits + (2 * code.n if witness is not None else 0) > limits["max_bits"]:
            raise WorkLimitExceeded("low-weight proof workspace exceeds max_bits")
        subject = source_id(code)
        if result["source"] != subject or result["record_id"] != low_weight_id(result):
            raise ValueError("low-weight source binding or record identifier mismatch")
        sector = result["sector"]
        if result["status"] == "budget_exhausted":
            return LowWeightCheckResult("no_evidence", meter.used, source=subject, sector=sector)
        if witness is not None:
            meter.charge(code.hx.nrows + code.hz.nrows + 2)
            if result["witness"]["source"] != subject or not verify_css_logical(code, witness):
                raise ValueError("counterexample fails independent syndrome/dual verification")
            return LowWeightCheckResult("checked_witness", meter.used, source=subject, sector=sector,
                                         upper_bound=witness.weight)
        same, opposite = (code.hx, code.hz) if sector == "x" else (code.hz, code.hx)
        incidence, _ = columns(code, opposite, limits, meter, reserved=coefficient_bits)
        obligations = reader_targets(incidence, result["through"], meter)
        if len(obligations) != len(result["memberships"]):
            raise ValueError("membership proof does not cover every regenerated obligation")
        for sites, support in zip(obligations, result["memberships"]):
            meter.charge(1 + len(support))
            actual = 0
            for index in support:
                actual ^= same.rows[index]
            if actual != sum(1 << site for site in sites):
                raise ValueError("membership coefficients do not prove the required stabilizer identity")
        return LowWeightCheckResult("checked_exclusion", meter.used, source=subject, sector=sector,
                                     excluded_through=result["through"])
    except WorkLimitExceeded as error:
        return LowWeightCheckResult("budget_exhausted", meter.used, str(error))
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as error:
        return LowWeightCheckResult("invalid_record", meter.used, str(error))


__all__ = ["LOW_WEIGHT_SCHEMA", "LOW_WEIGHT_DOMAIN", "low_weight_id", "LowWeightCheckResult",
           "certify_low_weight", "check_low_weight"]
