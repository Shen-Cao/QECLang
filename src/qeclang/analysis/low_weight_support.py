"""Bounded support extraction and data identities for low-weight exclusion."""
from hashlib import sha256
import json

from qeclang.algebra.binary import nonnegative_integer, set_bits
from qeclang.algebra.polynomial import ArithmeticBudget, WorkLimitExceeded
from qeclang.codes import CSSCode
from .certificate_io import presentation_fingerprint

LOW_WEIGHT_SCHEMA = "qeclang.low_weight.v1"
LOW_WEIGHT_DOMAIN = b"qeclang.low_weight.v1\x00"
FIELDS = frozenset(("schema", "record_id", "source", "sector", "through", "status", "steps",
                    "claim", "memberships", "witness", "reason"))


def low_weight_id(record):
    """Hash canonical record content except record_id; this does not verify it."""
    digest = sha256(LOW_WEIGHT_DOMAIN)
    body = {key: value for key, value in record.items() if key != "record_id"}
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    for chunk in encoder.iterencode(body):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def source_id(code):
    """Use the shared streaming identity, including check order and redundancy."""
    return presentation_fingerprint(code)


def options(code, max_steps, max_axis, max_bits, max_indices):
    if not isinstance(code, CSSCode):
        raise TypeError("low-weight analysis requires an explicitly packed positive CSSCode")
    if code.n.bit_length() > 4096:
        raise ValueError("source dimension exceeds the portable fingerprint integer limit")
    limits = {name: nonnegative_integer(value, name) for name, value in
              (("max_steps", max_steps), ("max_axis", max_axis),
               ("max_bits", max_bits), ("max_indices", max_indices))}
    return ArithmeticBudget(limits["max_steps"]), limits


def request(sector, through):
    if sector not in ("x", "z") or type(through) is not int or through not in (1, 2):
        raise ValueError("low-weight requests require sector x/z and through 1 or 2")


def base_budget(code, limits):
    n, mx, mz = code.n, code.hx.nrows, code.hz.nrows
    if max(n, mx, mz) > limits["max_axis"]:
        raise WorkLimitExceeded("low-weight input exceeds max_axis")
    cost = n * (mx + mz)
    if cost > limits["max_bits"]:
        raise WorkLimitExceeded("low-weight input exceeds max_bits")
    return cost


def columns(code, opposite, limits, meter, reserved=0):
    """Sparse incidence lists; no packed transpose or weight-pair enumeration."""
    cost = base_budget(code, limits)
    meter.charge(opposite.nrows)
    nonzeros = sum(row.bit_count() for row in opposite.rows)
    width = max(1, code.n.bit_length(), opposite.nrows.bit_length())
    cost += (3 * code.n + 2 * nonzeros) * width
    if cost + reserved > limits["max_bits"]:
        raise WorkLimitExceeded("column incidence workspace exceeds max_bits")
    meter.charge(code.n)
    result = [[] for _ in range(code.n)]
    for check, row in enumerate(opposite.rows):
        for site in set_bits(row):
            meter.charge()
            result[site].append(check)
    return result, cost


def producer_targets(columns, through, meter):
    groups = {}
    for site, column in enumerate(columns):
        meter.charge(1 + len(column))
        groups.setdefault(tuple(column), []).append(site)
    for site in groups.pop((), ()):
        meter.charge()
        yield 1 << site
    if through == 2:
        for sites in groups.values():
            representative = sites[0]
            for site in sites[1:]:
                meter.charge()
                yield (1 << site) | (1 << representative)


def reader_targets(columns, through, meter):
    """Regenerate all obligations by a separate representative/pair traversal."""
    zeros, pairs, representatives = [], [], {}
    for site, column in enumerate(columns):
        meter.charge(1 + (len(column) if through == 2 else 0))
        if not column:
            zeros.append(site)
        elif through == 2:
            signature = tuple(column)
            if signature in representatives:
                pairs.append((representatives[signature], site))
            else:
                representatives[signature] = site
    meter.charge(len(pairs) * max(1, len(pairs).bit_length()))
    pairs.sort()
    return [(site,) for site in zeros] + [(representative, site) for representative, site in pairs]


def assemble(source, sector, through, status, steps, *, memberships=None, witness=None, reason=None):
    claim = ({"predicate": "nontrivial_weight_exclusion", "through": through} if status == "excluded"
             else {"predicate": "distance_upper_bound", "upper_bound": witness["claim"]["distance_upper_bound"]}
             if status == "found" else None)
    result = {"schema": LOW_WEIGHT_SCHEMA, "source": source, "sector": sector, "through": through,
              "status": status, "steps": steps, "claim": claim, "memberships": memberships,
              "witness": witness, "reason": reason}
    result["record_id"] = low_weight_id(result)
    return result
