"""Data-only exact certificate records bound to a physical presentation.

These records certify rank or a logical distance upper bound. They do not
certify distance lower bounds, numerical optimality, or code equivalence.
"""
from dataclasses import asdict
from hashlib import sha256
import json
import re

from qeclang.algebra import BinaryMatrix
from qeclang.algebra.binary import nonnegative_integer, set_bits
from qeclang.codes import CSSCode, StabilizerCode
from qeclang.io import to_dict
from .certificates import (
    LogicalWitness, RankCertificate, verify_css_logical, verify_rank,
    verify_stabilizer_logical,
)

SCHEMA = "qeclang.certificate.v1"


def _matrix_record(matrix):
    return {"ncols": matrix.ncols, "rows": [hex(row) for row in matrix.rows]}


def presentation_fingerprint(source):
    """Hash a presentation, including check order/signs but excluding its name.

    This is neither a code-equivalence classifier nor an authenticity signature.
    """
    if isinstance(source, CSSCode):
        return _css_fingerprint(source)
    if isinstance(source, BinaryMatrix):
        data = {"kind": "binary_matrix", **_matrix_record(source)}
    else:
        data = to_dict(source)
        data.pop("name", None)
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _css_fingerprint(code):
    """Stream the established canonical CSS JSON identity using set-bit visits."""
    digest = sha256()
    for number, (key, matrix) in enumerate((("hx", code.hx), ("hz", code.hz))):
        digest.update(("{" if number == 0 else ",").encode() + ('"' + key + '":[').encode())
        for i, row in enumerate(matrix.rows):
            if i:
                digest.update(b",")
            digest.update(b"[")
            for j, site in enumerate(set_bits(row)):
                if j:
                    digest.update(b",")
                digest.update(str(site).encode())
            digest.update(b"]")
        digest.update(b"]")
    digest.update((',"kind":"css","n":' + str(code.n) + ',"schema":"qeclang.code.v1"}').encode())
    return digest.hexdigest()


def certificate_record(source, certificate):
    if isinstance(certificate, RankCertificate) and isinstance(source, BinaryMatrix):
        if not verify_rank(source, certificate):
            raise ValueError("invalid rank certificate")
        kind = "binary_rank"
        data = {key: _matrix_record(getattr(certificate, key)) for key in ("u", "v", "w", "z")}
        claim = {"rank": certificate.rank}
    elif isinstance(certificate, LogicalWitness):
        if isinstance(source, CSSCode) and verify_css_logical(source, certificate):
            kind = "css_logical"
        elif isinstance(source, StabilizerCode) and verify_stabilizer_logical(source, certificate):
            kind = "stabilizer_logical"
        else:
            raise ValueError("invalid logical certificate or incompatible source")
        data = asdict(certificate)
        data["operator"], data["dual"] = hex(certificate.operator), hex(certificate.dual)
        claim = {"distance_upper_bound": certificate.weight, "sector": certificate.sector}
    else:
        raise TypeError("unsupported certificate/source combination")
    return {"schema": SCHEMA, "source": presentation_fingerprint(source),
            "kind": kind, "claim": claim, "certificate": data}


def _word(value, width):
    if (not isinstance(value, str) or len(value) > max(1, (width+3)//4) + 2
            or re.fullmatch(r"0x[0-9a-f]+", value) is None):
        raise ValueError("invalid or oversized hexadecimal word")
    word = int(value, 16)
    if word.bit_length() > width:
        raise ValueError("word exceeds its declared width")
    return word


def verify_certificate_record(source, record, *, max_bits=100_000_000, max_axis=100_000):
    """Return False for malformed, oversized, mismatched or invalid evidence.

    max_bits bounds total parsed certificate matrix payload. max_axis also
    bounds empty matrix dimensions. Input JSON byte limits belong to the caller.
    """
    max_bits = nonnegative_integer(max_bits, "max_bits")
    max_axis = nonnegative_integer(max_axis, "max_axis")
    try:
        if (not isinstance(record, dict)
                or set(record) != {"schema", "source", "kind", "claim", "certificate"}
                or record["schema"] != SCHEMA
                or record["source"] != presentation_fingerprint(source)):
            return False
        budget = max_bits
        def matrix(data):
            nonlocal budget
            if not isinstance(data, dict) or set(data) != {"ncols", "rows"}:
                raise ValueError("invalid matrix record")
            n = nonnegative_integer(data["ncols"], "ncols")
            rows = data["rows"]
            if not isinstance(rows, list) or max(len(rows), n) > max_axis:
                raise ValueError("matrix axis exceeds limit")
            budget -= n * len(rows)
            if budget < 0:
                raise ValueError("matrix payload exceeds limit")
            return BinaryMatrix(tuple(_word(row, n) for row in rows), n)
        payload = record["certificate"]
        if record["kind"] == "binary_rank" and isinstance(source, BinaryMatrix):
            if not isinstance(payload, dict) or set(payload) != {"u", "v", "w", "z"}:
                return False
            cert = RankCertificate(*(matrix(payload[k]) for k in ("u", "v", "w", "z")))
            claim = record["claim"]
            return (isinstance(claim, dict) and set(claim) == {"rank"}
                    and type(claim["rank"]) is int and claim["rank"] == cert.rank
                    and verify_rank(source, cert))
        if not isinstance(payload, dict) or set(payload) != {"sector", "operator", "dual", "weight"}:
            return False
        width = source.n if record["kind"] == "css_logical" else 2 * source.n
        if width > max_axis or 2*width > max_bits:
            return False
        cert = LogicalWitness(payload["sector"], _word(payload["operator"], width),
                              _word(payload["dual"], width), payload["weight"])
        if record["claim"] != {"distance_upper_bound": cert.weight, "sector": cert.sector}:
            return False
        if type(record["claim"]["distance_upper_bound"]) is not int:
            return False
        if record["kind"] == "css_logical" and isinstance(source, CSSCode):
            return verify_css_logical(source, cert)
        if record["kind"] == "stabilizer_logical" and isinstance(source, StabilizerCode):
            return verify_stabilizer_logical(source, cert)
        return False
    except (ValueError, TypeError, KeyError, AttributeError):
        return False
