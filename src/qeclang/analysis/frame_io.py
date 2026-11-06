"""Portable logical frames with independently checked sparse completeness evidence."""
from hashlib import sha256
import json

from qeclang.algebra import BinaryMatrix, SparseBinaryMatrix
from qeclang.algebra.binary import nonnegative_integer
from qeclang.codes import CSSCode, SparseCSSCode
from .certificate_io import _matrix_record, _word
from .certificates import RankCertificate, certify_rank, verify_sparse_rank
from .frames import CSSLogicalFrame

SCHEMA = "qeclang.logical_frame.v1"


def _fingerprint(code):
    data = {"n": code.n, "hx": code.hx.rows, "hz": code.hz.rows}
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _check_source(code, max_axis, max_nonzeros):
    if not isinstance(code, SparseCSSCode):
        raise TypeError("frame import requires a validated SparseCSSCode presentation")
    if max(code.n, code.hx.nrows, code.hz.nrows) > max_axis:
        raise ValueError("frame presentation exceeds max_axis")
    if code.hx.nnz + code.hz.nnz > max_nonzeros:
        raise ValueError("frame presentation exceeds max_nonzeros")


def _identity_pairing(xs, zs, max_work):
    work = xs.nrows + zs.nrows + xs.nnz + zs.nnz
    if work > max_work:
        raise ValueError("logical pairing validation exceeds max_work")
    incidence = {}
    for i, support in enumerate(zs.rows):
        for site in support:
            incidence.setdefault(site, []).append(i)
    for support in xs.rows:
        for site in support:
            work += len(incidence.get(site, ()))
            if work > max_work:
                raise ValueError("logical pairing validation exceeds max_work")
    for i, support in enumerate(xs.rows):
        odd = set()
        for site in support:
            for j in incidence.get(site, ()):
                if j in odd:
                    odd.remove(j)
                else:
                    odd.add(j)
        if odd != {i}:
            return False
    return True


def import_logical_frame(code: SparseCSSCode, record, *, max_bits=100_000_000,
                         max_work=10_000_000, max_axis=100_000,
                         max_nonzeros=10_000_000) -> CSSLogicalFrame:
    """Validate a data-only record against sparse checks, with no elimination.

    Two rank factorizations prove k=n-rank(Hx)-rank(Hz); exact commutation and
    identity logical pairing then establish a complete quotient basis. Rank
    products use one temporary packed row at a time, never a packed source.

    ``max_bits`` bounds the combined dense certificate payload. ``max_nonzeros``
    separately bounds source incidences and parsed logical incidences; max_axis
    bounds all dimensions. ``max_work`` applies to each of five validation
    stages (two ranks, two commutation checks, one pairing check). These limits
    exclude object overhead and bigint operation costs. Input JSON byte limits
    must be enforced before parsing. Malformed, tampered, or oversized evidence
    raises ValueError; no partially validated frame is returned.
    """
    max_bits = nonnegative_integer(max_bits, "max_bits")
    max_work = nonnegative_integer(max_work, "max_work")
    max_axis = nonnegative_integer(max_axis, "max_axis")
    max_nonzeros = nonnegative_integer(max_nonzeros, "max_nonzeros")
    _check_source(code, max_axis, max_nonzeros)
    fields = {"schema", "source", "n", "k", "rank_x", "rank_z", "x_logicals", "z_logicals"}
    if not isinstance(record, dict) or set(record) != fields or record["schema"] != SCHEMA:
        raise ValueError("invalid logical frame schema or fields")
    if (type(record["n"]) is not int or record["n"] != code.n
            or type(record["k"]) is not int or not 0 <= record["k"] <= code.n
            or record["source"] != _fingerprint(code)):
        raise ValueError("logical frame source or dimension mismatch")
    remaining = max_bits
    def matrix(data):
        nonlocal remaining
        if not isinstance(data, dict) or set(data) != {"ncols", "rows"}:
            raise ValueError("invalid rank matrix record")
        ncols = nonnegative_integer(data["ncols"], "ncols")
        rows = data["rows"]
        if not isinstance(rows, list) or max(ncols, len(rows)) > max_axis:
            raise ValueError("rank matrix exceeds max_axis")
        remaining -= ncols * len(rows)
        if remaining < 0:
            raise ValueError("logical frame certificate payload exceeds max_bits")
        return BinaryMatrix(tuple(_word(row, ncols) for row in rows), ncols)
    def certificate(data):
        if not isinstance(data, dict) or set(data) != {"u", "v", "w", "z"}:
            raise ValueError("invalid rank certificate fields")
        return RankCertificate(*(matrix(data[key]) for key in ("u", "v", "w", "z")))
    cx, cz = certificate(record["rank_x"]), certificate(record["rank_z"])
    for source, proof in ((code.hx, cx), (code.hz, cz)):
        if not verify_sparse_rank(source, proof, max_bits=max_bits,
                                   max_work=max_work, max_axis=max_axis):
            raise ValueError("invalid sparse rank certificate")
    k = code.n - cx.rank - cz.rank
    if record["k"] != k:
        raise ValueError("logical frame dimension disagrees with certified ranks")
    nonzeros = 0
    def logicals(rows):
        nonlocal nonzeros
        if not isinstance(rows, list) or len(rows) != k:
            raise ValueError("logical frame must contain exactly k representatives per sector")
        for row in rows:
            if not isinstance(row, list):
                raise ValueError("logical support must be a list")
            nonzeros += len(row)
            if nonzeros > max_nonzeros:
                raise ValueError("logical supports exceed max_nonzeros")
        return SparseBinaryMatrix.from_supports(rows, code.n)
    xs, zs = logicals(record["x_logicals"]), logicals(record["z_logicals"])
    SparseCSSCode(code.hz, xs, max_validation_work=max_work)
    SparseCSSCode(code.hx, zs, max_validation_work=max_work)
    if not _identity_pairing(xs, zs, max_work):
        raise ValueError("logical frame pairing must be the identity")
    frame = object.__new__(CSSLogicalFrame)
    object.__setattr__(frame, "presentation", code)
    object.__setattr__(frame, "x_logicals", xs)
    object.__setattr__(frame, "z_logicals", zs)
    return frame


def export_logical_frame(code: CSSCode, frame: CSSLogicalFrame | None = None, *,
                         max_bits=100_000_000, max_work=10_000_000,
                         max_axis=100_000, max_nonzeros=10_000_000) -> dict:
    """Produce a JSON-compatible frame record and exact rank evidence.

    Evidence generation explicitly requires the packed backend. Verification
    during later import does not. Supplied frames must match the retained check
    presentation; names do not affect identity. No executable objects are stored.
    """
    if not isinstance(code, CSSCode):
        raise TypeError("frame export requires an explicitly packed CSSCode")
    max_bits = nonnegative_integer(max_bits, "max_bits")
    max_work = nonnegative_integer(max_work, "max_work")
    max_axis = nonnegative_integer(max_axis, "max_axis")
    max_nonzeros = nonnegative_integer(max_nonzeros, "max_nonzeros")
    if max(code.n, code.hx.nrows, code.hz.nrows) > max_axis:
        raise ValueError("frame presentation exceeds max_axis")
    if frame is None:
        frame = CSSLogicalFrame.from_packed(code, max_bits=max_bits,
                                            max_validation_work=max_work)
    elif not isinstance(frame, CSSLogicalFrame) or not frame.matches(code):
        raise ValueError("frame must be validated and match the CSS presentation")
    _check_source(frame.presentation, max_axis, max_nonzeros)
    if frame.x_logicals.nnz + frame.z_logicals.nnz > max_nonzeros:
        raise ValueError("logical supports exceed max_nonzeros")
    payload = sum(2 * h.rank() * (h.nrows + h.ncols) for h in (code.hx, code.hz))
    if payload > max_bits:
        raise ValueError("logical frame certificate payload exceeds max_bits")
    certificates = [certify_rank(h, max_bits=max_bits) for h in (code.hx, code.hz)]
    def evidence(proof):
        return {key: _matrix_record(getattr(proof, key)) for key in ("u", "v", "w", "z")}
    record = {"schema": SCHEMA, "source": _fingerprint(frame.presentation),
              "n": code.n, "k": frame.k,
              "rank_x": evidence(certificates[0]), "rank_z": evidence(certificates[1]),
              "x_logicals": [list(row) for row in frame.x_logicals.rows],
              "z_logicals": [list(row) for row in frame.z_logicals.rows]}
    import_logical_frame(frame.presentation, record, max_bits=max_bits, max_work=max_work,
                         max_axis=max_axis, max_nonzeros=max_nonzeros)
    return record


__all__ = ["export_logical_frame", "import_logical_frame"]
