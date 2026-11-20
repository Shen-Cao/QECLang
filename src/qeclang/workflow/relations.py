"""Exact relational checks for based CSS representation changes.

These checkers compare source and target data, never call the producing
conversion, and do not recompute rank or distance. Exact generator identities
imply the stated quotient/metric guarantees. Inputs belong to constructor-
validated dialects; unrelated structural admission remains the constructors'
responsibility. Decorative names have no bearing on these relations.
"""

from qeclang.algebra import BinaryMatrix, SparseBinaryMatrix
from qeclang.algebra.binary import set_bits
from qeclang.analysis.certificate_io import certificate_record as make_certificate_record
from qeclang.analysis.certificate_io import verify_certificate_record
from qeclang.analysis.certificates import LogicalWitness, verify_css_logical
from qeclang.codes import CSSCode, ChainComplex, SparseCSSCode


class RelationError(ValueError):
    """A requested source-to-target relationship failed exact verification."""


_COMMON = ("physical_qubit_count", "check_order", "check_labels", "check_weights",
           "syndrome_action", "logical_quotient", "hamming_metric")


def _matrix(matrix, sparse=False):
    expected = SparseBinaryMatrix if sparse else BinaryMatrix
    if not isinstance(matrix, expected):
        raise RelationError("matrix representation does not match its dialect")
    if type(matrix.ncols) is not int or matrix.ncols < 0 or not isinstance(matrix.rows, tuple):
        raise RelationError("malformed matrix dimensions or rows")
    for row in matrix.rows:
        if sparse:
            if not isinstance(row, tuple):
                raise RelationError("sparse support rows must be tuples")
            previous = -1
            for site in row:
                if type(site) is not int or not previous < site < matrix.ncols:
                    raise RelationError("sparse supports must be canonical in-range indices")
                previous = site
        elif type(row) is not int or row < 0 or row.bit_length() > matrix.ncols:
            raise RelationError("packed row exceeds its declared physical width")
    return matrix


def _css(code, sparse=False):
    if not isinstance(code, SparseCSSCode if sparse else CSSCode):
        raise RelationError("source or target is not in the required CSS dialect")
    hx, hz = _matrix(code.hx, sparse), _matrix(code.hz, sparse)
    if hx.ncols != hz.ncols:
        raise RelationError("CSS sectors disagree on physical width")
    return hx, hz


def _chain(chain):
    if not isinstance(chain, ChainComplex):
        raise RelationError("source or target is not a based ChainComplex")
    d1, d2 = _matrix(chain.d1), _matrix(chain.d2)
    if d1.ncols != d2.nrows:
        raise RelationError("chain boundaries disagree on the physical space")
    return d1, d2


def _same_packed(left, right):
    return left.shape == right.shape and left.rows == right.rows


def _transpose_matches(left, right):
    """Check a transpose identity without producing a transposed matrix."""
    if left.shape != (right.ncols, right.nrows):
        return False
    if sum(row.bit_count() for row in left.rows) != sum(row.bit_count() for row in right.rows):
        return False
    return all((right.rows[j] >> i) & 1 for i, row in enumerate(left.rows) for j in set_bits(row))


def _same_supports(packed, sparse):
    if packed.shape != sparse.shape:
        return False
    return all(row.bit_count() == len(support) and all((row >> j) & 1 for j in support)
               for row, support in zip(packed.rows, sparse.rows))


def _permutation(values, width):
    if not isinstance(values, (list, tuple)) or len(values) != width:
        raise RelationError("permutation must list every physical site exactly once")
    if any(type(site) is not int or not 0 <= site < width for site in values):
        raise RelationError("permutation entries must be in-range integer site indices")
    if len(set(values)) != width:
        raise RelationError("permutation is not a bijection")
    return tuple(values)


def _permuted_supports(source, target, permutation):
    if source.shape != target.shape:
        return False
    return all(old.bit_count() == new.bit_count() and
               all((new >> permutation[j]) & 1 for j in set_bits(old))
               for old, new in zip(source.rows, target.rows))


def check_relation(op, source, target, params) -> tuple[str, ...]:
    """Verify an allowed based relation and return its checked guarantees.

    Supported operations are css.chain, chain.css, css.sparse, sparse.css, and
    css.permute. Only css.permute takes parameters: ``{'permutation': [...]}``,
    where old site i maps to permutation[i]. Check order, redundancy, empty
    dimensions, and X/Z labels are significant; display names are ignored.
    General chain maps/basis changes are deliberately not accepted here.
    """
    operations = ("css.chain", "chain.css", "css.sparse", "sparse.css", "css.permute")
    if not isinstance(op, str) or op not in operations:
        raise RelationError("unsupported relational operation")
    expected_keys = {"permutation"} if op == "css.permute" else set()
    if not isinstance(params, dict) or set(params) != expected_keys:
        raise RelationError("parameters do not match the relational operation")
    if op in ("css.chain", "chain.css"):
        css, chain = (source, target) if op == "css.chain" else (target, source)
        hx, hz = _css(css)
        d1, d2 = _chain(chain)
        valid = _same_packed(hx, d1) and _transpose_matches(hz, d2)
    elif op in ("css.sparse", "sparse.css"):
        packed, sparse = (source, target) if op == "css.sparse" else (target, source)
        hx, hz = _css(packed)
        sx, sz = _css(sparse, sparse=True)
        valid = _same_supports(hx, sx) and _same_supports(hz, sz)
    else:
        hx, hz = _css(source)
        tx, tz = _css(target)
        permutation = _permutation(params["permutation"], hx.ncols)
        valid = (_permuted_supports(hx, tx, permutation)
                 and _permuted_supports(hz, tz, permutation))
    if not valid:
        raise RelationError("target does not implement the declared based operator relationship")
    basis = "qubit_bijection" if op == "css.permute" else "physical_basis_identity"
    return _COMMON + (basis,)


def transport_witness(source: CSSCode, target: CSSCode, permutation,
                      certificate_record: dict) -> dict:
    """Transport a bound CSS logical upper witness along a verified relabeling.

    Both the logical operator and its odd-pairing dual are permuted. The source
    record must verify against the exact source presentation; the returned
    record is independently verified and bound to the exact target presentation.
    This transports an upper-bound witness, not a distance lower bound.
    """
    check_relation("css.permute", source, target, {"permutation": permutation})
    permutation = _permutation(permutation, source.n)
    if (not isinstance(certificate_record, dict) or certificate_record.get("kind") != "css_logical"
            or not verify_certificate_record(source, certificate_record,
                                             max_axis=source.n, max_bits=2 * source.n)):
        raise RelationError("logical certificate is invalid or not bound to the source presentation")
    payload = certificate_record["certificate"]

    def relabel(word):
        return sum(1 << permutation[j] for j in set_bits(word))

    transported = LogicalWitness(payload["sector"], relabel(int(payload["operator"], 16)),
                                 relabel(int(payload["dual"], 16)), payload["weight"])
    if not verify_css_logical(target, transported):
        raise RelationError("transported logical witness fails verification against target")
    result = make_certificate_record(target, transported)
    if not verify_certificate_record(target, result, max_axis=target.n, max_bits=2 * target.n):
        raise RelationError("transported record failed independent target verification")
    return result


__all__ = ["RelationError", "check_relation", "transport_witness"]
