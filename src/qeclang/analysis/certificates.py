"""Small exact-arithmetic certificates, independently checkable without search.

Verification uses matrix identities or parity only, never elimination, an
optimizer, or the producing search. These Python checkers are not formal proofs
of the implementation. All coordinate conventions match the binary code IR.
"""
from dataclasses import dataclass

from qeclang.algebra import BinaryMatrix, vstack
from qeclang.algebra.binary import nonnegative_integer, validate_vector
from qeclang.codes import CSSCode, StabilizerCode


@dataclass(frozen=True)
class RankCertificate:
    """A = U V, W U = I_r, V Z = I_r certify rank(A)=r over GF(2)."""
    u: BinaryMatrix
    v: BinaryMatrix
    w: BinaryMatrix
    z: BinaryMatrix

    @property
    def rank(self):
        return self.u.ncols


def certify_rank(matrix: BinaryMatrix, *, max_bits=100_000_000):
    """Produce a factorization and one-sided inverses using exact elimination.

    The conservative payload guard includes input and auxiliary matrices before
    elimination. It excludes object overhead and does not bound elapsed time.
    """
    limit = nonnegative_integer(max_bits, "max_bits")
    m, n = matrix.shape
    bound = min(m, n)
    if m*n + 2*bound*(m+n) > limit:
        raise ValueError("rank certificate construction exceeds max_bits")
    v = matrix.row_basis()
    pivots = v.rref()[1]
    u = BinaryMatrix(tuple(sum(((row >> p) & 1) << j for j, p in enumerate(pivots))
                           for row in matrix.rows), v.nrows)
    w = BinaryMatrix(tuple(u.T.solve(1 << j) for j in range(v.nrows)), m)
    z = BinaryMatrix(tuple(v.solve(1 << j) for j in range(v.nrows)), n).T
    certificate = RankCertificate(u, v, w, z)
    if not verify_rank(matrix, certificate):
        raise ArithmeticError("constructed rank certificate failed verification")
    return certificate


def verify_rank(matrix: BinaryMatrix, certificate: RankCertificate):
    """Verify rank with multiplication only, including rank-zero/empty cases."""
    if not isinstance(matrix, BinaryMatrix) or not isinstance(certificate, RankCertificate):
        return False
    u, v, w, z = certificate.u, certificate.v, certificate.w, certificate.z
    if not all(isinstance(value, BinaryMatrix) for value in (u, v, w, z)):
        return False
    r = u.ncols
    if (r > min(matrix.shape) or u.shape != (matrix.nrows, r)
            or v.shape != (r, matrix.ncols) or w.shape != (r, matrix.nrows)
            or z.shape != (matrix.ncols, r)):
        return False
    return u @ v == matrix and w @ u == BinaryMatrix.identity(r) and v @ z == BinaryMatrix.identity(r)


@dataclass(frozen=True)
class LogicalWitness:
    """An operator and dual commuting operator with odd logical pairing.

    sector is x/z for CSS, or pauli for a stabilizer. General Pauli vectors
    use x | (z << n). ``weight`` is physical site weight, not packed bit count.
    """
    sector: str
    operator: int
    dual: int
    weight: int


def certify_css_logical(code: CSSCode, sector: str, operator: int):
    if sector not in ("x", "z"):
        raise ValueError("sector must be x or z")
    operator = validate_vector(operator, code.n)
    same, opposite = (code.hx, code.hz) if sector == "x" else (code.hz, code.hx)
    if opposite.matvec(operator):
        raise ValueError("operator has nonzero syndrome")
    equations = vstack(same, BinaryMatrix((operator,), code.n))
    try:
        dual = equations.solve(1 << same.nrows)
    except ValueError as exc:
        raise ValueError("operator is a stabilizer, not a nontrivial logical") from exc
    return LogicalWitness(sector, operator, dual, operator.bit_count())


def verify_css_logical(code: CSSCode, certificate: LogicalWitness):
    """Check a distance upper-bound witness using only syndrome and parity."""
    if not isinstance(certificate, LogicalWitness) or certificate.sector not in ("x", "z"):
        return False
    try:
        operator = validate_vector(certificate.operator, code.n)
        dual = validate_vector(certificate.dual, code.n)
    except (ValueError, TypeError):
        return False
    same, opposite = ((code.hx, code.hz) if certificate.sector == "x" else (code.hz, code.hx))
    return (type(certificate.weight) is int and certificate.weight == operator.bit_count()
            and opposite.matvec(operator) == 0 and same.matvec(dual) == 0
            and (operator & dual).bit_count() % 2 == 1)


def _symplectic_dual(vector, n):
    return (vector >> n) | ((vector & ((1 << n)-1)) << n)


def certify_stabilizer_logical(code: StabilizerCode, operator: int):
    operator = validate_vector(operator, 2*code.n)
    constraints = BinaryMatrix(tuple(_symplectic_dual(v, code.n) for v in code.checks.rows), 2*code.n)
    if constraints.matvec(operator):
        raise ValueError("operator has nonzero syndrome")
    equations = vstack(constraints, BinaryMatrix((_symplectic_dual(operator, code.n),), 2*code.n))
    try:
        dual = equations.solve(1 << constraints.nrows)
    except ValueError as exc:
        raise ValueError("operator is a stabilizer, not a nontrivial logical") from exc
    weight = ((operator & ((1 << code.n)-1)) | (operator >> code.n)).bit_count()
    return LogicalWitness("pauli", operator, dual, weight)


def verify_stabilizer_logical(code: StabilizerCode, certificate: LogicalWitness):
    if not isinstance(certificate, LogicalWitness) or certificate.sector != "pauli":
        return False
    try:
        op = validate_vector(certificate.operator, 2*code.n)
        dual = validate_vector(certificate.dual, 2*code.n)
    except (ValueError, TypeError):
        return False
    weight = ((op & ((1 << code.n)-1)) | (op >> code.n)).bit_count()
    constraints = BinaryMatrix(tuple(_symplectic_dual(v, code.n) for v in code.checks.rows), 2*code.n)
    return (type(certificate.weight) is int and certificate.weight == weight
            and constraints.matvec(op) == constraints.matvec(dual) == 0
            and (_symplectic_dual(op, code.n) & dual).bit_count() % 2 == 1)
