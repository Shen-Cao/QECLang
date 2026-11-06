"""Logical operator bases over GF(2), with physical coordinates retained."""
from dataclasses import dataclass

from qeclang.algebra import BinaryMatrix
from qeclang.codes import CSSCode, StabilizerCode


def quotient_basis(space: BinaryMatrix, subspace: BinaryMatrix) -> tuple[int, ...]:
    """Extend a subspace basis to a space basis; return quotient representatives."""
    if space.ncols != subspace.ncols:
        raise ValueError("ambient dimensions differ")
    basis = space.row_basis()
    if any(not basis.contains(row) for row in subspace.rows):
        raise ValueError("subspace is not contained in space")
    span = subspace.row_basis()
    representatives = []
    for row in basis.rows:
        if not span.contains(row):
            representatives.append(row)
            span = BinaryMatrix(span.rows + (row,), span.ncols).row_basis()
    return tuple(representatives)


@dataclass(frozen=True)
class CSSLogicals:
    """Paired representatives: x[i] dot z[j] = delta_ij over GF(2)."""
    x: tuple[int, ...]
    z: tuple[int, ...]


def css_logicals(code: CSSCode) -> CSSLogicals:
    xs = list(quotient_basis(code.hz.nullspace(), code.hx))
    zs = list(quotient_basis(code.hx.nullspace(), code.hz))
    paired_x, paired_z = [], []
    while xs:
        x = xs.pop(0)
        index = next(j for j, z in enumerate(zs) if (x & z).bit_count() % 2)
        z = zs.pop(index)
        xs = [v ^ x if (v & z).bit_count() % 2 else v for v in xs]
        zs = [v ^ z if (v & x).bit_count() % 2 else v for v in zs]
        paired_x.append(x)
        paired_z.append(z)
    return CSSLogicals(tuple(paired_x), tuple(paired_z))


def symplectic(a: int, b: int, n: int) -> int:
    mask = (1 << n) - 1
    return (((a & mask) & (b >> n)).bit_count()
            + ((a >> n) & (b & mask)).bit_count()) % 2


def stabilizer_logicals(code: StabilizerCode) -> tuple[tuple[int, int], ...]:
    """Return symplectically paired logicals, packed as x | (z << n)."""
    vectors = list(quotient_basis(code.centralizer(), code.checks))
    pairs = []
    while vectors:
        a = vectors.pop(0)
        index = next(i for i, b in enumerate(vectors) if symplectic(a, b, code.n))
        b = vectors.pop(index)
        vectors = [v ^ (a if symplectic(v, b, code.n) else 0)
                   ^ (b if symplectic(v, a, code.n) else 0) for v in vectors]
        pairs.append((a, b))
    return tuple(pairs)
