"""Check the physical action of a shared lift directly from its seed translations."""
from qeclang.codes import CSSCode
from qeclang.constructions import LiftedProduct
from .relations import RelationError


def comparison_units(lift):
    """Upper proxy for base-entry visits and coordinate operations on their terms."""
    a, b = lift.a, lift.b
    mx, mz = lift.check_counts
    profile = lift.locality()
    return (mx*(a.ncols + b.nrows + 1) + mz*(b.ncols + a.nrows + 1)
            + (profile.x_nonzeros + profile.z_nonzeros)*(len(a.orders)+1))


def _translate(site, term, orders, sign):
    result, stride = 0, 1
    for modulus, shift in zip(reversed(orders), reversed(term)):
        site, digit = divmod(site, modulus)
        result += ((digit + sign*shift) % modulus) * stride
        stride *= modulus
    return result


def _row_matches(row, blocks, site, orders, length):
    # Different base columns occupy disjoint blocks; canonical monomials within a
    # block map one site to distinct columns. Equal cardinality rules out extras.
    weight = 0
    for block, terms, sign in blocks:
        weight += len(terms)
        for term in terms:
            column = block*length + _translate(site, term, orders, sign)
            if not (row >> column) & 1:
                return False
    return row.bit_count() == weight


def check_lift(lift, target):
    """Verify Hx,Hz membership and weights in original VV/CC site order.

    Avoid construction of symbolic check matrices, packed/sparse lowering, rank,
    and distance. The caller bounds source, target, and comparison work.
    """
    if not isinstance(lift, LiftedProduct) or not isinstance(target, CSSCode):
        raise RelationError('lift.css requires a lifted product and a packed CSS target')
    mx, mz = lift.check_counts
    if target.hx.shape != (mx, lift.n) or target.hz.shape != (mz, lift.n):
        raise RelationError('lifted check dimensions disagree with the seeds')
    a, b = lift.a, lift.b
    vv = a.ncols*b.ncols
    length = a.group_size

    def x_blocks(i, j):
        for v in range(a.ncols):
            yield v*b.ncols+j, a.rows[i][v].terms, 1
        for c in range(b.nrows):
            yield vv+i*b.nrows+c, b.rows[c][j].terms, -1

    def z_blocks(v, c):
        for j in range(b.ncols):
            yield v*b.ncols+j, b.rows[c][j].terms, 1
        for i in range(a.nrows):
            yield vv+i*b.nrows+c, a.rows[i][v].terms, -1

    for matrix, width, base_blocks in ((target.hx, b.ncols, x_blocks),
                                       (target.hz, b.nrows, z_blocks)):
        for i, row in enumerate(matrix.rows):
            base, site = divmod(i, length)
            blocks = base_blocks(*divmod(base, width))
            if not _row_matches(row, blocks, site, a.orders, length):
                raise RelationError('lifted support differs from the specified regular action')
    return ('physical_site_order', 'x_and_z_check_labels', 'ordered_check_supports',
            'regular_group_action', 'check_weights')
