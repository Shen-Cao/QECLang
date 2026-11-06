"""Construct and decode a shared lift without materializing dense bit rows."""
from dataclasses import asdict
import json

from qeclang.constructions import LiftedProduct
from qeclang.decoding import BPDecoder
from qeclang.symbolic import GroupAlgebraElement as G, GroupAlgebraMatrix as M


def main():
    orders = (31,)
    one = G.from_terms(orders, [(0,)])
    x = G.from_terms(orders, [(1,)])
    zero = G.from_terms(orders, [])
    a = M(((one, x, zero), (zero, one, x)), 3, orders)
    b = M(((one, x),), 2, orders)
    construction = LiftedProduct(a, b)
    hx, hz = construction.sparse_checks()
    decoder = BPDecoder(hz, 0.01, max_iterations=30, fallback="osd0")
    result = decoder.decode(hz.matvec(1))
    print(json.dumps({"n": construction.n, "check_counts": construction.check_counts,
                      "locality": asdict(construction.locality()),
                      "sparse_check_nonzeros": hx.nnz + hz.nnz,
                      "decoder_status": result.status,
                      "syndrome_matched": result.success}))


if __name__ == "__main__":
    main()
