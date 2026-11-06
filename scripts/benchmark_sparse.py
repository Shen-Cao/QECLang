"""Compare sparse/packed syndrome products with explicit reproducible parameters."""
import argparse
from dataclasses import asdict
import json
from random import Random
from time import perf_counter

from qeclang.algebra import SparseBinaryMatrix
from qeclang.decoding.belief_propagation import BPDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if not 0 <= args.degree <= args.size or args.size < 1:
        parser.error("require 0 <= degree <= size and size >= 1")
    rng = Random(args.seed)
    sparse = SparseBinaryMatrix.from_supports(
        (rng.sample(range(args.size), args.degree) for _ in range(args.size)), args.size)
    bits = tuple(int(rng.random() < 0.01) for _ in range(args.size))
    packed = sparse.to_packed()
    word = sum(bit << j for j, bit in enumerate(bits))
    start = perf_counter()
    syndrome_bits = sparse.matvec_bits(bits)
    sparse_time = perf_counter() - start
    start = perf_counter()
    syndrome = packed.matvec(word)
    packed_time = perf_counter() - start
    assert syndrome == sum(bit << j for j, bit in enumerate(syndrome_bits))
    decoder = BPDecoder(sparse, 0.01, max_iterations=args.iterations)
    start = perf_counter()
    decoded = decoder.decode(syndrome)
    elapsed = perf_counter() - start
    summary = asdict(decoded)
    summary.pop("posterior_llrs")
    summary.pop("last_decision")
    summary.pop("error")
    print(json.dumps({"parameters": vars(args), "nonzeros": sparse.nnz,
                      "sparse_product_seconds": sparse_time,
                      "packed_product_seconds": packed_time,
                      "bp_seconds": elapsed, "bp": summary}))


if __name__ == "__main__":
    main()
