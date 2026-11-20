"""Compare independent recomputation and reuse of a CSS upper witness on the same target."""
import argparse
from dataclasses import asdict
import json
from random import Random
from statistics import median
from time import perf_counter_ns

from qeclang.algebra import BinaryMatrix
from qeclang.archive import InMemoryEvidenceStore, publish
from qeclang.archive.reuse import make_reuse_program
from qeclang.analysis import css_distance
from qeclang.analysis.certificates import certify_css_logical, verify_css_logical
from qeclang.codes import CSSCode
from qeclang.constructions import steane, toric
from qeclang.io import to_dict
from qeclang.provenance import runtime_provenance
from qeclang.workflow import ExecutionLimits, Program, execute, inspect_run
from qeclang.workflow.row_equivalence import row_maps, verify_row_equivalence


def source_program(code, weight, candidates):
    return Program.from_dict({'schema': 'qeclang.program.v1', 'nodes': [
        {'id': 'source', 'op': 'css.literal', 'inputs': [], 'params': {'code': to_dict(code)}},
        {'id': 'distance', 'op': 'css.distance', 'inputs': ['source'],
         'params': {'sector': 'x', 'max_weight': weight, 'max_candidates': candidates}},
        {'id': 'proof', 'op': 'css.witness', 'inputs': ['source', 'distance'], 'params': {}}
    ], 'outputs': ['proof']})


def measure(case, code, repeats, seed):
    runtime = runtime_provenance()
    def augment(matrix):
        return BinaryMatrix(tuple(reversed(matrix.rows)) + (matrix.rows[0] ^ matrix.rows[1],), matrix.ncols)
    target = CSSCode(augment(code.hx), augment(code.hz))
    limits = ExecutionLimits(max_candidates=100_000)
    src = source_program(code, 3, limits.max_candidates)
    original = execute(src, limits=limits)
    store = InMemoryEvidenceStore()
    receipt = publish(src, original, store)
    key = receipt['bundle_id']
    reference_weight = receipt['inspection']['claims'][-1]['bound']
    maps = row_maps(code, target)

    def direct(cap):
        instance = CSSCode(BinaryMatrix(target.hx.rows, target.n), BinaryMatrix(target.hz.rows, target.n))
        result = css_distance(instance, 'x', max_weight=3, max_candidates=cap)
        witness = certify_css_logical(instance, 'x', result.witness)
        if not verify_css_logical(instance, witness):
            raise ArithmeticError('invalid direct upper witness')
        return {'upper_bound': witness.weight, 'exclusions_through': result.searched_through,
                'candidates': result.candidates}

    def reuse():
        program = make_reuse_program(store, key, 'proof', target, maps)
        result = execute(program, limits=ExecutionLimits(max_candidates=0))
        inspection = inspect_run(program, result)
        if inspection['status'] != 'checked' or not inspection['claims']:
            raise ArithmeticError('reused output failed verification')
        upper = inspection['claims'][-1]['bound']
        if upper != reference_weight or result['ledger']['used']['candidates'] != 0:
            raise ArithmeticError('evidence was not reused with the expected bound and budget')
        return {'upper_bound': upper, 'candidates': 0, 'row_map_work': result['ledger']['used']['row_map_work']}

    def derive_maps():
        derived = row_maps(code, target)
        verify_row_equivalence(code, target, derived)
        return {'reference_indices': sum(len(row) for rows in derived.values() for row in rows)}

    phases = {'fresh_direct_witness': lambda: direct(0),
              'fresh_bounded_distance': lambda: direct(limits.max_candidates),
              'derive_and_check_maps': derive_maps, 'read_rebind_and_inspect': reuse}
    observations = {name: [] for name in phases}
    results = {name: function() for name, function in phases.items()}
    rng = Random(seed)
    for _ in range(repeats):
        order = list(phases)
        rng.shuffle(order)
        for name in order:
            start = perf_counter_ns()
            value = phases[name]()
            observations[name].append(perf_counter_ns()-start)
            if value != results[name]:
                raise ArithmeticError('phase result changed across repeats')
    if runtime_provenance() != runtime:
        raise RuntimeError('package contents or runtime changed during measurement')
    return {'case': case, 'n': code.n, 'k': code.k, 'runtime': runtime,
            'repeats': repeats, 'phase_order_seed': seed, 'execution_limits': asdict(limits),
            'initial_search_candidates': original['ledger']['used']['candidates'],
            'results': results, 'nanoseconds': observations,
            'medians_ns': {k: median(v) for k, v in observations.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--order-seed', type=int, default=17)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error('repeats must be between 1 and 100')
    selected = CSSCode(BinaryMatrix((116, 224, 79), 8), BinaryMatrix((205, 126, 166), 8))
    for name, code in [('steane', steane()), ('toric4', toric(4)), ('selected8', selected)]:
        print(json.dumps(measure(name, code, args.repeats, args.order_seed), allow_nan=False))


if __name__ == '__main__':
    main()
