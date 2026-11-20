"""Measure fresh packed/structured work and independent inspection; emit JSON Lines."""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter_ns

from qeclang.constructions import records
from qeclang.provenance import runtime_provenance
from qeclang.workflow import ExecutionLimits, InspectionLimits, Program, execute, inspect_run
from qeclang.workflow.artifacts import describe


def template(lift):
    def cell(*exponents):
        return [[e] for e in exponents]
    return {'schema': records.SCHEMA, 'kind': 'abelian_lifted_product', 'coefficient_field': 'GF(2)',
            'orders': [lift], 'action': records.ACTION, 'convention': records.CONVENTION,
            'physical_ordering': records.ORDERING,
            'a': {'shape': [2, 3], 'entries': [[cell(0, 1), cell(0), cell()],
                                              [cell(), cell(0, 1), cell(1)]]},
            'b': {'shape': [1, 2], 'entries': [[cell(0, 1), cell(0, 2)]]}}


def program(source, steps, *, materialize=False):
    nodes = [{'id': 'seed', 'op': 'lift.literal', 'inputs': [], 'params': {'construction': source}},
             {'id': 'parameters', 'op': 'lift.parameters', 'inputs': ['seed'],
              'params': {'backend': 'polynomial', 'max_steps': steps, 'record_operations': True}}]
    if materialize:
        nodes.append({'id': 'css', 'op': 'lift.css', 'inputs': ['seed'], 'params': {}})
    return Program.from_dict({'schema': 'qeclang.program.v1', 'nodes': nodes,
                              'outputs': ['parameters', 'css'] if materialize else ['parameters']})


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def payload(run):
    require(run['status'] == 'completed', f"structured execution failed: {run['status']}")
    return run['artifacts'][run['outputs']['parameters']]['payload']


def byte_size(value):
    return len(records.canonical_json(value).encode())


def benchmark(lift, args, limits, inspection):
    source = template(lift)
    parsed = program(source, limits.max_rank_steps)
    expected = dict(zip(('n', 'rank_x', 'rank_z', 'k'), (8*lift, 4*lift, 3*lift-1, lift+1)))
    def packed():
        code = records.from_dict(source).lower(max_bits=limits.max_matrix_bits,
                                               max_sites=limits.max_axis)
        rx, rz = code.hx.rank(), code.hz.rank()
        return code, {'n': code.n, 'rank_x': rx, 'rank_z': rz, 'k': code.n-rx-rz}
    reference_code, values = packed()
    require(values == expected, 'packed parameters disagree with the fixture identities')
    reference_run = execute(parsed, limits=limits)
    reported = payload(reference_run)
    require(reported['status'] == 'exact' and all(reported[k] == v for k, v in expected.items()),
            'structured and packed parameters disagree')
    reference_inspection = inspect_run(parsed, reference_run, limits=inspection)
    require(reference_inspection['status'] == 'checked', 'independent inspection failed')
    claims = reference_inspection['claims']
    require(len(claims) == 1 and claims[0]['predicate'] == 'logical_dimension' and
            claims[0]['source'] == records.digest(source) and claims[0]['value'] == expected['k'] and
            all(claims[0][k] == expected[k] for k in ('rank_x', 'rank_z')), 'dimension claim mismatch')
    calls = {'packed_lower_and_rank': packed, 'structured_execute': lambda: execute(parsed, limits=limits),
             'independent_inspection': lambda: inspect_run(parsed, reference_run, limits=inspection)}
    def validate(name, value):
        if name == 'packed_lower_and_rank':
            require(value[1] == expected and value[0] == reference_code, 'fresh packed result changed')
        else:
            require(value == (reference_run if name == 'structured_execute' else reference_inspection),
                    f'fresh {name} result changed')
    for _ in range(args.warmups):
        for name, call in calls.items():
            validate(name, call())
    observations, rng = [], Random(args.order_seed + lift)
    for repetition in range(args.repeats):
        order = list(calls)
        rng.shuffle(order)
        for name in order:
            start = perf_counter_ns()
            value = calls[name]()
            elapsed = (perf_counter_ns()-start)/1_000_000_000
            validate(name, value)
            observations.append({'repetition': repetition, 'mode': name, 'seconds': elapsed})
    small = replace(limits, max_matrix_bits=args.packed_cap)
    small_reader = replace(inspection, artifacts=replace(inspection.artifacts, max_matrix_bits=args.packed_cap))
    symbolic = execute(parsed, limits=small)
    refused = execute(program(source, limits.max_rank_steps, materialize=True), limits=small)
    checked = inspect_run(parsed, symbolic, limits=small_reader)
    require(payload(symbolic)['k'] == expected['k'] and checked['status'] == 'checked' and
            checked['claims'] == reference_inspection['claims'],
            'symbolic analysis failed under the small packed cap')
    require(refused['status'] == 'budget_exhausted' and refused['trace'][-1]['node'] == 'css' and
            refused['trace'][-1]['resource'] == 'matrix_bits', 'packed lowering did not refuse as intended')
    return {'lift': lift, 'case': 'repeated_root_even' if lift % 2 == 0 else 'odd_cyclic',
            'parameters': expected, 'program': parsed.to_dict(), 'program_fingerprint': parsed.fingerprint(),
            'median_seconds': {name: median(o['seconds'] for o in observations if o['mode'] == name) for name in calls},
            'observations': observations, 'packed_css_artifact_bytes': byte_size(describe('css', reference_code)),
            'structured_artifact_bytes': byte_size(reference_run['artifacts']),
            'structured_run_bytes': byte_size(reference_run), 'inspection_bytes': byte_size(reference_inspection),
            'inspection': reference_inspection, 'ledger': reference_run['ledger'],
            'refusal': {'packed_cap': args.packed_cap, 'symbolic_status': symbolic['status'],
                        'inspection_status': checked['status'], 'materialization_status': refused['status'],
                        'step': refused['trace'][-1]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lifts', type=int, nargs='+', default=[7, 31, 63])
    parser.add_argument('--even-lift', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--order-seed', type=int, default=17)
    parser.add_argument('--packed-cap', type=int, default=1024)
    args = parser.parse_args()
    if (args.repeats < 1 or args.warmups < 0 or args.packed_cap < 0 or args.even_lift % 2 or
            any(lift < 3 or lift > 127 for lift in args.lifts + [args.even_lift])):
        parser.error('require repeats>0, warmups/cap>=0, lifts in [3,127], and an even --even-lift')
    runtime, limits, inspection = runtime_provenance(), ExecutionLimits(), InspectionLimits()
    results = [benchmark(lift, args, limits, inspection) for lift in args.lifts + [args.even_lift]]
    require(runtime_provenance() == runtime, 'source/runtime changed during measurement')
    for result in results:
        print(json.dumps({'schema': 'qeclang.lift-workflow.benchmark.v1', 'runtime': runtime,
                          'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                          'configuration': vars(args), 'execution_limits': asdict(limits),
                          'inspection_limits': asdict(inspection), **result}, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
