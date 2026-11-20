"""Bounded qldpc==0.4.1 differential observations, never foreign proof claims.

Each foreign task runs in a separate process with a timeout. Sector distances
use fresh generic CSS objects, no supplied distance/equality promises, cutoff=0
and method='brute_force'. JIT/import/cache effects make timings diagnostic only.
"""
import argparse
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from importlib import metadata
import inspect
import json
from math import isfinite, isnan
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from qeclang.analysis import css_distance
from qeclang.analysis.certificate_io import presentation_fingerprint
from qeclang.catalog.references import build_reference, reference_cases
from qeclang.codes import CSSCode, same_css_presentation
from qeclang.io import from_dict, to_dict
from qeclang.provenance import runtime_provenance
from qeclang.transforms import permute_qubits
from qeclang.workflow.row_equivalence import row_maps, verify_row_equivalence

SCHEMA = 'qeclang.qldpc-comparison.v1'
PINNED_VERSION = '0.4.1'
FAMILIES = ('steane', 'hgp_open_2_3', 'toric_2', 'bicycle_cyclic_3')


@dataclass(frozen=True)
class ComparisonLimits:
    max_n: int = 13
    max_checks: int = 64
    max_elements: int = 1_000_000
    max_native_candidates: int = 100_000
    max_foreign_candidates: int = 65_536
    timeout_seconds: float = 60.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == 'timeout_seconds':
                if isinstance(value, bool) or not isfinite(value) or value <= 0:
                    raise ValueError('timeout_seconds must be finite and positive')
            elif type(value) is not int or value < 0:
                raise ValueError(f'{field.name} must be a nonnegative integer')
        if self.max_n > 20:
            raise ValueError('this small-instance comparison caps max_n at 20')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preflight(code, limits):
    require(isinstance(code, CSSCode), 'comparison requires positive binary CSSCode')
    require(code.n <= limits.max_n and code.hx.nrows + code.hz.nrows <= limits.max_checks,
            'comparison axes exceed configured limits')
    require((code.hx.nrows + code.hz.nrows)*code.n + code.hx.nrows*code.hz.nrows <= limits.max_elements,
            'comparison matrix payload exceeds max_elements')
    rx, rz = code.hx.rank(), code.hz.rank()
    k = code.n - rx - rz
    estimates = {'x': ((1 << k)-1) * (1 << rx), 'z': ((1 << k)-1) * (1 << rz)}
    require(max(estimates.values()) <= limits.max_foreign_candidates,
            'foreign exhaustive logical/stabilizer enumeration exceeds its candidate cap')
    return {'n': code.n, 'k': k, 'rank_x': rx, 'rank_z': rz, 'foreign_candidate_bounds': estimates}


def _versions():
    return {name: metadata.version(name) for name in ('qldpc', 'numpy', 'galois', 'numba')}


def _family(identifier):
    from qldpc import codes
    if identifier == 'steane':
        return codes.SteaneCode(), 'SteaneCode()'
    if identifier == 'hgp_open_2_3':
        return codes.HGPCode(codes.RepetitionCode(2), codes.RepetitionCode(3), set_logicals=False), 'HGPCode(RepetitionCode(2),RepetitionCode(3),set_logicals=False)'
    if identifier == 'toric_2':
        return codes.ToricCode(2, rotated=False), 'ToricCode(2,rotated=False)'
    if identifier == 'bicycle_cyclic_3':
        from sympy.abc import x, y
        return codes.BBCode({x: 3, y: 1}, 1+x, 1+x**2*y), 'BBCode({x:3,y:1},1+x,1+x**2*y)'
    raise ValueError('unknown independent family constructor')


def foreign_worker(request):
    """Worker protocol; exact searches and optional imports never run on import."""
    require(type(request) is dict and set(request) == {'task', 'payload', 'limits'}, 'invalid worker request')
    limits = ComparisonLimits(**request['limits'])
    versions = _versions()
    require(versions['qldpc'] == PINNED_VERSION, 'comparison requires pinned qldpc==0.4.1')
    from qeclang.interop.qldpc import from_qldpc, to_qldpc
    from qldpc.codes import CSSCode as ForeignCSS
    import qldpc.codes.distance as foreign_distance
    sources = {'css_class': sha256(Path(inspect.getsourcefile(ForeignCSS)).read_bytes()).hexdigest(),
               'distance_kernel': sha256(Path(foreign_distance.__file__).read_bytes()).hexdigest()}
    options = {'max_axis': max(limits.max_n, limits.max_checks), 'max_elements': limits.max_elements}
    if request['task'] == 'family':
        known = build_reference(request['payload'], max_qubits=limits.max_n,
                                 max_checks=limits.max_checks, max_bits=limits.max_elements)
        preflight(known, limits)
        external, constructor = _family(request['payload'])
        native = from_qldpc(external, **options)
        preflight(native, limits)
        return {'status': 'observed', 'code': to_dict(native), 'constructor': constructor,
                'family_distance_metadata_used': False, 'versions': versions, 'source_hashes': sources}
    require(request['task'] == 'matched', 'unsupported worker task')
    code = from_dict(request['payload'], max_qubits=limits.max_n,
                     max_checks=limits.max_checks, max_bits=limits.max_elements)
    baseline = preflight(code, limits)
    distances, timings = {}, {}
    foreign_parameters = None
    for sector in ('x', 'z'):
        external = to_qldpc(code, **options)  # Fresh object for EACH sector, even Hx=Hz.
        require(type(external) is ForeignCSS, 'foreign family overrides must not supply distances')
        require(same_css_presentation(code, from_qldpc(external, **options)), 'foreign presentation changed')
        observed = {'n': len(external), 'k': int(external.dimension),
                    'rank_x': int(external.code_x.rank), 'rank_z': int(external.code_z.rank)}
        require(all(observed[key] == baseline[key] for key in observed), 'foreign dimension/rank mismatch')
        foreign_parameters = observed
        count = ((1 << observed['k'])-1) * (1 << observed['rank_'+sector])
        require(count <= limits.max_foreign_candidates, 'foreign-side enumeration cap exceeded')
        known = external.get_distance_if_known(sector.upper())
        if observed['k'] == 0:
            require(isnan(float(known)), 'foreign k=0 convention changed')
            distances[sector], timings[sector] = None, None
            continue
        require(known is None, 'unexpected supplied/cached foreign sector distance')
        start = perf_counter()
        value = external.get_distance_exact(sector.upper(), cutoff=0, method='brute_force')
        timings[sector] = perf_counter() - start
        require(isfinite(value) and int(value) == value and 1 <= value <= code.n, 'invalid foreign finite distance')
        distances[sector] = int(value)
    return {'status': 'observed', **foreign_parameters, 'distances': distances, 'seconds': timings,
            'source': presentation_fingerprint(code), 'versions': versions, 'source_hashes': sources,
            'method': 'fresh generic CSSCode.get_distance_exact', 'cutoff': 0, 'enumerator': 'brute_force',
            'assurance': 'external_computational_result_not_a_certificate',
            'no_logicals_convention': 'undefined_nan_normalized_to_null' if code.k == 0 else None}


def foreign_call(task, payload, limits):
    request = {'task': task, 'payload': payload, 'limits': asdict(limits)}
    start = perf_counter()
    try:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
            input=json.dumps(request, allow_nan=False), text=True, capture_output=True,
            timeout=limits.timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'elapsed_seconds': perf_counter()-start}
    if completed.returncode or len(completed.stdout) > 1_000_000:
        return {'status': 'error', 'reason': completed.stderr[:2000] or 'worker output/exit failure'}
    try:
        result = json.loads(completed.stdout)
    except ValueError:
        return {'status': 'error', 'reason': 'foreign worker did not return one JSON object'}
    require(type(result) is dict, 'worker result must be an object')
    result['elapsed_seconds'] = perf_counter()-start
    return result


def compare_case(identifier, *, limits=ComparisonLimits()):
    case = next(item for item in reference_cases() if item.id == identifier)
    record = {'kind': 'matched_presentation', 'case': identifier, 'reference': case.to_dict()}
    try:
        code = build_reference(identifier, max_qubits=limits.max_n, max_checks=limits.max_checks,
                               max_bits=limits.max_elements)
        parameters = preflight(code, limits)
    except ValueError as error:
        return {**record, 'status': 'skipped_budget', 'reason': str(error)}
    start = perf_counter()
    native = {sector: asdict(css_distance(code, sector, max_weight=code.n,
                                        max_candidates=limits.max_native_candidates)) for sector in ('x', 'z')}
    record.update(source=presentation_fingerprint(code), native_parameters=parameters, native_distances=native,
                  native_distance_seconds=perf_counter()-start, code=to_dict(code))
    if any(value['status'] not in ('exact', 'no_logicals') for value in native.values()):
        return {**record, 'status': 'native_incomplete'}
    external = foreign_call('matched', to_dict(code), limits)
    record['foreign'] = external
    if external['status'] != 'observed':
        return {**record, 'status': 'external_'+external['status']}
    local = {sector: value['upper_bound'] for sector, value in native.items()}
    expectations = {'x': case.expected_dx, 'z': case.expected_dz}
    agreement = (external['source'] == record['source'] and external['distances'] == local and
                 all(external[key] == parameters[key] for key in ('n', 'k', 'rank_x', 'rank_z')))
    expected = (parameters['n'], parameters['k'], local) == (case.expected_n, case.expected_k, expectations)
    status = ('matched_no_logicals' if parameters['k'] == 0 else 'matched') if agreement and expected else 'mismatch'
    return {**record, 'status': status, 'reference_expectations_match': expected,
            'distance_comparison': 'not_applicable_no_logicals' if parameters['k'] == 0 else 'both_sectors_exact',
            'assurance': 'differential_agreement_not_formal_proof'}


def compare_family(identifier, *, limits=ComparisonLimits()):
    native = build_reference(identifier, max_qubits=limits.max_n, max_checks=limits.max_checks,
                              max_bits=limits.max_elements)
    preflight(native, limits)
    result = foreign_call('family', identifier, limits)
    row = {'kind': 'independent_family', 'case': identifier, 'foreign': result}
    if result['status'] != 'observed':
        return {**row, 'status': 'external_'+result['status']}
    other = from_dict(result['code'], max_qubits=limits.max_n,
                      max_checks=limits.max_checks, max_bits=limits.max_elements)
    candidates = [('physical_identity', list(range(native.n)))]
    if identifier == 'bicycle_cyclic_3':
        candidates.append(('cyclic_coordinate_inversion', [block*3 + (-site)%3 for block in range(2) for site in range(3)]))
    for convention, permutation in candidates:
        mapped = permute_qubits(native, permutation)
        try:
            maps = row_maps(mapped, other, max_axis=max(limits.max_n, limits.max_checks), max_bits=limits.max_elements)
            guarantees = verify_row_equivalence(mapped, other, maps)
        except ValueError:
            continue
        return {**row, 'status': 'verified_relation', 'coordinate_convention': convention,
                'permutation': permutation, 'maps': maps, 'guarantees': list(guarantees),
                'relation': 'same_presentation' if same_css_presentation(mapped, other) else 'same_generator_spaces',
                'source': presentation_fingerprint(native), 'target': presentation_fingerprint(other),
                'assurance': 'native_exact_relation_check_on_foreign_raw_matrices'}
    return {**row, 'status': 'relation_unresolved', 'reason': 'no declared coordinate relation verified; parameters are insufficient'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--cases', nargs='+', choices=[item.id for item in reference_cases()])
    parser.add_argument('--max-n', type=int, default=13)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--no-families', action='store_true')
    args = parser.parse_args()
    if args.worker:
        try:
            text = sys.stdin.read(1_000_001)
            require(len(text) <= 1_000_000, 'worker request exceeds byte proxy')
            result = foreign_worker(json.loads(text))
        except Exception as error:
            result = {'status': 'error', 'error_type': type(error).__name__, 'reason': str(error)[:2000]}
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    limits = ComparisonLimits(max_n=args.max_n, timeout_seconds=args.timeout)
    runtime = runtime_provenance()
    script_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    selected = args.cases or [item.id for item in reference_cases() if item.expected_n <= limits.max_n]
    rows = [compare_case(identifier, limits=limits) for identifier in selected]
    if not args.no_families:
        rows.extend(compare_family(identifier, limits=limits) for identifier in FAMILIES if identifier in selected)
    require(runtime_provenance() == runtime and sha256(Path(__file__).read_bytes()).hexdigest() == script_hash,
            'source/runtime changed during comparison')
    for row in rows:
        print(json.dumps({'schema': SCHEMA, 'runtime': runtime, 'script_sha256': script_hash,
                          'limits': asdict(limits), **row}, sort_keys=True, allow_nan=False))
    return 0 if all(row['status'] in ('matched', 'matched_no_logicals', 'verified_relation') for row in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())
