"""Small bounded comparison index for positive binary CSS codes in fixed coordinates."""
from dataclasses import asdict, dataclass

from qeclang.algebra.binary import nonnegative_integer
from qeclang.analysis.certificate_io import presentation_fingerprint
from qeclang.codes import CSSCode, same_css_presentation
from qeclang.constructions.records import digest
from qeclang.transforms import permute_qubits
from qeclang.workflow.relations import check_relation, RelationError
from qeclang.workflow.row_equivalence import row_maps, verify_row_equivalence


class LookupLimitExceeded(ValueError):
    """The requested comparison or index exceeds caller policy."""


@dataclass(frozen=True)
class LookupLimits:
    max_cases: int = 128
    max_axis: int = 256
    max_matrix_bits: int = 1_000_000
    max_map_bits: int = 1_000_000
    max_map_attempts: int = 16

    def __post_init__(self):
        for name, value in vars(self).items():
            nonnegative_integer(value, name)


def _check(code, limits):
    if not isinstance(code, CSSCode):
        raise TypeError('fixed-coordinate lookup requires positive binary packed CSSCode')
    n, mx, mz = code.n, code.hx.nrows, code.hz.nrows
    if max(n, mx, mz) > limits.max_axis or (mx+mz)*n + mx*mz > limits.max_matrix_bits:
        raise LookupLimitExceeded('lookup input exceeds matrix or axis budget')


def _row_identity(code):
    hx, hz = code.hx.row_basis(), code.hz.row_basis()
    key = digest({'scheme': 'qeclang.fixed_css_rowspace.v1', 'n': code.n,
                  'x_rref_rows': [hex(v) for v in hx.rows], 'z_rref_rows': [hex(v) for v in hz.rows]})
    return key, code.n-hx.nrows-hz.nrows


@dataclass(frozen=True)
class _Entry:
    identifier: str
    code: CSSCode
    presentation: str
    row_identity: str
    k: int


@dataclass(frozen=True, init=False)
class ReferenceIndex:
    """An in-memory snapshot; all references and maps retain check and sector order.

    Lookup covers identity coordinates or exactly one supplied site permutation.
    The row-space digest screens candidates. Positive equivalence reports require
    actual comparison of supports or checked two-way generator maps. Separate
    budgets bound the number and packed workspace of map derivations.
    """
    _entries: tuple[_Entry, ...]
    limits: LookupLimits
    snapshot: str

    def __init__(self, entries, *, limits=LookupLimits()):
        if not isinstance(limits, LookupLimits):
            raise TypeError('limits must be LookupLimits')
        cases, used = [], set()
        for identifier, code in entries:
            if len(cases) == limits.max_cases:
                raise LookupLimitExceeded('snapshot exceeds max_cases')
            if type(identifier) is not str or not 1 <= len(identifier) <= 128 or identifier in used:
                raise ValueError('reference IDs must be unique nonempty bounded strings')
            _check(code, limits)
            key, k = _row_identity(code)
            used.add(identifier)
            cases.append(_Entry(identifier, code, presentation_fingerprint(code), key, k))
        ordered = tuple(sorted(cases, key=lambda entry: entry.identifier))
        object.__setattr__(self, '_entries', ordered)
        object.__setattr__(self, 'limits', limits)
        object.__setattr__(self, 'snapshot', digest({'schema': 'qeclang.reference_index.v1',
                            'entries': [[e.identifier, e.presentation] for e in ordered]}))

    def lookup(self, query, *, permutation=None):
        _check(query, self.limits)
        if permutation is None:
            comparison = query
        else:
            if type(permutation) not in (list, tuple) or len(permutation) != query.n:
                raise ValueError('permutation must list each query site')
            permutation = list(permutation)
            comparison = permute_qubits(query, permutation)
            check_relation('css.permute', query, comparison, {'permutation': permutation})
        key, k = _row_identity(comparison)
        matches, parameters, unresolved = [], [], []
        attempts = 0
        for entry in self._entries:
            if comparison.n != entry.code.n or k != entry.k:
                continue
            if key != entry.row_identity:
                parameters.append(entry.identifier)
                continue
            if same_css_presentation(comparison, entry.code):
                kind = 'identical_ordered_checks'
                maps = None
            else:
                if attempts >= self.limits.max_map_attempts:
                    unresolved.append({'id': entry.identifier, 'reason': 'map attempt budget'})
                    continue
                attempts += 1
                try:
                    maps = row_maps(comparison, entry.code, max_bits=self.limits.max_map_bits,
                                    max_axis=self.limits.max_axis)
                except RelationError as error:
                    # A screen match alone never establishes equivalence.
                    unresolved.append({'id': entry.identifier, 'reason': str(error)})
                    continue
                verify_row_equivalence(comparison, entry.code, maps)
                kind = 'verified_generator_spaces'
            matches.append({'id': entry.identifier, 'relation': kind, 'target': entry.presentation,
                            'maps': maps})
        return {'schema': 'qeclang.reference_lookup.v1', 'snapshot': self.snapshot,
                'query': presentation_fingerprint(query), 'comparison': presentation_fingerprint(comparison),
                'n': comparison.n, 'k': k, 'row_space_key': key, 'permutation': permutation,
                'scope': 'positive_binary_css_fixed_sites' if permutation is None else 'one_supplied_qubit_permutation',
                'status': 'matched' if matches else 'unresolved' if unresolved else 'no_match_in_scope',
                'complete_in_scope': not unresolved,
                'matches': matches, 'parameter_matches_only': parameters, 'unresolved': unresolved,
                'limits': asdict(self.limits), 'map_attempts': attempts, 'references': len(self._entries)}


def default_index(*, limits=LookupLimits()):
    from .references import build_reference, reference_cases
    descriptors = reference_cases()
    if len(descriptors) > limits.max_cases:
        raise LookupLimitExceeded('reference corpus exceeds max_cases')
    for case in descriptors:
        n, mx, mz = case.expected_n, case.x_checks, case.z_checks
        if max(n, mx, mz) > limits.max_axis or (mx+mz)*n + mx*mz > limits.max_matrix_bits:
            raise LookupLimitExceeded('reference snapshot exceeds matrix or axis budget')
    return ReferenceIndex(((case.id, build_reference(case.id, max_qubits=limits.max_axis,
                max_checks=2*limits.max_axis, max_bits=limits.max_matrix_bits)) for case in descriptors), limits=limits)
