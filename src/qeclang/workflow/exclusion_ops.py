"""Execution and reader obligations for bounded, degeneracy-aware CSS exclusions."""
from qeclang.analysis.certificate_io import presentation_fingerprint, verify_certificate_record
from .budget import BudgetExceeded
from .lift_inspection import InspectionQuota
from .relations import RelationError


def _payload(source, exclusion, upper):
    """Combine evidence after both premises have been checked on this source."""
    if (exclusion['source'] != presentation_fingerprint(source) or upper['source'] != exclusion['source']
            or upper['claim']['sector'] != exclusion['sector']):
        raise RelationError('exclusion and upper witness must share a source and sector')
    low = exclusion['through'] + 1
    high = upper['claim']['distance_upper_bound']
    if low > high:
        raise RelationError('checked upper witness conflicts with checked exclusion')
    return {'schema': 'qeclang.workflow.distance_claim.v1', 'source': upper['source'],
            'sector': exclusion['sector'], 'lower_bound': low, 'upper_bound': high,
            'status': 'exact' if low == high else 'bounded'}


def combine_sectors(code, first, second):
    """Take the minimum sector distance using already verified finite intervals."""
    source = presentation_fingerprint(code)
    if (first['source'] != source or second['source'] != source
            or {first['sector'], second['sector']} != {'x', 'z'}):
        raise RelationError('full CSS distance requires one X and one Z claim on this presentation')
    low = min(first['lower_bound'], second['lower_bound'])
    high = min(first['upper_bound'], second['upper_bound'])
    return {'schema': 'qeclang.workflow.distance_claim.v1', 'source': source,
            'sector': 'pauli', 'lower_bound': low, 'upper_bound': high,
            'status': 'exact' if low == high else 'bounded'}


def execute_exclusion(op, args, params, ledger):
    from qeclang.analysis.low_weight import certify_low_weight, check_low_weight
    from .operations import UnavailableEvidence
    code = args[0]
    if op == 'css.full_distance':
        return combine_sectors(*args)
    options = dict(max_axis=ledger.limits.max_axis, max_bits=ledger.limits.max_basis_bits,
                   max_indices=ledger.limits.max_exclusion_indices)
    if op == 'css.exclude':
        result = certify_low_weight(code, params['sector'], through=params['through'],
                                    max_steps=min(params['max_steps'], ledger.remaining('exclusion_steps')),
                                    **options)
        ledger.charge('exclusion_steps', result['steps'])
        return result
    _, exclusion, upper = args
    if exclusion['status'] != 'excluded':
        raise UnavailableEvidence('combining distance bounds requires a completed exclusion proof')
    remaining = ledger.remaining('exclusion_verify_steps')
    review = check_low_weight(code, exclusion, max_steps=remaining, **options)
    ledger.charge('exclusion_verify_steps', review.steps)
    if review.status == 'budget_exhausted':
        raise _VerificationLimit(review.reason or 'low-weight verification limit exceeded')
    if (review.status != 'checked_exclusion' or upper['kind'] != 'css_logical'
            or not verify_certificate_record(code, upper, max_axis=code.n, max_bits=2*code.n)):
        raise RelationError('distance premises failed independent verification')
    return _payload(code, exclusion, upper)


class _VerificationLimit(BudgetExceeded):
    def __init__(self, message):
        self.detail = {'resource': 'exclusion_verification'}
        Exception.__init__(self, message)


def inspect_exclusion(node, code, record, key, reader):
    from qeclang.analysis.low_weight import check_low_weight
    params = node.params
    if (record['sector'] != node.sector or record['through'] != params['through']
            or record['source'] != presentation_fingerprint(code) or record['steps'] > params['max_steps']):
        raise RelationError('low-weight report differs from its request or source')
    review = check_low_weight(code, record, max_steps=reader.limits.max_exclusion_verify_steps
                              - reader.counts['exclusion_verify_steps'],
                              max_axis=reader.limits.artifacts.max_axis, max_bits=reader.limits.artifacts.max_matrix_bits,
                              max_indices=reader.limits.max_exclusion_indices)
    reader.charge('exclusion_verify_steps', review.steps)
    if review.status == 'budget_exhausted':
        raise InspectionQuota(review.reason or 'low-weight verification exceeded reader limits')
    if review.status == 'no_evidence' and record['status'] == 'budget_exhausted':
        return None
    fact = {'node': node.id, 'scope': 'instance', 'source': record['source'], 'sector': node.sector,
            'evidence': key}
    if review.status == 'checked_exclusion':
        return {**fact, 'predicate': 'nontrivial_weight_exclusion', 'through': record['through']}
    if review.status == 'checked_witness':
        return {**fact, 'predicate': 'distance_upper_bound', 'bound': record['claim']['upper_bound']}
    raise RelationError(review.reason or 'invalid low-weight evidence')


def inspect_combination(node, args, value, key, checked_exclusions, input_keys):
    # The earlier nodes' checks refer to the exact digests named by this edge.
    if input_keys[1] not in checked_exclusions:
        raise RelationError('combine_distance requires an independently checked exclusion')
    expected = _payload(*args)
    if expected != value:
        raise RelationError('stored distance claim differs from its verified premises')
    claim = {k: v for k, v in expected.items() if k != 'schema'}
    return {**claim, 'node': node.id, 'scope': 'instance', 'predicate': 'sector_distance',
            'value': claim['upper_bound'] if claim['status'] == 'exact' else None,
            'evidence': key, 'premises': input_keys[1:]}


def inspect_full(node, args, value, key, checked_intervals, input_keys):
    if not all(ref in checked_intervals for ref in input_keys[1:]):
        raise RelationError('full CSS distance requires two independently established interval premises')
    expected = combine_sectors(*args)
    if expected != value:
        raise RelationError('stored full CSS distance differs from its checked sector intervals')
    return {**{k: v for k, v in expected.items() if k != 'schema'}, 'node': node.id,
            'scope': 'instance', 'predicate': 'logical_distance', 'evidence': key,
            'value': value['upper_bound'] if value['status'] == 'exact' else None,
            'premises': input_keys[1:]}
