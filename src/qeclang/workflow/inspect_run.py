"""Check mathematical evidence in a saved CSS program run, independently of execution."""
from dataclasses import dataclass, field
import re

from qeclang.analysis.certificate_io import presentation_fingerprint, verify_certificate_record
from qeclang.algebra.binary import set_bits
from .import_artifact import ImportBudget, ImportLimitExceeded, ImportLimits, import_artifact
from .json_input import JSONLimitExceeded, JSONLimits, validate_json_data
from .program import Program
from .relations import BASED_RELATIONS, check_relation, check_witness_transport


@dataclass(frozen=True)
class InspectionLimits:
    max_relation_checks: int = 512
    max_certificates: int = 256
    record: JSONLimits = field(default_factory=lambda: JSONLimits(max_bytes=32_000_000,
                                max_integer_bits=4096, max_payload_items=8_000_000))
    artifacts: ImportLimits = field(default_factory=ImportLimits)

    def __post_init__(self):
        for value in (self.max_relation_checks, self.max_certificates):
            if type(value) is not int or value < 0:
                raise ValueError("inspection counters must be nonnegative integers")
        if not isinstance(self.record, JSONLimits) or not isinstance(self.artifacts, ImportLimits):
            raise TypeError("invalid inspection limit types")


_RUN_FIELDS = {"schema", "program", "status", "runtime", "verify_relations", "outputs",
               "artifacts", "ledger", "trace"}
_STEP_FIELDS = {"node", "op", "inputs", "status", "output", "evidence", "spent"}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _literal(code, record):
    _require(code.n == record['n'], "CSS literal width differs from program")
    for matrix, key in ((code.hx, 'hx'), (code.hz, 'hz')):
        raw = record[key]
        _require(matrix.nrows == len(raw), "CSS literal check count differs from program")
        for row, sites in zip(matrix.rows, raw):
            _require(set(set_bits(row)) == set(sites), "CSS literal support differs from program")


def _distance_binding(value, code, node):
    _require(value['source'] == presentation_fingerprint(code) and value['sector'] == node.sector,
             "distance report has a different source or sector")
    witness = value['witness']
    if witness is not None:
        _require(witness.bit_length() <= code.n, "reported witness exceeds physical width")
        _require(1 <= value['upper_bound'] <= code.n, "reported bounds exceed physical width")
    # These are elementary consistency tests, not a check of exhausted weight layers.
    params = node.params
    _require(value['candidates'] <= params['max_candidates'] and
             value['searched_through'] <= params['max_weight'], "reported search exceeds node request")


def _witness(value, code, report):
    _require(value['kind'] == 'css_logical' and value['claim']['sector'] == report['sector'],
             "witness kind/sector differs from distance report")
    _require(verify_certificate_record(code, value, max_axis=code.n, max_bits=2*code.n),
             "logical upper-bound certificate failed independent verification")
    _require(int(value['certificate']['operator'], 16) == report['witness'] and
             value['claim']['distance_upper_bound'] == report['upper_bound'],
             "witness does not certify the reported operator/upper bound")


def _inspect(program, record, limits):
    validate_json_data(record, limits=limits.record)
    _require(type(record) is dict and set(record) == _RUN_FIELDS and
             record['schema'] == 'qeclang.run.v1', "invalid run schema/fields")
    _require(record['program'] == program.fingerprint(), "program fingerprint mismatch")
    if record['status'] in ('budget_exhausted', 'invalid_input', 'verification_failed', 'evidence_unavailable'):
        return 'unsupported', [], [], [], 'inspection requires a completed run'
    _require(record['status'] == 'completed', 'unknown run status')
    trace, records = record['trace'], record['artifacts']
    _require(type(trace) is list and len(trace) == len(program.nodes), "incomplete or extra trace nodes")
    _require(type(records) is dict and len(records) <= len(trace), "invalid retained artifact map")
    budget = ImportBudget(max_artifact_bytes=limits.artifacts.max_bytes,
                          max_validation_work=limits.artifacts.max_validation_work)
    values, bindings, claims, relations, reported = {}, {}, [], [], []
    relation_count = certificate_count = 0
    for node, step in zip(program.nodes, trace):
        _require(type(step) is dict and set(step) == _STEP_FIELDS and step['status'] == 'completed',
                 "invalid completed trace step")
        _require((step['node'], step['op']) == (node.id, node.op), "trace instruction mismatch")
        refs = [bindings[i] for i in node.inputs]
        _require(refs == step['inputs'], "trace input digest mismatch")
        key = step['output']
        _require(type(key) is str and re.fullmatch('[0-9a-f]{64}', key) and key in records,
                 "missing or invalid trace output")
        artifact = records[key]
        _require(type(artifact) is dict and artifact.get('digest') == key
                 and artifact.get('kind') == node.result_type, "artifact key or type mismatch")
        if key not in values:
            values[key] = import_artifact(artifact, node.result_type, limits=limits.artifacts, budget=budget)
        value = values[key]
        args = [values[d] for d in refs]
        if node.op in BASED_RELATIONS or node.op == 'css.transport':
            relation_count += 1
            if relation_count > limits.max_relation_checks:
                raise _InspectionQuota("relation checks exceed max_relation_checks")
        if node.op in ('css.transport', 'css.witness'):
            certificate_count += 1
            if certificate_count > limits.max_certificates:
                raise _InspectionQuota("certificate nodes exceed max_certificates")
        if node.op == 'css.literal':
            _literal(value, node.params['code'])
        elif node.op in BASED_RELATIONS:
            guarantees = check_relation(node.op, args[0], value, node.params)
            relations.append({'node': node.id, 'operation': node.op, 'source': refs[0],
                              'target': key, 'guarantees': list(guarantees)})
        elif node.op == 'css.distance':
            _distance_binding(value, args[0], node)
            reported.append(node.id)
        elif node.op == 'css.witness':
            _witness(value, args[0], args[1])
        elif node.op == 'css.transport':
            check_witness_transport(args[0], args[1], node.params['permutation'], args[2], value)
            relations.append({'node': node.id, 'operation': node.op, 'source': refs[0],
                              'target': refs[1], 'source_witness': refs[2], 'target_witness': key})
        else:
            raise ValueError('inspection has no checker for this instruction')
        if node.op in ('css.transport', 'css.witness'):
            claims.append({'node': node.id, 'scope': 'instance', 'source': value['source'],
                           'predicate': 'distance_upper_bound', 'sector': node.sector,
                           'bound': value['claim']['distance_upper_bound'], 'evidence': key})
        bindings[node.id] = key
    _require(set(values) == set(records), 'unreferenced artifact')
    _require(record['outputs'] == {i: bindings[i] for i in program.outputs}, 'output binding mismatch')
    return 'checked', claims, relations, reported, None


class _InspectionQuota(Exception):
    pass


def inspect_run(program: Program, record, *, limits=InspectionLimits()):
    """Check graph bindings to the caller's program and verify relational/parity predicates.

    The checker trusts its own validators and input program. Success returns only
    newly checked finite-instance claims. A distance report alone establishes
    no lower bound, exactness or k=0 claim. Machine/runtime identity, historical
    spend, and producer-assigned assurance are outside this scientific check.
    It does not execute the program, solve equations, or enumerate error weights.
    """
    if not isinstance(program, Program) or not isinstance(limits, InspectionLimits):
        raise TypeError('inspect_run requires Program and InspectionLimits')
    try:
        status, claims, relations, reported, reason = _inspect(program, record, limits)
    except (_InspectionQuota, JSONLimitExceeded, ImportLimitExceeded) as exc:
        status, claims, relations, reported, reason = 'budget_exhausted', [], [], [], str(exc)
    except (TypeError, ValueError) as exc:
        status, claims, relations, reported, reason = 'invalid_record', [], [], [], str(exc)
    return {'schema': 'qeclang.inspection.v1', 'program': program.fingerprint(), 'status': status,
            'scope': 'based_relations_and_css_upper_witnesses', 'claims': claims,
            'relations': relations, 'reported_distance_nodes': reported, 'reason': reason,
            'unverified': ['distance_lower_bounds', 'distance_exactness', 'no_logicals',
                           'runtime', 'ledger', 'candidate_counts', 'recorded_assurance']}
