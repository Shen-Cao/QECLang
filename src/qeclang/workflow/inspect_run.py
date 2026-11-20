"""Check mathematical evidence in a saved CSS program run, independently of execution."""
from dataclasses import dataclass, field
import re

from qeclang.analysis.certificate_io import presentation_fingerprint, verify_certificate_record
from qeclang.algebra.binary import set_bits
from .import_artifact import ImportBudget, ImportLimitExceeded, ImportLimits, import_artifact
from .json_input import JSONLimitExceeded, JSONLimits, validate_json_data
from .program import Program
from .relations import BASED_RELATIONS, check_relation, check_witness_transport
from .lift_inspection import InspectionQuota, LiftInspection
from .reuse_ops import admit_witness
from qeclang.constructions.records import to_dict as construction_record


@dataclass(frozen=True)
class InspectionLimits:
    max_relation_checks: int = 512
    max_certificates: int = 256
    max_rank_steps: int = 1_000_000
    max_lift_comparison_units: int = 1_000_000
    max_row_map_work: int = 1_000_000
    max_blocks: int = 100_000
    max_symbolic_terms: int = 100_000
    max_lift_degree: int = 512
    max_polynomial_degree: int = 2048
    max_polynomial_operations: int = 10_000
    record: JSONLimits = field(default_factory=lambda: JSONLimits(max_bytes=32_000_000,
                                max_integer_bits=4096, max_payload_items=8_000_000))
    artifacts: ImportLimits = field(default_factory=ImportLimits)

    def __post_init__(self):
        for value in (self.max_relation_checks, self.max_certificates, self.max_rank_steps,
                      self.max_lift_comparison_units, self.max_row_map_work, self.max_blocks, self.max_symbolic_terms, self.max_lift_degree,
                      self.max_polynomial_degree, self.max_polynomial_operations):
            if type(value) is not int or value < 0:
                raise ValueError("inspection counters must be nonnegative integers")
        if not isinstance(self.record, JSONLimits) or not isinstance(self.artifacts, ImportLimits):
            raise TypeError("invalid inspection limit types")


_RUN_FIELDS = {"schema", "program", "status", "runtime", "verify_relations", "outputs",
               "artifacts", "ledger", "trace"}
_STEP_FIELDS = {"node", "op", "inputs", "status", "output", "evidence", "spent"}
_STOPPED = frozenset(('budget_exhausted', 'invalid_input', 'verification_failed', 'evidence_unavailable'))
_STOP_FIELDS = frozenset(('node', 'op', 'inputs', 'status', 'reason', 'spent'))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _spent_shape(value):
    """Validate recorded counter syntax without certifying historical usage."""
    _require(type(value) is dict and all(type(k) is str and bool(k) and type(v) is int and v >= 0
                                       for k, v in value.items()), 'invalid recorded spent counters')


def _stop_marker(node, step, status, bindings):
    """Bind a terminal refusal marker; its explanation/cost remains unverified."""
    _require(type(step) is dict and _STOP_FIELDS <= step.keys(), 'invalid terminal stop marker')
    _require((step['node'], step['op'], step['status']) == (node.id, node.op, status),
             'terminal marker differs from next program node or run status')
    _require(step['inputs'] == [bindings[i] for i in node.inputs], 'terminal input digest mismatch')
    _require(type(step['reason']) is str, 'terminal reason must be a string')
    _spent_shape(step['spent'])
    if status == 'budget_exhausted':
        _require(type(step.get('resource')) is str and bool(step['resource']),
                 'budget stop requires a resource name')
        extra = {'resource'} if step['resource'] == 'artifact_admission' else {'resource', 'requested', 'available'}
        _require(set(step) == _STOP_FIELDS | extra, 'invalid budget stop fields')
        if 'requested' in extra:
            _require(all(type(step[key]) is int and step[key] >= 0 for key in ('requested', 'available')),
                     'budget stop amounts must be nonnegative integers')
    else:
        _require(set(step) == _STOP_FIELDS, 'non-budget stop contains unexpected fields')


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


class _Reader:
    """Keep the reader's charged units even when the scientific check refuses."""
    def __init__(self, limits):
        self.limits = limits
        self.imports = ImportBudget(max_artifact_bytes=limits.artifacts.max_bytes,
                                    max_validation_work=limits.artifacts.max_validation_work)
        self.lifts = LiftInspection(limits.max_rank_steps, limits.max_lift_comparison_units)
        self.counts = {'relation_checks': 0, 'certificates': 0, 'row_map_work': 0}
        self.prefix_nodes = ()
        self.stop = None

    def charge(self, key, amount=1):
        if self.counts[key] + amount > getattr(self.limits, 'max_' + key):
            raise InspectionQuota(f'inspection exceeds max_{key}')
        self.counts[key] += amount

    def summary(self):
        return {**self.imports.used, **self.counts,
                'rank_steps': self.limits.max_rank_steps - self.lifts.remaining_steps,
                'lift_comparison_units': self.limits.max_lift_comparison_units - self.lifts.remaining_comparison_units}


def _inspect(program, record, reader, allow_prefix):
    limits = reader.limits
    validate_json_data(record, limits=limits.record)
    _require(type(record) is dict and set(record) == _RUN_FIELDS and
             record['schema'] == 'qeclang.run.v1', "invalid run schema/fields")
    _require(record['program'] == program.fingerprint(), "program fingerprint mismatch")
    stopped = record['status'] in _STOPPED if type(record['status']) is str else False
    if stopped and not allow_prefix:
        return 'unsupported', [], [], [], 'inspection requires a completed run'
    _require(stopped or record['status'] == 'completed', 'unknown run status')
    trace, records = record['trace'], record['artifacts']
    if stopped:
        _require(type(trace) is list and 1 <= len(trace) <= len(program.nodes),
                 'stopped run must contain a prefix and exactly one terminal marker')
        completed = trace[:-1]
    else:
        _require(type(trace) is list and len(trace) == len(program.nodes), "incomplete or extra trace nodes")
        completed = trace
    _require(type(records) is dict and len(records) <= len(completed), "invalid retained artifact map")
    values, bindings, claims, relations, reported = {}, {}, [], [], []
    for node, step in zip(program.nodes, completed):
        _require(type(step) is dict and set(step) == _STEP_FIELDS and step['status'] == 'completed',
                 "invalid completed trace step")
        _require((step['node'], step['op']) == (node.id, node.op), "trace instruction mismatch")
        if stopped:
            _spent_shape(step['spent'])
        refs = [bindings[i] for i in node.inputs]
        _require(refs == step['inputs'], "trace input digest mismatch")
        key = step['output']
        _require(type(key) is str and re.fullmatch('[0-9a-f]{64}', key) and key in records,
                 "missing or invalid trace output")
        artifact = records[key]
        _require(type(artifact) is dict and artifact.get('digest') == key
                 and artifact.get('kind') == node.result_type, "artifact key or type mismatch")
        if key not in values:
            values[key] = import_artifact(artifact, node.result_type, limits=limits.artifacts, budget=reader.imports)
        value = values[key]
        args = [values[d] for d in refs]
        if node.op in BASED_RELATIONS or node.op in ('css.transport', 'lift.css', 'css.rebase_witness'):
            reader.charge('relation_checks')
        is_witness = node.result_type == 'witness'
        if (is_witness or
                (node.op == 'lift.parameters' and node.params['record_operations'] and value['status'] == 'exact')):
            reader.charge('certificates')
        if node.op == 'css.literal':
            _literal(value, node.params['code'])
        elif node.op == 'lift.literal':
            _require(construction_record(value) == node.params['construction'], 'lift literal differs from program')
        elif node.op == 'lift.parameters':
            verified = reader.lifts.parameters(node, args[0], value, key, limits)
            if verified is not None:
                claims.append(verified)
        elif node.op in BASED_RELATIONS or node.op == 'lift.css':
            guarantees = (reader.lifts.expansion(args[0], value) if node.op == 'lift.css'
                          else check_relation(node.op, args[0], value, node.params))
            relations.append({'node': node.id, 'operation': node.op, 'source': refs[0],
                              'target': key, 'guarantees': list(guarantees)})
        elif node.op == 'css.distance':
            _distance_binding(value, args[0], node)
            reported.append(node.id)
        elif node.op == 'css.witness':
            _witness(value, args[0], args[1])
        elif node.op == 'css.import_witness':
            _require(value == node.params['certificate'], 'stored witness differs from program input')
            admit_witness(args[0], value)
        elif node.op == 'css.rebase_witness':
            from .row_equivalence import check_rebound, row_equivalence_work
            maps = node.params
            reader.charge('row_map_work', row_equivalence_work(args[0], args[1], maps))
            check_rebound(args[0], args[1], maps, args[2], value)
            relations.append({'node': node.id, 'operation': node.op, 'source': refs[0], 'target': refs[1],
                              'source_witness': refs[2], 'target_witness': key,
                              'relation': 'equal_positive_css_row_spaces_in_fixed_physical_basis'})
        elif node.op == 'css.transport':
            check_witness_transport(args[0], args[1], node.params['permutation'], args[2], value)
            relations.append({'node': node.id, 'operation': node.op, 'source': refs[0],
                              'target': refs[1], 'source_witness': refs[2], 'target_witness': key})
        else:
            raise ValueError('inspection has no checker for this instruction')
        if is_witness:
            claims.append({'node': node.id, 'scope': 'instance', 'source': value['source'],
                           'predicate': 'distance_upper_bound', 'sector': node.sector,
                           'bound': value['claim']['distance_upper_bound'], 'evidence': key})
        bindings[node.id] = key
    if stopped:
        terminal = program.nodes[len(completed)]
        _stop_marker(terminal, trace[-1], record['status'], bindings)
    _require(set(values) == set(records), 'unreferenced artifact')
    expected_outputs = {i: bindings[i] for i in program.outputs if i in bindings}
    _require(record['outputs'] == expected_outputs, 'output binding mismatch')
    if stopped:
        reader.prefix_nodes = tuple(bindings)
        reader.stop = {'node': terminal.id, 'status': record['status']}
    return 'checked_prefix' if stopped else 'checked', claims, relations, reported, None


def inspect_run(program: Program, record, *, limits=InspectionLimits(), allow_prefix=False):
    """Check graph bindings to the caller's program and verify relational/parity predicates.

    The checker trusts its own validators and input program. Success returns only
    newly checked finite-instance claims. A distance report alone establishes
    no lower bound, exactness or k=0 claim. Machine/runtime identity, historical
    spend, and producer-assigned assurance are outside this scientific check.
    It does not execute the program, solve equations, or enumerate error weights.

    allow_prefix=True additionally accepts a structurally valid completed prefix
    ending in one recognized stop marker. Only retained completed-node evidence
    is checked. The terminal node/suffix is never certified; its reported cause
    and spend are unverified even when the preceding prefix passes inspection.
    checked_prefix_nodes identifies admitted, bound records; only the emitted
    claims identify independently verified mathematical predicates.
    """
    if not isinstance(program, Program) or not isinstance(limits, InspectionLimits):
        raise TypeError('inspect_run requires Program and InspectionLimits')
    if type(allow_prefix) is not bool:
        raise TypeError('allow_prefix must be boolean')
    reader = _Reader(limits)
    try:
        status, claims, relations, reported, reason = _inspect(program, record, reader, allow_prefix)
    except (InspectionQuota, JSONLimitExceeded, ImportLimitExceeded) as exc:
        status, claims, relations, reported, reason = 'budget_exhausted', [], [], [], str(exc)
    except (TypeError, ValueError) as exc:
        status, claims, relations, reported, reason = 'invalid_record', [], [], [], str(exc)
    structured = any(n.op.startswith('lift.') for n in program.nodes)
    scope = ('based_relations_css_upper_witnesses_and_traced_ranks' if structured
             else 'based_relations_and_css_upper_witnesses')
    checked_nodes = {c['node'] for c in claims}
    inspected_nodes = ({node.id for node in program.nodes} if status == 'checked'
                       else set(reader.prefix_nodes) if status == 'checked_prefix' else set())
    result = {'schema': 'qeclang.inspection.v1', 'program': program.fingerprint(), 'status': status,
            'scope': scope, 'claims': claims, 'reader_usage': reader.summary(),
            'relations': relations, 'reported_distance_nodes': reported, 'reason': reason,
            'unverified_parameter_nodes': [n.id for n in program.nodes if n.op == 'lift.parameters'
                                          and n.id in inspected_nodes and n.id not in checked_nodes],
            'unverified': ['distance_lower_bounds', 'distance_exactness', 'no_logicals',
                           'runtime', 'ledger', 'candidate_counts', 'recorded_assurance']}

    if status == 'checked_prefix':
        result.update(checked_prefix_nodes=list(reader.prefix_nodes), unverified_stop=reader.stop.copy())
        result['unverified'].extend(('stop_cause', 'stop_spend'))
    if any(n.id in inspected_nodes and n.op == 'css.import_witness' and 'origin' in n.params
           for n in program.nodes):
        result['unverified'].append('witness_origin_metadata')
    return result
