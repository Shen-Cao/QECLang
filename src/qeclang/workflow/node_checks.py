"""Mathematical obligations for each typed node in a scientifically inspected trace."""
from qeclang.analysis.certificate_io import presentation_fingerprint, verify_certificate_record
from qeclang.algebra.binary import set_bits
from .relations import BASED_RELATIONS, check_relation, check_witness_transport
from .reuse_ops import admit_witness
from .exclusion_ops import inspect_exclusion, inspect_combination, inspect_full
from qeclang.constructions.records import to_dict as construction_record


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


def inspect_node(node, args, refs, value, key, reader, claims, relations, reported):
    """Accumulate fresh findings after the caller checks the graph edge bindings."""
    limits = reader.limits
    if node.op in BASED_RELATIONS or node.op in ('css.transport', 'lift.css', 'css.rebase_witness', 'css.transport_distance'):
        reader.charge('relation_checks')
    is_witness = node.result_type == 'witness'
    if (is_witness or node.op in ('css.exclude', 'css.import_exclusion') or
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
    elif node.op in ('css.exclude', 'css.import_exclusion'):
        fact = inspect_exclusion(node, args[0], value, key, reader)
        if fact is not None:
            claims.append(fact)
            if fact['predicate'] == 'nontrivial_weight_exclusion':
                reader.checked_exclusions.add(key)
    elif node.op == 'css.combine_distance':
        claims.append(inspect_combination(node, args, value, key, reader.checked_exclusions, refs))
        reader.checked_intervals.add(key)
    elif node.op == 'css.full_distance':
        claims.append(inspect_full(node, args, value, key, reader.checked_intervals, refs))
        reader.checked_intervals.add(key)
    elif node.op == 'css.transport_distance':
        from .distance_transport import check_transported_interval, transport_interval_work
        _require(refs[2] in reader.checked_intervals, 'transport requires an independently established interval')
        params = node.params
        work = transport_interval_work(args[0], args[1], params, max_axis=reader.limits.artifacts.max_axis)
        reader.charge('interval_transport_work', work)
        check_transported_interval(args[0], args[1], args[2], value, params,
                                   max_axis=reader.limits.artifacts.max_axis, max_work=work)
        relations.append({'node': node.id, 'operation': node.op, 'relation': params['relation'],
                          'source': refs[0], 'target': refs[1], 'source_interval': refs[2], 'target_interval': key})
        claims.append({**{k: v for k, v in value.items() if k != 'schema'},
                       'node': node.id, 'scope': 'instance', 'evidence': key,
                       'predicate': 'logical_distance' if node.sector == 'pauli' else 'sector_distance',
                       'value': value['upper_bound'] if value['status'] == 'exact' else None,
                       'premises': refs})
        reader.checked_intervals.add(key)
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
