"""Execution budgets for a structured lifted-product program path."""
from dataclasses import fields
from io import StringIO
import json

from qeclang.analysis.structured import analyze_lifted, unsupported_reason
from qeclang.constructions.records import from_dict
from .json_input import JSONLimitExceeded, JSONLimits, loads_json


def bound_symbolic(lift, ledger):
    a, b = lift.a, lift.b
    columns = a.ncols*b.ncols + a.nrows*b.nrows
    rows = a.nrows*b.ncols + a.ncols*b.nrows
    blocks = rows*columns
    ledger.bound('blocks', max(blocks, max(a.nrows, a.ncols, b.nrows, b.ncols)**2))
    profile = lift.locality()
    # Every retained regular block contributes |G| times its support size to
    # the expanded nonzero count. Bound the number of terms in symbolic checks.
    ledger.bound('symbolic_terms', (profile.x_nonzeros + profile.z_nonzeros)//a.group_size)


def evaluate_lift(op, args, params, ledger, program_limits):
    if op == 'lift.literal':
        from .lift_types import construction_limits
        return from_dict(params['construction'], limits=construction_limits(program_limits))
    lift = args[0]
    if op == 'lift.css':
        ledger.matrices(lift.n, *lift.check_counts)
        bound_symbolic(lift, ledger)
        return lift.lower(max_bits=ledger.limits.max_matrix_bits,
                          max_blocks=ledger.limits.max_blocks, max_sites=ledger.limits.max_axis)
    if unsupported_reason(params['backend'], lift.a.orders) is None:
        bound_symbolic(lift, ledger)
    result = analyze_lifted(lift, backend=params['backend'],
                            max_steps=min(params['max_steps'], ledger.remaining('rank_steps')),
                            max_blocks=ledger.limits.max_blocks, max_entries=ledger.limits.max_blocks,
                            max_degree=ledger.limits.max_lift_degree,
                            max_polynomial_degree=ledger.limits.max_polynomial_degree,
                            max_operations=ledger.limits.max_polynomial_operations,
                            max_component_entries=ledger.limits.max_blocks,
                            max_component_axis=ledger.limits.max_component_axis,
                            max_field_degree=ledger.limits.max_field_degree,
                            max_characters=ledger.limits.max_characters,
                            record_operations=params['record_operations'])
    ledger.charge('rank_steps', result.steps or 0)
    if params['record_operations'] and result.status == 'exact':
        if any(getattr(proof, 'operations', None) is None for proof in (result.x_evidence, result.z_evidence)):
            raise ArithmeticError('exact polynomial result omitted a requested rank trace')
    # Stream existing dataclasses and tuples without deep-copying a whole trace
    # before the byte guard. The final artifact envelope has its own ledger charge.
    admission = JSONLimits(max_bytes=min(10_000_000, ledger.limits.max_artifact_bytes),
                           max_integer_bits=4096)
    encoder = json.JSONEncoder(separators=(',', ':'), ensure_ascii=True, allow_nan=False,
                               default=lambda obj: {f.name: getattr(obj, f.name) for f in fields(obj)})
    buffer = StringIO()
    for chunk in encoder.iterencode(result):
        if buffer.tell() + len(chunk) > admission.max_bytes:
            raise JSONLimitExceeded('symbolic result serialization exceeds byte limit')
        buffer.write(chunk)
    return loads_json(buffer.getvalue(), limits=admission)
