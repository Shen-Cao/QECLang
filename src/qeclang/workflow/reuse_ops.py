"""Runtime and inspection obligations for CSS evidence supplied by another program."""
from qeclang.analysis.certificate_io import verify_certificate_record
from .relations import RelationError


def admit_witness(code, record):
    if (type(record) is not dict or record.get('kind') != 'css_logical'
            or not verify_certificate_record(code, record, max_axis=code.n, max_bits=2*code.n)):
        raise RelationError('imported logical witness fails source binding or parity verification')
    return record


def apply_reuse(op, args, params, ledger):
    if op == 'css.import_witness':
        ledger.charge('witness_imports', 1)
        return admit_witness(args[0], params['certificate'])
    from .row_equivalence import rebind_witness, row_equivalence_work
    ledger.charge('transports', 1)
    ledger.charge('row_map_work', row_equivalence_work(args[0], args[1], params))
    return rebind_witness(args[0], args[1], params, args[2])
