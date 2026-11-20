"""Call the reference kernels after the checks required by each program operation."""
from dataclasses import asdict

from qeclang.algebra.binary import set_bits
from qeclang.analysis import css_distance
from qeclang.analysis.certificate_io import certificate_record, presentation_fingerprint
from qeclang.analysis.certificates import certify_css_logical
from qeclang.codes import SparseCSSCode
from qeclang.io import from_dict
from qeclang.transforms import permute_qubits
from .relations import transport_witness
from .reuse_ops import apply_reuse


class UnavailableEvidence(Exception):
    """A valid earlier computation supplied no witness for this request."""


def _literal(params, ledger):
    record = params["code"]
    n, hx, hz = record.get("n"), record.get("hx"), record.get("hz")
    if type(n) is not int or n < 0 or not isinstance(hx, list) or not isinstance(hz, list):
        raise ValueError("invalid CSS literal dimensions")
    ledger.matrices(n, len(hx), len(hz))
    return from_dict(record, max_qubits=ledger.limits.max_axis, max_checks=ledger.limits.max_axis,
                     max_bits=ledger.limits.max_matrix_bits)


def _sparse(source, ledger):
    # Same incidence proxy as SparseCSSCode. Reserve it before converting either matrix.
    degrees = source.hz.T.rows
    work = source.hx.nrows + source.hz.nrows + source.hx.nnz + source.hz.nnz
    work += sum(degrees[j].bit_count() for row in source.hx.rows for j in set_bits(row))
    ledger.charge("validation_work", work)
    return SparseCSSCode.from_packed(source, max_validation_work=work)


def evaluate(op, args, params, ledger):
    if op == "css.literal":
        return _literal(params, ledger)
    if op in ('css.import_witness', 'css.rebase_witness'):
        return apply_reuse(op, args, params, ledger)
    source = args[0]
    if op == "css.chain":
        return source.to_chain_complex()
    if op == "chain.css":
        return source.to_css()
    if op == "css.sparse":
        return _sparse(source, ledger)
    if op == "sparse.css":
        return source.to_packed(max_bits=ledger.limits.max_matrix_bits)
    if op == "css.permute":
        return permute_qubits(source, params["permutation"])
    if op == "css.distance":
        ledger.basis(source)
        allowed = min(params["max_candidates"], ledger.remaining("candidates"))
        result = css_distance(source, params["sector"], max_weight=params["max_weight"],
                              max_candidates=allowed)
        ledger.charge("candidates", result.candidates)
        return {**asdict(result), "source": presentation_fingerprint(source), "sector": params["sector"]}
    if op == "css.witness":
        distance = args[1]
        if distance["source"] != presentation_fingerprint(source):
            raise ValueError("distance result belongs to a different CSS presentation")
        if distance["witness"] is None:
            raise UnavailableEvidence("the distance result contains no finite logical witness")
        ledger.basis(source)
        cert = certify_css_logical(source, distance["sector"], distance["witness"])
        if cert.weight != distance["upper_bound"]:
            raise ArithmeticError("distance witness disagrees with its upper bound")
        return certificate_record(source, cert)
    if op == "css.transport":
        ledger.charge("transports", 1)
        return transport_witness(source, args[1], params["permutation"], args[2])
    raise AssertionError(f"unhandled validated operation {op}")
