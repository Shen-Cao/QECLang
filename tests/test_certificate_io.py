from copy import deepcopy
import json

from qeclang.algebra import BinaryMatrix as M
from qeclang.analysis import css_distance, stabilizer_distance
from qeclang.analysis.certificates import certify_css_logical, certify_rank, certify_stabilizer_logical
from qeclang.analysis.certificate_io import certificate_record, verify_certificate_record
from qeclang.constructions import five_qubit, steane


def test_portable_records_and_mutations():
    code = steane()
    matrix = code.hx
    stab = five_qubit()
    pairs = [(matrix, certify_rank(matrix)),
             (code, certify_css_logical(code, "x", css_distance(code, "x").witness)),
             (stab, certify_stabilizer_logical(stab, stabilizer_distance(stab).witness))]
    for source, certificate in pairs:
        record = json.loads(json.dumps(certificate_record(source, certificate)))
        assert verify_certificate_record(source, record)
        assert not verify_certificate_record(source, dict(record, source="wrong"))
        assert not verify_certificate_record(source, dict(record, schema="unknown"))
        assert not verify_certificate_record(source, record, max_bits=0)
        assert not verify_certificate_record(source, dict(record, claim={}))
        assert not verify_certificate_record(source, None)


def test_rank_input_guards_and_wrong_source():
    matrix = M((5, 6), 3)
    record = certificate_record(matrix, certify_rank(matrix))
    assert not verify_certificate_record(M((4, 6), 3), record)
    for key, value in (("ncols", True), ("ncols", 10**12), ("rows", ["0x"+"f"*1000])):
        bad = deepcopy(record)
        bad["certificate"]["u"][key] = value
        assert not verify_certificate_record(matrix, bad)
    bad = deepcopy(record)
    bad["claim"]["rank"] = True
    assert not verify_certificate_record(matrix, bad)
