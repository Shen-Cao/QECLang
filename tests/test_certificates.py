from dataclasses import replace
from itertools import product
from random import Random

import pytest

from qeclang.algebra import BinaryMatrix as M
from qeclang.analysis import css_distance, stabilizer_distance
from qeclang.analysis.certificates import (
    certify_css_logical, certify_rank, certify_stabilizer_logical,
    verify_css_logical, verify_rank, verify_stabilizer_logical,
)
from qeclang.constructions import five_qubit, steane, toric


def test_rank_certificates_exhaustive_and_tamper():
    for m, n in product(range(4), repeat=2):
        for rows in product(range(1 << n), repeat=m):
            matrix = M(rows, n)
            certificate = certify_rank(matrix)
            assert certificate.rank == matrix.rank()
            assert verify_rank(matrix, certificate)
            if m and n:
                tampered = M((matrix.rows[0] ^ 1,) + matrix.rows[1:], n)
                assert not verify_rank(tampered, certificate)
    with pytest.raises(ValueError):
        certify_rank(M.identity(3), max_bits=1)


def test_rank_verification_needs_no_elimination(monkeypatch):
    rng = Random(52)
    matrix = M(tuple(rng.randrange(1 << 12) for _ in range(10)), 12)
    certificate = certify_rank(matrix)
    def forbidden(*args, **kwargs):
        raise AssertionError("verification must not eliminate")
    monkeypatch.setattr(M, "rank", forbidden)
    monkeypatch.setattr(M, "solve", forbidden)
    monkeypatch.setattr(M, "row_basis", forbidden)
    monkeypatch.setattr(M, "__matmul__", forbidden)
    assert verify_rank(matrix, certificate)


def test_css_certificates_independent_of_row_membership(monkeypatch):
    code = steane()
    certificates = [certify_css_logical(code, sector, css_distance(code, sector).witness)
                    for sector in ("x", "z")]
    def forbidden(*args):
        raise AssertionError("verifier must not test row membership")
    monkeypatch.setattr(M, "contains", forbidden)
    for certificate in certificates:
        assert verify_css_logical(code, certificate)
        assert not verify_css_logical(code, replace(certificate, weight=2))
        assert not verify_css_logical(code, replace(certificate, dual=0))
        assert not verify_css_logical(code, replace(certificate, operator=-1))


def test_degenerate_and_invalid_logicals():
    for code in (steane(), toric(2)):
        for row in code.hx.rows:
            with pytest.raises(ValueError):
                certify_css_logical(code, "x", row)
    with pytest.raises(ValueError, match="syndrome"):
        certify_css_logical(steane(), "x", 1)
    code = five_qubit()
    witness = stabilizer_distance(code).witness
    certificate = certify_stabilizer_logical(code, witness)
    assert certificate.weight == 3
    assert verify_stabilizer_logical(code, certificate)
    assert not verify_stabilizer_logical(code, replace(certificate, dual=0))
    for row in code.checks.rows:
        with pytest.raises(ValueError):
            certify_stabilizer_logical(code, row)
