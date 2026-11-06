from copy import deepcopy
import json
from random import Random

import pytest

from qeclang.algebra import BinaryMatrix as M, SparseBinaryMatrix as S
from qeclang.analysis import CSSLogicalFrame, export_logical_frame, import_logical_frame
from qeclang.analysis.certificates import RankCertificate, certify_rank, verify_sparse_rank
from qeclang.analysis.frame_io import SCHEMA, _fingerprint
from qeclang.codes import CSSCode, SparseCSSCode
from qeclang.decoding import CSSBPDecoder, simulate_css_memory


def steane():
    h = M((85, 102, 120), 7)
    return CSSCode(h, h)


def test_sparse_rank_verifier_differential_and_limits():
    rng = Random(49152)
    for m in range(5):
        for n in range(5):
            matrix = M(tuple(rng.randrange(1 << n) for _ in range(m)), n)
            certificate = certify_rank(matrix)
            assert verify_sparse_rank(S.from_packed(matrix), certificate)
    matrix = S(((0,),), 1)
    proof = certify_rank(M((1,), 1))
    with pytest.raises(ValueError, match='max_bits'):
        verify_sparse_rank(matrix, proof, max_bits=3)
    with pytest.raises(ValueError, match='max_work'):
        verify_sparse_rank(matrix, proof, max_work=0)
    with pytest.raises(ValueError, match='max_axis'):
        verify_sparse_rank(matrix, proof, max_axis=0)
    assert not verify_sparse_rank(S(((),), 1), proof)
    assert not verify_sparse_rank(matrix, RankCertificate(M((0,), 1), proof.v, proof.w, proof.z))
    assert not verify_sparse_rank(matrix, RankCertificate(None, None, None, None))


def test_sparse_rank_never_allocates_full_product(monkeypatch):
    m, n = 1500, 2000
    source = S(((n-1,),) + ((),) * (m-1), n)
    proof = RankCertificate(M((1,) + (0,) * (m-1), 1), M((1 << (n-1),), n),
                            M((1,), m), M((0,) * (n-1) + (1,), 1))
    def forbidden(*args, **kwargs):
        raise AssertionError('packed matrix multiplication or source lowering is forbidden')
    monkeypatch.setattr(M, '__matmul__', forbidden)
    monkeypatch.setattr(S, 'to_packed', forbidden)
    # Certificate payload is far smaller than the 3,000,000-bit source shape.
    assert verify_sparse_rank(source, proof, max_bits=7000, max_axis=2000, max_work=10_000)


def test_portable_frame_roundtrip_independent_of_elimination(monkeypatch):
    code = steane()
    source = SparseCSSCode.from_packed(code)
    original = CSSLogicalFrame.from_packed(code)
    record = json.loads(json.dumps(export_logical_frame(code, original)))
    decoder = CSSBPDecoder(source, .05, .05, max_iterations=5)
    baseline = simulate_css_memory(source, .05, trials=100, seed=51,
                                   decoder=decoder, logical_frame=original)
    def forbidden(*args, **kwargs):
        raise AssertionError('import must not lower the source or perform elimination')
    for method in ('rank', 'row_basis', 'nullspace', 'solve', 'contains', '__matmul__'):
        monkeypatch.setattr(M, method, forbidden)
    monkeypatch.setattr(S, 'to_packed', forbidden)
    monkeypatch.setattr(SparseCSSCode, 'to_packed', forbidden)
    recovered = import_logical_frame(source, record)
    assert recovered == original
    assert simulate_css_memory(source, .05, trials=100, seed=51,
                               decoder=decoder, logical_frame=recovered) == baseline


@pytest.mark.parametrize('tamper', [
    lambda r: r.update(source='0'*64),
    lambda r: r.update(k=0),
    lambda r: r.update(k=True),
    lambda r: r.update(extra='unknown'),
    lambda r: r['rank_x']['u']['rows'].__setitem__(0, '0x0'),
    lambda r: r['rank_x']['v'].update(ncols=8),
    lambda r: r.update(x_logicals=[]),
    lambda r: r.update(x_logicals=[[0]]),
    lambda r: r.update(z_logicals=[[]]),
    lambda r: r.update(z_logicals=[[0, 0]]),
    lambda r: r.update(z_logicals=[[True]]),
])
def test_tampered_or_incomplete_evidence_is_rejected(tamper):
    code = steane()
    record = deepcopy(export_logical_frame(code))
    tamper(record)
    with pytest.raises(ValueError):
        import_logical_frame(SparseCSSCode.from_packed(code), record)


def test_frame_import_resource_guards():
    code = steane()
    source = SparseCSSCode.from_packed(code)
    record = export_logical_frame(code)
    for limits in ({'max_bits': 1}, {'max_work': 0}, {'max_axis': 1}, {'max_nonzeros': 1}):
        with pytest.raises(ValueError):
            import_logical_frame(source, record, **limits)
    oversized = deepcopy(record)
    oversized['rank_x']['u']['rows'][0] = '0x' + 'f' * 100
    with pytest.raises(ValueError, match='hexadecimal'):
        import_logical_frame(source, oversized)
    with pytest.raises(TypeError, match='SparseCSSCode'):
        import_logical_frame(code, record)


def test_complete_sparse_frame_can_import_with_zero_dense_payload():
    # A valid externally supplied certificate for all free qubits. Logical
    # supports are sparse identities; no dense n-by-n matrix is constructed.
    n = 64
    source = SparseCSSCode(S((), n), S((), n))
    rank_zero = {'u': {'ncols': 0, 'rows': []}, 'v': {'ncols': n, 'rows': []},
                 'w': {'ncols': 0, 'rows': []}, 'z': {'ncols': 0, 'rows': ['0x0'] * n}}
    record = {'schema': SCHEMA, 'source': _fingerprint(source), 'n': n, 'k': n,
              'rank_x': rank_zero, 'rank_z': deepcopy(rank_zero),
              'x_logicals': [[i] for i in range(n)], 'z_logicals': [[i] for i in range(n)]}
    frame = import_logical_frame(source, record, max_bits=0, max_nonzeros=2*n)
    assert frame.k == n and frame.n == n
    assert frame.residual_is_trivial(0, 0)
    assert not frame.residual_is_trivial(1, 0)
    assert not frame.residual_is_trivial(0, 1 << (n-1))


def test_empty_and_no_logical_frame_roundtrip():
    for code in (CSSCode(M((), 0), M((), 0)), CSSCode(M((3,), 2), M((3,), 2))):
        record = export_logical_frame(code)
        recovered = import_logical_frame(SparseCSSCode.from_packed(code), record)
        assert recovered.k == 0
        assert recovered.residual_is_trivial(0, 0)
