from dataclasses import FrozenInstanceError
from itertools import product
from types import SimpleNamespace

import pytest

from qeclang.algebra import BinaryMatrix as M, SparseBinaryMatrix as S
from qeclang.analysis import CSSLogicalFrame
from qeclang.analysis.logicals import CSSLogicals
from qeclang.codes import CSSCode, SparseCSSCode
from qeclang.decoding import CSSBPDecoder, css_decode, simulate_css_memory


def steane():
    h = M((85, 102, 120), 7)
    return CSSCode(h, h)


def test_frame_factory_and_exhaustive_stabilizer_membership():
    codes = (CSSCode(M((15,), 4), M((15,), 4)),
             CSSCode(M((3,), 2), M((3,), 2)),
             CSSCode(M((), 3), M((), 3)),
             CSSCode(M((3, 6), 3), M((), 3)))
    for code in codes:
        frame = CSSLogicalFrame.from_packed(code)
        assert frame.n == code.n and frame.k == code.k
        assert isinstance(frame.presentation, SparseCSSCode)
        assert isinstance(frame.x_logicals, S) and isinstance(frame.z_logicals, S)
        assert frame.matches(code) and frame.matches(SparseCSSCode.from_packed(code))
        for x, z in product(range(1 << code.n), repeat=2):
            expected = code.hx.contains(x) and code.hz.contains(z)
            assert frame.residual_is_trivial(x, z) == expected


def test_constructor_and_mutation_cannot_inject_unvalidated_frame():
    code = steane()
    with pytest.raises(TypeError, match='from_packed'):
        CSSLogicalFrame()
    with pytest.raises(TypeError, match='from_packed'):
        CSSLogicalFrame(SparseCSSCode.from_packed(code), S((), 7), S((), 7))
    frame = CSSLogicalFrame.from_packed(code)
    with pytest.raises(FrozenInstanceError):
        frame.x_logicals = S((), 7)
    with pytest.raises(TypeError, match='explicitly packed'):
        CSSLogicalFrame.from_packed(SparseCSSCode.from_packed(code))
    with pytest.raises(ValueError, match='max_bits'):
        CSSLogicalFrame.from_packed(code, max_bits=1)
    with pytest.raises(ValueError, match='max_validation_work'):
        CSSLogicalFrame.from_packed(code, max_validation_work=0)


def test_factory_rejects_incomplete_and_incorrect_produced_basis(monkeypatch):
    import qeclang.analysis.frames as frames
    code = steane()
    monkeypatch.setattr(frames, 'css_logicals', lambda _: CSSLogicals((), ()))
    with pytest.raises(ArithmeticError, match='complete'):
        CSSLogicalFrame.from_packed(code)
    monkeypatch.setattr(frames, 'css_logicals', lambda _: CSSLogicals((1,), (1,)))
    with pytest.raises(ArithmeticError, match='syndrome'):
        CSSLogicalFrame.from_packed(code)
    monkeypatch.setattr(frames, 'css_logicals', lambda _: CSSLogicals((85,), (85,)))
    with pytest.raises(ArithmeticError, match='pairing'):
        CSSLogicalFrame.from_packed(code)


def test_frame_rejects_nonzero_syndrome_and_invalid_corrections():
    code = CSSCode(M((3,), 2), M((3,), 2))
    frame = CSSLogicalFrame.from_packed(code)
    assert frame.k == 0
    assert frame.residual_is_trivial(3, 0)
    assert not frame.residual_is_trivial(1, 0)
    valid = css_decode(code, 0, 0)
    assert frame.logical_success(3, 0, valid)
    assert not frame.logical_success(1, 0, valid)
    failed = SimpleNamespace(success=False)
    assert not frame.logical_success(0, 0, failed)
    invalid = SimpleNamespace(success=True, x=SimpleNamespace(error=4), z=SimpleNamespace(error=0))
    with pytest.raises(ValueError):
        frame.logical_success(0, 0, invalid)
    with pytest.raises(ValueError):
        frame.residual_is_trivial(-1, 0)
    empty = CSSLogicalFrame.from_packed(CSSCode(M((), 0), M((), 0)), max_bits=0)
    assert empty.n == empty.k == 0 and empty.residual_is_trivial(0, 0)


def test_frame_preserves_presentation_identity():
    code = steane()
    frame = CSSLogicalFrame.from_packed(code)
    reordered = CSSCode(M(tuple(reversed(code.hx.rows)), 7), code.hz)
    assert not frame.matches(reordered)
    assert frame.matches(CSSCode(code.hx, code.hz, 'different name'))
    assert not frame.matches(object())
    with pytest.raises(ValueError, match='matching'):
        simulate_css_memory(reordered, .05, trials=10, seed=4, logical_frame=frame)


def test_sparse_simulation_has_no_hidden_elimination_or_lowering(monkeypatch):
    code = steane()
    sparse = SparseCSSCode.from_packed(code)
    frame = CSSLogicalFrame.from_packed(code)
    decoder = CSSBPDecoder(sparse, .05, .05, max_iterations=10)
    expected = simulate_css_memory(code, .08, trials=300, seed=24, decoder=decoder)
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected lowering or elimination during sparse frame analysis')
    for method in ('rank', 'contains', 'solve', 'row_basis', 'nullspace'):
        monkeypatch.setattr(M, method, forbidden)
    monkeypatch.setattr(S, 'to_packed', forbidden)
    monkeypatch.setattr(SparseCSSCode, 'to_packed', forbidden)
    actual = simulate_css_memory(sparse, .08, trials=300, seed=24,
                                  decoder=decoder, logical_frame=frame)
    assert actual.failures == expected.failures
    assert actual.decoder_failures == expected.decoder_failures
    assert actual.logical_failures == expected.logical_failures
    assert actual.logical_analysis == 'validated_logical_frame'
    assert expected.logical_analysis == 'exact_row_space'
    assert actual.confidence_interval == expected.confidence_interval
    assert actual == simulate_css_memory(sparse, .08, trials=300, seed=24,
                                         decoder=decoder, logical_frame=frame)
    with pytest.raises(ValueError, match='logical_frame'):
        simulate_css_memory(sparse, .08, trials=1, seed=0, decoder=decoder)
    with pytest.raises(ValueError, match='explicit decoder'):
        simulate_css_memory(sparse, .08, trials=1, seed=0, logical_frame=frame)


def test_frame_counts_converged_logical_failures():
    code = steane()
    frame = CSSLogicalFrame.from_packed(code)
    error = 1 << 6
    correction = CSSBPDecoder(SparseCSSCode.from_packed(code), .05, .05).decode(0, code.hz.matvec(error))
    assert correction.success
    assert not frame.logical_success(error, 0, correction)
