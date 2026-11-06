from dataclasses import FrozenInstanceError
from itertools import product
from random import Random

import pytest

from qeclang.algebra import BinaryMatrix as M, SparseBinaryMatrix as S
from qeclang.codes import CSSCode, SparseCSSCode, same_css_presentation
from qeclang.decoding import CSSBPDecoder, simulate_css_memory


def steane():
    h = M((85, 102, 120), 7)
    return CSSCode(h, h, 'Steane')


def test_sparse_css_preserves_presentation_and_syndrome():
    code = steane()
    sparse = SparseCSSCode.from_packed(code)
    assert sparse.n == code.n
    assert same_css_presentation(code, sparse)
    assert sparse.to_packed() == code
    for x, z in product(range(1 << code.n), (0, 1, 7, 31, 127)):
        assert sparse.syndrome(x, z) == (code.hx.matvec(z), code.hz.matvec(x))
    sx, sz = sparse.syndrome_bits([1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0])
    expected = sparse.syndrome(5, 2)
    assert (sum(v << i for i, v in enumerate(sx)), sum(v << i for i, v in enumerate(sz))) == expected
    with pytest.raises(AttributeError):
        _ = sparse.k
    with pytest.raises(FrozenInstanceError):
        sparse.name = 'other'


def test_commutation_and_work_budget_are_exact():
    h = S(((0, 1),), 2)
    assert SparseCSSCode(h, h, max_validation_work=8).validation_work == 8
    with pytest.raises(ValueError, match='max_validation_work'):
        SparseCSSCode(h, h, max_validation_work=7)
    with pytest.raises(ValueError, match='do not commute'):
        SparseCSSCode(S(((0,),), 2), h)
    with pytest.raises(ValueError, match='same number'):
        SparseCSSCode(S((), 2), S((), 3))
    with pytest.raises(TypeError, match='SparseBinaryMatrix'):
        SparseCSSCode(M((), 2), M((), 2))
    with pytest.raises(ValueError):
        SparseCSSCode(h, h, max_validation_work=True)


def test_no_hidden_dense_or_rank_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected packed materialization or rank')
    monkeypatch.setattr(S, 'to_packed', forbidden)
    monkeypatch.setattr(M, 'rank', forbidden)
    # Large physical dimension with small supports must not allocate n columns.
    large = SparseCSSCode(S(((0, 999_999_999),), 1_000_000_000),
                          S(((0, 999_999_999),), 1_000_000_000))
    assert large.n == 1_000_000_000
    assert large.locality.max_check_weight == 2
    assert large.locality.max_qubit_degree == 2
    assert large.syndrome(0, 0) == (0, 0)
    h = S(((0, 1),), 2)
    small = SparseCSSCode(h, h)
    assert CSSBPDecoder(small, .1, .1, max_iterations=2).decode(0, 0).success
    assert same_css_presentation(small, CSSCode(M((3,), 2), M((3,), 2)))


def test_lowering_guard_includes_validation_product():
    zero_checks = S(((),) * 20, 1)
    sparse = SparseCSSCode(zero_checks, zero_checks)
    # Forty check payload bits plus a 20-by-20 validation product.
    with pytest.raises(ValueError, match='max_bits'):
        sparse.to_packed(max_bits=439)
    packed = sparse.to_packed(max_bits=440)
    assert packed.k == 1
    empty = SparseCSSCode(S((), 0), S((), 0), max_validation_work=0)
    assert empty.to_packed(max_bits=0).k == 0
    assert empty.locality.max_check_weight == empty.locality.max_qubit_degree == 0


def test_random_commutation_agrees_with_packed_product():
    rng = Random(8361)
    for n in range(6):
        for _ in range(20):
            hx = M(tuple(rng.randrange(1 << n) for _ in range(rng.randrange(5))), n)
            hz = M(tuple(rng.randrange(1 << n) for _ in range(rng.randrange(5))), n)
            if (hx @ hz.T).is_zero:
                sparse = SparseCSSCode(S.from_packed(hx), S.from_packed(hz))
                assert sparse.to_packed() == CSSCode(hx, hz)
                expected = hx.nrows + hz.nrows + hx.nnz + hz.nnz
                expected += sum((a & b).bit_count() for a in hx.rows for b in hz.rows)
                assert sparse.validation_work == expected
            else:
                with pytest.raises(ValueError, match='do not commute'):
                    SparseCSSCode(S.from_packed(hx), S.from_packed(hz))


def test_sparse_bp_and_explicit_simulation_analysis():
    packed = steane()
    sparse = SparseCSSCode.from_packed(packed)
    sparse_decoder = CSSBPDecoder(sparse, .05, .05, fallback='osd0')
    packed_decoder = CSSBPDecoder(packed, .05, .05, fallback='osd0')
    for sx, sz in product(range(8), repeat=2):
        assert sparse_decoder.decode(sx, sz) == packed_decoder.decode(sx, sz)
    with pytest.raises(ValueError, match='explicit packed analysis'):
        simulate_css_memory(sparse, .05, trials=20, seed=37, decoder=sparse_decoder)
    a = simulate_css_memory(packed, .05, trials=100, seed=37, decoder=sparse_decoder)
    b = simulate_css_memory(packed, .05, trials=100, seed=37, decoder=packed_decoder)
    assert a == b


def test_presentation_matching_preserves_row_order_and_redundancy():
    packed = CSSCode(M((3, 6), 3), M((), 3))
    sparse = SparseCSSCode.from_packed(packed)
    assert same_css_presentation(packed, sparse)
    assert same_css_presentation(sparse, CSSCode(packed.hx, packed.hz, 'renamed'))
    reordered = SparseCSSCode(S(((1, 2), (0, 1)), 3), S((), 3))
    assert not same_css_presentation(packed, reordered)
    redundant = SparseCSSCode(S(((0, 1), (1, 2), (0, 2)), 3), S((), 3))
    assert not same_css_presentation(sparse, redundant)
    assert not same_css_presentation(sparse, object())


def test_exact_decoding_rejects_implicit_sparse_analysis():
    from qeclang.decoding import css_decode, css_erasure_decode, css_logical_success
    sparse = SparseCSSCode.from_packed(steane())
    correction = CSSBPDecoder(sparse).decode(0, 0)
    for operation in (lambda: css_decode(sparse, 0, 0),
                      lambda: css_erasure_decode(sparse, 0, 0, []),
                      lambda: css_logical_success(sparse, 0, 0, correction)):
        with pytest.raises(TypeError, match='explicitly lower'):
            operation()
