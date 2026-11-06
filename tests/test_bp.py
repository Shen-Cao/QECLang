from itertools import product
from math import isfinite

import pytest

from qeclang.algebra import BinaryMatrix as M, SparseBinaryMatrix
from qeclang.codes import CSSCode
from qeclang.decoding.belief_propagation import BPDecoder, CSSBPDecoder


def test_repetition_tree_recovers_each_single_error():
    checks = M((3, 6, 12, 24), 5)
    decoder = BPDecoder(checks, 0.05, min_sum_scale=1.0)
    for site in range(5):
        error = 1 << site
        result = decoder.decode(checks.matvec(error))
        assert result.success and result.status == 'converged'
        assert result.error == error
        assert result.syndrome_residual == 0
        assert result.iterations <= decoder.max_iterations
    assert decoder.decode(0).iterations == 0


def test_per_site_priors_and_zero_iterations():
    checks = M((3,), 2)
    decoder = BPDecoder(checks, [0.9, 0.1], max_iterations=0)
    result = decoder.decode(1)
    assert result.status == 'converged' and result.error == 1
    failed = decoder.decode(0)
    assert failed.status == 'not_converged' and failed.error is None
    assert failed.last_decision == 1 and failed.syndrome_residual == 1
    assert failed.iterations == 0


def test_osd_order_zero_keeps_reliable_decisions():
    checks = M((3,), 2)
    decoder = BPDecoder(checks, [0.49, 0.001], max_iterations=0, fallback='osd0')
    result = decoder.decode(1)
    assert result.status == 'osd0' and result.error == 1
    assert result.success and result.syndrome_residual == 0
    # Reliability chooses site 1 instead when the channel favors that error.
    other = BPDecoder(checks, [0.001, 0.49], max_iterations=0, fallback='osd0').decode(1)
    assert other.error == 2
    # A reliable high-probability error is retained outside the pivot correction.
    retained = BPDecoder(checks, [0.49, 0.999], max_iterations=0, fallback='osd0').decode(0)
    assert retained.error == 3


def test_fallback_consistency_for_all_small_syndromes():
    from random import Random
    rng = Random(783)
    for n in range(1, 7):
        for _ in range(5):
            checks = M(tuple(rng.randrange(1 << n) for _ in range(4)), n)
            decoder = BPDecoder(checks, 0.1, max_iterations=2, fallback='osd0')
            possible = {checks.matvec(v) for v in range(1 << n)}
            for syndrome in range(1 << checks.nrows):
                result = decoder.decode(syndrome)
                assert result.success == (syndrome in possible)
                if result.success:
                    assert checks.matvec(result.error) == syndrome
                else:
                    assert result.status == 'inconsistent'


def test_sparse_packed_equivalence_reuse_and_damping():
    checks = M((7, 14, 13), 4)
    sparse = SparseBinaryMatrix.from_packed(checks)
    packed_decoder = BPDecoder(checks, [0.1, 0.2, 0.3, 0.4], damping=0.4,
                               max_iterations=5, fallback='osd0')
    sparse_decoder = BPDecoder(sparse, [0.1, 0.2, 0.3, 0.4], damping=0.4,
                               max_iterations=5, fallback='osd0')
    for syndrome in range(8):
        result = packed_decoder.decode(syndrome)
        assert sparse_decoder.decode(syndrome) == result
        assert packed_decoder.decode(syndrome) == result
        assert all(isfinite(llr) for llr in result.posterior_llrs)
    assert sum(map(len, sparse_decoder.edges_by_check)) == checks.nnz
    assert sum(map(len, sparse_decoder.edges_by_variable)) == checks.nnz


def test_singletons_zero_rows_and_extreme_probabilities():
    checks = M((1, 2, 0), 2)
    decoder = BPDecoder(checks, [5e-324, 1-1e-16])
    for syndrome in range(4):
        result = decoder.decode(syndrome)
        assert result.status == 'converged' and result.error == syndrome
        assert all(isfinite(v) for v in result.posterior_llrs)
    assert decoder.decode(4).status == 'inconsistent'
    assert BPDecoder(M((1, 1), 1)).decode(1).status == 'inconsistent'
    empty = BPDecoder(M((), 0), 0.1)
    assert empty.decode(0).success and empty.decode(0).error == 0
    free = BPDecoder(M((), 3), [0.9, 0.1, 0.8])
    assert free.decode(0).error == 5


def test_css_sectors_match_exact_syndromes_and_single_error_logicals():
    h = M((0b1010101, 0b1100110, 0b1111000), 7)
    code = CSSCode(h, h)
    decoder = CSSBPDecoder(code, 0.05, 0.05, min_sum_scale=0.4, fallback='osd0')
    for site, label in product(range(code.n), (1, 2, 3)):
        x, z = (label & 1) << site, (label >> 1) << site
        sx, sz = code.hx.matvec(z), code.hz.matvec(x)
        result = decoder.decode(sx, sz)
        assert result.success
        assert code.hz.matvec(result.x.error) == sz
        assert code.hx.matvec(result.z.error) == sx
        assert code.hx.contains(result.x.error ^ x)
        assert code.hz.contains(result.z.error ^ z)


def test_fallback_materialization_limit_and_failure_status():
    checks = M((3,), 2)
    limit = BPDecoder(checks, 0.1, max_iterations=0, fallback='osd0', max_osd_bits=1)
    result = limit.decode(1)
    assert result.status == 'fallback_limit' and not result.success
    result = BPDecoder(checks, 0.1, max_iterations=3).decode(1)
    assert result.status == 'not_converged' and result.iterations == 3
    assert result.error is None and result.syndrome_residual == 1


@pytest.mark.parametrize('probabilities', [0, 1, float('nan'), [0.1], [0.1, 1]])
def test_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        BPDecoder(M((3,), 2), probabilities)


@pytest.mark.parametrize('options', [{'max_iterations': -1}, {'max_iterations': True},
                                     {'min_sum_scale': 0}, {'min_sum_scale': 1.1},
                                     {'damping': 1}, {'damping': -0.1},
                                     {'llr_clip': float('inf')}, {'llr_clip': 0},
                                     {'fallback': 'unknown'}, {'max_osd_bits': -1}])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        BPDecoder(M((3,), 2), **options)


def test_invalid_syndrome():
    decoder = BPDecoder(M((3,), 2))
    for syndrome in (-1, 2, True):
        with pytest.raises(ValueError):
            decoder.decode(syndrome)


def test_convergence_is_not_a_logical_success_guarantee():
    h = M((85, 102, 120), 7)
    code = CSSCode(h, h)
    decoder = CSSBPDecoder(code, 0.05, 0.05)
    error = 1 << 6
    result = decoder.decode(0, h.matvec(error))
    assert result.success and result.x.status == 'converged'
    assert h.matvec(result.x.error) == h.matvec(error)
    assert not h.contains(result.x.error ^ error)


def test_memory_simulation_decoder_selection_and_provenance():
    from qeclang.decoding import simulate_css_memory
    h = M((85, 102, 120), 7)
    code = CSSCode(h, h)
    decoder = CSSBPDecoder(code, 0.05, 0.05, fallback='osd0')
    a = simulate_css_memory(code, 0.05, trials=200, seed=91, decoder=decoder)
    b = simulate_css_memory(code, 0.05, trials=200, seed=91, decoder=decoder)
    assert a == b
    assert a.decoder_name == 'css_normalized_min_sum'
    assert a.decoder_configuration['x']['fallback'] == 'osd0'
    assert a.decoder_configuration['x']['probabilities'] == [0.05] * 7
    assert a.decoder_configuration['z']['max_iterations'] == 50
    assert a.logical_failures > 0 and a.decoder_failures == 0
    ordinary = simulate_css_memory(code, 0.05, trials=20, seed=91)
    explicit_default = simulate_css_memory(code, 0.05, trials=20, seed=91, decoder=None)
    assert ordinary == explicit_default
    assert ordinary.decoder_name == 'css_minimum_weight'
    assert ordinary.decoder_configuration['max_candidates_per_sector'] == 1_000_000
    different = CSSCode(M((), 7), M((), 7))
    with pytest.raises(ValueError, match='presentation'):
        simulate_css_memory(different, 0.1, trials=1, seed=1, decoder=decoder)
    decoder.x.max_iterations = 10
    assert a.decoder_configuration['x']['max_iterations'] == 50
