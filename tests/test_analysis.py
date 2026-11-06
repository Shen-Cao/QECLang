from itertools import combinations

import pytest

from qeclang.algebra import BinaryMatrix as M
from qeclang.codes import CSSCode, Pauli, StabilizerCode
from qeclang.analysis import (css_distance, css_logicals, css_profile,
                              css_erasure_correctable, stabilizer_distance,
                              stabilizer_erasure_correctable, stabilizer_logicals,
                              tanner_four_cycles, quotient_basis)
from qeclang.analysis.logicals import symplectic


def steane():
    h = M((0b1010101, 0b1100110, 0b1111000), 7)
    return CSSCode(h, h)


def five_qubit():
    # XZZXI, IXZZX, XIXZZ, ZXIXZ, with site 0 at the left.
    return StabilizerCode(5, tuple(Pauli(x, z) for x, z in
                                  ((0b01001, 0b00110), (0b10010, 0b01100),
                                   (0b00101, 0b11000), (0b01010, 0b10001))))


def test_steane_distances_and_pairing():
    code = steane()
    logicals = css_logicals(code)
    assert len(logicals.x) == len(logicals.z) == 1
    assert (logicals.x[0] & logicals.z[0]).bit_count() % 2 == 1
    for sector in ('x', 'z'):
        result = css_distance(code, sector)
        assert result.distance == 3
        assert result.lower_bound == result.upper_bound == 3
        assert result.witness.bit_count() == 3
    assert stabilizer_distance(code.to_stabilizer()).distance == 3


def test_asymmetric_and_no_logical_codes():
    repetition = CSSCode(M((), 3), M((3, 6), 3))
    assert css_distance(repetition, 'x').distance == 3
    assert css_distance(repetition, 'z').distance == 1
    bell = CSSCode(M((3,), 2), M((3,), 2))
    result = css_distance(bell, 'x')
    assert result.status == 'no_logicals'
    assert result.distance is None and result.witness is None
    assert css_erasure_correctable(bell, [0, 1])
    assert stabilizer_distance(bell.to_stabilizer()).status == 'no_logicals'


def test_distance_interruption_never_overstates_lower_bound():
    code = steane()
    zero = css_distance(code, 'x', max_candidates=0)
    assert zero.status == 'budget_exhausted' and zero.lower_bound == 1
    partial = css_distance(code, 'x', max_candidates=8)
    assert partial.status == 'budget_exhausted'
    assert partial.searched_through == 1 and partial.lower_bound == 2
    assert partial.candidates == 8
    limited = css_distance(code, 'x', max_weight=1)
    assert limited.status == 'weight_limit' and limited.lower_bound == 2
    assert limited.upper_bound >= 3
    with pytest.raises(ValueError):
        css_distance(code, 'y')
    with pytest.raises(ValueError):
        css_distance(code, 'x', max_candidates=-1)


def test_five_qubit_distance_pairing_and_erasures():
    code = five_qubit()
    assert code.k == 1
    assert stabilizer_distance(code).distance == 3
    a, b = stabilizer_logicals(code)[0]
    assert symplectic(a, b, 5) == 1
    for sites in combinations(range(5), 2):
        assert stabilizer_erasure_correctable(code, sites)
    assert not stabilizer_erasure_correctable(code, range(5))


def test_css_erasure_matches_stabilizer_and_logical_support():
    code = steane()
    stabilizer = code.to_stabilizer()
    for mask in range(1 << code.n):
        support = [i for i in range(code.n) if mask >> i & 1]
        assert css_erasure_correctable(code, support) == stabilizer_erasure_correctable(stabilizer, support)
    for sites in combinations(range(7), 2):
        assert css_erasure_correctable(code, sites)
    logical = css_distance(code, 'x').witness
    assert not css_erasure_correctable(code, [i for i in range(7) if logical >> i & 1])
    with pytest.raises(ValueError):
        css_erasure_correctable(code, [0, 0])
    with pytest.raises(ValueError):
        css_erasure_correctable(code, [7])


def test_profile_and_multiple_logicals():
    code = CSSCode(M((15,), 4), M((15,), 4))
    logicals = css_logicals(code)
    assert len(logicals.x) == 2
    assert [[(x & z).bit_count() % 2 for z in logicals.z]
            for x in logicals.x] == [[1, 0], [0, 1]]
    profile = css_profile(code)
    assert (profile.n, profile.k, profile.rate) == (4, 2, 0.5)
    assert (profile.max_check_weight, profile.max_qubit_degree) == (4, 2)
    assert tanner_four_cycles(M((15, 15), 4)) == 6
    assert quotient_basis(M.identity(3), M((1,), 3)) == (2, 4)
    with pytest.raises(ValueError):
        quotient_basis(M((1,), 3), M((2,), 3))


def test_empty_code_and_free_qubits():
    empty = CSSCode(M((), 0), M((), 0))
    assert css_profile(empty).rate is None
    assert css_distance(empty, 'x').status == 'no_logicals'
    assert css_erasure_correctable(empty, [])
    free = CSSCode(M((), 3), M((), 3))
    assert css_distance(free, 'x').distance == 1
    pairs = stabilizer_logicals(free.to_stabilizer())
    assert len(pairs) == 3
    for i, (a, b) in enumerate(pairs):
        for j, (c, d) in enumerate(pairs):
            assert symplectic(a, d, 3) == (i == j)
            assert not symplectic(a, c, 3) and not symplectic(b, d, 3)


def test_milp_matches_exact_and_preserves_solver_status():
    pytest.importorskip('scipy')
    from qeclang.analysis import css_distance_milp
    code = steane()
    for sector in ('x', 'z'):
        result = css_distance_milp(code, sector)
        assert result.status == 'solver_optimal'
        assert result.upper_bound == css_distance(code, sector).distance == 3
        assert result.witness.bit_count() == 3
        assert result.solver_lower_bound == pytest.approx(3)
    limited = css_distance_milp(code, 'x', time_limit=0)
    assert limited.status == 'limit'
    assert limited.subproblems_solved == 0 and limited.solver_lower_bound == 1
    multiple = CSSCode(M((15,), 4), M((15,), 4))
    result = css_distance_milp(multiple, 'x')
    assert result.subproblems_total == result.subproblems_solved == 2
    assert result.status == 'solver_optimal' and result.upper_bound == 2


def test_subsystem_distances_against_full_pauli_oracle():
    from qeclang.codes import SubsystemCode
    from qeclang.analysis import subsystem_distance
    gauge = tuple(Pauli.from_label(s) for s in ('XXII', 'IIXX', 'ZIZI', 'IZIZ'))
    code = SubsystemCode(4, gauge)
    assert (code.k, code.r) == (1, 1)
    for kind in ('dressed', 'bare'):
        result = subsystem_distance(code, kind)
        trivial, constraints = ((code.checks, code.center) if kind == 'dressed'
                                else (code.center, code.checks))
        expected = min(((v & 15) | (v >> 4)).bit_count() for v in range(1 << 8)
                       if not trivial.contains(v)
                       and all(not symplectic(v, s, 4) for s in constraints.rows))
        assert result.distance == expected == 2
    gauge_only = SubsystemCode(1, (Pauli(1, 0), Pauli(0, 1)))
    assert subsystem_distance(gauge_only).status == 'no_logicals'
    assert subsystem_distance(gauge_only, 'bare').status == 'no_logicals'
    with pytest.raises(ValueError):
        subsystem_distance(code, 'invalid')


def test_random_small_css_against_exhaustive_definition():
    from random import Random
    rng = Random(94821)
    for n in range(1, 6):
        for _ in range(5):
            hx = M(tuple(rng.randrange(1 << n) for _ in range(rng.randrange(n + 1))), n)
            kernel = hx.nullspace().rows
            hzrows = []
            for _ in range(rng.randrange(n + 1)):
                v = 0
                for row in kernel:
                    if rng.randrange(2):
                        v ^= row
                hzrows.append(v)
            code = CSSCode(hx, M(tuple(hzrows), n))
            for sector, checks, trivial in (('x', code.hz, code.hx),
                                            ('z', code.hx, code.hz)):
                weights = [v.bit_count() for v in range(1 << n)
                           if checks.matvec(v) == 0 and not trivial.contains(v)]
                exact = min(weights) if weights else None
                assert css_distance(code, sector).distance == exact
                for budget in (0, 1, 4):
                    bounded = css_distance(code, sector, max_candidates=budget)
                    if exact is not None:
                        assert bounded.lower_bound <= exact <= bounded.upper_bound
                        assert bounded.candidates <= budget
                        assert not trivial.contains(bounded.witness)
                        assert checks.matvec(bounded.witness) == 0
            for mask in range(1 << n):
                sites = [i for i in range(n) if mask >> i & 1]
                logical_supported = any(
                    v & ~mask == 0 and checks.matvec(v) == 0 and not trivial.contains(v)
                    for checks, trivial in ((code.hx, code.hz), (code.hz, code.hx))
                    for v in range(1 << n))
                assert css_erasure_correctable(code, sites) == (not logical_supported)


def test_random_subsystem_distances_against_exhaustive_definition():
    from random import Random
    from qeclang.codes import SubsystemCode
    from qeclang.analysis import subsystem_distance
    rng = Random(51295)
    for _ in range(25):
        n = 3
        code = SubsystemCode(n, tuple(Pauli(rng.randrange(8), rng.randrange(8))
                                      for _ in range(rng.randrange(6))))
        for kind in ('bare', 'dressed'):
            trivial, constraints = ((code.checks, code.center) if kind == 'dressed'
                                    else (code.center, code.checks))
            weights = [((v & 7) | (v >> 3)).bit_count() for v in range(64)
                       if not trivial.contains(v)
                       and all(not symplectic(v, s, n) for s in constraints.rows)]
            expected = min(weights) if weights else None
            assert subsystem_distance(code, kind).distance == expected
