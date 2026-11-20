import pytest

from qeclang.algebra.polynomial import ArithmeticBudget, WorkLimitExceeded
from qeclang.analysis.structured import analyze_lifted
from qeclang.constructions.lifted import LiftedProduct
from qeclang.symbolic import GroupAlgebraElement as G, GroupAlgebraMatrix as M
from qeclang.symbolic.abelian_analysis import abelian_rank
from qeclang.symbolic.cyclic_analysis import cyclic_rank
from qeclang.symbolic.mixed_abelian import abelian_rank_mixed
from qeclang.symbolic.repeated_root import cyclic_rank_general


def seeds(orders, rectangular=False):
    a = G.from_terms(orders, [(0,) * len(orders), (1,) * len(orders)])
    b = G.from_terms(orders, [(0,) * len(orders), (2,) * len(orders)])
    return LiftedProduct(M(((a, b, a),) if rectangular else ((a,),), 3 if rectangular else 1, orders),
                         M(((b,),), 1, orders))


CASES = (("crt", cyclic_rank, (7,)), ("polynomial", cyclic_rank_general, (6,)),
         ("abelian", abelian_rank, (3, 5)), ("mixed-abelian", abelian_rank_mixed, (2, 3)))


@pytest.mark.parametrize("backend,rank,orders", CASES)
def test_injected_meter_keeps_invocation_local_counts(backend, rank, orders):
    matrix = seeds(orders).symbolic_checks()[0]
    standalone = rank(matrix)
    meter = ArithmeticBudget()
    meter.charge(37)
    shared = rank(matrix, _meter=meter)
    assert shared.rank == standalone.rank == matrix.lower().rank()
    assert shared.steps == standalone.steps
    assert meter.used == 37 + shared.steps
    with pytest.raises(TypeError):
        rank(matrix, _meter=object())


@pytest.mark.parametrize("backend,rank,orders", CASES)
def test_shared_steps_record_partial_x_and_z_refusals(backend, rank, orders):
    construction = seeds(orders)
    complete = analyze_lifted(construction, backend=backend)
    assert complete.status == "exact"
    assert complete.steps == complete.x_evidence.steps + complete.z_evidence.steps
    for maximum in (1, complete.x_evidence.steps + 1, complete.steps - 1):
        stopped = analyze_lifted(construction, backend=backend, max_steps=maximum)
        assert stopped.status == "budget_exhausted"
        assert 0 < stopped.steps <= maximum
        assert stopped.k is stopped.rank_x is stopped.rank_z is None
    assert analyze_lifted(construction, backend=backend, max_steps=complete.steps).status == "exact"


@pytest.mark.parametrize("backend,rank,orders", CASES)
def test_z_shape_preflight_keeps_completed_x_spend(backend, rank, orders):
    construction = seeds(orders, rectangular=True)
    complete = analyze_lifted(construction, backend=backend)
    # Hx has shape 1x4, Hz 3x4: only Hz exceeds this entry ceiling.
    stopped = analyze_lifted(construction, backend=backend, max_entries=8)
    assert stopped.status == "budget_exhausted"
    assert stopped.steps == complete.x_evidence.steps > 0
    assert stopped.k is None


@pytest.mark.parametrize("backend,rank,orders", CASES)
def test_shape_refusal_and_zero_budget_do_not_invent_work(backend, rank, orders):
    construction = seeds(orders)
    for limits in ({"max_entries": 0}, {"max_blocks": 0}, {"max_steps": 0}):
        stopped = analyze_lifted(construction, backend=backend, **limits)
        assert stopped.status == "budget_exhausted" and stopped.steps == 0
    matrix = construction.symbolic_checks()[0]
    meter = ArithmeticBudget()
    meter.charge(9)
    options = {"max_seed_entries": 0} if backend == "mixed-abelian" else {"max_entries": 0}
    with pytest.raises(WorkLimitExceeded):
        rank(matrix, _meter=meter, **options)
    assert meter.used == 9


def test_noncharge_failures_preserve_metered_search_work():
    degree = analyze_lifted(seeds((7,)), backend="abelian", max_field_degree=2)
    assert degree.status == "budget_exhausted" and degree.steps > 0
    one, x2, zero = (G.from_terms((3,), terms) for terms in ([(0,)], [(2,)], []))
    construction = LiftedProduct(M(((one, x2), (x2, zero)), 2, (3,)), M(((zero,),), 1, (3,)))
    growing = analyze_lifted(construction, backend="polynomial", max_polynomial_degree=3)
    assert growing.status == "budget_exhausted" and growing.steps > 0
    assert "intermediate" in growing.reason
    operations = analyze_lifted(construction, backend="polynomial", max_operations=0)
    assert operations.status == "budget_exhausted" and operations.steps > 0
    assert "operations" in operations.reason


def test_packed_remains_explicitly_unmetered_and_unsupported_is_zero():
    construction = seeds((2,))
    assert analyze_lifted(construction, backend="crt").steps == 0
    assert analyze_lifted(construction, backend="packed").steps is None
    assert analyze_lifted(construction, backend="packed", max_bits=0).steps is None


def test_failed_bulk_reservation_does_not_charge_the_rejected_request():
    meter = ArithmeticBudget(5)
    meter.charge(2)
    with pytest.raises(WorkLimitExceeded):
        meter.charge(4)
    assert meter.used == 2
