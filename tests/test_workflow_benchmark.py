import importlib.util
import json
from pathlib import Path

import pytest

from qeclang.workflow import ExecutionLimits, execute

_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_workflow.py"
_SPEC = importlib.util.spec_from_file_location("qeclang_workflow_benchmark", _PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


@pytest.mark.parametrize("case", benchmark.CASES)
@pytest.mark.parametrize("candidate_limit", [0, 100_000])
def test_manual_baseline_produces_the_same_artifacts_under_the_same_search_cap(case, candidate_limit):
    _, program = benchmark.make_case(case)
    limits = ExecutionLimits(max_candidates=candidate_limit)
    expected = benchmark.materialize(program, benchmark.direct_kernels(program, limits))
    for checked in (False, True):
        record = execute(program, limits=limits, verify_relations=checked)
        assert record["status"] == "completed"
        assert record["artifacts"] == expected["artifacts"]
        assert record["outputs"] == expected["outputs"]


def test_measurement_records_every_phase_without_speed_assumptions(monkeypatch):
    tick = 0
    def clock():
        nonlocal tick
        tick += 1000
        return tick
    monkeypatch.setattr(benchmark, "perf_counter_ns", clock)
    result = benchmark.benchmark_case("steane-exact", repeats=2, warmups=0, order_seed=5)
    assert result["kind"] == "timing"
    assert result["distance"]["status"] == "exact"
    assert result["inspection"]["status"] == "checked"
    assert len(result["timings"]) == 8 and len(result["observations"]) == 16
    for timing in result["timings"].values():
        assert timing["observations"] == 2
        assert timing["median_seconds"] == pytest.approx(1e-6)
    assert result["checked_ledger"]["used"]["optional_checks"] > 0
    assert result["unchecked_ledger"]["used"]["optional_checks"] == 0
    assert result["unchecked_ledger"]["used"]["transports"] == 1
    json.dumps(result, allow_nan=False)


def test_rehashed_scalar_preserving_corruption_exposes_both_verification_gates():
    import qeclang.workflow.executor as executor
    original = executor.evaluate
    result = benchmark.corruption_fixture()
    assert executor.evaluate is original
    assert result["source_parameters"] == result["target_parameters"] == {"n": 4, "k": 2}
    assert result["source"]["hx"] != result["wrong_target"]["hx"]
    assert result["artifact_hashes_recomputed"]
    assert result["checked_status"] == "verification_failed"
    assert result["unchecked_status"] == "completed"
    assert result["inspection"]["status"] == "invalid_record"
    assert result["mandatory_transport_status"] == "verification_failed"
    assert result["mandatory_transport_step"]["op"] == "css.transport"


@pytest.mark.parametrize("options", [{"repeats": 0}, {"repeats": True},
                                     {"warmups": -1}, {"order_seed": False}])
def test_invalid_measurement_controls_are_rejected(options):
    with pytest.raises(ValueError):
        benchmark.benchmark_case("steane-exact", **options)
