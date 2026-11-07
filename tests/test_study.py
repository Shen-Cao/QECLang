from dataclasses import replace
import json
import sqlite3

import pytest

from qeclang.analysis.certificates import LogicalWitness, verify_css_logical
from qeclang.constructions.records import from_dict
from qeclang.experiments import study
from qeclang.experiments.study import StudyConfig, read_study, run_study
from qeclang.experiments.study_evaluation import provenance
from qeclang.experiments.study_storage import pareto


@pytest.fixture(autouse=True)
def stable_test_provenance(monkeypatch):
    monkeypatch.setattr(study, "provenance", lambda: {"source_sha256": "test-source", "runtime": "test"})


def database_rows(path):
    with sqlite3.connect(path) as connection:
        return {table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                for table in ("metadata", "evaluations", "trials")}


@pytest.mark.parametrize("strategy", ["random", "mutate"])
def test_resume_matches_uninterrupted_trial_sequence(tmp_path, strategy):
    config = StudyConfig(lift=7, terms=2, seed=43, strategy=strategy, restart_interval=3,
                         max_weight=1, max_candidates=100)
    resumed, fresh = tmp_path / "resume.sqlite", tmp_path / "fresh.sqlite"
    assert run_study(resumed, config, target_trials=3).trials == 3
    assert run_study(resumed, config, target_trials=8).trials == 8
    run_study(fresh, config, target_trials=8)
    assert database_rows(resumed) == database_rows(fresh)
    if strategy == "mutate":
        proposals = [json.loads(row[2]) for row in database_rows(resumed)["trials"]]
        assert [p["kind"] for p in proposals[:4]] == ["restart", "mutation", "mutation", "restart"]
        for proposal in proposals:
            assert len(set(proposal["a_support"])) == len(set(proposal["b_support"])) == 2
        for row in read_study(resumed).pareto:
            code = from_dict(row["construction"]).lower()
            for witness in row["evaluation"]["logical_witnesses"].values():
                assert verify_css_logical(code, LogicalWitness(**witness))


def test_cache_and_total_target_semantics(tmp_path, monkeypatch):
    path = tmp_path / "cached.sqlite"
    config = StudyConfig(lift=3, terms=0, strategy="mutate")
    original, calls = study.evaluate_candidate, []

    def counted(construction, cfg):
        calls.append(1)
        return original(construction, cfg)

    monkeypatch.setattr(study, "evaluate_candidate", counted)
    result = run_study(path, config, target_trials=5)
    assert (result.trials, result.unique_evaluations, result.cache_hits) == (5, 1, 4)
    assert len(calls) == 1
    assert run_study(path, config, target_trials=5) == result
    assert run_study(path, config, target_trials=3) == result
    assert len(calls) == 1
    assert len(result.pareto) == 1


def test_evaluation_and_trial_rollback_together(tmp_path, monkeypatch):
    path = tmp_path / "interrupted.sqlite"
    config = StudyConfig(lift=7, terms=2, max_weight=1)
    original = study.storage.insert_trial

    def fail_second(connection, trial, *args):
        if trial == 1:
            raise RuntimeError("simulated interruption after evaluation insert")
        return original(connection, trial, *args)

    monkeypatch.setattr(study.storage, "insert_trial", fail_second)
    with pytest.raises(RuntimeError, match="interruption"):
        run_study(path, config, target_trials=4)
    partial = read_study(path)
    assert partial.trials == partial.unique_evaluations == 1
    monkeypatch.setattr(study.storage, "insert_trial", original)
    assert run_study(path, config, target_trials=4).trials == 4
    fresh = tmp_path / "fresh.sqlite"
    run_study(fresh, config, target_trials=4)
    assert database_rows(path) == database_rows(fresh)


def test_configuration_and_provenance_mismatches_never_mutate_database(tmp_path, monkeypatch):
    path = tmp_path / "fixed.sqlite"
    config = StudyConfig(lift=3, terms=1)
    run_study(path, config, target_trials=2)
    before = database_rows(path)
    for changed in (replace(config, seed=1), replace(config, max_weight=3),
                    replace(config, backend="polynomial")):
        with pytest.raises(ValueError, match="mismatch"):
            run_study(path, changed, target_trials=4)
        assert database_rows(path) == before
    monkeypatch.setattr(study, "provenance", lambda: {"source_sha256": "new-source"})
    with pytest.raises(ValueError, match="provenance mismatch"):
        run_study(path, config, target_trials=4)
    assert database_rows(path) == before


@pytest.mark.parametrize("config,status,distance_status", [
    (StudyConfig(lift=4, terms=2), "unsupported", "parameters_unavailable"),
    (StudyConfig(lift=3, terms=1, max_steps=0), "budget_exhausted", "parameters_unavailable"),
    (StudyConfig(lift=3, terms=1), "exact", "no_logicals"),
    (StudyConfig(lift=3, terms=0, max_distance_n=0), "exact", "size_limit"),
    (StudyConfig(lift=3, terms=0, max_bits=0), "exact", "expansion_budget_exhausted"),
])
def test_explicit_noneligible_statuses(tmp_path, config, status, distance_status):
    path = tmp_path / "status.sqlite"
    result = run_study(path, config, target_trials=1)
    assert not result.pareto
    evaluation = json.loads(database_rows(path)["evaluations"][0][4])
    assert evaluation["parameters"]["status"] == status
    assert evaluation["distance_status"] == distance_status
    assert evaluation["objective"] is None


def test_polynomial_even_backend_and_lower_bound_objective(tmp_path):
    path = tmp_path / "even.sqlite"
    config = StudyConfig(lift=4, terms=2, backend="polynomial", max_candidates=0)
    result = run_study(path, config, target_trials=1)
    assert result.pareto
    evaluation = result.pareto[0]["evaluation"]
    assert evaluation["parameters"]["status"] == "exact"
    assert evaluation["objective"][1] == min(evaluation["distance_x"]["lower_bound"],
                                            evaluation["distance_z"]["lower_bound"])
    assert evaluation["objective"][1] == 1


@pytest.mark.parametrize("backend,lift", [("abelian", 3), ("mixed-abelian", 4)])
def test_studies_accept_abelian_backends_and_persist_budgets(tmp_path, backend, lift):
    path = tmp_path / "abelian.sqlite"
    config = StudyConfig(lift=lift, terms=2, backend=backend, max_candidates=0)
    result = run_study(path, config, target_trials=1)
    assert result.pareto[0]["evaluation"]["parameters"]["method"] == backend
    assert result.pareto[0]["evaluation"]["parameters"]["status"] == "exact"
    with pytest.raises(ValueError, match="mismatch"):
        run_study(path, replace(config, max_component_axis=99), target_trials=2)


def test_distance_expansion_respects_configured_axis_budget(tmp_path):
    path = tmp_path / "axis.sqlite"
    config = StudyConfig(lift=3, terms=0, max_sites=5)
    result = run_study(path, config, target_trials=1)
    assert not result.pareto
    evaluation = json.loads(database_rows(path)["evaluations"][0][4])
    assert evaluation["parameters"]["status"] == "exact"
    assert evaluation["distance_status"] == "expansion_budget_exhausted"


def test_pareto_never_uses_upper_bounds():
    def record(key, objective, upper):
        return {"cache_key": key, "evaluation": {"objective": objective,
                "distance_x": {"upper_bound": upper}}}
    records = [record("a", [2, 2, -4, -4], 100), record("b", [2, 3, -4, -4], 3),
               record("c", None, 1000), record("d", [4, 1, -4, -4], 1)]
    assert {r["cache_key"] for r in pareto(records)} == {"b", "d"}


def test_study_limits_and_portable_provenance(tmp_path):
    for kwargs in ({"lift": 0}, {"terms": 8}, {"strategy": "other"},
                   {"restart_interval": 0}, {"max_weight": True}, {"seed": 1.0}):
        with pytest.raises(ValueError):
            StudyConfig(**kwargs)
    path = tmp_path / "absent.sqlite"
    with pytest.raises(ValueError):
        run_study(path, StudyConfig(max_trials=2), target_trials=3)
    assert not path.exists()
    empty = run_study(path, StudyConfig(), target_trials=0)
    assert empty.trials == empty.unique_evaluations == 0
    actual = provenance()
    assert len(actual["source_sha256"]) == 64
    assert all("/" not in str(value) for value in actual.values())
