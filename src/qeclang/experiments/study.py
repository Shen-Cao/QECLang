"""Bounded, deterministic, resumable studies of cyclic scalar lifted products."""
from random import Random
import json

from qeclang.constructions.lifted import LiftedProduct
from qeclang.constructions.records import digest, fingerprint, from_dict, to_dict
from qeclang.symbolic import GroupAlgebraElement, GroupAlgebraMatrix

from .study_evaluation import StudyConfig, evaluate_candidate, provenance
from . import study_storage as storage
from .study_storage import StudySummary, read_study


def _candidate(support_a, support_b, lift):
    def matrix(support):
        element = GroupAlgebraElement.from_terms((lift,), ((i,) for i in support))
        return GroupAlgebraMatrix(((element,),), 1, (lift,))
    return LiftedProduct(matrix(support_a), matrix(support_b))


def _proposal(config, trial, history):
    rng_seed = digest({"seed": config.seed, "trial": trial,
                       "algorithm": "cyclic-scalar-fixed-support.v1"})
    rng = Random(int(rng_seed, 16))
    parent = None
    eligible = [record for record in history if record["evaluation"]["objective"] is not None]
    mutate = config.strategy == "mutate" and trial % config.restart_interval and eligible
    if mutate:
        parent = max(eligible, key=lambda record: (
            record["evaluation"]["objective"][1], record["evaluation"]["objective"][0],
            record["evaluation"]["objective"][2], record["evaluation"]["objective"][3],
            -record["trial"]))
        code = from_dict(parent["construction"])
        supports = [sorted(t[0] for t in code.a.rows[0][0].terms),
                    sorted(t[0] for t in code.b.rows[0][0].terms)]
        kind = "unchanged"
        if 0 < config.terms < config.lift:
            axis = rng.randrange(2)
            support = set(supports[axis])
            removed = rng.choice(sorted(support))
            added = rng.choice([i for i in range(config.lift) if i not in support])
            supports[axis] = sorted((support - {removed}) | {added})
            kind = "mutation"
    else:
        supports = [sorted(rng.sample(range(config.lift), config.terms)) for _ in range(2)]
        kind = "random" if config.strategy == "random" else "restart"
    construction = _candidate(*supports, config.lift)
    proposal = {"kind": kind, "parent_trial": None if parent is None else parent["trial"],
                "a_support": supports[0], "b_support": supports[1], "rng_seed_sha256": rng_seed}
    return construction, proposal


def run_study(path, config: StudyConfig, *, target_trials: int) -> StudySummary:
    """Reach a total trial count, preserving every completed transaction.

    Resuming with changed configuration, source contents, or Python runtime is
    rejected. Each evaluation and its trial commit together. An exception rolls
    back the current trial; prior trials remain resumable. This API performs no
    network actions or process execution.
    """
    if not isinstance(config, StudyConfig):
        raise TypeError("config must be StudyConfig")
    if type(target_trials) is not int or not 0 <= target_trials <= config.max_trials:
        raise ValueError("target_trials must be within the configured trial budget")
    connection = storage.connect(path)
    try:
        evaluation_fingerprint = storage.initialize(connection, config, provenance())
        while True:
            connection.execute("BEGIN IMMEDIATE")
            try:
                trial = storage.trial_count(connection)
                if trial >= target_trials:
                    connection.commit()
                    break
                history = storage.records(connection) if config.strategy == "mutate" else ()
                construction, proposal = _proposal(config, trial, history)
                construction_fingerprint = fingerprint(construction)
                key = digest({"construction": construction_fingerprint,
                              "evaluation": evaluation_fingerprint})
                cached = connection.execute("SELECT evaluation_json FROM evaluations WHERE cache_key=?",
                                            (key,)).fetchone()
                if cached is None:
                    evaluation = evaluate_candidate(construction, config)
                    storage.insert_evaluation(connection, key, construction_fingerprint,
                                              evaluation_fingerprint, to_dict(construction), evaluation)
                else:
                    json.loads(cached[0])
                storage.insert_trial(connection, trial, key, proposal, cached is not None)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return storage.summary(connection)
    finally:
        connection.close()


__all__ = ["StudyConfig", "StudySummary", "run_study", "read_study"]
