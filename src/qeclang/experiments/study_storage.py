"""SQLite persistence for atomic candidate evaluation and trial records."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3

from qeclang.constructions.records import canonical_json, digest

SCHEMA = "qeclang.search-study.v1"


def connect(path, *, readonly=False):
    if readonly:
        connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(path, isolation_level=None)
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialize(connection, config, provenance):
    expected = {"schema": SCHEMA, "config": asdict(config), "provenance": provenance}
    expected["config_fingerprint"] = digest(expected["config"])
    expected["evaluation_fingerprint"] = digest({"evaluation": config.evaluation_config(),
                                                 "provenance": provenance})
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        current = dict(connection.execute("SELECT key,value FROM metadata"))
        if current:
            if set(current) != set(expected):
                raise ValueError("study metadata schema mismatch")
            for key, value in expected.items():
                if current[key] != canonical_json(value):
                    raise ValueError(f"study {key} mismatch; use a new database")
        else:
            connection.executemany("INSERT INTO metadata VALUES (?,?)",
                                   ((key, canonical_json(value)) for key, value in expected.items()))
        connection.execute("""CREATE TABLE IF NOT EXISTS evaluations (
            cache_key TEXT PRIMARY KEY, construction_fingerprint TEXT NOT NULL,
            evaluation_fingerprint TEXT NOT NULL, construction_json TEXT NOT NULL,
            evaluation_json TEXT NOT NULL)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS trials (
            trial INTEGER PRIMARY KEY CHECK(trial >= 0), cache_key TEXT NOT NULL,
            proposal_json TEXT NOT NULL, cache_hit INTEGER NOT NULL CHECK(cache_hit IN (0,1)),
            FOREIGN KEY(cache_key) REFERENCES evaluations(cache_key))""")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return expected["evaluation_fingerprint"]


def trial_count(connection):
    count, maximum = connection.execute("SELECT COUNT(*),COALESCE(MAX(trial),-1) FROM trials").fetchone()
    if count != maximum + 1:
        raise ValueError("study trial sequence is not contiguous")
    return count


def records(connection):
    query = """SELECT t.trial,e.cache_key,e.construction_json,e.evaluation_json
               FROM trials t JOIN evaluations e ON e.cache_key=t.cache_key ORDER BY t.trial"""
    return [{"trial": trial, "cache_key": key, "construction": json.loads(construction),
             "evaluation": json.loads(evaluation)}
            for trial, key, construction, evaluation in connection.execute(query)]


def insert_evaluation(connection, key, fingerprint, evaluation_fingerprint, construction, evaluation):
    connection.execute("INSERT INTO evaluations VALUES (?,?,?,?,?)",
                       (key, fingerprint, evaluation_fingerprint, canonical_json(construction),
                        canonical_json(evaluation)))


def insert_trial(connection, trial, key, proposal, cache_hit):
    connection.execute("INSERT INTO trials VALUES (?,?,?,?)",
                       (trial, key, canonical_json(proposal), int(cache_hit)))


def pareto(records):
    eligible = {}
    for record in records:
        if record["evaluation"]["objective"] is not None:
            eligible.setdefault(record["cache_key"], record)
    front = []
    for key, candidate in eligible.items():
        score = candidate["evaluation"]["objective"]
        dominated = any(other_key != key and
                        all(a >= b for a, b in zip(other["evaluation"]["objective"], score)) and
                        any(a > b for a, b in zip(other["evaluation"]["objective"], score))
                        for other_key, other in eligible.items())
        if not dominated:
            front.append(candidate)
    return tuple(front)


@dataclass(frozen=True)
class StudySummary:
    trials: int
    unique_evaluations: int
    cache_hits: int
    pareto: tuple[dict, ...]


def summary(connection):
    return StudySummary(trial_count(connection),
                        connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0],
                        connection.execute("SELECT COALESCE(SUM(cache_hit),0) FROM trials").fetchone()[0],
                        pareto(records(connection)))


def read_study(path):
    connection = connect(path, readonly=True)
    try:
        return summary(connection)
    finally:
        connection.close()
