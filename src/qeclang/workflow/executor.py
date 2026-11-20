"""Serial execution and replay of closed, typed QEC programs."""
from collections import Counter

from qeclang.constructions.records import canonical_json
from qeclang.provenance import runtime_provenance
from .artifacts import describe
from .budget import BudgetExceeded, ExecutionLimits, Ledger
from .operations import UnavailableEvidence, evaluate
from .lift_operations import evaluate_lift
from .lift_relations import comparison_units, check_lift
from .json_input import JSONLimitExceeded
from .program import Program
from .relations import BASED_RELATIONS, RelationError, check_relation


def _evidence(op, args, params, value, ledger, verify):
    if op in BASED_RELATIONS or op == 'lift.css':
        if not verify:
            return {"assurance": "not_requested"}
        if not ledger.remaining("optional_checks"):
            return {"assurance": "budget_exhausted", "resource": "optional_checks"}
        if op == 'lift.css':
            work = comparison_units(args[0])
            if work > ledger.remaining('lift_comparison_units'):
                return {'assurance': 'budget_exhausted', 'resource': 'lift_comparison_units'}
            ledger.charge('lift_comparison_units', work)
        ledger.charge("optional_checks", 1)
        guarantees = (check_lift(args[0], value) if op == 'lift.css'
                      else check_relation(op, args[0], value, params))
        return {"assurance": "relation_checked", "verifier": op + ".relation.v1",
                "guarantees": list(guarantees)}
    if op in ("css.witness", "css.transport"):
        return {"assurance": "independently_checked", "scope": "instance",
                "predicate": "distance_upper_bound", "source": value["source"],
                "sector": value["claim"]["sector"], "bound": value["claim"]["distance_upper_bound"],
                "verifier": "qeclang.certificate.v1/css_logical"}
    if op == "css.distance":
        return {"assurance": "producer_computed", "scope": "instance", "predicate": "sector_distance",
                "source": value["source"], "sector": value["sector"], "search_status": value["status"],
                "lower_bound": value["lower_bound"], "upper_bound": value["upper_bound"]}
    if op == 'lift.parameters':
        return {'assurance': 'producer_computed' if value['status'] == 'exact' else 'unavailable',
                'scope': 'instance', 'predicate': 'logical_dimension', 'analysis_status': value['status'],
                'source': value['construction_fingerprint'], 'value': value['k'], 'method': value['method']}
    return {"assurance": "core_validated"}


def execute(program: Program, *, limits=ExecutionLimits(), verify_relations=True):
    """Evaluate nodes in declaration order, retaining completed work on refusal.

    A bounded distance interval is a usable result. Optional relation checks
    may be skipped for lack of verification budget without suppressing a value.
    Witness production/transport always verify certificates. Invalid evidence
    or a failed relation stops execution before publishing the affected value.
    """
    if not isinstance(program, Program) or not isinstance(limits, ExecutionLimits):
        raise TypeError("execute requires a parsed Program and ExecutionLimits")
    if type(verify_relations) is not bool:
        raise TypeError("verify_relations must be boolean")
    ledger = Ledger(limits)
    consumers = Counter(i for node in program.nodes for i in node.inputs)
    runtime = runtime_provenance()
    trace, artifacts, values, ids = [], {}, {}, {}
    status = "completed"
    for node in program.nodes:
        start = ledger.used.copy()
        step = {"node": node.id, "op": node.op, "inputs": [ids[i] for i in node.inputs]}
        try:
            params = node.params
            args = [values[i] for i in node.inputs]
            value = (evaluate_lift(node.op, args, params, ledger, program.limits)
                     if node.op.startswith('lift.') else evaluate(node.op, args, params, ledger))
            evidence = _evidence(node.op, args, params, value, ledger, verify_relations)
            try:
                artifact = describe(node.result_type, value)
            except JSONLimitExceeded:
                raise
            except (ValueError, TypeError) as exc:
                raise ArithmeticError(f"{node.op} produced an invalid artifact") from exc
            key = artifact["digest"]
            if key not in artifacts:
                ledger.charge("artifact_bytes", len(canonical_json(artifact).encode()))
                artifacts[key] = artifact
            if consumers[node.id]:
                values[node.id] = value
            ids[node.id] = key
            step.update(status="completed", output=key, evidence=evidence)
            for i in node.inputs:
                consumers[i] -= 1
                if not consumers[i]:
                    del values[i]
        except BudgetExceeded as exc:
            status = "budget_exhausted"
            step.update(status=status, reason=str(exc), **exc.detail)
        except JSONLimitExceeded as exc:
            status = "budget_exhausted"
            step.update(status=status, reason=str(exc), resource="artifact_admission")
        except RelationError as exc:
            status = "verification_failed"
            step.update(status=status, reason=str(exc))
        except UnavailableEvidence as exc:
            status = "evidence_unavailable"
            step.update(status=status, reason=str(exc))
        except ValueError as exc:
            status = "invalid_input"
            step.update(status=status, reason=str(exc))
        step["spent"] = {k: ledger.used[k]-v for k, v in start.items()}
        trace.append(step)
        if status != "completed":
            break
    return {"schema": "qeclang.run.v1", "program": program.fingerprint(), "status": status,
            "runtime": runtime, "verify_relations": verify_relations,
            "ledger": ledger.record(), "trace": trace, "artifacts": artifacts,
            "outputs": {i: ids[i] for i in program.outputs if i in ids}}


def replay(program: Program, record: dict, *, limits=ExecutionLimits(), verify_relations=True):
    """Check strict reproducibility, under the caller's limits and assurance policy.

    Compare canonical JSON, which distinguishes boolean and integer fields.
    A match replays a particular run, including a refusal or an interval.
    """
    if not isinstance(record, dict) or record.get("schema") != "qeclang.run.v1":
        return False
    if record.get("program") != program.fingerprint():
        return False
    if record.get("runtime") != runtime_provenance():
        return False
    ledger = record.get("ledger")
    if not isinstance(ledger, dict) or ledger.get("limits") != vars(limits):
        return False
    if record.get("verify_relations") is not verify_relations:
        return False
    actual = execute(program, limits=limits, verify_relations=verify_relations)
    try:
        return canonical_json(actual) == canonical_json(record)
    except (TypeError, ValueError, RecursionError):
        return False
