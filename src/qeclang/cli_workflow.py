"""Command line boundary for versioned, data-only QEC programs."""
import argparse
import json
from pathlib import Path

from qeclang.workflow import ExecutionLimits, Program, execute, replay
from qeclang.workflow.json_input import JSONLimitExceeded, JSONLimits, load_json


def register(commands):
    parser = commands.add_parser("run", help="execute a typed workflow or replay its saved record")
    parser.add_argument("program", type=Path)
    saved = parser.add_mutually_exclusive_group()
    saved.add_argument("--replay", type=Path, metavar="RECORD")
    saved.add_argument("--inspect", type=Path, metavar="RECORD",
                       help="independently check saved relations and upper-bound witnesses")
    parser.add_argument("--max-inspection-relations", type=int, default=512)
    parser.add_argument("--max-inspection-certificates", type=int, default=256)
    parser.add_argument("--max-record-bytes", type=int, default=32_000_000)
    parser.add_argument("--no-relation-checks", action="store_true")
    for name, default in vars(ExecutionLimits()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)


def _read_record(path, limit):
    return load_json(path, limits=JSONLimits(max_bytes=limit, max_integer_bits=4096,
                                            max_payload_items=8_000_000))


def run(args: argparse.Namespace):
    try:
        limits = ExecutionLimits(**{k: getattr(args, k) for k in vars(ExecutionLimits())})
        policy = {"limits": limits, "verify_relations": not args.no_relation_checks}
        program = Program.load(args.program)
        if args.inspect is not None:
            from qeclang.workflow.inspect_run import InspectionLimits, inspect_run
            from qeclang.workflow.import_artifact import ImportLimits
            inspection = InspectionLimits(max_relation_checks=args.max_inspection_relations,
                max_certificates=args.max_inspection_certificates,
                record=JSONLimits(max_bytes=args.max_record_bytes, max_integer_bits=4096,
                                  max_payload_items=8_000_000),
                artifacts=ImportLimits(max_bytes=limits.max_artifact_bytes, max_axis=limits.max_axis,
                    max_matrix_bits=limits.max_matrix_bits, max_validation_work=limits.max_validation_work))
            result = inspect_run(program, _read_record(args.inspect, args.max_record_bytes), limits=inspection)
            ok = result['status'] == 'checked'
        elif args.replay is None:
            result = execute(program, **policy)
            ok = result["status"] == "completed"
        else:
            record = _read_record(args.replay, args.max_record_bytes)
            ok = replay(program, record, **policy)
            result = {"schema": "qeclang.replay.v1", "matches": ok, "program": program.fingerprint()}
    except JSONLimitExceeded as exc:
        result = {"schema": "qeclang.error.v1", "status": "budget_exhausted", "reason": str(exc)}
        ok = False
    except (ValueError, OSError, RecursionError) as exc:
        result = {"schema": "qeclang.error.v1", "status": "invalid_request", "reason": str(exc)}
        ok = False
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if ok else 2
