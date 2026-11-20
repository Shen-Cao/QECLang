"""Command line boundary for versioned, data-only QEC programs."""
import argparse
import json
from pathlib import Path

from qeclang.workflow import ExecutionLimits, Program, execute, replay
from qeclang.workflow.json_input import JSONLimits, load_json


def register(commands):
    parser = commands.add_parser("run", help="execute a typed workflow or replay its saved record")
    parser.add_argument("program", type=Path)
    parser.add_argument("--replay", type=Path, metavar="RECORD")
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
        if args.replay is None:
            result = execute(program, **policy)
            ok = result["status"] == "completed"
        else:
            record = _read_record(args.replay, args.max_record_bytes)
            ok = replay(program, record, **policy)
            result = {"schema": "qeclang.replay.v1", "matches": ok, "program": program.fingerprint()}
    except (ValueError, OSError, RecursionError) as exc:
        result = {"schema": "qeclang.error.v1", "status": "invalid_request", "reason": str(exc)}
        ok = False
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if ok else 2
