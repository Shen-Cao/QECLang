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
    parser.add_argument("--allow-prefix", action="store_true", help="inspect completed nodes before a reported refusal")
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
        if args.allow_prefix and args.inspect is None:
            raise ValueError('--allow-prefix applies only to inspection')
        if args.inspect is not None:
            from qeclang.workflow.inspect_run import InspectionLimits, inspect_run
            from qeclang.workflow.import_artifact import ImportLimits
            inspection = InspectionLimits(max_relation_checks=args.max_inspection_relations,
                max_certificates=args.max_inspection_certificates,
                max_rank_steps=limits.max_rank_steps, max_blocks=limits.max_blocks,
                max_symbolic_terms=limits.max_symbolic_terms, max_lift_degree=limits.max_lift_degree,
                max_polynomial_degree=limits.max_polynomial_degree,
                max_polynomial_operations=limits.max_polynomial_operations,
                max_lift_comparison_units=limits.max_lift_comparison_units,
                max_row_map_work=limits.max_row_map_work,
                max_interval_transport_work=limits.max_interval_transport_work,
                max_exclusion_verify_steps=limits.max_exclusion_verify_steps,
                max_exclusion_indices=limits.max_exclusion_indices,
                record=JSONLimits(max_bytes=args.max_record_bytes, max_integer_bits=4096,
                                  max_payload_items=8_000_000),
                artifacts=ImportLimits(max_bytes=limits.max_artifact_bytes, max_axis=limits.max_axis,
                    max_matrix_bits=limits.max_matrix_bits, max_validation_work=limits.max_validation_work,
                    max_low_weight_indices=limits.max_exclusion_indices))
            result = inspect_run(program, _read_record(args.inspect, args.max_record_bytes), limits=inspection,
                                 allow_prefix=args.allow_prefix)
            ok = result['status'] in ('checked', 'checked_prefix')
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
