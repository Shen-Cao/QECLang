"""Opt-in SQLite publication and lookup for checked workflow bundles."""
from dataclasses import replace
import json
from pathlib import Path


def register(commands):
    archive = commands.add_parser('archive', help='explicitly publish or read workflow evidence')
    verbs = archive.add_subparsers(dest='archive_command', required=True)
    put = verbs.add_parser('put', help='inspect and atomically publish one program/run pair')
    get = verbs.add_parser('get', help='fetch a bundle by digest and check it independently')
    reuse = verbs.add_parser('reuse', help='create a program that rebinds a saved CSS upper witness')
    for parser in (put, get, reuse):
        parser.add_argument('database', type=Path)
        parser.add_argument('--allow-prefix', action='store_true')
        parser.add_argument('--max-bundle-bytes', type=int, default=32_000_000)
        parser.add_argument('--max-reader-steps', type=int, default=1_000_000,
                            help='maximum arithmetic units for verification of symbolic rank proofs')
        parser.add_argument('--max-reader-exclusion-steps', type=int, default=1_000_000,
                            help='maximum reader work units for low-weight exclusion evidence')
    put.add_argument('program', type=Path)
    put.add_argument('record', type=Path)
    get.add_argument('key')
    get.add_argument('--expected-program', metavar='SHA256')
    get.add_argument('--include-record', action='store_true', help='include the program and run in the response')
    reuse.add_argument('key')
    reuse.add_argument('witness_node')
    reuse.add_argument('target', type=Path, help='target positive CSS code record')
    reuse.add_argument('maps', type=Path, help='explicit forward and reverse generator maps')


def run(args):
    from .archive import (ArchiveLimits, ArchiveRefused, SQLiteEvidenceStore, publish, retrieve)
    from .archive.protocol import EvidenceStoreLimitExceeded
    from .workflow import Program
    from .workflow.json_input import JSONLimitExceeded, load_json
    import sqlite3
    ok = False
    try:
        defaults = ArchiveLimits()
        limits = replace(defaults, json=replace(defaults.json, max_bytes=args.max_bundle_bytes),
                         inspection=replace(defaults.inspection, max_rank_steps=args.max_reader_steps,
                                            max_exclusion_verify_steps=args.max_reader_exclusion_steps))
        store = SQLiteEvidenceStore(args.database, readonly=args.archive_command != 'put',
                                    max_bundle_bytes=limits.json.max_bytes)
        if args.archive_command == 'put':
            result = publish(Program.load(args.program, limits=limits.program),
                             load_json(args.record, limits=limits.json), store,
                             limits=limits, allow_prefix=args.allow_prefix)
        elif args.archive_command == 'reuse':
            from .archive.reuse import make_reuse_program
            from .io import from_dict
            raw = load_json(args.target, limits=limits.json)
            if not isinstance(raw, dict) or raw.get('kind') != 'css':
                raise ValueError('archive reuse requires a positive CSS target')
            target = from_dict(raw, max_qubits=limits.inspection.artifacts.max_axis,
                               max_checks=limits.inspection.artifacts.max_axis,
                               max_bits=limits.inspection.artifacts.max_matrix_bits)
            maps = load_json(args.maps, limits=limits.json)
            result = make_reuse_program(store, args.key, args.witness_node, target, maps,
                                        allow_prefix=args.allow_prefix, limits=limits).to_dict()
        else:
            result = retrieve(store, args.key, expected_program=args.expected_program,
                              limits=limits, allow_prefix=args.allow_prefix)
            if not args.include_record:
                result.pop('program')
                result.pop('run')
        ok = True
    except ArchiveRefused as exc:
        result = {'status': 'refused', 'inspection': exc.inspection}
    except (JSONLimitExceeded, EvidenceStoreLimitExceeded) as exc:
        result = {'status': 'budget_exhausted', 'reason': str(exc)}
    except KeyError:
        result = {'status': 'not_found'}
    except (ValueError, OSError, sqlite3.Error) as exc:
        result = {'status': 'invalid_request', 'reason': str(exc)}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if ok else 2
