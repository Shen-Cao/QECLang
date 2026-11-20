"""Opt-in SQLite publication and lookup for checked workflow bundles."""
from dataclasses import replace
import json
from pathlib import Path


def register(commands):
    archive = commands.add_parser('archive', help='explicitly publish or read workflow evidence')
    verbs = archive.add_subparsers(dest='archive_command', required=True)
    put = verbs.add_parser('put', help='inspect and atomically publish one program/run pair')
    get = verbs.add_parser('get', help='fetch a bundle by digest and check it independently')
    for parser in (put, get):
        parser.add_argument('database', type=Path)
        parser.add_argument('--allow-prefix', action='store_true')
        parser.add_argument('--max-bundle-bytes', type=int, default=32_000_000)
        parser.add_argument('--max-reader-steps', type=int, default=1_000_000,
                            help='maximum arithmetic units for verification of symbolic rank proofs')
    put.add_argument('program', type=Path)
    put.add_argument('record', type=Path)
    get.add_argument('key')
    get.add_argument('--expected-program', metavar='SHA256')
    get.add_argument('--include-record', action='store_true', help='include the program and run in the response')


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
                         inspection=replace(defaults.inspection, max_rank_steps=args.max_reader_steps))
        store = SQLiteEvidenceStore(args.database, readonly=args.archive_command == 'get',
                                    max_bundle_bytes=limits.json.max_bytes)
        if args.archive_command == 'put':
            result = publish(Program.load(args.program, limits=limits.program),
                             load_json(args.record, limits=limits.json), store,
                             limits=limits, allow_prefix=args.allow_prefix)
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
