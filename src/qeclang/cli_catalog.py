"""Inspect a small versioned reference corpus under an explicit equality relation."""
import json
from pathlib import Path

from .catalog import LookupLimits, default_index
from .catalog.index import LookupLimitExceeded
from .io import from_dict, to_dict
from .workflow.json_input import JSONLimits, load_json


def register(commands):
    catalog = commands.add_parser('catalog', help='compare positive CSS presentations under stated relations')
    sub = catalog.add_subparsers(dest='catalog_command', required=True)
    sub.add_parser('list', help='list reference metadata')
    show = sub.add_parser('show', help='emit the exact support record for one reference')
    show.add_argument('identifier')
    lookup = sub.add_parser('lookup', help='look up equal generator spaces in a named coordinate frame')
    lookup.add_argument('query', type=Path)
    lookup.add_argument('--permutation', type=Path, help='optional JSON list mapping old query sites to reference sites')
    lookup.add_argument('--max-map-attempts', type=int, default=16)
    lookup.add_argument('--max-map-bits', type=int, default=1_000_000)
    lookup.add_argument('--max-axis', type=int, default=256)


def run(args):
    from .catalog.references import build_reference, reference_cases
    try:
        if args.catalog_command == 'list':
            result = {'schema': 'qeclang.reference_list.v1', 'cases': [case.to_dict() for case in reference_cases()]}
        elif args.catalog_command == 'show':
            result = to_dict(build_reference(args.identifier))
        else:
            limits = LookupLimits(max_map_attempts=args.max_map_attempts, max_map_bits=args.max_map_bits,
                                  max_axis=args.max_axis)
            data = load_json(args.query, limits=JSONLimits(max_bytes=10_000_000, max_integer_bits=4096))
            if not isinstance(data, dict) or data.get('kind') != 'css':
                raise ValueError('catalog lookup requires a positive CSS support record')
            query = from_dict(data, max_qubits=limits.max_axis, max_checks=limits.max_axis,
                              max_bits=limits.max_matrix_bits)
            permutation = load_json(args.permutation) if args.permutation else None
            result = default_index(limits=limits).lookup(query, permutation=permutation)
        status = 0
    except LookupLimitExceeded as error:
        result = {'status': 'budget_exhausted', 'reason': str(error)}
        status = 2
    except (ValueError, TypeError, OSError, KeyError) as error:
        result = {'status': 'invalid_request', 'reason': str(error)}
        status = 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return status
