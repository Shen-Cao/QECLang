"""Carry an inspected CSS upper witness into a different measurement presentation."""
from pathlib import Path

from qeclang.algebra import BinaryMatrix
from qeclang.archive import InMemoryEvidenceStore, publish
from qeclang.archive.reuse import make_reuse_program
from qeclang.codes import CSSCode
from qeclang.io import from_dict
from qeclang.workflow import ExecutionLimits, Program, execute, inspect_run
from qeclang.workflow.row_equivalence import row_maps


def main():
    source_program = Program.load(Path(__file__).parent / 'programs' / 'transport.json')
    run = execute(source_program)
    target_record = run['artifacts'][run['outputs']['final']]['payload']
    original = from_dict(target_record)
    # One redundant generator changes the measurement presentation and syndrome width.
    expanded = BinaryMatrix(original.hx.rows + (original.hx.rows[0] ^ original.hx.rows[1],), original.n)
    target = CSSCode(expanded, original.hz)
    store = InMemoryEvidenceStore()
    key = publish(source_program, run, store)['bundle_id']
    maps = row_maps(original, target)
    program = make_reuse_program(store, key, 'transported', target, maps)
    new_run = execute(program, limits=ExecutionLimits(max_candidates=0, max_basis_bits=0))
    result = inspect_run(program, new_run)
    if result['status'] != 'checked':
        raise RuntimeError(result)
    claim = result['claims'][-1]
    print(f"Rebound {claim['sector'].upper()} logical upper bound {claim['bound']}; "
          f"{new_run['ledger']['used']['candidates']} new search candidates.")


if __name__ == '__main__':
    main()
