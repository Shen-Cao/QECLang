from copy import deepcopy

from qeclang.algebra import BinaryMatrix as M
from qeclang.analysis.certificate_io import certificate_record
from qeclang.analysis.certificates import certify_css_logical
from qeclang.codes import CSSCode
from qeclang.io import to_dict
from qeclang.workflow import ExecutionLimits, InspectionLimits, Program, execute, inspect_run


def fixture():
    source = CSSCode(M((3,12),4), M((3,),4))
    target = CSSCode(M((15,3,3),4), M((3,0),4))
    record = certificate_record(source, certify_css_logical(source, 'x', 4))
    maps = dict(x_forward=[[0,1],[0],[0]], x_reverse=[[1],[0,1]], z_forward=[[0],[]], z_reverse=[[0]])
    return {'schema':'qeclang.program.v1', 'nodes':[
        {'id':'src', 'op':'css.literal', 'inputs':[], 'params':{'code':to_dict(source)}},
        {'id':'imported', 'op':'css.import_witness', 'inputs':['src'], 'params':{'certificate':record}},
        {'id':'dst', 'op':'css.literal','inputs':[], 'params':{'code':to_dict(target)}},
        {'id':'out', 'op':'css.rebase_witness','inputs':['src','dst','imported'],'params':maps}], 'outputs':['out']}


def test_checked_reuse_needs_no_rank_solve_or_enumeration(monkeypatch):
    p = Program.from_dict(fixture())
    def forbidden(*args, **kwargs):
        raise AssertionError('explicit maps and supplied witnesses do not need elimination/search')
    for name in ('rank','solve','row_basis','nullspace','rref','contains'):
        monkeypatch.setattr(M, name, forbidden)
    monkeypatch.setattr('qeclang.workflow.operations.css_distance', forbidden)
    result = execute(p, verify_relations=False, limits=ExecutionLimits(max_candidates=0,max_basis_bits=0))
    assert result['status'] == 'completed', result['trace'][-1]
    used = result['ledger']['used']
    assert used['row_map_work'] > 0 and used['witness_imports'] == used['transports'] == 1
    assert not used['candidates'] and not used['optional_checks']
    inspection = inspect_run(p, result)
    assert inspection['status'] == 'checked', inspection
    first, second = inspection['claims']
    assert first['bound'] == second['bound'] == 1 and first['source'] != second['source']
    assert inspect_run(p, result, limits=InspectionLimits(max_row_map_work=0))['status'] == 'budget_exhausted'
    assert execute(p, limits=ExecutionLimits(max_row_map_work=0))['status'] == 'budget_exhausted'
    assert execute(p, limits=ExecutionLimits(max_witness_imports=0))['status'] == 'budget_exhausted'


def test_source_and_reverse_relation_are_mandatory_with_optional_assurance_disabled():
    data = fixture()
    bad_relation = deepcopy(data)
    bad_relation['nodes'][3]['params']['x_reverse'][1] = [0]
    for definition, expected in [(bad_relation,'out'),(data,'imported')]:
        if definition is data:
            definition['nodes'][1]['params']['certificate']['source'] = '0'*64
        result = execute(Program.from_dict(definition), verify_relations=False)
        assert result['status'] == 'verification_failed'
        assert result['trace'][-1]['node'] == expected
        assert not result['outputs']


def test_reuse_cli_fetches_checked_evidence_and_emits_a_data_only_program(tmp_path, capsys):
    import json
    from pathlib import Path
    from qeclang.archive import SQLiteEvidenceStore, publish
    from qeclang.cli import main
    from qeclang.workflow.row_equivalence import row_maps
    from qeclang.io import from_dict
    src = Program.load(Path(__file__).resolve().parents[1] / 'examples/programs/transport.json')
    run = execute(src)
    db = tmp_path / 'evidence.sqlite'
    key = publish(src, run, SQLiteEvidenceStore(db))['bundle_id']
    code = from_dict(run['artifacts'][run['outputs']['final']]['payload'])
    target = CSSCode(M(code.hx.rows + (code.hx.rows[0],), code.n), code.hz)
    target_path, map_path = tmp_path/'target.json', tmp_path/'maps.json'
    target_path.write_text(json.dumps(to_dict(target)))
    map_path.write_text(json.dumps(row_maps(code, target)))
    assert main(['archive', 'reuse', str(db), key, 'transported', str(target_path), str(map_path)]) == 0
    program = Program.loads(capsys.readouterr().out)
    reused = execute(program, limits=ExecutionLimits(max_candidates=0, max_basis_bits=0))
    review = inspect_run(program, reused)
    assert review['status'] == 'checked', review
    assert review['claims'][-1]['bound'] == 3
    assert reused['ledger']['used']['candidates'] == 0
