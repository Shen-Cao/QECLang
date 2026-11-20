from copy import deepcopy
from pathlib import Path


from qeclang.analysis.certificate_io import presentation_fingerprint
from qeclang.io import from_dict, to_dict
from qeclang.transforms import permute_qubits
from qeclang.workflow import Program, ExecutionLimits, InspectionLimits, execute, inspect_run
from qeclang.workflow.artifacts import describe

EXAMPLE = Path(__file__).resolve().parents[1]/'examples/programs/exact_steane.json'


def program_data():
    data = Program.load(EXAMPLE).to_dict()
    src = from_dict(data['nodes'][0]['params']['code'])
    perm = [1,2,3,4,5,6,0]
    target = permute_qubits(src, perm)
    data['nodes'] += [dict(id='target', op='css.literal', inputs=[], params={'code':to_dict(target)}),
                      dict(id='mapped', op='css.transport_distance', inputs=['code','target','distance'],
                           params={'relation':'permutation', 'permutation':perm})]
    data['outputs'] = ['mapped']
    return data, target


def test_transfer_full_distance_via_declared_qubit_bijection(monkeypatch):
    data, target = program_data()
    p = Program.from_dict(data)
    run = execute(p, verify_relations=False, limits=ExecutionLimits(max_candidates=0))
    assert run['status'] == 'completed', run['trace'][-1]
    assert run['ledger']['used']['transports'] == 1
    import qeclang.workflow.distance_transport as implementation
    def forbidden(*a, **kw):
        raise AssertionError('checker must not rerun the transport producer')
    monkeypatch.setattr(implementation, 'transport_interval_conclusion', forbidden)
    inspected = inspect_run(p, run)
    assert inspected['status'] == 'checked', inspected
    claim = inspected['claims'][-1]
    assert (claim['predicate'],claim['sector'],claim['value']) == ('logical_distance','pauli',3)
    assert claim['source'] == presentation_fingerprint(target)
    assert inspected['reader_usage']['interval_transport_work'] > 0
    assert inspect_run(p, run, limits=InspectionLimits(max_interval_transport_work=0))['status'] == 'budget_exhausted'


def test_rehashed_target_or_interval_substitution():
    data, _ = program_data()
    p = Program.from_dict(data)
    run = execute(p)
    forged = deepcopy(run)
    old = forged['outputs']['mapped']
    false = dict(forged['artifacts'][old]['payload'], lower_bound=2, status='bounded')
    new = describe('distance_claim', false)
    del forged['artifacts'][old]
    forged['artifacts'][new['digest']] = new
    forged['outputs']['mapped'] = forged['trace'][-1]['output'] = new['digest']
    assert inspect_run(p, forged)['status'] == 'invalid_record'
    bad = deepcopy(data)
    # Both valid codes have the same distance; the declared identity map is wrong.
    bad['nodes'][-1]['params']['permutation'] = list(range(7))
    assert execute(Program.from_dict(bad), verify_relations=False)['status'] == 'verification_failed'


def test_import_exclusion_validates_even_without_optional_checks():
    p = Program.load(EXAMPLE)
    run = execute(p)
    exclusion = run['artifacts'][run['trace'][1]['output']]['payload']
    raw = p.to_dict()
    raw['nodes'] = [raw['nodes'][0], dict(id='proof',op='css.import_exclusion',inputs=['code'],
                                         params={'record':exclusion})]
    raw['outputs'] = ['proof']
    p = Program.from_dict(raw)
    imported = execute(p, verify_relations=False, limits=ExecutionLimits(max_exclusion_steps=0))
    assert imported['status'] == 'completed'
    assert imported['ledger']['used']['exclusion_steps'] == 0
    review = inspect_run(p, imported)
    assert review['claims'][0]['predicate'] == 'nontrivial_weight_exclusion'
    raw['nodes'][1]['params']['record']['record_id'] = 'f'*64
    damaged = execute(Program.from_dict(raw), verify_relations=False)
    assert damaged['status'] == 'verification_failed'


def test_archive_cli_reconstructs_checked_premises_without_rank_or_search(tmp_path, capsys, monkeypatch):
    import json
    from qeclang.archive import SQLiteEvidenceStore, publish
    from qeclang.cli import main
    from qeclang.algebra import BinaryMatrix
    src = Program.load(EXAMPLE)
    db = tmp_path/'claims.sqlite'
    key = publish(src, execute(src), SQLiteEvidenceStore(db))['bundle_id']
    raw, target = program_data()
    target_path, relation_path = tmp_path/'target.json', tmp_path/'relation.json'
    target_path.write_text(json.dumps(to_dict(target)))
    relation_path.write_text(json.dumps(raw['nodes'][-1]['params']))
    def forbidden(*args, **kwargs):
        raise AssertionError('a checked archive handoff must not regenerate the source proof')
    for name in ('rank','row_basis','solve','nullspace'):
        monkeypatch.setattr(BinaryMatrix, name, forbidden)
    monkeypatch.setattr('qeclang.analysis.low_weight.certify_low_weight', forbidden)
    monkeypatch.setattr('qeclang.workflow.operations.css_distance', forbidden)
    assert main(['archive', 'reuse-distance', str(db), key, 'distance', str(target_path),
                 str(relation_path)]) == 0
    new = Program.loads(capsys.readouterr().out)
    run = execute(new, limits=ExecutionLimits(max_candidates=0, max_exclusion_steps=0, max_basis_bits=0))
    checked = inspect_run(new, run)
    assert checked['status'] == 'checked' and checked['claims'][-1]['value'] == 3
    assert run['ledger']['used']['exclusion_steps'] == run['ledger']['used']['candidates'] == 0
