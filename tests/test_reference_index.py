from copy import deepcopy

import pytest

from qeclang.algebra import BinaryMatrix as M
from qeclang.catalog import LookupLimits, ReferenceIndex
from qeclang.codes import CSSCode, StabilizerCode
from qeclang.constructions import steane
from qeclang.transforms import permute_qubits
from qeclang.workflow.row_equivalence import verify_row_equivalence


def test_fixed_coordinate_equivalence_with_parameter_only_foil():
    source = CSSCode(M((3,),4), M((12,),4))
    reordered = CSSCode(M((3,3),4), M((0,12),4))
    foil = CSSCode(M((5,),4), M((10,),4))
    entries = [('original',source),('redundant',reordered),('foil',foil)]
    index = ReferenceIndex(entries)
    result = index.lookup(source)
    assert result['status'] == 'matched'
    assert result['matches'][0]['relation'] == 'identical_ordered_checks'
    assert result['parameter_matches_only'] == ['foil']
    assert verify_row_equivalence(source,reordered,result['matches'][1]['maps'])
    assert index.snapshot == ReferenceIndex(reversed(entries)).snapshot
    changed = deepcopy(result)
    changed['matches'][1]['maps']['x_forward'] = []
    assert index.lookup(source) == result
    permuted = index.lookup(source,permutation=[0,2,1,3])
    assert [m['id'] for m in permuted['matches']] == ['foil']
    assert permuted['query'] != permuted['comparison']


def test_budget_refusal_and_no_match_are_different():
    code = steane()
    target = CSSCode(M(code.hx.rows+(0,),code.n),code.hz)
    for limits in (LookupLimits(max_map_bits=0),LookupLimits(max_map_attempts=0)):
        result = ReferenceIndex([('s',target)],limits=limits).lookup(code)
        assert result['status'] == 'unresolved' and not result['matches']
        assert result['unresolved'][0]['id'] == 's'
    p = [1,2,3,4,5,6,0]
    transformed = permute_qubits(code,p)
    assert ReferenceIndex([('t',transformed)]).lookup(code)['status'] == 'no_match_in_scope'
    assert ReferenceIndex([('t',transformed)]).lookup(code,permutation=p)['status'] == 'matched'


def test_index_input_guards():
    with pytest.raises(ValueError):
        ReferenceIndex([('s',steane())],limits=LookupLimits(max_axis=1))
    with pytest.raises(ValueError):
        ReferenceIndex([('s',steane()), ('s',steane())])
    with pytest.raises(TypeError):
        ReferenceIndex([('s',StabilizerCode(0,()))])
    with pytest.raises(ValueError):
        ReferenceIndex([]).lookup(steane(),permutation=[0,1,2,3,4,5,5])


def test_cli_snapshot_and_lookup(tmp_path, capsys):
    import json
    from qeclang.cli import main
    assert main(['catalog', 'list']) == 0
    cases = json.loads(capsys.readouterr().out)['cases']
    assert all(case['parameter_status'] == 'reference_expectations' for case in cases)
    assert main(['catalog', 'show', 'css_4_2_2']) == 0
    code = capsys.readouterr().out
    path = tmp_path / 'query.json'
    path.write_text(code)
    assert main(['catalog', 'lookup', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['snapshot'] and result['references'] == len(cases)
    assert 'disjoint_checks_4_2_1' in result['parameter_matches_only']
    assert 'css_4_2_2' in {m['id'] for m in result['matches']}
    assert main(['catalog', 'lookup', str(path), '--max-axis', '4']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'budget_exhausted'
