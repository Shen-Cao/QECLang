from copy import deepcopy
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import sys

import pytest

from qeclang.analysis.certificate_io import certificate_record
from qeclang.analysis.certificates import certify_css_logical
from qeclang.constructions import steane
from qeclang.io import dump

spec = importlib.util.spec_from_file_location("lean_export", Path(__file__).parents[1] /
                                             "scripts" / "verify_lean_witness.py")
exporter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = exporter
spec.loader.exec_module(exporter)


def inputs(tmp_path):
    code = steane()
    record = certificate_record(code, certify_css_logical(code, "x", 7))
    code_path, cert_path = tmp_path / "code.json", tmp_path / "cert.json"
    dump(code, code_path)
    cert_path.write_text(json.dumps(record))
    return code, record, code_path, cert_path


def test_export_is_literal_only_and_compiler_invocation_is_bounded(tmp_path):
    _, record, code_path, cert_path = inputs(tmp_path)
    calls, emitted = [], []
    def compiler(args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 60
        assert kwargs["cwd"].name == "formal"
        if args[1:3] == ["env", "lean"]:
            emitted.append(Path(args[-1]).read_text())
    result = exporter.verify_files(code_path, cert_path, runner=compiler)
    assert result["status"] == "lean_literal_verified"
    assert result["source"] == record["source"]
    assert len(calls) == 2
    assert "theorem zeroSyndrome" in emitted[0]
    assert "by decide" in emitted[0]
    assert "native_decide" not in emitted[0]
    assert not Path(calls[-1][0][-1]).exists()


def test_untrusted_names_are_never_interpolated(tmp_path):
    code, record, code_path, cert_path = inputs(tmp_path)
    data = json.loads(code_path.read_text())
    data["name"] = '"\naxiom falseProof : False\n'
    code_path.write_text(json.dumps(data))
    renamed, same_record = exporter.read_inputs(code_path, cert_path)
    source = exporter.render_lean_witness(renamed, same_record)
    assert "falseProof" not in source
    assert "axiom" not in source
    assert source == exporter.render_lean_witness(code, record)


@pytest.mark.parametrize("limits", [exporter.ReplayLimits(max_input_bytes=10),
    exporter.ReplayLimits(max_qubits=6), exporter.ReplayLimits(max_checks=5),
    exporter.ReplayLimits(max_work=100), exporter.ReplayLimits(max_source_bytes=100)])
def test_limits_before_compiler(tmp_path, limits):
    _, _, code_path, cert_path = inputs(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("compiler should not run")
    with pytest.raises(ValueError):
        exporter.verify_files(code_path, cert_path, limits=limits, runner=forbidden)


def test_bad_records_and_injection_rejected(tmp_path):
    code, record, _, _ = inputs(tmp_path)
    for mutate in (lambda r: r.update(source="0"*64),
                   lambda r: r["certificate"].update(dual="0x1"),
                   lambda r: r["certificate"].update(operator="0x7; axiom evil : False")):
        bad = deepcopy(record)
        mutate(bad)
        with pytest.raises(ValueError):
            exporter.render_lean_witness(code, bad)


@pytest.mark.skipif(os.environ.get("QECLANG_TEST_LEAN") != "1", reason="optional Lean integration")
def test_real_lean_accepts_valid_literal_and_rejects_tampering(tmp_path):
    code, record, code_path, cert_path = inputs(tmp_path)
    assert exporter.verify_files(code_path, cert_path)["status"] == "lean_literal_verified"
    source = exporter.render_lean_witness(code, record)
    source = source.replace(
        "def candidate : List Bool := [true, true, true, false, false, false, false]",
        "def candidate : List Bool := [true, false, false, false, false, false, false]")
    target = tmp_path / "Tampered.lean"
    target.write_text(source)
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(["lake", "env", "lean", str(target)],
                       cwd=Path(__file__).parents[1] / "formal", check=True,
                       capture_output=True, text=True, timeout=60)
