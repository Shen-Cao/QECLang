"""Replay a bounded CSS logical certificate as Boolean literals in Lean.

Python validates the code and record first. Lean checks the concrete parity and
nonmembership facts; Python parsing/translation and the quantum semantic bridge
remain outside the formal proof. Requires an installed QECLang and Lean toolchain.
"""
import argparse
from dataclasses import dataclass, fields
import json
from pathlib import Path
import subprocess
import tempfile

from qeclang.analysis.certificate_io import verify_certificate_record
from qeclang.codes import CSSCode
from qeclang.io import from_dict


@dataclass(frozen=True)
class ReplayLimits:
    max_input_bytes: int = 1_000_000
    max_qubits: int = 256
    max_checks: int = 512
    max_work: int = 1_000_000
    max_source_bytes: int = 1_000_000
    timeout_seconds: int = 60

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field.name} must be a positive integer")


def _budget(n, mx, mz, limits):
    if type(n) is not int or n < 0 or n > limits.max_qubits:
        raise ValueError("code width exceeds replay limit")
    if mx + mz > limits.max_checks:
        raise ValueError("check count exceeds replay limit")
    # Scalar parity-operation upper estimate, including code commutation checks.
    work = n * (mx*mz + mx + mz + 2) + mx + mz
    if work > limits.max_work:
        raise ValueError("replay exceeds max_work")
    # Conservative bound before allocating text, including declarations/proofs.
    if 7*n*(mx+mz+2) + 4*(mx+mz) + 4096 > limits.max_source_bytes:
        raise ValueError("Lean literals exceed max_source_bytes")


def _read_json(path, limits):
    def unique(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError("duplicate JSON object key")
            data[key] = value
        return data
    with Path(path).open("rb") as stream:
        content = stream.read(limits.max_input_bytes + 1)
    if len(content) > limits.max_input_bytes:
        raise ValueError("input exceeds max_input_bytes")
    return json.loads(content, object_pairs_hook=unique)


def read_inputs(code_path, certificate_path, *, limits=ReplayLimits()):
    data = _read_json(code_path, limits)
    if not isinstance(data, dict) or data.get("kind") != "css":
        raise ValueError("Lean replay currently requires a CSS code record")
    hx, hz = data.get("hx"), data.get("hz")
    if not isinstance(hx, list) or not isinstance(hz, list):
        raise ValueError("CSS checks must be lists")
    _budget(data.get("n"), len(hx), len(hz), limits)
    code = from_dict(data, max_qubits=limits.max_qubits, max_checks=limits.max_checks,
                     max_bits=limits.max_work)
    record = _read_json(certificate_path, limits)
    if not verify_certificate_record(code, record, max_bits=limits.max_work,
                                     max_axis=limits.max_qubits):
        raise ValueError("logical certificate is invalid or does not match the code")
    return code, record


def render_lean_witness(code, record, *, limits=ReplayLimits()):
    if not isinstance(code, CSSCode):
        raise TypeError("Lean replay currently requires CSSCode")
    _budget(code.n, code.hx.nrows, code.hz.nrows, limits)
    if not verify_certificate_record(code, record, max_bits=limits.max_work,
                                     max_axis=limits.max_qubits):
        raise ValueError("logical certificate is invalid or does not match the code")
    payload = record["certificate"]
    if record["kind"] != "css_logical":
        raise ValueError("Lean replay requires a CSS logical certificate")
    same, opposite = ((code.hx, code.hz) if payload["sector"] == "x"
                      else (code.hz, code.hx))
    operator, dual = int(payload["operator"], 16), int(payload["dual"], 16)

    def bits(word):
        return "[" + ", ".join("true" if (word >> j) & 1 else "false"
                                for j in range(code.n)) + "]"
    def rows(matrix):
        return "[" + ",\n ".join(bits(row) for row in matrix.rows) + "]"
    source = f"""import QECLang.CssChecker
set_option maxRecDepth 4096
set_option maxHeartbeats 4000000
namespace QECLangReplay
open QECLang
def width : Nat := {code.n}
def sameRows : List (List Bool) := {rows(same)}
def oppositeRows : List (List Bool) := {rows(opposite)}
def candidate : List Bool := {bits(operator)}
def dual : List Bool := {bits(dual)}
theorem accepted : checkCSSRaw width sameRows oppositeRows candidate dual = true := by decide
theorem nonmembership : ¬ RawInSpan width sameRows candidate :=
  checkCSSRaw_not_in_span width sameRows oppositeRows candidate dual accepted
theorem zeroSyndrome : ∃ (rs : List (Bits width)) (v : Bits width),
    parseRows width oppositeRows = some rs ∧ parse width candidate = some v ∧
    Orthogonal rs v := by
  obtain ⟨rs, v, hr, hv, ho, _⟩ :=
    checkCSSRaw_sound width sameRows oppositeRows candidate dual accepted
  exact ⟨rs, v, hr, hv, ho⟩
end QECLangReplay
"""
    if len(source.encode("utf-8")) > limits.max_source_bytes:
        raise ValueError("Lean source exceeds max_source_bytes")
    return source


def verify_files(code_path, certificate_path, *, limits=ReplayLimits(), runner=subprocess.run):
    code, record = read_inputs(code_path, certificate_path, limits=limits)
    source = render_lean_witness(code, record, limits=limits)
    project = Path(__file__).resolve().parents[1] / "formal"
    runner(["lake", "build", "QECLang"], cwd=project, check=True,
           timeout=limits.timeout_seconds, capture_output=True, text=True)
    with tempfile.TemporaryDirectory(prefix="qeclang-lean-") as directory:
        target = Path(directory) / "Witness.lean"
        target.write_text(source, encoding="utf-8")
        runner(["lake", "env", "lean", str(target)], cwd=project, check=True,
               timeout=limits.timeout_seconds, capture_output=True, text=True)
    return {"status": "lean_literal_verified", "source": record["source"],
            "sector": record["certificate"]["sector"],
            "python_checked_weight": record["certificate"]["weight"],
            "formal_scope": "literal parity, zero syndrome, row-span nonmembership"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", type=Path)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args()
    try:
        result = verify_files(args.code, args.certificate,
                              limits=ReplayLimits(timeout_seconds=args.timeout_seconds))
    except (ValueError, TypeError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Lean replay failed: {exc}\n{getattr(exc, 'stderr', '') or ''}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
