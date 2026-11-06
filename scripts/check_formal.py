"""Build Lean proofs, run fixtures, and audit the explicitly allowed kernel axioms."""
from pathlib import Path
import re
import subprocess


def main():
    project = Path(__file__).resolve().parents[1] / "formal"
    for source in project.rglob("*.lean"):
        if ".lake" in source.parts:
            continue
        if re.search(r"\b(?:sorry|admit|axiom|unsafe)\b", source.read_text()):
            raise SystemExit(f"Untrusted declaration or proof placeholder in {source.name}")
    subprocess.run(["lake", "build"], cwd=project, check=True)
    subprocess.run(["lake", "exe", "certificate_fixtures"], cwd=project, check=True)
    audit = subprocess.run(["lake", "env", "lean", "Audit.lean"], cwd=project,
                           check=True, capture_output=True, text=True)
    print(audit.stdout, end="")
    lines = [line for line in audit.stdout.splitlines() if "depends on axioms:" in line]
    if len(lines) != 7 or any(not line.endswith("[propext]") for line in lines):
        raise SystemExit("Unexpected theorem axiom dependencies; inspect formal/Audit.lean")


if __name__ == "__main__":
    main()
