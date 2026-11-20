"""Content-based provenance for replayable computations."""
from hashlib import sha256
from pathlib import Path
import platform

from qeclang import __version__


def runtime_provenance():
    """Identify source contents and runtime without host paths or identifiers."""
    root = Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        name = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(content).to_bytes(8, "big") + content)
    return {"package_version": __version__, "source_sha256": digest.hexdigest(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version()}
