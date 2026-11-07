"""Configuration, provenance, and conservative evaluation for search studies."""
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from pathlib import Path
import platform

from qeclang import __version__
from qeclang.analysis.certificates import certify_css_logical, verify_css_logical
from qeclang.analysis.distance import css_distance
from qeclang.analysis.structured import BACKENDS, analyze_lifted


@dataclass(frozen=True)
class StudyConfig:
    lift: int = 7
    terms: int = 3
    seed: int = 0
    strategy: str = "random"
    restart_interval: int = 10
    backend: str = "crt"
    max_trials: int = 10_000
    max_proposal_sites: int = 100_000
    max_degree: int = 512
    max_steps: int = 10_000_000
    max_entries: int = 1_000_000
    max_blocks: int = 1_000_000
    max_bits: int = 10_000_000
    max_polynomial_degree: int = 4096
    max_operations: int = 100_000
    max_sites: int = 1_000_000
    max_characters: int = 10_000
    max_field_degree: int = 64
    max_field_candidates: int = 4096
    max_factors: int = 16
    max_component_entries: int = 1_000_000
    max_component_axis: int = 10_000
    max_distance_n: int = 128
    max_weight: int = 2
    max_candidates: int = 100_000

    def __post_init__(self):
        for field in fields(self):
            if field.name in ("strategy", "backend"):
                continue
            value = getattr(self, field.name)
            if type(value) is not int or (field.name != "seed" and value < 0):
                raise ValueError(f"{field.name} must be an integer with valid sign")
        if not 1 <= self.lift <= self.max_proposal_sites or self.terms > self.lift:
            raise ValueError("lift/term count exceeds proposal domain")
        if self.strategy not in ("random", "mutate"):
            raise ValueError("strategy must be random or mutate")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if self.restart_interval == 0:
            raise ValueError("restart_interval must be positive")

    def evaluation_config(self):
        excluded = {"lift", "terms", "seed", "strategy", "restart_interval",
                    "max_trials", "max_proposal_sites"}
        return {key: value for key, value in asdict(self).items() if key not in excluded}


def provenance():
    """Identify source contents and runtime without host paths or identifiers."""
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        name = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(content).to_bytes(8, "big") + content)
    return {"package_version": __version__, "source_sha256": digest.hexdigest(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "proposal_algorithm": "cyclic-scalar-fixed-support.v1",
            "evaluation_algorithm": "exact-parameters-bounded-distance.v1"}


def evaluate_candidate(construction, config: StudyConfig):
    """Persist bounds and independent logical witnesses; never infer distance."""
    parameters = analyze_lifted(
        construction, backend=config.backend, max_degree=config.max_degree,
        max_steps=config.max_steps, max_entries=config.max_entries,
        max_blocks=config.max_blocks, max_bits=config.max_bits,
        max_polynomial_degree=config.max_polynomial_degree, max_operations=config.max_operations,
        max_sites=config.max_sites, max_characters=config.max_characters,
        max_field_degree=config.max_field_degree, max_field_candidates=config.max_field_candidates,
        max_factors=config.max_factors, max_component_entries=config.max_component_entries,
        max_component_axis=config.max_component_axis)
    locality = asdict(construction.locality())
    result = {"parameters": asdict(parameters), "locality": locality,
              "distance_status": None, "distance_x": None, "distance_z": None,
              "logical_witnesses": {}, "objective": None}
    if parameters.status != "exact":
        result["distance_status"] = "parameters_unavailable"
        return result
    if parameters.k == 0:
        result["distance_status"] = "no_logicals"
        return result
    if construction.n > config.max_distance_n:
        result["distance_status"] = "size_limit"
        return result
    mx, mz = construction.check_counts
    if (construction.n * (mx + mz) + mx * mz > config.max_bits or
            max(construction.n, mx, mz) > config.max_sites):
        result["distance_status"] = "expansion_budget_exhausted"
        return result
    code = construction.lower(max_bits=config.max_bits, max_blocks=config.max_blocks,
                              max_sites=config.max_sites)
    if code.k != parameters.k:
        raise ArithmeticError("symbolic and packed dimensions disagree")
    for sector in ("x", "z"):
        distance = css_distance(code, sector, max_weight=config.max_weight,
                                max_candidates=config.max_candidates)
        result[f"distance_{sector}"] = asdict(distance)
        if distance.witness is not None:
            witness = certify_css_logical(code, sector, distance.witness)
            if not verify_css_logical(code, witness):
                raise ArithmeticError("distance witness certificate failed verification")
            result["logical_witnesses"][sector] = asdict(witness)
    lower = min(result["distance_x"]["lower_bound"], result["distance_z"]["lower_bound"])
    result["distance_status"] = "evaluated"
    result["objective"] = [parameters.k, lower, -locality["max_check_weight"],
                           -locality["max_qubit_degree"]]
    return result
