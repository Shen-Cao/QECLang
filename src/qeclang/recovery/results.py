"""Numerical evidence from fixed-encoding recovery optimization."""
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class PrimalDiagnostics:
    trace_preservation_residual: float
    hermiticity_residual: float
    minimum_eigenvalue: float
    objective_imaginary_residual: float

    def feasible(self, tolerance):
        return (self.trace_preservation_residual <= tolerance
                and self.hermiticity_residual <= tolerance
                and self.minimum_eigenvalue >= -tolerance
                and self.objective_imaginary_residual <= tolerance)


@dataclass(frozen=True)
class DualDiagnostics:
    hermiticity_residual: float
    minimum_slack_eigenvalue: float
    objective_imaginary_residual: float

    def feasible(self, tolerance):
        return (self.hermiticity_residual <= tolerance
                and self.minimum_slack_eigenvalue >= -tolerance
                and self.objective_imaginary_residual <= tolerance)


@dataclass(frozen=True)
class RecoveryResult:
    """Primal/dual objectives are numerical evidence, not rigorous bounds.

    ``choi`` is the raw recovery Choi matrix in received-input ⊗ logical-output
    order. It is not silently projected onto the CPTP set. The objective uses
    maximally mixed logical input, and is not worst-case or diamond fidelity.
    """
    status: str
    solver: str
    logical_dimension: int
    received_dimension: int
    validation_tolerance: float
    primal_status: str | None = None
    dual_status: str | None = None
    primal_objective: float | None = None
    dual_objective: float | None = None
    numerical_gap: float | None = None
    primal_diagnostics: PrimalDiagnostics | None = None
    dual_diagnostics: DualDiagnostics | None = None
    choi: np.ndarray | None = field(default=None, repr=False, compare=False)
    dual_variable: np.ndarray | None = field(default=None, repr=False, compare=False)
    provenance: dict = field(default_factory=dict)
    messages: tuple[str, ...] = ()
    method: str = "fixed-encoding maximally-mixed entanglement-fidelity SDP"

    def summary(self):
        """JSON-compatible scalar diagnostics and provenance, excluding arrays."""
        from dataclasses import fields
        result = {}
        for item in fields(self):
            if item.name in ("choi", "dual_variable"):
                continue
            value = getattr(self, item.name)
            result[item.name] = asdict(value) if item.name.endswith("diagnostics") and value else value
        return result

    def as_channel(self, *, repair_tolerance=None, tolerance=1e-8, **limits):
        """Explicitly convert an available candidate; conversion revalidates CPTP.

        Repair, when requested, is reported by the channel's conversion report.
        It does not preserve the SDP objective automatically; evaluate the
        resulting channel independently against the original encoding/noise.
        """
        from qeclang.semantic.channels import KrausChannel
        if self.choi is None:
            raise ValueError("no recovery Choi matrix is available")
        return KrausChannel.from_choi(self.choi, input_dim=self.received_dimension,
                                     output_dim=self.logical_dimension,
                                     repair_tolerance=repair_tolerance,
                                     tolerance=tolerance, **limits)
