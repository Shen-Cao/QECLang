"""Optional dense semantic model for finite-dimensional subspace/subsystem codes."""
from .dense import CorrectabilityResult, IsometryCode, from_stabilizer, pauli_matrix
from .channels import (KrausChannel, amplitude_damping, dephasing, depolarizing,
                       entanglement_fidelity, validate_choi)

__all__ = ["CorrectabilityResult", "IsometryCode", "from_stabilizer", "pauli_matrix",
           "KrausChannel", "amplitude_damping", "dephasing", "depolarizing",
           "entanglement_fidelity", "validate_choi"]
