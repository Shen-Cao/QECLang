"""Optional dense semantic model for finite-dimensional subspace/subsystem codes."""
from .dense import CorrectabilityResult, IsometryCode, from_stabilizer, pauli_matrix
from .channels import (KrausChannel, amplitude_damping, dephasing, depolarizing,
                       entanglement_fidelity, validate_choi)
from .encoding import EncodingPreparation, encoding_channel, prepare_encoding

__all__ = ["CorrectabilityResult", "IsometryCode", "from_stabilizer", "pauli_matrix",
           "KrausChannel", "amplitude_damping", "dephasing", "depolarizing",
           "entanglement_fidelity", "validate_choi", "EncodingPreparation",
           "encoding_channel", "prepare_encoding"]
