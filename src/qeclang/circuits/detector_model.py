"""Independent binary fault mechanisms mapped to detectors and logical flips."""
from dataclasses import InitVar, dataclass
from hashlib import sha256
import json
from math import isfinite
from numbers import Real

from qeclang.algebra import SparseBinaryMatrix
from qeclang.algebra.binary import nonnegative_integer

SCHEMA = "qeclang.detector_model.v1"


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("fault probabilities must be real numbers")
    value = float(value)
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("fault probabilities must be finite and in [0,1]")
    return value


def _limits(max_faults, max_detectors, max_observables, max_nonzeros):
    return tuple(nonnegative_integer(value, name) for value, name in zip(
        (max_faults, max_detectors, max_observables, max_nonzeros),
        ("max_faults", "max_detectors", "max_observables", "max_nonzeros")))


@dataclass(frozen=True)
class DetectorModel:
    """H f gives detector events; L f gives logical flips, both over GF(2).

    Columns are independent Bernoulli mechanisms, not physical qubits. A column
    may touch any number of detectors, only logical observables, or neither.
    No graphlike decomposition, merging, or probability approximation is used.
    Limits bound original axes/incidences before copying the probability vector;
    they exclude Python object overhead and do not bound elapsed time.
    """
    checks: SparseBinaryMatrix
    observables: SparseBinaryMatrix
    probabilities: tuple[float, ...]
    max_faults: InitVar[int] = 1_000_000
    max_detectors: InitVar[int] = 1_000_000
    max_observables: InitVar[int] = 1_000_000
    max_nonzeros: InitVar[int] = 10_000_000

    def __post_init__(self, max_faults, max_detectors, max_observables, max_nonzeros):
        nf, nd, no, nnz = _limits(max_faults, max_detectors, max_observables, max_nonzeros)
        if not isinstance(self.checks, SparseBinaryMatrix) or not isinstance(self.observables, SparseBinaryMatrix):
            raise TypeError("detector and observable maps must be SparseBinaryMatrix objects")
        if self.checks.ncols != self.observables.ncols:
            raise ValueError("detector and observable maps must share the fault columns")
        if self.checks.ncols > nf or self.checks.nrows > nd or self.observables.nrows > no:
            raise ValueError("detector model exceeds its axis limits")
        if self.checks.nnz + self.observables.nnz > nnz:
            raise ValueError("detector model exceeds max_nonzeros")
        probabilities = []
        for probability in self.probabilities:
            if len(probabilities) >= self.checks.ncols:
                raise ValueError("probability count must equal the number of fault columns")
            probabilities.append(_probability(probability))
        if len(probabilities) != self.checks.ncols:
            raise ValueError("probability count must equal the number of fault columns")
        object.__setattr__(self, "probabilities", tuple(probabilities))

    @property
    def num_faults(self):
        return self.checks.ncols

    @property
    def num_detectors(self):
        return self.checks.nrows

    @property
    def num_observables(self):
        return self.observables.nrows

    def syndrome(self, faults):
        return self.checks.matvec(faults)

    def logical(self, faults):
        return self.observables.matvec(faults)

    def to_dict(self):
        return {"schema": SCHEMA,
                "checks": {"ncols": self.num_faults, "rows": [list(r) for r in self.checks.rows]},
                "observables": {"ncols": self.num_faults, "rows": [list(r) for r in self.observables.rows]},
                "probabilities": list(self.probabilities)}

    def fingerprint(self):
        """Hash the complete ordered H,L,p presentation; not an equivalence test."""
        content = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(content.encode()).hexdigest()

    @classmethod
    def from_dict(cls, data, *, max_faults=1_000_000, max_detectors=1_000_000,
                  max_observables=1_000_000, max_nonzeros=10_000_000):
        nf, nd, no, nnz = _limits(max_faults, max_detectors, max_observables, max_nonzeros)
        if (not isinstance(data, dict) or set(data) != {"schema", "checks", "observables", "probabilities"}
                or data["schema"] != SCHEMA):
            raise ValueError("invalid detector model schema or fields")
        used = 0
        def matrix(record, max_rows):
            nonlocal used
            if not isinstance(record, dict) or set(record) != {"ncols", "rows"}:
                raise ValueError("invalid sparse detector matrix record")
            ncols = nonnegative_integer(record["ncols"], "ncols")
            rows = record["rows"]
            if ncols > nf or not isinstance(rows, list) or len(rows) > max_rows:
                raise ValueError("detector matrix exceeds axis limits")
            for row in rows:
                if not isinstance(row, list):
                    raise ValueError("detector matrix supports must be lists")
                used += len(row)
                if used > nnz:
                    raise ValueError("detector model exceeds max_nonzeros")
            return SparseBinaryMatrix.from_supports(rows, ncols)
        checks = matrix(data["checks"], nd)
        observables = matrix(data["observables"], no)
        if not isinstance(data["probabilities"], list):
            raise ValueError("probabilities must be a list")
        return cls(checks, observables, data["probabilities"], nf, nd, no, nnz)

    @classmethod
    def from_errors(cls, errors, *, num_detectors, num_observables,
                    max_faults=1_000_000, max_detectors=1_000_000,
                    max_observables=1_000_000, max_nonzeros=10_000_000):
        """Build columns from (probability, detector IDs, observable IDs).

        Repeated target IDs cancel modulo two. max_nonzeros also bounds the raw
        input target visits, including duplicates, before row allocation.
        """
        nf, nd, no, nnz = _limits(max_faults, max_detectors, max_observables, max_nonzeros)
        detectors = nonnegative_integer(num_detectors, "num_detectors")
        observables = nonnegative_integer(num_observables, "num_observables")
        if detectors > nd or observables > no:
            raise ValueError("detector model exceeds axis limits")
        columns, probabilities, visited = [], [], 0
        for error in errors:
            if len(columns) >= nf:
                raise ValueError("detector model exceeds max_faults")
            if not isinstance(error, (tuple, list)) or len(error) != 3:
                raise ValueError("errors must contain probability and two target sequences")
            probability, detector_ids, observable_ids = error
            supports = []
            for targets, size in ((detector_ids, detectors), (observable_ids, observables)):
                parity = set()
                for target in targets:
                    visited += 1
                    if visited > nnz:
                        raise ValueError("fault targets exceed max_nonzeros")
                    target = nonnegative_integer(target, "target index")
                    if target >= size:
                        raise ValueError("fault target exceeds declared dimension")
                    if target in parity:
                        parity.remove(target)
                    else:
                        parity.add(target)
                supports.append(tuple(sorted(parity)))
            columns.append(tuple(supports))
            probabilities.append(_probability(probability))
        hrows = [[] for _ in range(detectors)]
        lrows = [[] for _ in range(observables)]
        for fault, (ds, ls) in enumerate(columns):
            for target in ds:
                hrows[target].append(fault)
            for target in ls:
                lrows[target].append(fault)
        return cls(SparseBinaryMatrix.from_supports(hrows, len(columns)),
                   SparseBinaryMatrix.from_supports(lrows, len(columns)), tuple(probabilities),
                   nf, nd, no, nnz)


__all__ = ["DetectorModel"]
