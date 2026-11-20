"""Per-execution counters and shape limits for the serial workflow interpreter."""
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class ExecutionLimits:
    max_candidates: int = 100_000
    max_optional_checks: int = 256
    max_transports: int = 256
    max_artifact_bytes: int = 8_000_000
    max_validation_work: int = 1_000_000
    max_axis: int = 1024
    max_matrix_bits: int = 8_000_000
    max_basis_bits: int = 8_000_000

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field.name} must be a nonnegative integer")


class BudgetExceeded(Exception):
    def __init__(self, resource, requested, remaining):
        self.detail = dict(resource=resource, requested=requested, available=remaining)
        super().__init__(f"{resource} requires {requested}, available {remaining}")


class Ledger:
    """Candidates/checks/retained bytes/sparse validation units accumulate.

    Matrix/basis payload estimates and individual dimensions are per-operation
    ceilings. Report their largest attempted value, including refused attempts.
    """
    def __init__(self, limits):
        self.limits = limits
        self.used = {k: 0 for k in ("candidates", "optional_checks", "transports",
                                    "artifact_bytes", "validation_work")}
        self.peak_requested = {k: 0 for k in ("axis", "matrix_bits", "basis_bits")}

    def remaining(self, resource):
        return getattr(self.limits, "max_" + resource) - self.used[resource]

    def charge(self, resource, count):
        if type(count) is not int or count < 0:
            raise ValueError("invalid resource charge")
        available = self.remaining(resource)
        if count > available:
            raise BudgetExceeded(resource, count, available)
        self.used[resource] += count

    def bound(self, resource, size):
        if type(size) is not int or size < 0:
            raise ValueError("invalid resource request")
        self.peak_requested[resource] = max(size, self.peak_requested[resource])
        limit = getattr(self.limits, "max_" + resource)
        if size > limit:
            raise BudgetExceeded(resource, size, limit)

    def matrices(self, n, mx, mz):
        self.bound("axis", max(n, mx, mz))
        self.bound("matrix_bits", (mx + mz)*n + mx*mz)

    def basis(self, code):
        self.bound("basis_bits", 8*code.n**2 + 2*(code.hx.nrows + code.hz.nrows)*(code.n+1))

    def record(self):
        return {"limits": asdict(self.limits), "used": self.used.copy(),
                "peak_requested": self.peak_requested.copy()}
