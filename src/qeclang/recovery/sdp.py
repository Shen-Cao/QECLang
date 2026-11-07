"""Optional CVXPY recovery optimization for a fixed finite subspace encoding.

Fletcher, Shor and Win: https://arxiv.org/abs/quant-ph/0606035 . The encoding
and noise are fixed. Joint encoder/recovery optimization is not claimed convex.
"""
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from math import isfinite

import numpy as np

from qeclang import __version__
from qeclang.algebra.binary import nonnegative_integer
from qeclang.semantic.dense import IsometryCode
from .results import DualDiagnostics, PrimalDiagnostics, RecoveryResult


def _finite(value):
    if not np.isfinite(value).all():
        raise ArithmeticError("nonfinite recovery optimization intermediate")
    return value


def _snapshot(value):
    value = np.array(_finite(value), dtype=complex, copy=True)
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


def _fingerprint(matrices):
    digest = sha256()
    for matrix in matrices:
        digest.update(json.dumps(list(matrix.shape)).encode())
        digest.update(np.asarray(matrix, dtype="<c16").tobytes(order="C"))
    return digest.hexdigest()


def recovery_objective(code, noise, *, max_choi_dimension=256, max_elements=1_000_000):
    """Build C=Σ vec_F((Na V)†) vec_F((Na V)†)† / K².

    The Choi index is received_index*K+logical_index. ``max_elements`` bounds
    the coefficient matrix and each dense input/intermediate payload separately,
    not total solver workspace. Inputs are rechecked at their existing tolerance.
    """
    from qeclang.semantic.channels import KrausChannel
    if not isinstance(code, IsometryCode) or not isinstance(noise, KrausChannel):
        raise TypeError("recovery requires IsometryCode and KrausChannel")
    if code.gauge_dimension != 1:
        raise ValueError("subsystem recovery requires an explicit gauge preparation; only subspace encoding is supported")
    limit = nonnegative_integer(max_choi_dimension, "max_choi_dimension")
    elements = nonnegative_integer(max_elements, "max_elements")
    v = code.isometry
    k, received = code.logical_dimension, noise.output_dim
    dimension = k * received
    if (dimension > limit or max(dimension**2, v.size, noise.input_dim**2,
                                 k**2, received*k) > elements):
        raise ValueError("recovery model exceeds dense allocation limits")
    if noise.input_dim != v.shape[0]:
        raise ValueError("noise input dimension does not match encoded physical space")
    _finite(v)
    with np.errstate(over="ignore", invalid="ignore"):
        gram = _finite(v.conj().T @ v)
    if np.max(np.abs(gram - np.eye(k))) > code.tolerance:
        raise ValueError("encoding is not isometric at its declared tolerance")
    coefficient = np.zeros((dimension, dimension), dtype=complex)
    gram_noise = np.zeros((noise.input_dim, noise.input_dim), dtype=complex)
    with np.errstate(over="ignore", invalid="ignore"):
        for operator in noise.kraus:
            gram_noise = _finite(gram_noise + operator.conj().T @ operator)
            effective = _finite(operator @ v)
            vector = effective.conj().T.reshape(-1, order="F") / k
            coefficient = _finite(coefficient + np.outer(vector, vector.conj()))
    if np.max(np.abs(gram_noise - np.eye(noise.input_dim))) > noise.tolerance:
        raise ValueError("noise is not trace preserving at its declared tolerance")
    return _snapshot(coefficient)


def _primal_metrics(choi, coefficient, received, logical):
    trace_output = _finite(np.trace(choi.reshape(received, logical, received, logical), axis1=1, axis2=3))
    objective = _finite(np.vdot(coefficient, choi))
    hermitian = _finite((choi + choi.conj().T) / 2)
    diagnostics = PrimalDiagnostics(float(np.max(_finite(np.abs(trace_output - np.eye(received))))),
                                    float(np.max(_finite(np.abs(choi - choi.conj().T)))),
                                    float(_finite(np.linalg.eigvalsh(hermitian))[0]),
                                    float(abs(objective.imag)))
    return float(objective.real), diagnostics


def _dual_metrics(value, coefficient, logical):
    slack = _finite(np.kron(value, np.eye(logical)) - coefficient)
    objective = _finite(np.trace(value))
    diagnostics = DualDiagnostics(float(np.max(_finite(np.abs(value - value.conj().T)))),
                                  float(_finite(np.linalg.eigvalsh(_finite((slack + slack.conj().T)/2)))[0]),
                                  float(abs(objective.imag)))
    return float(objective.real), diagnostics


def _options(solver, supplied):
    if supplied is not None and not isinstance(supplied, dict):
        raise TypeError("solver_options must be a dictionary")
    options = {"eps": 1e-7, "max_iters": 10_000} if solver == "SCS" else {}
    options.update({} if supplied is None else supplied)
    if any(name in options for name in ("solver", "solver_path", "gp", "qcp")):
        raise ValueError("solver_options cannot replace the selected SDP formulation or solver")
    return json.loads(json.dumps(options, allow_nan=False))


def optimize_recovery(code, noise, *, solver="SCS", solver_options=None,
                      validation_tolerance=1e-6, solve_dual=True,
                      max_choi_dimension=256, max_elements=1_000_000):
    """Solve primal and optionally dual SDPs, then independently check residuals.

    Recovery maps received physical states into logical states. The primal is
    max Tr(CJ), J>=0, Tr_output J=I; the dual is min Tr(Y), Y⊗I-C>=0.
    No solver fallback or repair is implicit. SCS defaults to 10,000 iterations;
    other solvers use their own defaults unless options are supplied. Solver
    canonicalization/workspace and elapsed time are not bounded by max_elements.
    """
    if not isinstance(solver, str) or not solver:
        raise ValueError("solver must be a nonempty name")
    if (isinstance(validation_tolerance, bool) or not isfinite(validation_tolerance)
            or validation_tolerance <= 0):
        raise ValueError("validation_tolerance must be finite and positive")
    if type(solve_dual) is not bool:
        raise ValueError("solve_dual must be boolean")
    options = _options(solver, solver_options)
    coefficient = recovery_objective(code, noise, max_choi_dimension=max_choi_dimension,
                                      max_elements=max_elements)
    try:
        import cvxpy as cp
    except ImportError as error:
        raise ImportError("recovery optimization requires qeclang[sdp]") from error
    k, received = code.logical_dimension, noise.output_dim
    try:
        solver_version = version({"SDPA": "sdpa-python"}.get(solver, solver.lower()))
    except PackageNotFoundError:
        solver_version = None
    provenance = {"qeclang_version": __version__, "cvxpy_version": cp.__version__,
                  "numpy_version": np.__version__, "solver_version": solver_version,
                  "encoding_sha256": _fingerprint((code.isometry,)),
                  "kraus_representation_sha256": _fingerprint(noise.kraus),
                  "solver_options": options, "solve_dual": solve_dual,
                  "max_choi_dimension": max_choi_dimension, "max_elements": max_elements,
                  "input_state": "maximally_mixed_logical", "choi_order": "received_tensor_logical"}
    result = RecoveryResult("solver_error", solver, k, received, validation_tolerance,
                            provenance=provenance)
    choi = cp.Variable((1, 1)) if k*received == 1 else cp.Variable((k*received, k*received), hermitian=True)
    partial_trace = sum(choi[j::k, j::k] for j in range(k))
    primal = cp.Problem(cp.Maximize(cp.real(cp.trace(coefficient @ choi))),
                        [choi >> 0, partial_trace == np.eye(received)])
    try:
        primal.solve(solver=solver, **options)
    except cp.error.SolverError as error:
        return replace(result, messages=(str(error),))
    result = replace(result, primal_status=str(primal.status))
    if choi.value is None:
        return replace(result, messages=("solver returned no primal candidate",))
    try:
        candidate = _snapshot(choi.value)
        with np.errstate(over="raise", invalid="raise"):
            objective, diagnostics = _primal_metrics(candidate, coefficient, received, k)
    except (ArithmeticError, np.linalg.LinAlgError) as error:
        return replace(result, status="invalid_solution", messages=(str(error),))
    result = replace(result, choi=candidate, primal_objective=objective,
                     primal_diagnostics=diagnostics)
    if not solve_dual:
        status = "primal_only" if diagnostics.feasible(validation_tolerance) else "invalid_solution"
        return replace(result, status=status)
    dual = cp.Variable((1, 1)) if received == 1 else cp.Variable((received, received), hermitian=True)
    dual_problem = cp.Problem(cp.Minimize(cp.real(cp.trace(dual))),
                              [cp.kron(dual, np.eye(k)) - coefficient >> 0])
    try:
        dual_problem.solve(solver=solver, **options)
    except cp.error.SolverError as error:
        return replace(result, messages=(str(error),))
    result = replace(result, dual_status=str(dual_problem.status))
    if dual.value is None:
        return replace(result, messages=("solver returned no dual candidate",))
    try:
        dual_candidate = _snapshot(dual.value)
        with np.errstate(over="raise", invalid="raise"):
            dual_objective, dual_diagnostics = _dual_metrics(dual_candidate, coefficient, k)
            gap = float(_finite(dual_objective - objective))
    except (ArithmeticError, np.linalg.LinAlgError) as error:
        return replace(result, status="invalid_solution", messages=(str(error),))
    feasible = diagnostics.feasible(validation_tolerance) and dual_diagnostics.feasible(validation_tolerance)
    tight = abs(gap) <= 10*validation_tolerance*max(1, abs(objective), abs(dual_objective))
    status = ("numerical_optimal" if feasible and tight and primal.status == dual_problem.status == cp.OPTIMAL
              else "numerical_inaccurate" if feasible else "invalid_solution")
    return replace(result, status=status, dual_variable=dual_candidate,
                   dual_objective=dual_objective, dual_diagnostics=dual_diagnostics, numerical_gap=gap)
