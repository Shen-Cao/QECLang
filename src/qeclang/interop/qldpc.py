"""Optional, version-pinned interchange of raw positive binary CSS checks.

Only physical column order and retained X/Z generator rows cross this boundary.
Distance/dimension caches, logical frames, layouts, optional bias-tailoring hints
and measurement/decoder semantics do not. No external rank/distance method is
called. Foreign getters can execute or allocate before exposing array shapes;
the adapter is not a sandbox for arbitrary Python objects.
"""
from importlib import import_module, metadata
from inspect import getattr_static
from numbers import Integral

from qeclang.algebra import BinaryMatrix
from qeclang.algebra.binary import nonnegative_integer, set_bits
from qeclang.codes import CSSCode

TESTED_QLDPC_VERSION = "0.4.1"
_SIGN_FIELDS = tuple(prefix + name for prefix in ("", "_") for name in
                    ("signs", "phases", "signs_x", "signs_z", "phases_x", "phases_z",
                     "stabilizer_signs", "stabilizer_phases"))
_MISSING = object()


class QldpcInteropError(ValueError):
    """Unsupported or inconsistent data at the pinned qLDPC boundary."""


def _dependencies():
    try:
        installed = metadata.version("qldpc")
    except metadata.PackageNotFoundError as error:
        raise ImportError(f"this adapter requires the optional qldpc=={TESTED_QLDPC_VERSION} package") from error
    if installed != TESTED_QLDPC_VERSION:
        raise QldpcInteropError(f"qldpc=={TESTED_QLDPC_VERSION} is required; found {installed}")
    qldpc = import_module("qldpc")
    if qldpc.__version__ != TESTED_QLDPC_VERSION:
        raise QldpcInteropError("loaded qldpc module disagrees with the pinned distribution version")
    return qldpc, import_module("numpy"), import_module("galois")


def _limits(max_axis, max_elements):
    return (nonnegative_integer(max_axis, "max_axis"),
            nonnegative_integer(max_elements, "max_elements"))


def _shape_budget(mx, mz, n, max_axis, max_elements):
    if max(mx, mz, n) > max_axis:
        raise QldpcInteropError("qLDPC matrix dimensions exceed max_axis")
    if (mx + mz) * n + mx * mz > max_elements:
        raise QldpcInteropError("qLDPC checks and commutation payload exceed max_elements")


def _unsigned_state(code, np):
    # qLDPC 0.4.1 CSS matrices have no sign columns. Do not silently discard
    # recognizable extensions that attempt to attach signed generator data.
    if any(getattr_static(code, name, _MISSING) is not _MISSING for name in _SIGN_FIELDS):
        raise QldpcInteropError("signed/phase-extended CSS objects are outside this unsigned adapter")
    state = vars(code)
    flag = state.get("_is_subsystem_code")
    if flag is not None and type(flag) is not bool:
        raise QldpcInteropError("foreign subsystem flag must be boolean or unset")
    if flag:
        raise QldpcInteropError("subsystem/gauge presentations are unsupported")
    gauge_dimension = state.get("gauge_dimension")
    if gauge_dimension is not None and (isinstance(gauge_dimension, bool)
            or not isinstance(gauge_dimension, Integral) or gauge_dimension != 0):
        raise QldpcInteropError("foreign cached gauge dimension is unsupported")
    gauge_ops = state.get("_gauge_ops")
    if gauge_ops is not None and (not isinstance(gauge_ops, np.ndarray)
                                 or gauge_ops.ndim != 2 or gauge_ops.shape[0] != 0):
        raise QldpcInteropError("foreign cached gauge operators are unsupported")


def _foreign_arrays(code, qldpc, np, galois, max_axis, max_elements):
    if not isinstance(code, qldpc.codes.CSSCode):
        raise QldpcInteropError("foreign object must be a qldpc.codes.CSSCode")
    _unsigned_state(code, np)
    if code.field is not galois.GF2:
        raise QldpcInteropError("only the exact binary field GF(2) is supported")
    # Access only the raw X/Z arrays: len(code), code.matrix and several gauge
    # properties can materialize larger symplectic matrices or run elimination.
    x, z = code.matrix_x, code.matrix_z
    if (type(x) is not galois.GF2 or type(z) is not galois.GF2 or
            x.ndim != 2 or z.ndim != 2 or x.dtype.kind not in "iu" or z.dtype.kind not in "iu"):
        raise QldpcInteropError("foreign checks require two-dimensional integer GF(2) arrays")
    if x.shape[1] != z.shape[1]:
        raise QldpcInteropError("foreign CSS sectors disagree on physical column count")
    _shape_budget(x.shape[0], z.shape[0], x.shape[1], max_axis, max_elements)
    return x.view(np.ndarray), z.view(np.ndarray)


def _subsystem_flag(code):
    flag = code.is_subsystem_code
    if type(flag) is not bool or flag:
        raise QldpcInteropError("foreign code is a subsystem/gauge code or has an invalid flag")


def _pack(array):
    rows = []
    for row in array:
        word = 0
        for site, item in enumerate(row):
            value = int(item)
            if value not in (0, 1):
                raise QldpcInteropError("foreign GF(2) buffer contains nonbinary entries")
            word |= value << site
        rows.append(word)
    return BinaryMatrix(tuple(rows), array.shape[1])


def _matches(array, matrix):
    return array.shape == matrix.shape and all(
        int(value) == ((row >> j) & 1)
        for row, values in zip(matrix.rows, array) for j, value in enumerate(values))


def to_qldpc(code: CSSCode, *, max_axis=4096, max_elements=10_000_000):
    """Create a fresh generic qLDPC CSSCode with the exact retained check arrays.

    Shapes are guarded before optional imports or dense allocation. Pass no
    distance/equality promises or logical metadata; qLDPC may independently
    recognize identical X/Z matrices. Signed/general stabilizer and sparse inputs
    require a separate explicit conversion and are not accepted implicitly.
    """
    max_axis, max_elements = _limits(max_axis, max_elements)
    if not isinstance(code, CSSCode):
        raise QldpcInteropError("export requires a positive packed CSSCode")
    _shape_budget(code.hx.nrows, code.hz.nrows, code.n, max_axis, max_elements)
    qldpc, np, galois = _dependencies()
    arrays = []
    for matrix in (code.hx, code.hz):
        array = np.zeros(matrix.shape, dtype=np.uint8)
        for i, row in enumerate(matrix.rows):
            for j in set_bits(row):
                array[i, j] = 1
        arrays.append(array)
    external = qldpc.codes.CSSCode(*arrays, field=2, is_subsystem_code=False,
                                   promise_equal_distance_xz=False)
    x, z = _foreign_arrays(external, qldpc, np, galois, max_axis, max_elements)
    if not _matches(x, code.hx) or not _matches(z, code.hz):
        raise QldpcInteropError("foreign constructor changed physical/check ordering or values")
    _subsystem_flag(external)
    return external


def from_qldpc(external, *, max_axis=4096, max_elements=10_000_000) -> CSSCode:
    """Import raw unsigned GF(2) CSS checks, ignoring all foreign parameter hints.

    Snapshot integer buffers after shape preflight, validate every binary entry,
    and recheck commutation with the exact local CSS constructor. Row order,
    redundancy and empty dimensions survive. Existing foreign arrays/getters,
    JIT/import overhead and temporary workspace are outside max_elements.
    """
    max_axis, max_elements = _limits(max_axis, max_elements)
    qldpc, np, galois = _dependencies()
    x, z = _foreign_arrays(external, qldpc, np, galois, max_axis, max_elements)
    # Keep the original integer dtype until binary validation; casting first
    # could silently turn an invalid 256 into uint8 zero.
    hx, hz = _pack(x.copy()), _pack(z.copy())
    _subsystem_flag(external)
    try:
        return CSSCode(hx, hz)
    except ValueError as error:
        raise QldpcInteropError("foreign positive CSS checks do not commute") from error


__all__ = ["TESTED_QLDPC_VERSION", "QldpcInteropError", "to_qldpc", "from_qldpc"]
