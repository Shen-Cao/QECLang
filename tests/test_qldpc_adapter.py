import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qeclang.algebra import BinaryMatrix as B
from qeclang.analysis.distance import css_distance
from qeclang.codes import CSSCode, Pauli, SparseCSSCode, StabilizerCode
from qeclang.constructions import steane
from qeclang.interop.qldpc import (TESTED_QLDPC_VERSION, QldpcInteropError,
                                  from_qldpc, to_qldpc)


@pytest.fixture
def foreign():
    qldpc = pytest.importorskip("qldpc")
    if qldpc.__version__ != TESTED_QLDPC_VERSION:
        pytest.skip("real interoperability tests require the pinned qldpc version")
    return qldpc, pytest.importorskip("numpy"), pytest.importorskip("galois")


def test_optional_module_import_is_lazy():
    import qeclang

    script = "import sys; import qeclang.interop.qldpc; assert 'qldpc' not in sys.modules; assert 'galois' not in sys.modules"
    environment = os.environ.copy()
    root = str(Path(qeclang.__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", script], env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_export_type_and_limits_precede_optional_import(monkeypatch):
    import qeclang.interop.qldpc as adapter

    def forbidden():
        raise AssertionError("invalid export reached optional dependencies")

    monkeypatch.setattr(adapter, "_dependencies", forbidden)
    with pytest.raises(QldpcInteropError, match="positive"):
        to_qldpc(StabilizerCode(1, (Pauli.from_label("-Z"),)))
    with pytest.raises(QldpcInteropError, match="positive"):
        to_qldpc(SparseCSSCode.from_packed(steane()))
    with pytest.raises(QldpcInteropError, match="axis"):
        to_qldpc(steane(), max_axis=6)
    with pytest.raises(QldpcInteropError, match="elements"):
        to_qldpc(steane(), max_elements=50)
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError):
            to_qldpc(steane(), max_axis=bad)


def test_wrong_or_missing_distribution_never_falls_back(monkeypatch):
    import qeclang.interop.qldpc as adapter

    monkeypatch.setattr(adapter.metadata, "version", lambda name: "0.4.2")
    monkeypatch.setattr(adapter, "import_module", lambda name: pytest.fail("loaded unsupported backend"))
    with pytest.raises(QldpcInteropError, match="0.4.1"):
        to_qldpc(steane())

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(adapter.metadata, "version", missing)
    with pytest.raises(ImportError, match="optional"):
        to_qldpc(steane())


def test_real_generic_roundtrips_preserve_all_rows_and_empty_shapes(foreign):
    qldpc, np, galois = foreign
    codes = (steane(), CSSCode(B((3, 12, 3, 0), 5), B((15, 0, 15), 5)),
             CSSCode(B((), 0), B((), 0)), CSSCode(B((0, 0), 0), B((0,), 0)),
             CSSCode(B((), 3), B((), 3)), CSSCode(B((), 3), B((3, 6), 3)))
    for source in codes:
        external = to_qldpc(source)
        assert type(external) is qldpc.codes.CSSCode
        assert external.field is galois.GF2
        assert external.matrix_x.shape == source.hx.shape and external.matrix_z.shape == source.hz.shape
        restored = from_qldpc(external)
        assert restored.hx == source.hx and restored.hz == source.hz
        assert restored.name == ""
        for field in ("_distance", "_distance_bound", "_distance_x", "_distance_z"):
            assert getattr(external, field) is None
    # Inbound externally constructed empty dimensions must survive too.
    external = qldpc.codes.CSSCode(np.zeros((2, 0), dtype=np.uint8),
                                  np.zeros((1, 0), dtype=np.uint8), field=2)
    restored = from_qldpc(external)
    assert restored.hx.shape == (2, 0) and restored.hz.shape == (1, 0)


@pytest.mark.parametrize("family", ["steane", "hgp", "toric", "c4"])
def test_real_external_constructors_use_raw_coordinates(foreign, family):
    qldpc, np, _ = foreign
    external = {
        "steane": lambda: qldpc.codes.SteaneCode(),
        "hgp": lambda: qldpc.codes.HGPCode(qldpc.codes.RepetitionCode(3), set_logicals=False),
        "toric": lambda: qldpc.codes.ToricCode(2, rotated=False),
        "c4": lambda: qldpc.codes.C4Code(),
    }[family]()
    imported = from_qldpc(external)
    exported = to_qldpc(imported)
    assert np.array_equal(external.matrix_x, exported.matrix_x)
    assert np.array_equal(external.matrix_z, exported.matrix_z)
    assert imported.k == {"steane": 1, "hgp": 1, "toric": 2, "c4": 2}[family]
    # No foreign layout or suggested bias-tailoring transformation was applied.
    assert imported.n == external.matrix_x.shape[1]


def test_distance_dimension_and_equality_promises_are_not_imported(foreign):
    qldpc, np, _ = foreign
    external = qldpc.codes.CSSCode(np.zeros((0, 3), dtype=np.uint8),
                                  np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
                                  field=2, is_subsystem_code=False, promise_equal_distance_xz=True)
    external._distance_x, external._distance_z, external._dimension = 99, 88, 17
    imported = from_qldpc(external)
    assert imported.k == 1
    assert css_distance(imported, "x").distance == 3
    assert css_distance(imported, "z").distance == 1
    fresh = to_qldpc(imported)
    assert fresh._distance_x is fresh._distance_z is fresh._dimension is None
    assert fresh._equal_distance_xz is False


def test_different_qudit_fields_and_subsystems_rejected(foreign):
    qldpc, np, _ = foreign
    for field in (3, 4):
        external = qldpc.codes.CSSCode(np.zeros((0, 2), dtype=np.uint8),
                                      np.zeros((0, 2), dtype=np.uint8), field=field)
        with pytest.raises(QldpcInteropError, match=r"GF\(2\)"):
            from_qldpc(external)
    subsystem = qldpc.codes.BaconShorCode(2, set_logicals=False)
    with pytest.raises(QldpcInteropError, match="subsystem"):
        from_qldpc(subsystem)
    declared_gauge = qldpc.codes.CSSCode(np.zeros((0, 1), dtype=np.uint8),
                                       np.zeros((0, 1), dtype=np.uint8), is_subsystem_code=True)
    with pytest.raises(QldpcInteropError, match="subsystem"):
        from_qldpc(declared_gauge)
    with pytest.raises(QldpcInteropError, match="CSSCode"):
        from_qldpc(qldpc.codes.FiveQubitCode())


def test_false_subsystem_flag_does_not_replace_local_commutation_check(foreign):
    qldpc, np, _ = foreign
    external = qldpc.codes.CSSCode(np.array([[1]], dtype=np.uint8),
                                  np.array([[1]], dtype=np.uint8), is_subsystem_code=False)
    with pytest.raises(QldpcInteropError, match="commute"):
        from_qldpc(external)


def test_nonbinary_buffers_and_unsupported_sign_extensions_are_rejected(foreign):
    _, np, _ = foreign
    external = to_qldpc(steane())
    external.matrix_x.view(np.ndarray)[0, 0] = 2
    with pytest.raises(QldpcInteropError, match="nonbinary"):
        from_qldpc(external)
    external = to_qldpc(steane())
    external.signs_x = [-1, 1, 1]
    with pytest.raises(QldpcInteropError, match="signed"):
        from_qldpc(external)
    external = to_qldpc(steane())
    external._code_x._matrix = np.array(external.matrix_x, dtype=float)
    with pytest.raises(QldpcInteropError, match=r"GF\(2\) arrays"):
        from_qldpc(external)


def test_cached_gauge_semantics_and_inconsistent_sector_width_rejected(foreign):
    _, _, galois = foreign
    external = to_qldpc(steane())
    external.gauge_dimension = 1
    with pytest.raises(QldpcInteropError, match="gauge"):
        from_qldpc(external)
    external = to_qldpc(steane())
    external._gauge_ops = galois.GF2.Ones((2, 14))
    with pytest.raises(QldpcInteropError, match="gauge"):
        from_qldpc(external)
    external = to_qldpc(steane())
    external._code_z._matrix = galois.GF2.Zeros((3, 8))
    with pytest.raises(QldpcInteropError, match="column count"):
        from_qldpc(external)


def test_limits_precede_foreign_subsystem_computation_and_copy(foreign, monkeypatch):
    qldpc, np, _ = foreign

    class Foreign(qldpc.codes.CSSCode):
        @property
        def is_subsystem_code(self):
            raise AssertionError("subsystem arithmetic happened before shape guard")

    external = Foreign(np.zeros((2, 8), dtype=np.uint8), np.zeros((3, 8), dtype=np.uint8))
    with pytest.raises(QldpcInteropError, match="axis"):
        from_qldpc(external, max_axis=7)
    with pytest.raises(QldpcInteropError, match="elements"):
        from_qldpc(external, max_elements=45)


def test_adapter_never_asks_for_foreign_rank_distance_logicals_or_full_matrix(foreign):
    qldpc, np, _ = foreign

    class Foreign(qldpc.codes.CSSCode):
        @property
        def matrix(self):
            raise AssertionError("full symplectic matrix must not be requested")
        @property
        def dimension(self):
            raise AssertionError("foreign dimension must not be requested")
        def get_distance(self, *args, **kwargs):
            raise AssertionError("foreign distance must not be requested")
        def get_logical_ops(self, *args, **kwargs):
            raise AssertionError("foreign logicals must not be requested")

    external = Foreign(np.zeros((0, 3), dtype=np.uint8),
                       np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8), is_subsystem_code=False)
    assert from_qldpc(external).hz.rows == (3, 6)


def test_import_and_export_buffers_are_independent(foreign):
    _, np, _ = foreign
    original = steane()
    external = to_qldpc(original)
    imported = from_qldpc(external)
    external.matrix_x.view(np.ndarray)[:] = 0
    assert original.hx == imported.hx
    assert original.hx.rows == (85, 102, 120)
