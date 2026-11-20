"""Small immutable reference recipes; parameter metadata is not runtime evidence."""
from dataclasses import dataclass

from qeclang.algebra import BinaryMatrix
from qeclang.algebra.binary import nonnegative_integer
from qeclang.codes import CSSCode
from qeclang.constructions import bicycle, hypergraph_product, repetition_check, shor, steane, toric
from qeclang.symbolic import GroupAlgebraElement

REFERENCE_SCHEMA = "qeclang.reference_case.v1"
_BINARY_ORDER = "Physical site i is packed bit i; rows retain the stated generator order."
_PRODUCT_ORDER = ("VV block first, then CC; tensor pairs use the second factor as the "
                  "fast index. Check rows retain the constructor's tensor order.")
_STEANE = ("https://arxiv.org/abs/quant-ph/9601029",)
_SHOR = ("https://doi.org/10.1103/PhysRevA.52.R2493",)
_TORIC = ("https://arxiv.org/abs/quant-ph/9707021",)
_HGP = ("https://arxiv.org/abs/0903.0566",)
_BICYCLE = ("https://arxiv.org/abs/quant-ph/0304161",)


@dataclass(frozen=True)
class ReferenceCase:
    id: str
    title: str
    category: str
    source_type: str
    expected_n: int
    expected_k: int
    expected_dx: int | None
    expected_dz: int | None
    x_checks: int
    z_checks: int
    recipe: str
    physical_order: str
    citation_urls: tuple[str, ...] = ()
    citation_scope: str = "Local algebraic fixture; no claim of a new code or literature identity."
    version: int = 1

    @property
    def case_id(self):
        return self.id

    @property
    def expected_distance(self):
        if self.expected_dx is None or self.expected_dz is None:
            return None
        return min(self.expected_dx, self.expected_dz)

    def to_dict(self):
        """Detached JSON metadata; reading it does not run parameter verification."""
        return {"schema": REFERENCE_SCHEMA, "id": self.id, "version": self.version,
                "title": self.title, "category": self.category,
                "scope": "positive_binary_css_finite_instance",
                "parameter_status": "reference_expectations",
                "expected": {"n": self.expected_n, "k": self.expected_k,
                             "d_x": self.expected_dx, "d_z": self.expected_dz,
                             "d": self.expected_distance},
                "check_counts": {"x": self.x_checks, "z": self.z_checks},
                "recipe": self.recipe, "physical_order": self.physical_order,
                "provenance": {"source_type": self.source_type,
                               "citations": list(self.citation_urls), "scope": self.citation_scope}}


def _case(identifier, title, category, source, parameters, checks, recipe, order=_BINARY_ORDER,
          citations=(), scope="Local algebraic fixture; no claim of a new code or literature identity."):
    return ReferenceCase(identifier, title, category, source, *parameters, *checks,
                         recipe, order, citations, scope)


_CASES = (
    _case("bell_redundant", "Bell state with redundant checks", "boundary_fixture", "algebraic_fixture",
          (2, 0, None, None), (3, 3), "Hx=(3,3,0), Hz=(3,0,3), n=2."),
    _case("bicycle_cyclic_3", "Six-site cyclic bicycle fixture", "structured_fixture", "algebraic_fixture",
          (6, 2, 2, 2), (3, 3), "bicycle(A,A*), A=1+x in GF(2)[C3]; x^3=1.",
          "Two group-site blocks; within each block sites are residues 0,1,2; row r uses columns r+term mod 3.",
          _BICYCLE, "Construction B motivates [C,C^T]; this fixed six-site seed is a local fixture, not a reported performance case."),
    _case("css_4_2_2", "Four-site distance-two CSS fixture", "metric_foil", "algebraic_fixture",
          (4, 2, 2, 2), (1, 1), "Hx=(15), Hz=(15), n=4."),
    _case("disjoint_checks_4_2_1", "Disjoint-check distance-one foil", "metric_foil", "algebraic_fixture",
          (4, 2, 1, 1), (1, 1), "Hx=(3), Hz=(12), n=4."),
    _case("empty", "Zero-site CSS boundary", "boundary_fixture", "algebraic_fixture",
          (0, 0, None, None), (0, 0), "Hx and Hz both have shape 0 by 0."),
    _case("free_2", "Two unprotected qubits", "boundary_fixture", "algebraic_fixture",
          (2, 2, 1, 1), (0, 0), "Hx and Hz both have shape 0 by 2."),
    _case("hgp_open_2_3", "Open rectangular repetition product", "structured_fixture", "algebraic_fixture",
          (8, 1, 3, 2), (3, 4), "hypergraph_product(repetition_check(2), repetition_check(3)).",
          _PRODUCT_ORDER, _HGP,
          "Hypergraph-product construction; this particular rectangular seed and its sector convention are native fixtures."),
    _case("repetition_x_checks_3", "Three-site X-check repetition", "asymmetric_fixture", "algebraic_fixture",
          (3, 1, 1, 3), (2, 0), "Hx=repetition_check(3)=(3,6); Hz has shape 0 by 3."),
    _case("repetition_z_checks_3", "Three-site Z-check repetition", "asymmetric_fixture", "algebraic_fixture",
          (3, 1, 3, 1), (0, 2), "Hx has shape 0 by 3; Hz=repetition_check(3)=(3,6)."),
    _case("shor", "Shor nine-qubit code", "standard_code", "analytic_literature",
          (9, 1, 3, 3), (2, 6), "shor(); Hx=(63,504), Hz=(3,6,24,48,192,384).",
          _BINARY_ORDER, _SHOR, "Nine-qubit encoding correcting arbitrary one-qubit errors; native checks fix our physical and sector order."),
    _case("steane", "Steane seven-qubit code", "standard_code", "analytic_literature",
          (7, 1, 3, 3), (3, 3), "steane(); Hx=Hz=(85,102,120).",
          _BINARY_ORDER, _STEANE, "Seven-qubit CSS construction; native Hamming-check ordering is fixed by the recipe."),
    _case("steane_redundant", "Steane with zero and redundant generators", "presentation_fixture", "algebraic_fixture",
          (7, 1, 3, 3), (6, 6), "For H=(85,102,120), use Hx=Hz=H+(H[0]^H[1],0,H[0]).",
          _BINARY_ORDER, _STEANE, "Underlying Steane construction only; redundant presentation is a local fixture."),
    _case("toric_2", "Periodic two-by-two product", "topological_code", "analytic_literature",
          (8, 2, 2, 2), (4, 4), "toric(2): hypergraph product of two cyclic repetition_check(2) matrices.",
          _PRODUCT_ORDER, _TORIC, "Periodic toric-code construction and parameters; the size-two product retains duplicate cyclic checks."),
    _case("toric_3", "Periodic three-by-three product", "topological_code", "analytic_literature",
          (18, 2, 3, 3), (9, 9), "toric(3): hypergraph product of two cyclic repetition_check(3) matrices.",
          _PRODUCT_ORDER, _TORIC, "Periodic toric-code construction and parameters; native product ordering fixes the presentation."),
)


def reference_cases() -> tuple[ReferenceCase, ...]:
    """Return immutable descriptors in stable lexical-ID order, without building codes."""
    return _CASES


def _construct(identifier, max_bits):
    if identifier == "steane":
        return steane()
    if identifier == "steane_redundant":
        matrix = steane().hx
        expanded = BinaryMatrix(matrix.rows + (matrix.rows[0] ^ matrix.rows[1], 0, matrix.rows[0]), 7)
        return CSSCode(expanded, expanded, name="Steane redundant presentation")
    if identifier == "shor":
        return shor()
    if identifier in ("toric_2", "toric_3"):
        return toric(2 if identifier == "toric_2" else 3, max_bits=max_bits)
    if identifier == "hgp_open_2_3":
        return hypergraph_product(repetition_check(2), repetition_check(3), max_bits=max_bits)
    if identifier == "bicycle_cyclic_3":
        a = GroupAlgebraElement.from_terms((3,), ((0,), (1,)))
        return bicycle(a, a.involution(), max_bits=max_bits)
    if identifier in ("repetition_x_checks_3", "repetition_z_checks_3"):
        checks, empty = repetition_check(3), BinaryMatrix((), 3)
        return CSSCode(checks, empty) if identifier == "repetition_x_checks_3" else CSSCode(empty, checks)
    rows = {"empty": (0, (), ()), "free_2": (2, (), ()),
            "bell_redundant": (2, (3, 3, 0), (3, 0, 3)),
            "css_4_2_2": (4, (15,), (15,)), "disjoint_checks_4_2_1": (4, (3,), (12,))}
    n, x, z = rows[identifier]
    return CSSCode(BinaryMatrix(x, n), BinaryMatrix(z, n), name=identifier)


def build_reference(case_id, *, max_qubits=64, max_checks=64, max_bits=1_000_000) -> CSSCode:
    """Regenerate one fixed recipe after shape preflight; no analysis or storage.

    max_checks bounds total X+Z generators, including redundant and zero rows.
    max_bits bounds check payload plus the CSS validation product and is also
    forwarded to product constructors. It is not a peak-memory or time bound.
    Declared n/k/d metadata remains an expectation; build does not certify it.
    """
    if type(case_id) is not str:
        raise TypeError("reference case ID must be a string")
    limits = [nonnegative_integer(value, name) for value, name in
              ((max_qubits, "max_qubits"), (max_checks, "max_checks"), (max_bits, "max_bits"))]
    case = next((item for item in _CASES if item.id == case_id), None)
    if case is None:
        raise KeyError(case_id)
    n, mx, mz = case.expected_n, case.x_checks, case.z_checks
    if n > limits[0] or mx + mz > limits[1] or (mx + mz) * n + mx * mz > limits[2]:
        raise ValueError("reference construction exceeds the requested size limits")
    code = _construct(case.id, limits[2])
    if (code.n, code.hx.nrows, code.hz.nrows) != (n, mx, mz):
        raise ArithmeticError("reference construction shape differs from its versioned recipe")
    return code


__all__ = ["REFERENCE_SCHEMA", "ReferenceCase", "reference_cases", "build_reference"]
