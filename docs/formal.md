# Formal parity-witness checker

The Lean project proves a narrow statement needed by QEC logical-witness verification: if a binary vector `dual` has even parity against every supplied generator row and odd parity against a candidate, that candidate is outside the rows' GF(2) span.

The CSS-sector wrapper additionally certifies zero syndrome against the opposite-sector rows. A quantum logical witness also needs a valid quantum-code interpretation and the appropriate quotient. The formal development does not prove that semantic bridge, the claimed physical weight, minimum distance, or a family-level distance theorem.

## Definitions and theorem chain

`formal/QECLang/Parity.lean` defines an indexed finite binary vector `Bits n`, componentwise XOR, and parity dot product. `InSpan rows value` is an inductively generated XOR span: begin with the zero vector and repeatedly XOR a supplied generator.

- `dot_xor` proves distributivity of the parity pairing over XOR.
- `span_pairs_zero` proves that every vector in the span pairs evenly with a dual orthogonal to all generators.
- `odd_pairing_not_in_span` derives nonmembership from an odd pairing.

`formal/QECLang/Checker.lean` supplies the executable Boolean checker. `parse` accepts a raw Boolean list only when its length equals the declared width; `parseRows` applies the same check to every generator. `checkRaw n rows value dual` rejects any length mismatch before evaluating the parity conditions.

`checkRaw_not_in_span` proves that a true checker result excludes `RawInSpan` membership under this explicit parsing interpretation. `checkRaw_lengths` separately proves that a true result implies every input vector has the declared width. `checkRaw_sound` exposes the typed parsed vectors and the parity properties used by the proof.

`formal/QECLang/CssChecker.lean` defines `checkCSSRaw n same opposite value dual`. It validates the opposite check widths, verifies that the candidate pairs evenly with all opposite-sector rows, and invokes the separating-dual checker. `checkCSSRaw_sound` proves acceptance implies well-shaped parsed vectors, zero syndrome, and nonmembership in the same-sector row span. `checkCSSRaw_lengths` exposes all width guarantees. A fixture explicitly shows that a noncentralizer candidate can pass the nonmembership checker but is rejected by the CSS wrapper.

## Fixtures and build

`Fixtures.lean` uses the repository's Steane check rows in little-endian site order. The weight-three vector on sites `{0,1,2}` serves as both candidate and separating dual. Kernel-evaluated examples establish acceptance of this witness and rejection of a nonorthogonal dual, a stabilizer row, malformed lengths, and the zero-width case. The executable repeats those checks and exits unsuccessfully if a fixture fails.

Run `python scripts/check_formal.py` from the repository root. The project pins Lean `4.28.0`, uses the bundled core/Std libraries, and has no external Lake packages. The script builds the project, runs the executable fixtures, checks for proof placeholders or custom axiom declarations, and audits the exported theorem dependencies. Build instructions use ordinary `lake` resolution and contain no machine-specific paths. See the official [Lean reference](https://lean-lang.org/doc/reference/latest/) and [Lake documentation](https://lean-lang.org/doc/reference/latest/Build-Tools-and-Distribution/Lake/).

## Trusted boundary

The current theorem audit reports only Lean's standard propositional-extensionality axiom, `propext`; there are no custom axioms, admitted proofs, or native-decision proof shortcuts. The proof is checked by the pinned Lean kernel. The compiled executable additionally relies on the Lean compiler and runtime. The audit script enforces the declared axiom dependency set rather than describing the development as axiom-free.

The Python certificate implementation uses integer bitsets; Lean uses explicit Boolean vectors. Their intended order agrees, and the small fixture corresponds to the same Steane instance, but no theorem currently connects Python integer parsing, JSON serialization, generated Lean input, or Python's implementation to these Lean definitions. Passing Python tests alone does not invoke the Lean theorem. The fixture executable consumes built-in examples. The separate replay script below checks concrete literals exported from bounded JSON inputs; the translation itself remains part of the trusted boundary.

## Replay a Python witness in Lean

After installing QECLang, run:

```sh
python scripts/verify_lean_witness.py results/code.json results/logical-certificate.json
```

The inputs use the existing `qeclang.code.v1` CSS schema and `qeclang.certificate.v1` logical-certificate schema. The script checks byte/dimension/work budgets before code construction or literal rendering, validates the source fingerprint and certificate using Python, then writes a temporary Lean source containing only validated natural-number and Boolean literals. Arbitrary names, comments, hexadecimal strings, or other user text are never inserted into Lean code. Both compiler invocations have timeouts; generated source is removed afterward.

Lean checks `checkCSSRaw = true` by kernel reduction (`decide`), then applies the proved soundness theorem to establish concrete zero-syndrome and nonmembership statements. Successful output is labeled `lean_literal_verified`. The weight remains explicitly `python_checked_weight`, because no Lean Hamming-weight theorem is currently included. The source fingerprint binds the Python record to the physical presentation but is not a formal hashing theorem or authenticity signature.

The Python exporter remains trusted to translate that input faithfully. A formal parser/serialization correspondence would reduce this boundary further. Matrix-rank and polynomial-factorization certificates remain outside the Lean proof's scope. Optional exporter/compiler integration tests can be run with `QECLANG_TEST_LEAN=1 python -m pytest tests/test_lean_export.py`; ordinary Python tests use a fake compiler and do not require Lean.
