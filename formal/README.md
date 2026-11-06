# Lean parity certificates

This standalone Lean 4 project proves the soundness of a finite binary parity
checker for row-span nonmembership and zero CSS-sector syndrome. The toolchain is pinned in `lean-toolchain`;
there are no external Lake packages or Mathlib dependency.

From the repository root:

```sh
python scripts/check_formal.py
```

Or from this directory:

```sh
lake build
lake exe certificate_fixtures
lake env lean Audit.lean
```

See [the formal contract](../docs/formal.md) for the theorem statements and
trusted boundary. Build products and the empty generated Lake manifest are
ignored. The executable runs concrete fixtures. `scripts/verify_lean_witness.py` separately
validates Python JSON certificates and replays bounded Boolean literals in Lean;
its Python translation remains outside the formal proof.
