# Structured workflow evaluation

This bounded experiment compares a fresh packed reference calculation with a
structured polynomial-rank workflow and its independent scientific inspection.
It also checks that symbolic dimension analysis remains usable when a deliberately
small packed-materialization cap refuses expansion. It does not measure distance,
code quality, a decoder threshold, or a general compiler speedup.

Run from the repository root after installing QECLang:

```bash
mkdir -p output
python scripts/benchmark_lift_workflow.py \
  --lifts 7 31 63 --even-lift 8 --repeats 5 --warmups 1 --order-seed 17 \
  --packed-cap 1024 > output/structured_workflow_evaluation.jsonl
```

The script uses the dependency-free reference implementations. It emits four
JSON Lines by default, retaining every measured duration, medians, exact policy
limits, the finite construction/program, source/runtime identity, and the script
content hash. Generated observations belong in the ignored `output/` directory;
no host-specific configuration or timing baseline is required by the repository.

## Fixture and correctness checks

Over `GF(2)[x]/(x^L+1)`, the rectangular seeds are

\[
A=\begin{bmatrix}1+x&1&0\\0&1+x&x\end{bmatrix},\qquad
B=\begin{bmatrix}1+x&1+x^2\end{bmatrix}.
\]

The minor of A on columns 2 and 3 has determinant x, a unit in this quotient
ring. Thus A is surjective. Adding `(1+x)` times B's first column to its second
reduces B to `[1+x,0]`. Multiplication by `1+x` has rank `L-1`, including even L,
since its gcd with `x^L+1` has degree one. These identities give the fixture's
lifted-product parameters:

| L | n | rank(Hx) | rank(Hz) | k |
| --- | --- | --- | --- | --- |
| 7 | 56 | 28 | 20 | 8 |
| 31 | 248 | 124 | 92 | 32 |
| 63 | 504 | 252 | 188 | 64 |
| 8, repeated-root case | 64 | 32 | 23 | 9 |

The script requires all three sources of dimension information to agree:

1. The analytic fixture identities `n=8L`, `rank(Hx)=4L`, `rank(Hz)=3L-1`, `k=L+1`.
2. Fresh packed expansion and exact GF(2) rank computation.
3. A traced polynomial structured result and the inspector's independently
   replayed `logical_dimension` claim, including its source fingerprint and both
   sector ranks.

The even case uses polynomial unimodular diagonalization, with no squarefree
field assumption. These constructions are dimension-analysis fixtures; the table
makes no assertion about useful distance or asymptotic goodness.

## Timed scopes

- **`packed_lower_and_rank`** includes construction-record deserialization,
  fresh binary lowering, and both exact ranks. It does not emit workflow records.
- **`structured_execute`** runs the already parsed two-node program: lifted
  literal followed by polynomial parameter analysis with operation traces. It
  includes the executor's source provenance, ledger, and artifact generation.
- **`independent_inspection`** checks a retained reference run. Each invocation
  admits fresh artifact objects and replays the polynomial evidence; it does not
  rerun the producer's rank search or expand the binary checks.

Each measured call is checked outside its timed interval. Packed objects are
rebuilt rather than reusing cached matrix ranks. Structured execution creates a
fresh run, while inspection deliberately reuses the same serialized mathematical
evidence. One warmup per scope precedes five observations by default. Scope order
is shuffled deterministically per repetition. Program parsing and final output
serialization are outside these measurements. A changed package source/runtime
fingerprint during measurement aborts the experiment.

These are warm single-process observations, with ordinary allocator, filesystem
cache, and timing noise. The three scopes do different work. In particular,
comparing packed computation with structured execution alone does not include
independent verification of the latter. The report computes no speedup ratio and
supports no hardware-independent performance claim.

## Artifact sizes and refusal experiment

Byte counts use canonical UTF-8 JSON, with separate fields for one packed CSS
artifact, all retained structured artifacts, the complete structured run, and the
inspection report. The formats contain different information: structured records
retain seeds and polynomial replay evidence. Their byte counts are neither peak
memory measurements nor compression ratios for equivalent output formats.

The refusal experiment uses the same 1,024-bit packed payload cap for every
fixture. A symbolic-only program must still complete, and inspection under the
same small packed-artifact cap must emit the same checked dimension claim. A
second program requesting `lift.css` must refuse that node with
`budget_exhausted` on `matrix_bits`. The refused run's step and requested/available
amounts are retained. The independent claim comes from the separately completed
symbolic-only run; default inspection requires a completed run. The optional prefix verifier can also check saved evidence preceding an expansion refusal.

This checks explicit representation choice under a declared resource policy.
The payload cap excludes object overhead and temporary or solver workspace. It
is not evidence of actual out-of-memory behavior, and no absence-of-expansion
claim is inferred from timing alone. See [workflow evaluation](workflow_evaluation.md)
for the earlier relational baseline and [evidence](evidence.md) for assurance scope.
