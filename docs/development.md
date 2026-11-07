# Development

Use a virtual environment and install from the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,sdp]'
python -m pytest
python -m ruff check .
git diff --check
```

The core is dependency-free; NumPy/SciPy and CVXPY/SCS are optional numerical tools.
CI runs the complete development checks on Python 3.11, 3.12 and 3.13. Dependency
minimums describe compatibility, not a bit-for-bit numerical environment. For
published experiments, archive an environment lock, source revision, input
manifest and solver versions together with generated data.

Tests compare exact algebra against exhaustive small spaces, check known code
parameters, verify Pauli signs and invalid presentations, replay logical
witnesses, and compare bounded search with independent finite oracles. Numerical
Knill–Laflamme, channel, MILP and recovery SDP tests remain explicitly numerical. New backends should
pass the same invariant checks before timing comparisons are meaningful.

## Report

The report sources are split across `report/sections/` and use public references
in `report/bibliography.bib`. Build in an ignored directory using a TeX distribution
with pdfLaTeX and BibTeX; see `report/README.md` for the commands. Do not commit
PDFs, auxiliary files, or downloaded papers. This report is a design and research
foundation, not a claim that every proposed dialect is implemented.

## Changes and experiments

Keep algebra, code definitions, constructions, algorithms, and experiment
orchestration in separate modules. Maintain explicit coordinate conventions.
Do not change matrix bases silently where physical weights matter. Validate
shape, field, commutation and phase obligations at representation boundaries.
Keep exact bounds, numerical solver evidence, and conjectures distinct in APIs
and documents.

Use conventional commit subjects such as `feat(algebra): ...`, `test(distance):
...`, and `docs(research): ...`. Explain implementation tradeoffs in commit
messages. Commit reproducible inputs and scripts; put generated data in
`results/`, build artifacts in `build/`, and optional downloaded references in
`references/`. Check `.gitignore` and staged changes before committing. Public
DOI/arXiv links, not paths into a local cache, belong in tracked citations.
