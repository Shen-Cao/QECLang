# Resumable structured search

`qeclang.experiments.study` runs bounded searches over scalar cyclic
`LiftedProduct` seeds. Both polynomials have the same fixed number of distinct
monomials. Their exact structured descriptions are stored; fingerprints identify
descriptions, not equivalence classes of codes. No optimality or asymptotic code
family claim follows from sampling these finite instances.

```python
from qeclang.experiments.study import StudyConfig, run_study, read_study

config = StudyConfig(lift=15, terms=3, seed=17, strategy="mutate",
                     restart_interval=10, backend="crt", max_weight=2,
                     max_candidates=100_000, max_distance_n=128)
summary = run_study("output/study.sqlite", config, target_trials=100)
summary = run_study("output/study.sqlite", config, target_trials=200)
print(summary.trials, summary.unique_evaluations, len(summary.pareto))
```

Create the output directory first. The target is a **total** trial count. Repeating
the same request performs no additional trials; a target below the completed
count leaves completed work intact. `max_trials` caps the total. There is no
network access, automatic publication, or execution of generated programs.

## Proposals and resumption

Each trial derives its RNG seed from the configured integer seed, trial index,
and proposal algorithm identifier. Random proposals sample each support without
replacement. Sampling between trials is with replacement, so duplicate
constructions are expected and cached.

Mutation selects the previously evaluated parent with the largest verified
distance lower bound, then largest exact logical dimension, then smallest check
weight and incidence degree. Ties select the earliest trial. One uniformly
selected seed replaces one existing exponent with an absent exponent, preserving
term count. Every `restart_interval` trials, including trial zero, starts a new
random proposal. If no eligible parent exists, the proposal is also random.
Empty/full supports cannot mutate and yield an explicitly unchanged proposal.
This is a deterministic greedy local-search baseline, not Pareto-population
evolution or an LLM research agent.

SQLite stores metadata, evaluations, and trials in separate tables. A new
evaluation and its trial commit in one transaction. Interruption rolls back the
current trial while preserving previous commits. Resumption reconstructs the
same proposal sequence from committed history. A write transaction serializes
concurrent writers; evaluation occurs inside that transaction.

The full configuration must match on resume. Source contents are hashed using
portable package-relative filenames and file contents; package and Python
implementation/version are also recorded. Source or runtime changes require a
new database. The source hash includes uncommitted package changes and does not
depend on a Git commit or host path. Cache keys combine the construction
fingerprint with evaluation settings and provenance. A cached result is reused
only within those exact semantics.

## Evidence and objectives

Choose the parameter backend explicitly: `crt` supports odd cyclic lifts,
`polynomial` also handles repeated-root/even lifts, and `packed` permits binary
expansion. Unsupported rings and exhausted algebra budgets are recorded. There
is no silent backend substitution.

The evaluator stores exact `n,k` when available, algebraic evidence, and exact
locality statistics. Distance analysis runs only after successful parameter
analysis and within `max_distance_n` and `max_bits`. Enumeration budgets apply
separately to X and Z sectors. Each sector retains its own exact/weight-limit/
budget-exhausted status, lower and upper bounds, and logical witness. An
independently checked dual-pairing certificate accompanies each upper-bound
witness. These certificates establish upper bounds; excluded weight layers
establish lower bounds.

The Pareto objective maximizes exact `k` and the smaller of the two verified
distance **lower** bounds, while minimizing maximum check weight and total
qubit incidence degree. Upper bounds never substitute for achieved distance.
Entries with unavailable parameters, skipped distance work, or no protected
logicals are explicitly ineligible. Budget-limited distance searches remain
eligible when they established a conservative lower bound. A code with `k=0`
is labeled `no_logicals`, not assigned a favorable artificial distance.

Scalar seeds with fixed term counts often have identical locality objectives;
the stored locality remains useful when comparing separate studies. The Pareto
front deduplicates identical cached descriptions but performs no physical-code
equivalence reduction. Parent selection is lexicographic as described above;
the reported front uses componentwise Pareto dominance.

## Cost limits

Proposal domain size, trials, polynomial degrees, algebra work, seed/block sizes,
packed payload, and distance candidates have explicit bounds in `StudyConfig`.
These are operation/storage controls, not wall-clock guarantees. Pareto-front
construction is quadratic in the number of distinct eligible evaluations, and
mutation reads prior records. This baseline targets modest bounded studies;
large campaigns need indexed frontier maintenance and more compact histories.
Database files and generated reports belong in ignored output directories.
