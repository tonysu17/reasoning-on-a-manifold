# PT-B2 — Does post-training move the behaviour-concentration map? (pre-registration)

**Date sealed:** 2026-08-20, committed before any PT-B2 statistic is computed.
**Cost:** $0. Local CPU only, on already-extracted activations. No pod, no
annotation, no API. **Authorisation:** Tony, in chat, 2026-08-20.
**Relation to PT-B1:** independent. Separate estimand, separate seal. PT-B1
measures generated *behaviour*; PT-B2 measures *second-moment representational
structure*. Neither result may be pooled with the other, and PT-B2 does not
modify, upgrade, or reinterpret RQ1, RQ3, or PT-B1.

## 1. Motivation

RQ1's one multiplicity-corrected positive is a **concentration map**: which
behaviour × depth cells carry more linear variance concentration than a
chain-stratified reference (11 of 20 cells at depths {11,14,17,20,27}).
RQ3 established that safety post-training displaces representations
(location) with bounded rotation and contraction (shape). What has never been
asked is whether post-training moves the *map itself* — whether the
second-moment structure that RQ1 found behaviour-specific, and against which
the Chapter-8 steering operator was built, is altered by a training-time
intervention. PT-B2 asks exactly that, and only that.

## 2. Fixed inputs

Seven checkpoints' stored activations, all with **byte-identical row indices**
(verified: sha256 prefix `e26dedf8122d123b`, 37,851 rows in every one —
backtracking 10,267 / uncertainty-estimation 16,728 / example-testing 5,829 /
adding-knowledge 5,027):

| Role | Directory |
|---|---|
| base | `data/activations/R1-1.5B/` |
| safety seeds 42/43/44 | `data/activations/R1-1.5B-lora-safety1000{,-s43,-s44}/` |
| control seeds 42/43/44 | `data/activations/R1-1.5B-lora-control1000{,-s43,-s44}/` |

Layers: **12 and 16 only** — the only depths extracted for the adapter arms.
These do **not** overlap RQ1's depths {11,14,17,20,27}, so every PT-B2 number
is computed fresh and no PT-B2 cell may be compared numerically to an RQ1 cell.
The RQ1 map is context, never a baseline.

## 3. Estimator

The RQ1 statistic, unchanged: **fixed-top-ten variance concentration**,
`src.nulls.top_k_variance_ratio(k=10)` — the cumulative explained-variance
ratio of the first ten principal components of a behaviour's activation cloud.

For tractability PT-B2 evaluates it as
`sum(top-10 eigenvalues of the centred covariance) / trace(covariance)`,
which is the same estimand by definition. This implementation is **validated
against the reference function before any result is produced**: agreement to
`< 1e-6` absolute on every (checkpoint, layer, behaviour) cell, recorded in the
output. Any cell exceeding that tolerance aborts the run.

This is one of four distinct estimands in this project. PT-B2 speaks only about
**fixed-top-ten variance concentration**. It is not correlation dimension, not
participation ratio, not PCA variance-threshold dimension, and "subspace" and
"manifold" are not synonyms for it.

## 4. Environment-validity rule (binding)

PT-B1's identity gate established that at these effect sizes, activation
comparisons **across extraction sessions are invalid** — library/hardware drift
moved activations roughly six times further than the intervention itself
(`PTB1_AMENDMENT_3_2026-08-20.md`). PT-B2 is designed around that:

- **Primary contrast is within-session.** For each seed, the safety and control
  arms were trained and extracted in the same session (seed 42 in the runB /
  consol session; seeds 43 and 44 in the Spark seed-replication session), so
  the safety-minus-control difference is environment-matched *within each seed*
  and seeds are only combined after differencing.
- **Base-referenced values are descriptive only.** The base extraction is from a
  different session, so base-versus-adapter numbers are reported as context
  with the cross-session caveat adjacent, never as a matched contrast and never
  as a primary endpoint.

## 5. Endpoints

**Primary (confirmatory, Holm over 8 cells = 4 behaviours × 2 layers).**
For behaviour *b* and layer *l*, the per-seed difference

D(b,l,s) = stat(safety_s, b, l) − stat(control_s, b, l),  s ∈ {42,43,44}

and the estimand is the mean over the three seeds. Two inferences are recorded,
both pre-committed, with the **more conservative governing any claim**:

1. **Seed permutation (registered primary).** Exact permutation of the six
   recipe labels within seed-pairs. A 3-versus-3 paired design has a two-sided
   floor of **p = 0.25** (2/8 sign assignments); this is a known and stated
   power ceiling, not a result. A cell may therefore be called *resolved* only
   under (2), with (1) reported alongside.
2. **Chain bootstrap (magnitude).** Resample chains with replacement, restrict
   to their rows, recompute D on the identical resampled row set for every arm
   (rows are byte-matched across checkpoints, so the design is exactly paired),
   B = 2,000, seed 20260820, BCa, two-sided empirical p, Holm over the 8 cells.

**Secondary — the concentration map (descriptive).** For each of the 7
checkpoints × 2 layers × 4 behaviours (56 cells): the statistic plus a
**chain-stratified permutation null** (`src.nulls.chain_stratified_permutation_null`
semantics: permute behaviour labels within each chain, recompute), n = 1,000,
seed 42, upper tail. Reported as the per-checkpoint map with raw p and the
specificity margin (real − null mean). No Holm claim attaches to the map; it is
reported as a map, and the smallest attainable p is 1/1001.

**Tertiary (descriptive):** the 3×3 seed-pair sign grid per cell; and the
per-checkpoint statistic table.

## 6. Non-claims

No PT-B2 output may claim: that post-training changes reasoning behaviour (that
is PT-B1's estimand, not this one); that concentration is a subspace, manifold,
intrinsic dimension, or causal mechanism; that a changed map implies a changed
steering response; that base-versus-adapter differences are matched; that any
PT-B2 cell replicates or contradicts an RQ1 cell (different depths); or
seed-population inference beyond a 3-versus-3 design. A null result is local to
this statistic, these two depths, this model, and these recipes.

## 7. Outputs

`results/ptb2/`: `ptb2_analysis.json`, `PTB2_REPORT.md`, and
`provenance/ptb2.json` binding every input `.npy` and `row_index.json` by
sha256, the estimator-validation deltas, seeds, and library versions.
Thesis disposition is a later decision by Tony; nothing here admits PT-B2 to
the dissertation.
