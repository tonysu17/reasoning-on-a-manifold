# Phase-2 arm-differential missingness bounding — analysis specification

**Date sealed:** 2026-08-19 (committed before the bounding analysis is run).
**Status marker:** post-hoc sensitivity specification. The complete-case
Phase-2 estimates in `results/ph2/analysis/ph2_analysis.json` were computed
and visible on 2026-08-13, before this spec was written; this document
therefore cannot pre-register the estimands. What it does fix in advance is
the **imputation rules, endpoint list, and decision vocabulary** of the
bounding analysis, so that no imputation scheme can be chosen after seeing
which scheme flatters which contrast.

## Motivation (from the sealed closeout)

`.codex/out/PH2_ANNOTATION_CLOSEOUT_2026-08-13.md` records arm-differential
unresolvedness — vanilla arms 5–7% versus steered arms 15–26% — and requires a
bounding/sensitivity analysis before any prevalence endpoint is cited.
Full-chain endpoints (looped, truncated, boxed, length, damage gates) never
depended on annotation coverage and are outside this analysis's scope.

## Scope: the bounded endpoints

1. **A2 adjunct vanilla prevalence contrasts** (the currently cited
   prevalence endpoints): paired per-task differences target−base for
   `prev_{backtracking, uncertainty-estimation, example-testing,
   adding-knowledge}`, roles star1 and deepscaler.
2. **Primary steered suppression cells**: the per-task
   `energy_matched_floor − transported_raw_suppress` difference in
   `prev_backtracking` per role (the delta-floor point estimate), where both
   arms carry steered-class missingness. The floor-adjusted, bootstrapped
   delta-floor verdicts are already downgraded by the sealed sensitivity gate
   and are not upgraded or re-litigated here; this analysis only reports how
   missingness moves the point estimate.
3. `bt_per_1k` is excluded from sharp bounds (it has no a-priori upper
   bound); it receives quantile scenarios only, with that caveat stated.

## Row-level missingness definition (mirrors the authoritative extractor)

A planned row is **unresolved** exactly when `src.delta_floor
.per_task_fraction` (steered cells; gen-row universe joined to annotation
rows) or the A2 per-task builder (vanilla cells; annotation-row universe)
would skip it: no annotation row, `annotation_complete` is `False`, coverage
gate excludes it, or empty annotations. Everything else is **resolved** with
a `behaviour_fraction` value in `[0, 1]`.

## Fixed imputation rules

Let a cell's per-task value be the mean of its rows' behaviour fractions,
with unresolved rows imputed as below; tasks unresolved in one arm re-enter
the paired universe under imputation (complete-case silently drops them).

1. **Complete-case anchor.** Reproduce the current estimates; the analysis
   aborts if the A2 anchors do not match `ph2_analysis.json` to within
   1e-9 (n and diff_mean).
2. **Manski worst-case bounds.** Unresolved values ∈ [0, 1]; the interval is
   [D_lo, D_hi] with the minuend arm's missing rows at 0 and subtrahend's at
   1 (and conversely). Sharp, assumption-free, expected to be wide.
3. **Tipping point.** With the other arm's unresolved rows imputed at that
   arm's observed complete-case mean, the imputed constant m* ∈ [0, 1] for
   this arm's unresolved rows at which the paired mean difference crosses 0,
   reported for both arms. m* outside [0, 1] ⇒ the sign cannot be flipped by
   that arm's missingness alone under this scenario.
4. **Behaviour-dense scenarios.** Both arms' unresolved rows imputed at
   their own arm's observed q50, q75, and q90 (the closeout's stated concern
   is that missing rows are plausibly the most behaviour-dense).

No other imputation scheme may be reported. All computations are
deterministic (no bootstrap; bounds and scenarios apply to point estimates,
and CI-level bounds are declared out of scope).

## Decision vocabulary (fixed in advance)

- **sign-robust (worst case)** — the Manski interval excludes 0.
- **sign-robust (tipping)** — neither arm's m* lies in [0, 1].
- **sign-robust (behaviour-dense)** — the sign is unchanged in all three
  quantile scenarios.
- **missingness-fragile** — none of the above holds.

Citation rule: a prevalence contrast may be cited only as complete-case with
its per-arm unresolved counts adjacent, plus its strongest applicable
robustness label (or "missingness-fragile"). No imputed value is a result.

## Outputs

`ph2_missingness_bounds.py` (repo root) writes
`results/ph2/analysis/missingness_bounds.json` and
`results/ph2/analysis/MISSINGNESS_BOUNDS.md`, both deterministic.
