# Binding caveats — Venhoff fixed-offset operator bridge (hybrid, 250 rows)

**Written 2026-08-19 at preservation time. These caveats travel with every use
of the numbers in this directory.** Evidence status: **provisional post-hoc
operator-sensitivity study**. This is not a Venhoff et al. replication (tasks,
prompts, baselines, annotation scheme, layers, and some directions differ), and
it is not evidence of useful or selective control.

1. **Dirty-tree execution.** Generation records analysis commit `fa3a59a` with
   `git_dirty: true`. The exact executed code is not uniquely recoverable from
   the repository; the committed implementation (`07f_venhoff_fixed_bridge.py`,
   `src/steered_inference.py` constant-offset modes, `src/venhoff_bridge_eval.py`)
   is the preserved state at commit time, hash-bound in
   `venhoff_bridge_report.json` → `analysis_code`.
2. **Annotation incomplete.** The campaign exhausted its permitted attempts with
   `stage_complete: false`: 250 planned rows, 193 coverage-complete at
   annotation, 190 final metric-resolved, 60 unresolved (51 coverage-incomplete,
   6 annotation-incomplete, 3 corrected token-alignment failures). Under the
   declared missingness contract every incomplete cell is ineligible for
   confirmatory significance and enters the Holm family at `p=1`; all
   suppression estimates are **complete-case only**. Even the example-testing
   worst-case missing-pair bound (the only one excluding 0) does not upgrade the
   result.
3. **Unresolved source-direction provenance.** The backtracking and
   example-testing directions reuse the current E1 vectors, whose own lineage is
   recorded as unresolved provenance; that uncertainty propagates to every cell
   using them.
4. **Mixed-vintage cells.** Uncertainty and adding-knowledge use directions
   and/or layers of a different vintage from the thesis's projective result, so
   those two cells cannot isolate operator semantics; only backtracking and
   example testing are close operator/write-dose sensitivity cells.
5. **Generation-regime change and damage concern (descriptive, ungated).** The
   fixed write materially changed generation: mean steered lengths
   730.9/978.4/986.8/1000.0 tokens across the four arms versus the shared
   baseline; repetition rose from .186 to .272/.466/.525/.760; only 6% of
   example-testing and 0% of adding-knowledge chains closed `</think>`
   (adding-knowledge always hit the 1,000-token cap). There is **no matched
   random-direction control and no task-correctness guard** in this bridge, so
   behaviour-suppression cannot be separated from generic degradation.
6. **Interpretation ceiling.** Fixed-versus-projective application semantics and
   delivered dose are a *leading explanation* of the earlier thesis-vs-Venhoff
   effect-size discrepancy (especially for uncertainty), but the bridge does not
   uniquely identify that cause. The negatively steered endpoints landing near
   the released Venhoff 1.5B endpoints while baselines differ is one reason
   effect sizes differ despite similar endpoints.
7. **The sibling directory**
   `R1-1.5B__venhoff_constant_all4_user_publishednorm/` contains only the
   validated 50-row shared baseline plus four ignored steered rows; it is a
   superseded partial artefact, **not** a second completed result.

Primary artefact: `venhoff_bridge_report.json` (schema, complete-case
estimates, BCa intervals, worst-case bounds, missingness accounting, source and
analysis-code hashes).
