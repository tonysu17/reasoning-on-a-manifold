# Repository inventory — what's current vs. stale

**Written 2026-06-06.** Purpose: after the heavy 2026-06-05 audit + fixes, keep
**new/valid** content from being confused with **old/stale/invalid** content.
This file classifies the scripts, results, and docs. For `data/` the existing
[`data/MANIFEST.md`](data/MANIFEST.md) is authoritative (canonical / intermediate
/ deprecated) — this inventory only summarizes it.

## The one date that matters: **2026-06-05**

The audit (`AUDIT.md`) fixed the intrinsic-dimension, curvature, PCA-repro­ducibility,
and CBS-geometry code on **2026-06-05**. Any *geometry / steering* output produced
**before** that date is from biased or non-reproducible estimators and is **stale —
must be regenerated**. Outputs that don't touch those estimators (base-model
verification, chain-quality, annotation distribution) are unaffected.

> **Outstanding action (AUDIT.md §6 #1):** regenerate the quarantined geometry /
> robustness results on the fixed code, then refresh the TwoNN dims in
> `PROGRESS.md`. Driver: `run_rerun_local.sh`. Until then, **no geometry number
> in this repo is citable.**

---

## `results/` — sorted 2026-06-06

| Location | Status | Use it? |
|----------|--------|---------|
| `base_model_verification/` | **VALID** — weight-cosine check, no estimator | ✅ yes |
| `quality_reports/` | **VALID** — chain token/truncation stats, no estimator | ✅ yes |
| `plots/`, `pilot_diagnostics.json`, `pilot_per_sentence.csv` | **PILOT** (May 22) — valid for the 20-chain pilot, superseded by full corpus | ⚠️ pilot only |
| `supervisor_meeting/` | **MIXED, kept intact** — May-28 deliverable; see per-figure table below | ⚠️ partial |
| `_STALE_pre_fix_20260605/` | **STALE / INVALID** — 12 dirs moved here 2026-06-06; pre-fix estimators. See its `README.md` | ❌ regenerate |
| `tier1_robustness/R1-1.5B/geometry_nulls_layer27.*` | **SUPERSEDED (2026-06-12)** — the 06-08 keystone-null run predates the CF-13 duplicate-row fix; its twoNN "dims" (0.168!) are zero-distance artifacts and its p=0.0000 cells are unsmoothed. Re-run `tier1_geometry_nulls.py` after re-extraction | ❌ re-run |
| `_archive_run1_20260528_094332/` | **ARCHIVED run1** — even older (pre May-28 cap fix); your prior archive | ❌ historical |
| `_archive_run1_20260528_094733/` | Empty archive dir (cruft from an aborted archive) | — |

**Quarantined into `_STALE_pre_fix_20260605/`** (nothing deleted; restore with
`mv results/_STALE_pre_fix_20260605/<dir> results/`):
`geometric/`, `robustness/`, `pca/`, `steering_vectors/`, `composition/`,
`saturation_predictions/`, `cross_layer/`, `triangulation/`, `clustering/`,
`cbs/`, `trajectory/`, `power_analysis/` (the last is independently invalid — all-NaN table).

### `supervisor_meeting/` per-figure status

Kept intact (it's a self-contained, clearly-dated bundle). Banners added to
`SUMMARY.md`, `CHECKPOINT.md`, `PHASE_5B_DEEP_DIVE.md`.

| Figure | Status | Why |
|--------|--------|-----|
| `fig3_annotation_dist.png` | ✅ VALID | annotation label distribution — no estimator |
| `fig4_chain_quality.png` | ✅ VALID | chain token/truncation stats |
| `fig5_pipeline_status.png` | ✅ VALID | status diagram |
| `viz1_curvature_inflates_dim.png`, `viz2_swiss_roll.png`, `viz3_intrinsic_estimators.png` | ✅ VALID | synthetic methodology explainers, not data results |
| `fig1_layer_sweep.png` | ❌ stale | PCA `d_eff`/PR sweep — regenerate with PCA |
| `fig2_deep_pca.png` | ❌ stale | PCA (non-reproducible solver) |
| `fig6_intrinsic_vs_pca.png` | ❌ stale | biased TwoNN intrinsic dim |
| `fig7_curvature_diagnostics.png` | ❌ stale | confounded curvature ratio |
| `fig8_null_hierarchy.png` | ❌ stale | built on the geometry estimates |
| `fig9_scorecard.png` | ❌ stale | summarizes the stale geometry verdicts |
| `*.html` (Jun-5 renders) | ⚠️ stale render | pre-date the banners above; re-run `render_html.py` to refresh |

---

## Scripts — almost all CURRENT (the audit touched them 2026-06-05)

No script currently *produces* wrong output; the wrong outputs were the May
results already quarantined above. Phase runners and `src/` are current.

**Phase pipeline (current):** `01`–`13` runners, `build_phase6.py`,
`compute_layer_triangulation.py`, `predict_saturation.py`,
`robustness_geometry.py`, `compare_annotators.py`, plus all of `src/`
(incl. the fixed `curvature.py`, `intrinsic_dim.py`, `pca.py`, `steering.py`,
`nulls.py`, `cbs/geometry.py`, and `config.py` as the single config source).

**Figure / report generators (current):** `make_explainer_figs.py`,
`make_fresh_figures.py`, `render_html.py`. ⚠️ Re-running these now will read
whatever results exist — run them only *after* regenerating geometry, or they'll
re-bake stale numbers.

**Regeneration / orchestration (current, relevant):**
- `run_rerun_local.sh` — local-CPU geometry regeneration (Phase 5→5c→triangulation→5d→5b). **This is the driver for the outstanding action.**
- `run_multiannotator_pipeline.sh` — 3-annotator robustness pipeline.
- `run_remaining_phases.sh` — cluster-side phase driver (uses `~/reasoning-on-manifold`, cluster venv paths).

**Done / one-shot, stable (keep as record):** `verify_base_model.py`,
`check_chain_quality.py`, `04_cleanup_tasks.py`, `verify_annotation_completeness.py`,
`power_analysis_curvature.py` (now fixed to fail loud — re-run pending).

**Pilot / early-stage (historical, superseded by the full pipeline):**
`00_pilot_gate.py`, `validate_pilot_lengths.py`.

**New scaffolding (untracked, valid):** `02b_generate_baseline_chains.py`
(awaits cluster run), `04b_extract_annotator.py`, `05d_subtype_clustering.py`,
`src/model_adapters.py` + `tests/test_gpt_oss_integration.py` (gpt-oss support).

> Naming note: there are two `04_` runners — `04_cleanup_tasks.py` (Phase 1.5,
> task dedup) and `04_extract_activations.py` (Phase 4, activations). Different
> phases sharing a prefix; not a duplicate.

---

## `data/` — see `data/MANIFEST.md` (authoritative)

- **Canonical (use these):** `tasks_final.json`, `chains_R1-1.5B.json`,
  `annotated_R1-1.5B.json`, `activations/R1-1.5B/`.
- **Deprecated — do not use:** `tasks.json` (stale pre-cleanup corpus),
  `chains_R1-1.5B_BAD_2048cap.json` (already name-flagged; wrong 2048 cap).
- Everything else is **intermediate / pilot / provenance** — kept, labeled in
  `MANIFEST.md`, not moved.

---

## Docs — which to trust for what

| Doc | Role | Freshness |
|-----|------|-----------|
| `AUDIT.md` | **Current authority** for software/numerical bug status + open actions | ✅ 2026-06-05 |
| `CONFOUNDS_AND_REMEDIATION.md` | **Current authority** for *scientific* confounds + negative results + the sequenced fix plan (gate → tiers) | ✅ 2026-06-06 |
| `README.md` | Entry point (rewritten in the audit) | ✅ 2026-06-05 |
| `INVENTORY.md` (this file) | File/output classification | ✅ 2026-06-06 |
| `data/MANIFEST.md` | `data/` classification | ✅ current |
| `PROGRESS.md` | History + pipeline narrative — **but its empirical numbers are stale** (banner added) | ⚠️ partial |
| `GPU_GUIDE.md` | Cluster/GPU setup | stable (Mar 25) |
