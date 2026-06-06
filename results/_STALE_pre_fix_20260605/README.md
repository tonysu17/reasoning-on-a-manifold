# STALE results — produced before the 2026-06-05 estimator fixes

**Do not use these numbers or figures.** Everything in this folder was generated
**before** the 2026-06-05 audit (`AUDIT.md`) corrected the intrinsic-dimension,
curvature, PCA-reproducibility, and CBS-geometry code. The outputs here were
produced by the **biased / non-reproducible** estimators and must be
**regenerated** on the fixed code before they can be trusted or cited.

Nothing has been deleted — this is a quarantine, not a trash can. To restore any
folder, move it back up one level (`mv <dir> ../`).

## Why each was quarantined

| Folder | Generated | Depends on (fixed 2026-06-05) |
|--------|-----------|-------------------------------|
| `geometric/` | May 25–28 | `intrinsic_dim.twoNN_estimate` (biased high ~35–50%), `curvature.local_vs_global_dim_ratio` (confounded), geodesic symmetrization |
| `robustness/` | May 30 | same three estimators, via `robustness_geometry.py` |
| `pca/` | May 25–28 | `pca.py` randomized SVD solver, no seed → non-reproducible components |
| `steering_vectors/` | May 25–28 | downstream of the non-reproducible PCA |
| `composition/` | May 25 | downstream of `steering_vectors/` |
| `saturation_predictions/` | May 25 | downstream of curvature + steering |
| `cross_layer/` | May 25–28 | PCA subspaces (reproducibility) |
| `triangulation/` | May 28 | PCA / geometry inputs |
| `clustering/` | May 28 | PCA `d_eff` peaks |
| `cbs/` | May 27 | `cbs/geometry.local_intrinsic_dim` (biased ~3–4×), `09_cbs_geometry` `hash()` non-determinism |
| `trajectory/` | May 27 | CBS geometry pipeline (regenerate with `cbs/`) |
| `power_analysis/` | May 24 | **independently invalid** — all-NaN `power_table.csv`; the "> max tested" claim is downstream of NaN, not a real null (AUDIT #6) |

## The concrete stale numbers

`geometric/R1-1.5B-smoke/summary_layer27.md` reports TwoNN intrinsic dims of
**9.4 / 16.8 / 13.7 / 27.9** and local/global curvature ratios of **0.13–0.24**.
Per `AUDIT.md` §2 (#1, #2) these are artifacts of the biased estimators — on
*perfectly flat* synthetic data the old curvature code scored 0.29–0.85 and the
old TwoNN over-estimated by 35–50%. The fixed estimators recover the planted
dimension on known-answer synthetic manifolds (`tests/test_curvature.py`,
`tests/test_intrinsic_dim.py`).

## How to regenerate (the outstanding action, AUDIT.md §6 #1)

`run_rerun_local.sh` drives the local CPU regeneration (Phase 5 → 5c →
triangulation → 5d → 5b). It re-creates `results/{pca,cross_layer,triangulation,
clustering,geometric}/` from scratch. The CBS geometry (`cbs/`, `trajectory/`)
and `robustness/` regenerate via `08`–`13` and `robustness_geometry.py`. After
regenerating, refresh the TwoNN dims quoted in `PROGRESS.md`.

> An even-older run ("run1") is already archived separately at
> `results/_archive_run1_20260528_094332/` — that one predates even the
> May-28 cap fix. This quarantine is the *second* (May 25–30) run.
