"""
src/cbs/comparison.py — Cross-model baseline replication (M6).

Purpose
-------
Compare the CBS-pipeline outputs from the distilled model (R1-Distill-Qwen-1.5B)
against the empirically-verified base (Qwen-2.5-Math-1.5B; see
`results/base_model_verification/verdict.md`). The distilled-vs-base
contrast is the empirically-loaded test of the "distillation creates the
geometry" vs "distillation reveals a pre-existing geometry" hypotheses.

Prerequisites
-------------
Extension A pipeline on QwenMath-1.5B: Phase 2b chain generation, Phase 3
annotation, Phase 4 activation extraction. M6 reads the same per-layer
geometry results + trajectory summaries the runners emit for R1-Distill.

Validation
----------
* All M2 / M3 / M4 sanity checks must pass on baseline data.
* Cross-model differences must survive bootstrap CI.
* Structured-null mode (synthesis §M6.4): if the baseline produces too
  few tier-3 sentences (< 3% rate) the cross-model comparison reports
  that as a finding rather than forcing parallel statistical tests on
  incommensurable sample sizes.

Milestone
---------
M6 (synthesis §M6). Build-now: orchestration code + cross-model helpers.
Run-phase: Extension A pipeline + M1-M4 reruns on the baseline.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Per-statistic cross-model comparison ──────────────────────────────────

def cross_model_compare(
    r1_results: dict,
    base_results: dict,
    *,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> dict:
    """Per geometric statistic: report
        {value_r1, value_base, delta, p_value, p_method}.
        p_method is "two_sample_bootstrap" when both sides persisted their
        effect-size bootstrap distributions, else "normal_approx_from_ci".

    Inputs are the JSON outputs of `09_cbs_geometry.py` for the two models.
    The function joins per (layer, behaviour, statistic, label_scheme) and
    emits a delta with a bootstrap p-value derived from the bootstrap-CI
    midpoint difference. Records with a missing counterpart are flagged.
    """
    rng = np.random.default_rng(seed)
    r1_records = {(r["layer"], r["behaviour"], r["statistic"],
                   r["label_scheme"]): r
                  for r in r1_results.get("results", [])
                  if r.get("labels_source", "real") not in
                     {"shuffle_control", "reversal_control"}}
    base_records = {(r["layer"], r["behaviour"], r["statistic"],
                     r["label_scheme"]): r
                    for r in base_results.get("results", [])
                    if r.get("labels_source", "real") not in
                       {"shuffle_control", "reversal_control"}}
    keys = sorted(set(r1_records) | set(base_records))
    out_records: list[dict] = []
    for k in keys:
        layer, behaviour, statistic, label_scheme = k
        r1 = r1_records.get(k)
        bs = base_records.get(k)
        if r1 is None or bs is None:
            out_records.append({
                "layer": layer, "behaviour": behaviour,
                "statistic": statistic, "label_scheme": label_scheme,
                "missing": ("r1" if r1 is None else "base"),
            })
            continue
        v_r1 = float(r1.get("effect_size", float("nan")))
        v_bs = float(bs.get("effect_size", float("nan")))
        delta = v_r1 - v_bs
        ci_r1 = r1.get("effect_size_ci95", [float("nan"), float("nan")])
        ci_bs = bs.get("effect_size_ci95", [float("nan"), float("nan")])
        boots_r1 = r1.get("effect_size_boots")
        boots_bs = bs.get("effect_size_boots")

        if boots_r1 and boots_bs:
            # GENUINE two-sample bootstrap: resample each model's effect-size
            # bootstrap distribution independently (the two models are
            # independent — there is no pairing across models) and form the
            # difference distribution. Two-sided p = fraction on the wrong side
            # of zero, doubled. This is what #15 was waiting on (09 now persists
            # effect_size_boots). See AUDIT.md §5.
            a = np.asarray(boots_r1, dtype=float)
            b = np.asarray(boots_bs, dtype=float)
            ia = rng.integers(0, a.shape[0], size=n_bootstrap)
            ib = rng.integers(0, b.shape[0], size=n_bootstrap)
            diff = a[ia] - b[ib]
            p = float(min(1.0, 2.0 * min((diff <= 0).mean(), (diff >= 0).mean())))
            method = "two_sample_bootstrap"
        elif all(np.isfinite(x) for x in ci_r1 + ci_bs) and delta != 0:
            # Fallback: NORMAL approximation from the CIs (not a bootstrap).
            # Reached only when the producer didn't persist bootstrap arrays.
            sd_r1 = max(1e-9, (ci_r1[1] - ci_r1[0]) / 3.92)
            sd_bs = max(1e-9, (ci_bs[1] - ci_bs[0]) / 3.92)
            draws_r1 = rng.normal(loc=v_r1, scale=sd_r1, size=n_bootstrap)
            draws_bs = rng.normal(loc=v_bs, scale=sd_bs, size=n_bootstrap)
            deltas = draws_r1 - draws_bs
            p = float(np.mean(deltas * np.sign(delta) <= 0))
            method = "normal_approx_from_ci"
        else:
            p = float("nan")
            method = "none"
        out_records.append({
            "layer": layer, "behaviour": behaviour,
            "statistic": statistic, "label_scheme": label_scheme,
            "value_r1": v_r1, "value_base": v_bs,
            "delta": delta, "p_value": p, "p_method": method,
            "ci95_r1": ci_r1, "ci95_base": ci_bs,
        })
    return {"records": out_records,
            "n_compared": int(sum(1 for r in out_records
                                  if "delta" in r)),
            "n_missing": int(sum(1 for r in out_records if "missing" in r))}


# ── Trajectory-distribution Wasserstein ────────────────────────────────────

def trajectory_wasserstein(
    r1_success: np.ndarray,
    r1_failure: np.ndarray,
    base_success: np.ndarray,
    base_failure: np.ndarray,
) -> dict:
    """2-Wasserstein distance between (success | failure) trajectory-stat
    distributions, in each model. Uses POT (Python Optimal Transport) when
    available; falls back to 1-D sort-based exact computation per dimension.

    Returns: {
      "w2_r1_success_vs_r1_failure": float,
      "w2_base_success_vs_base_failure": float,
      "w2_r1_vs_base_success": float,
      "w2_r1_vs_base_failure": float,
      "backend": "pot" | "sort1d",
    }
    """
    def _w2(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return float("nan")
        try:
            import ot
            a2 = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(-1, 1)
            b2 = b.reshape(-1, b.shape[-1]) if b.ndim > 1 else b.reshape(-1, 1)
            M = ot.dist(a2, b2)
            wa = np.ones(a2.shape[0]) / a2.shape[0]
            wb = np.ones(b2.shape[0]) / b2.shape[0]
            return float(np.sqrt(ot.emd2(wa, wb, M)))
        except ImportError:
            # 1-D sort-based Wasserstein on each feature, averaged.
            a1 = a.reshape(-1, 1) if a.ndim == 1 else a
            b1 = b.reshape(-1, 1) if b.ndim == 1 else b
            ds: list[float] = []
            for j in range(a1.shape[1]):
                sa = np.sort(a1[:, j])
                sb = np.sort(b1[:, j])
                # Resample to common length via interpolation.
                n = max(sa.size, sb.size)
                qa = np.interp(np.arange(n) / max(1, n - 1),
                               np.arange(sa.size) / max(1, sa.size - 1),
                               sa)
                qb = np.interp(np.arange(n) / max(1, n - 1),
                               np.arange(sb.size) / max(1, sb.size - 1),
                               sb)
                ds.append(float(np.sqrt(np.mean((qa - qb) ** 2))))
            return float(np.mean(ds))

    backend = "pot"
    try:
        import ot  # noqa: F401
    except ImportError:
        backend = "sort1d"

    return {
        "w2_r1_success_vs_r1_failure": _w2(np.asarray(r1_success),
                                            np.asarray(r1_failure)),
        "w2_base_success_vs_base_failure": _w2(np.asarray(base_success),
                                                np.asarray(base_failure)),
        "w2_r1_vs_base_success": _w2(np.asarray(r1_success),
                                      np.asarray(base_success)),
        "w2_r1_vs_base_failure": _w2(np.asarray(r1_failure),
                                      np.asarray(base_failure)),
        "backend": backend,
    }


# ── Cross-model classifier transfer ────────────────────────────────────────

def cross_model_classifier(
    r1_train: np.ndarray,
    r1_labels: np.ndarray,
    base_test: np.ndarray,
    base_labels: np.ndarray,
    *,
    seed: int = 0,
) -> float:
    """Train logistic regression on R1 success/failure trajectory features;
    test on base; return transfer accuracy. Returns NaN when sklearn is
    unavailable."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return float("nan")
    if r1_train.size == 0 or base_test.size == 0:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, solver="liblinear",
                             random_state=seed)
    clf.fit(r1_train, r1_labels)
    return float(clf.score(base_test, base_labels))


__all__ = [
    "cross_model_compare",
    "trajectory_wasserstein",
    "cross_model_classifier",
]
