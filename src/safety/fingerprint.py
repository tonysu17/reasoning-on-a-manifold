"""De-confounded recipe-fingerprint scoring (red-team F2 + F6).

``geometry_analysis.analyze_recipes`` fits the refusal direction and reports
Cohen's d / AUROC *in-sample*, then picks ``best_layer`` by argmax of that same d.
That double-dips twice: the direction is fit to maximise the separation it is then
scored on (severe at N << d, e.g. N≈300 vs d=2880), and the layer is selected on
the statistic under test (winner's curse). The recipe ranking could be selection
noise. This module supplies the honest replacements, leaving the existing API
untouched so its tests still pass:

  * :func:`separation_heldout` — grouped K-fold: fit the direction on train rows,
    score on held-out rows, grouped by chain/pair so a chain cannot straddle the
    split (the CF-15 discipline applied to the fingerprint).
  * :func:`separation_permutation_null` — label-permutation null for Cohen's d, so
    "the recipe carves a sharp axis" is tested against chance at this
    dimensionality rather than asserted.
  * :func:`bootstrap_separation_ci` — stimulus bootstrap CI on the held-out d.
  * :func:`layer_at_fraction` — pick the analysis layer by *pre-registered*
    fractional depth, not by argmax of d.

F6 length controls (effort changes CoT length, which moves dimension/PR estimates):
  * :func:`subsample_to_min` — equalise row counts across effort levels before any
    dimension/participation-ratio estimate.
  * :func:`length_normalised_engagement` — engagement per token, not per chain, so
    "engagement scales with effort" is not a token-count artefact.

Pure numpy; reuses ``refusal_direction``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.safety.refusal_direction import _auroc, _cohens_d, project, refusal_direction


# ── Grouped held-out scoring (F2) ─────────────────────────────────────────────

def _fold_of_groups(group_ids: Sequence, n_folds: int, seed: int) -> dict:
    """Assign each unique group id to a fold via a seeded shuffle (dependency-free
    LCG so it is reproducible without numpy's global RNG state)."""
    uniq = sorted(set(group_ids), key=lambda x: str(x))
    state = (seed * 2654435761 + 1) & 0xFFFFFFFF
    for i in range(len(uniq) - 1, 0, -1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        j = state % (i + 1)
        uniq[i], uniq[j] = uniq[j], uniq[i]
    return {g: (k % n_folds) for k, g in enumerate(uniq)}


def separation_heldout(
    harmful: np.ndarray,
    harmless: np.ndarray,
    groups_harmful: Sequence,
    groups_harmless: Sequence,
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """Grouped K-fold separation: fit the refusal direction on the training rows
    of each fold and score Cohen's d / AUROC on the held-out rows.

    Groups (chain id or pair id) are folded jointly across harmful and harmless,
    so a chain's rows never appear in both train and test. Returns the mean and
    per-fold held-out d/AUROC. Folds with an empty test side are skipped.
    """
    harmful = np.asarray(harmful, float)
    harmless = np.asarray(harmless, float)
    gh = list(groups_harmful)
    gl = list(groups_harmless)
    if len(gh) != harmful.shape[0] or len(gl) != harmless.shape[0]:
        raise ValueError("group arrays must match row counts")

    fold_of = _fold_of_groups(list(gh) + list(gl), n_folds, seed)
    fh = np.array([fold_of[g] for g in gh])
    fl = np.array([fold_of[g] for g in gl])

    ds, aucs = [], []
    for k in range(n_folds):
        tr_h, te_h = harmful[fh != k], harmful[fh == k]
        tr_l, te_l = harmless[fl != k], harmless[fl == k]
        if len(te_h) == 0 or len(te_l) == 0 or len(tr_h) < 2 or len(tr_l) < 2:
            continue
        direction = refusal_direction(tr_h, tr_l)  # fit on TRAIN only
        ds.append(_cohens_d(project(te_h, direction), project(te_l, direction)))
        aucs.append(_auroc(project(te_h, direction), project(te_l, direction)))
    if not ds:
        return {"cohens_d_mean": float("nan"), "auroc_mean": float("nan"),
                "cohens_d_folds": [], "n_folds_used": 0}
    return {
        "cohens_d_mean": float(np.mean(ds)),
        "auroc_mean": float(np.mean(aucs)),
        "cohens_d_folds": [round(float(x), 4) for x in ds],
        "n_folds_used": len(ds),
    }


def separation_permutation_null(
    harmful: np.ndarray,
    harmless: np.ndarray,
    *,
    n_perm: int = 1000,
    seed: int = 0,
) -> dict:
    """Label-permutation null for the in-sample Cohen's d of the refusal axis.

    Pools harmful+harmless, repeatedly reshuffles the labels, refits the direction
    on the shuffled split and recomputes the (in-sample) d. The smoothed p-value
    (Phipson–Smyth: ``(#null >= obs + 1)/(n_perm + 1)``) says whether the real
    separation exceeds what diff-of-means manufactures from noise at this
    dimensionality. A large d with p≈1 is the N≪d artefact this guards against.
    """
    harmful = np.asarray(harmful, float)
    harmless = np.asarray(harmless, float)
    nh = harmful.shape[0]
    pooled = np.vstack([harmful, harmless])
    n = pooled.shape[0]

    def _d(a, b):
        direction = refusal_direction(a, b)
        return _cohens_d(project(a, direction), project(b, direction))

    obs = _d(harmful, harmless)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(n)
        a, b = pooled[idx[:nh]], pooled[idx[nh:]]
        null[i] = _d(a, b)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"observed_d": float(obs), "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "p_value": float(p)}


def bootstrap_separation_ci(
    harmful: np.ndarray,
    harmless: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Stimulus-bootstrap CI on the in-sample Cohen's d (resample rows on each
    side with replacement). Wide CIs flag an unstable fingerprint."""
    harmful = np.asarray(harmful, float)
    harmless = np.asarray(harmless, float)
    rng = np.random.default_rng(seed)
    nh, nl = harmful.shape[0], harmless.shape[0]
    ds = np.empty(n_boot)
    for i in range(n_boot):
        a = harmful[rng.integers(0, nh, nh)]
        b = harmless[rng.integers(0, nl, nl)]
        direction = refusal_direction(a, b)
        ds[i] = _cohens_d(project(a, direction), project(b, direction))
    lo = float(np.percentile(ds, 100 * alpha / 2))
    hi = float(np.percentile(ds, 100 * (1 - alpha / 2)))
    return {"cohens_d_mean": float(ds.mean()), "ci_low": lo, "ci_high": hi}


def layer_at_fraction(layers: Sequence[int], fraction: float) -> int:
    """The layer nearest a fractional depth — the pre-registered alternative to
    argmax-of-d layer selection. ``fraction=0.75`` on 24 layers → layer 18."""
    layers = sorted(int(L) for L in layers)
    if not layers:
        raise ValueError("no layers")
    target = layers[0] + fraction * (layers[-1] - layers[0])
    return min(layers, key=lambda L: abs(L - target))


# ── F6 length controls ────────────────────────────────────────────────────────

def subsample_to_min(by_key: dict, *, seed: int = 0) -> dict:
    """Subsample every matrix in ``by_key`` to the minimum row count across keys,
    so intrinsic-dimension / participation-ratio estimates compared across effort
    levels are not confounded by sample size. Seeded and deterministic."""
    arrays = {k: np.asarray(v) for k, v in by_key.items()}
    counts = [a.shape[0] for a in arrays.values() if a.size]
    if not counts:
        return {k: a for k, a in arrays.items()}
    m = min(counts)
    rng = np.random.default_rng(seed)
    out = {}
    for k, a in arrays.items():
        if a.shape[0] <= m:
            out[k] = a
        else:
            out[k] = a[np.sort(rng.choice(a.shape[0], m, replace=False))]
    return out


def length_normalised_engagement(
    projection_sum_by_effort: dict,
    token_count_by_effort: dict,
    *,
    baseline: Optional[float] = None,
) -> dict:
    """Engagement per token, not per chain (F6): summed safety-direction
    projection over the generated DSR spans divided by the spans' token count,
    optionally minus a benign-CoT baseline. This is the corrected
    ``effort_engagement`` — the built one reads the prompt token and never sees
    the chain, so its "growth with effort" would be a token-count artefact.
    """
    out = {}
    for eff, proj_sum in projection_sum_by_effort.items():
        ntok = token_count_by_effort.get(eff, 0)
        val = (proj_sum / ntok) if ntok else float("nan")
        if baseline is not None and val == val:
            val = val - baseline
        out[eff] = float(val)
    return out


__all__ = [
    "separation_heldout", "separation_permutation_null", "bootstrap_separation_ci",
    "layer_at_fraction", "subsample_to_min", "length_normalised_engagement",
]
