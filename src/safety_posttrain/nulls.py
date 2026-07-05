"""Gating nulls for the spillover geometry diff (pre-registered 2026-07-03).

Implements the analysis gates of ``METHODOLOGY_SAFETY_SPILLOVER_2026-07-03.md``
so that no principal angle is ever reported against an implicit floor of zero:

- ``parity_check``            : the two activation dirs must describe the SAME rows in
                                the SAME order (chain_id / annotation_index / char_offset)
                                AND the same ``token_start`` values — the latter catches
                                any tokenizer divergence (e.g. the STAR1 extra-BOS bug,
                                which shifts token_start by +1) mechanically.
- ``matched_angle_observed``  : mean principal angle base-vs-post computed on S
                                matched-size subsamples (kills the n-asymmetry bias).
- ``permutation_angle_null``  : the finite-sample angle floor — pool base+post rows,
                                draw two disjoint size-m groups, angle between their
                                top-k subspaces, repeated; observed is reported as an
                                EXCESS over this null with a smoothed permutation p.
- ``split_half_floor``        : within-model sanity floor (angles between disjoint
                                halves of the SAME model's rows).
- ``paired_displacement``     : per-row (post - base) displacement; coherence ratio
                                ||mean d|| / mean ||d|| against a pairing-destroyed
                                (shuffled) null. Requires parity.

Pure numpy; deterministic under ``seed``. The surprisal control (teacher-forcing
drift vs per-span NLL) and the annotator-swap check need forward passes /
re-extraction and are NOT implemented here; the runner must list them as
not-yet-applied gates in its report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .spillover import subspace_basis

# row_index fields that must match one-for-one across the two extractions
_ROW_ID_FIELDS = ("chain_id", "annotation_index", "char_offset")
_TOKEN_FIELD = "token_start"


# ── parity ───────────────────────────────────────────────────────────────────

def parity_check(base_dir: Path | str, post_dir: Path | str,
                 behaviours: list[str]) -> dict:
    """Verify the two extractions describe identical rows in identical order."""
    out: dict = {"ok": True, "behaviours": {}}
    for d, name in ((Path(base_dir), "base"), (Path(post_dir), "post")):
        if not (d / "row_index.json").exists():
            return {"ok": False, "error": f"{name} dir has no row_index.json ({d})"}
    rb = json.loads((Path(base_dir) / "row_index.json").read_text())["rows"]
    rp = json.loads((Path(post_dir) / "row_index.json").read_text())["rows"]
    for b in behaviours:
        eb, ep = rb.get(b, []), rp.get(b, [])
        entry = {"n_base": len(eb), "n_post": len(ep),
                 "row_identity_ok": True, "token_start_ok": True,
                 "first_mismatch": None}
        if len(eb) != len(ep):
            entry["row_identity_ok"] = False
            entry["first_mismatch"] = "row count differs"
        else:
            for i, (a, c) in enumerate(zip(eb, ep)):
                if any(a.get(f) != c.get(f) for f in _ROW_ID_FIELDS):
                    entry["row_identity_ok"] = False
                    entry["first_mismatch"] = f"row {i}: id fields differ"
                    break
                if a.get(_TOKEN_FIELD) != c.get(_TOKEN_FIELD):
                    entry["token_start_ok"] = False
                    entry["first_mismatch"] = (
                        f"row {i}: token_start {a.get(_TOKEN_FIELD)} vs "
                        f"{c.get(_TOKEN_FIELD)} (tokenizer divergence?)")
                    break
        if not (entry["row_identity_ok"] and entry["token_start_ok"]):
            out["ok"] = False
        out["behaviours"][b] = entry
    return out


# ── angles with a real floor ─────────────────────────────────────────────────

def _mean_topk_angle(A: np.ndarray, B: np.ndarray, k: int) -> float:
    Ba, Bb = subspace_basis(A, k), subspace_basis(B, k)
    kk = min(Ba.shape[0], Bb.shape[0])
    s = np.clip(np.linalg.svd(Ba[:kk] @ Bb[:kk].T, compute_uv=False), -1.0, 1.0)
    return float(np.degrees(np.arccos(s)).mean())


def matched_angle_observed(Xb: np.ndarray, Xp: np.ndarray, k: int, m: int,
                           n_rep: int, rng: np.random.Generator) -> float:
    """Mean principal angle over ``n_rep`` matched-size (m) subsample draws."""
    vals = []
    for _ in range(n_rep):
        ib = rng.choice(Xb.shape[0], size=m, replace=False)
        ip = rng.choice(Xp.shape[0], size=m, replace=False)
        vals.append(_mean_topk_angle(Xb[ib], Xp[ip], k))
    return float(np.mean(vals))


def permutation_angle_null(Xb: np.ndarray, Xp: np.ndarray, k: int, m: int,
                           n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """SECONDARY diagnostic: two disjoint size-m groups from the pooled rows.

    CAUTION (found in synthetic validation, 2026-07-04): under a genuine
    subspace difference the pooled distribution's top-k cut can become
    degenerate (vanishing eigengap), inflating this null and MASKING real
    effects. Use ``within_model_null`` as the primary floor; keep this only
    as an exchangeability diagnostic.
    """
    pool = np.concatenate([Xb, Xp], axis=0)
    n = pool.shape[0]
    vals = np.empty(n_perm)
    for j in range(n_perm):
        idx = rng.permutation(n)
        vals[j] = _mean_topk_angle(pool[idx[:m]], pool[idx[m:2 * m]], k)
    return vals


def within_model_null(Xb: np.ndarray, Xp: np.ndarray, k: int, m: int,
                      n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """PRIMARY angle floor: disjoint size-m subsample pairs WITHIN one model.

    Alternates draws within base and within post (both marginals contribute),
    each requiring 2m <= n rows. The resulting distribution is the sampling
    floor for a top-k angle at size m when there is NO distribution change.
    """
    vals = []
    for j in range(n_draws):
        X = Xb if j % 2 == 0 else Xp
        if X.shape[0] < 2 * m:
            X = Xp if j % 2 == 0 else Xb        # fall back to the larger side
        if X.shape[0] < 2 * m:
            raise ValueError(f"within_model_null needs 2*m={2*m} rows; "
                             f"have {Xb.shape[0]}/{Xp.shape[0]}")
        idx = rng.permutation(X.shape[0])
        vals.append(_mean_topk_angle(X[idx[:m]], X[idx[m:2 * m]], k))
    return np.asarray(vals)


def split_half_floor(X: np.ndarray, k: int, n_rep: int,
                     rng: np.random.Generator) -> float:
    """Within-model floor: mean angle between disjoint halves of X."""
    n = X.shape[0]
    h = n // 2
    vals = []
    for _ in range(n_rep):
        idx = rng.permutation(n)
        vals.append(_mean_topk_angle(X[idx[:h]], X[idx[h:2 * h]], k))
    return float(np.mean(vals))


def gated_angle_report(Xb: np.ndarray, Xp: np.ndarray, k: int, m: int,
                       n_perm: int, n_rep: int, seed: int) -> dict:
    """Observed matched-n angle vs the WITHIN-MODEL floor, with smoothed p.

    ``m`` is clamped so that disjoint within-model pairs exist (2m <= n on at
    least one side; both sides where possible). The pooled-permutation value
    is reported as a secondary diagnostic only (see ``permutation_angle_null``).
    """
    rng = np.random.default_rng(seed)
    m = min(m, Xb.shape[0], Xp.shape[0],
            max(Xb.shape[0], Xp.shape[0]) // 2)
    observed = matched_angle_observed(Xb, Xp, k, m, n_rep, rng)
    null = within_model_null(Xb, Xp, k, m, n_perm, rng)
    p = (1 + int((null >= observed).sum())) / (1 + n_perm)
    perm = permutation_angle_null(Xb, Xp, k, m, min(n_perm, 100), rng)
    return {
        "k": k, "matched_n": m, "n_rep": n_rep, "n_perm": n_perm,
        "observed_deg": round(observed, 3),
        "null_mean_deg": round(float(null.mean()), 3),
        "null_sd_deg": round(float(null.std()), 3),
        "excess_deg": round(observed - float(null.mean()), 3),
        "p_within": round(p, 5),
        "pooled_perm_mean_deg": round(float(perm.mean()), 3),
        "split_half_base_deg": round(split_half_floor(Xb, k, max(5, n_rep), rng), 3)
            if Xb.shape[0] >= 2 * m else None,
        "split_half_post_deg": round(split_half_floor(Xp, k, max(5, n_rep), rng), 3)
            if Xp.shape[0] >= 2 * m else None,
    }


# ── paired displacement ──────────────────────────────────────────────────────

def paired_displacement(Xb: np.ndarray, Xp: np.ndarray, n_perm: int,
                        seed: int) -> dict:
    """Row-paired drift: coherence of (post - base) vs a sign-flip null.

    Coherence = ||mean d|| / mean ||d||: 1.0 for a uniform translation, ~0 for
    isotropic per-row noise. The null randomly flips the sign of each row's
    displacement (destroying any common direction while keeping every norm),
    which is the exact null for "the displacements share a direction".
    """
    if Xb.shape != Xp.shape:
        raise ValueError("paired_displacement requires parity (equal shapes)")
    rng = np.random.default_rng(seed)
    d = Xp.astype(np.float64) - Xb.astype(np.float64)
    norms = np.linalg.norm(d, axis=1)
    mean_norm = float(norms.mean())
    coherence = float(np.linalg.norm(d.mean(axis=0)) / mean_norm) if mean_norm > 0 else 0.0
    base_scale = float(np.linalg.norm(Xb.astype(np.float64), axis=1).mean())

    null = np.empty(n_perm)
    for j in range(n_perm):
        signs = rng.choice((-1.0, 1.0), size=d.shape[0])
        num = np.linalg.norm((d * signs[:, None]).mean(axis=0))
        null[j] = num / mean_norm if mean_norm > 0 else 0.0
    p = (1 + int((null >= coherence).sum())) / (1 + n_perm)
    return {
        "mean_disp_norm": round(mean_norm, 4),
        "mean_disp_over_base_norm": round(mean_norm / base_scale, 5) if base_scale > 0 else None,
        "coherence": round(coherence, 4),
        "coherence_null_mean": round(float(null.mean()), 4),
        "coherence_p_perm": round(p, 5),
    }
