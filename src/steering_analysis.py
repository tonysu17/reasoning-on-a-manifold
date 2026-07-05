"""
Phase 7 steering ANALYSIS — pure, summary-consuming, no model.

The single-vs-manifold comparison is only fundable if the comparison is made
**at matched on-target effect**, not at equal α. The manifold vector is a
renormalised projection of the single direction, so at the same α the two arms
deliver DIFFERENT perturbation magnitudes — equal α ≠ equal perturbation, and
"manifold has lower repetition at α=1" is then uninterpretable. This module
implements the matched-effect machinery the adversarial review demanded:

  (a) interpolate each method's damage-vs-effect curve and read collateral
      damage at a MATCHED on-target suppression (`collateral_at_matched_effect`);
  (b) the effect-vs-damage Pareto frontier (`pareto_frontier`);
  (c) a PAIRED bootstrap over tasks (BCa, 1000 resamples) of the matched-effect
      damage difference between two methods (`paired_bootstrap_matched_effect`);
  (d) Holm–Bonferroni across the 4 behaviours on one pre-registered summary
      statistic (`holm_bonferroni`, `compare_across_behaviours`).

Inputs are the dicts produced by `src.evaluation.aggregate_results` (on-target
mean + repetition/degenerate/mean_n_tokens/leakage) and, optionally,
`src.evaluation.aggregate_accuracy` (accuracy_drop_vs_vanilla). The paired
bootstrap additionally needs PER-TASK effect/damage values (the summaries only
carry cell means); `per_task_curves` extracts those from the same re-annotated
records, and the bootstrap also accepts pre-built per-task curves directly so it
is unit-testable on synthetic data with no model and no annotation.

Nothing here generates, runs, or judges — it is arithmetic over summaries.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Reuse the project's Holm step-down rather than re-deriving it (cbs is a
# sibling library; evaluation.py is the off-limits module, not this one).
from src.cbs.geometry import holm_correction

#: Default damage axes pulled from an aggregate_results cell, in the sign
#: convention "higher = more damage". mean_n_tokens is handled separately
#: (its damage direction is ambiguous — both collapse and runaway are bad).
DAMAGE_KEYS = ("repetition_rate", "degenerate_rate", "leakage_mean")

VANILLA = "vanilla"


# ── On-target effect + damage extraction from the summary ─────────────────────

def on_target_effect(cell_mean: Optional[float],
                     vanilla_mean: Optional[float]) -> Optional[float]:
    """Suppression effect = vanilla_fraction − steered_fraction (subtract mode).

    Positive = the behaviour was suppressed (the intended direction of the
    subtract-mode arms). None if either fraction is missing (an all-failed cell).
    """
    if cell_mean is None or vanilla_mean is None:
        return None
    return float(vanilla_mean - cell_mean)


def damage_scalar(cell: dict,
                  damage_keys: Sequence[str] = DAMAGE_KEYS,
                  accuracy_cell: Optional[dict] = None,
                  weights: Optional[dict] = None) -> Optional[float]:
    """Collapse a cell's damage axes into one scalar (higher = worse).

    Sums the present damage axes (each already in [0, 1]-ish, higher = worse)
    plus ``accuracy_drop_vs_vanilla`` from *accuracy_cell* when given. ``weights``
    optionally rescales individual axes. Returns None if NO damage axis is
    present (so the caller can skip the point rather than score a spurious 0).

    NOTE: this is a convenience aggregate for the Pareto/plotting path. The
    PRE-REGISTERED primary endpoint should be ONE axis (the caller passes a
    ``damage_fn`` selecting it) — summing axes is a secondary, descriptive view.
    """
    weights = weights or {}
    vals = []
    for k in damage_keys:
        v = cell.get(k)
        if v is not None:
            vals.append(float(v) * float(weights.get(k, 1.0)))
    if accuracy_cell is not None:
        drop = accuracy_cell.get("accuracy_drop_vs_vanilla")
        if drop is not None:
            vals.append(float(drop) * float(weights.get("accuracy_drop_vs_vanilla", 1.0)))
    if not vals:
        return None
    return float(np.sum(vals))


@dataclass
class MethodCurve:
    """A method's (on-target effect → damage) curve, sorted by effect.

    ``effects`` and ``damages`` are matched-length arrays sorted by increasing
    effect; ``alphas`` keeps the α each point came from (for provenance/Pareto
    labelling). Built per (behaviour, method) by ``method_curve``.
    """
    behaviour: str
    method: str
    effects: np.ndarray
    damages: np.ndarray
    alphas: np.ndarray

    def __len__(self) -> int:
        return int(self.effects.shape[0])


def method_curve(
    summary: dict,
    behaviour: str,
    method: str,
    damage_fn: Callable[[dict], Optional[float]] = damage_scalar,
    accuracy_summary: Optional[dict] = None,
    vanilla_method: str = VANILLA,
) -> MethodCurve:
    """Build the effect→damage curve for one (behaviour, method) from a summary.

    Effect is suppression vs the behaviour's vanilla cell (α=0). Damage is
    ``damage_fn(cell)`` (default: summed damage axes). Points with a missing
    effect OR missing damage are dropped. Sorted by increasing effect.
    """
    beh_block = summary.get(behaviour, {})
    van = beh_block.get(vanilla_method, {})
    # vanilla lives at α=0 (single shared baseline expanded per behaviour).
    van_alpha = min(van.keys(), key=float) if van else None
    van_mean = van.get(van_alpha, {}).get("mean") if van_alpha is not None else None

    acc_block = (accuracy_summary or {}).get(behaviour, {}).get(method, {})
    eff, dmg, alph = [], [], []
    for alpha, cell in sorted(beh_block.get(method, {}).items(), key=lambda kv: float(kv[0])):
        e = on_target_effect(cell.get("mean"), van_mean)
        d = damage_fn(cell, accuracy_cell=acc_block.get(alpha)) \
            if _accepts_accuracy(damage_fn) else damage_fn(cell)
        if e is None or d is None:
            continue
        eff.append(e); dmg.append(d); alph.append(float(alpha))
    order = np.argsort(eff) if eff else np.array([], dtype=int)
    return MethodCurve(
        behaviour=behaviour, method=method,
        effects=np.asarray(eff, dtype=float)[order],
        damages=np.asarray(dmg, dtype=float)[order],
        alphas=np.asarray(alph, dtype=float)[order],
    )


def _accepts_accuracy(fn: Callable) -> bool:
    """True if *fn* takes an ``accuracy_cell`` kwarg (so we can thread accuracy)."""
    try:
        import inspect
        return "accuracy_cell" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


# ── (a) collateral at a matched on-target effect ──────────────────────────────

def collateral_at_matched_effect(
    curve: MethodCurve, target_effect: float,
    extrapolate: bool = False,
) -> Optional[float]:
    """Interpolate *curve*'s damage at a MATCHED on-target suppression.

    Linear interpolation on the (effect → damage) curve. Returns None if the
    method never reaches *target_effect* (its max effect is below it) and
    ``extrapolate`` is False — the honest answer is "this method cannot deliver
    that much suppression", not an invented damage value. With ``extrapolate``
    the endpoints are clamped (held flat) — use only for robustness checks.

    This is THE comparison primitive: damage is always read at equal effect, so
    "equal α ≠ equal perturbation" can never contaminate it.
    """
    if len(curve) == 0:
        return None
    e, d = curve.effects, curve.damages
    if len(curve) == 1:
        return float(d[0]) if (extrapolate or np.isclose(e[0], target_effect)) else None
    lo, hi = float(e.min()), float(e.max())
    if target_effect < lo or target_effect > hi:
        if not extrapolate:
            return None
        target_effect = min(max(target_effect, lo), hi)
    # np.interp needs strictly increasing x; collapse duplicate-effect points to
    # their mean damage so a flat region doesn't break the interpolation.
    ue, ud = _dedup_increasing(e, d)
    if ue.shape[0] == 1:
        return float(ud[0])
    return float(np.interp(target_effect, ue, ud))


def _dedup_increasing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average y over duplicate x and return strictly increasing (x, y)."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    ux = np.unique(x)
    if ux.shape[0] == x.shape[0]:
        return x, y
    uy = np.array([y[x == v].mean() for v in ux])
    return ux, uy


def matched_effect_grid(curves: Sequence[MethodCurve],
                        n: int = 25) -> np.ndarray:
    """A grid of on-target effect levels COMMON to every curve (their overlap).

    The matched comparison is only defined where every method actually reaches
    that suppression, so the grid runs from max(min effect) to min(max effect)
    over the curves. Empty if ANY curve is empty (a method with no usable points
    cannot be matched) or if the curves don't overlap.
    """
    if not curves or any(len(c) == 0 for c in curves):
        return np.array([])
    lo = max(float(c.effects.min()) for c in curves)
    hi = min(float(c.effects.max()) for c in curves)
    if hi <= lo:
        return np.array([lo]) if np.isclose(hi, lo) else np.array([])
    return np.linspace(lo, hi, n)


# ── (b) effect-vs-damage Pareto frontier ──────────────────────────────────────

@dataclass
class ParetoPoint:
    method: str
    alpha: float
    effect: float
    damage: float


def pareto_frontier(curves: Sequence[MethodCurve]) -> list[ParetoPoint]:
    """Non-dominated (effect ↑, damage ↓) points across all method curves.

    A point dominates another if it has effect ≥ and damage ≤ with at least one
    strict. The frontier is the set dominated by nothing else; ties (identical
    effect AND damage) are all kept. Returned sorted by increasing effect.
    """
    pts: list[ParetoPoint] = []
    for c in curves:
        for e, d, a in zip(c.effects, c.damages, c.alphas):
            pts.append(ParetoPoint(c.method, float(a), float(e), float(d)))
    frontier: list[ParetoPoint] = []
    for p in pts:
        dominated = False
        for q in pts:
            if q is p:
                continue
            if (q.effect >= p.effect and q.damage <= p.damage
                    and (q.effect > p.effect or q.damage < p.damage)):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    return sorted(frontier, key=lambda p: (p.effect, p.damage))


def dominates(a_curve: MethodCurve, b_curve: MethodCurve,
              grid_n: int = 25) -> Optional[bool]:
    """Does method A Pareto-dominate B on the matched-effect overlap?

    Returns True if A's interpolated damage is ≤ B's at EVERY matched effect on
    their overlap (strictly less somewhere), False if B is ever strictly better,
    None if the curves don't overlap. A convenience for "is the manifold curve
    uniformly less destructive at equal effect".
    """
    if len(a_curve) == 0 or len(b_curve) == 0:
        return None
    grid = matched_effect_grid([a_curve, b_curve], n=grid_n)
    if grid.size == 0:
        return None
    da_list = [collateral_at_matched_effect(a_curve, t) for t in grid]
    db_list = [collateral_at_matched_effect(b_curve, t) for t in grid]
    if any(v is None for v in da_list) or any(v is None for v in db_list):
        return None
    da = np.asarray(da_list, dtype=float)
    db = np.asarray(db_list, dtype=float)
    if np.all(da <= db + 1e-12) and np.any(da < db - 1e-12):
        return True
    if np.all(db <= da + 1e-12) and np.any(db < da - 1e-12):
        return False
    return bool(np.all(da <= db + 1e-12))  # weak A-domination (or exact tie)


# ── (c) paired BCa bootstrap over tasks of the matched-effect difference ───────

@dataclass
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    n_tasks: int
    n_resamples: int
    method: str = "BCa"
    sign_consistent: bool = field(default=False)  # CI excludes 0

    def excludes_zero(self) -> bool:
        return (self.ci_low > 0) or (self.ci_high < 0)


def _bca_interval(theta_hat: float, boot: np.ndarray, jack: np.ndarray,
                  ci: float = 0.95) -> tuple[float, float]:
    """Bias-corrected & accelerated (BCa) CI from bootstrap + jackknife replicates.

    Efron's BCa: bias-correction z0 from the fraction of bootstrap replicates
    below the point estimate; acceleration a from the jackknife skew. Falls back
    to the percentile interval when the normal-CDF machinery degenerates (e.g.
    every bootstrap replicate identical, or a zero-variance jackknife).
    """
    from scipy.stats import norm
    boot = np.asarray(boot, dtype=float)
    n_b = boot.shape[0]
    alpha = (1.0 - ci) / 2.0
    # Bias-correction z0.
    prop = float(np.mean(boot < theta_hat))
    if prop <= 0.0 or prop >= 1.0:
        lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
        return float(lo), float(hi)
    z0 = norm.ppf(prop)
    # Acceleration from jackknife.
    jack = np.asarray(jack, dtype=float)
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    zlo, zhi = norm.ppf(alpha), norm.ppf(1.0 - alpha)

    def _adj(z):
        denom = 1.0 - a * (z0 + z)
        if denom == 0:
            denom = 1e-12
        return norm.cdf(z0 + (z0 + z) / denom)

    a_lo, a_hi = _adj(zlo), _adj(zhi)
    lo, hi = np.quantile(boot, [a_lo, a_hi])
    return float(lo), float(hi)


def paired_bootstrap_matched_effect(
    per_task_curves_a: dict,
    per_task_curves_b: dict,
    target_effect: float,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    extrapolate: bool = False,
) -> Optional[BootstrapResult]:
    """Paired BCa bootstrap of (damage_B − damage_A) at a MATCHED effect.

    *per_task_curves_a/b* map ``task_id -> MethodCurve`` (one method's curve
    measured on that single task). Both dicts must share task ids (the pairing).
    Each bootstrap resample draws task ids WITH replacement, pools the resampled
    tasks' points into one aggregate curve per method (so the curve is rebuilt
    on the resample, the correct unit being the task), interpolates damage at
    *target_effect*, and records ``damage_B − damage_A``. Positive ⇒ B more
    destructive at equal effect ⇒ A (e.g. manifold) wins.

    The point estimate is the same statistic on the full sample; the jackknife
    (leave-one-task-out) drives the BCa acceleration. Returns None if the pooled
    full-sample curves don't both reach *target_effect* (nothing to compare),
    unless ``extrapolate``.
    """
    tasks = sorted(set(per_task_curves_a) & set(per_task_curves_b))
    if len(tasks) < 2:
        raise ValueError("paired bootstrap needs >= 2 shared tasks")
    rng = np.random.default_rng(seed)

    def _stat(task_subset: Sequence[str]) -> Optional[float]:
        ca = _pool_curves([per_task_curves_a[t] for t in task_subset])
        cb = _pool_curves([per_task_curves_b[t] for t in task_subset])
        da = collateral_at_matched_effect(ca, target_effect, extrapolate=extrapolate)
        db = collateral_at_matched_effect(cb, target_effect, extrapolate=extrapolate)
        if da is None or db is None:
            return None
        return db - da

    theta_hat = _stat(tasks)
    if theta_hat is None:
        return None

    boot = []
    n = len(tasks)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        s = _stat([tasks[i] for i in idx])
        if s is not None:
            boot.append(s)
    boot = np.asarray(boot, dtype=float)
    if boot.size < 2:
        return None

    # Leave-one-out jackknife for the acceleration term.
    jack = []
    for i in range(n):
        s = _stat([tasks[j] for j in range(n) if j != i])
        if s is not None:
            jack.append(s)
    jack = np.asarray(jack, dtype=float)
    if jack.size < 2:
        lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    else:
        lo, hi = _bca_interval(theta_hat, boot, jack, ci=ci)

    res = BootstrapResult(
        estimate=float(theta_hat), ci_low=float(lo), ci_high=float(hi),
        n_tasks=n, n_resamples=int(boot.size))
    res.sign_consistent = res.excludes_zero()
    return res


def _pool_curves(curves: Sequence[MethodCurve]) -> MethodCurve:
    """Pool several per-task curves of the SAME method into one curve.

    Concatenates points then averages damage within each shared α (the α grid is
    common across tasks — every task is steered at the same α set), so the pooled
    curve is the across-task mean effect/damage per α. This is what a bootstrap
    resample's aggregate curve should be.
    """
    by_alpha_e: dict = defaultdict(list)
    by_alpha_d: dict = defaultdict(list)
    method = curves[0].method if curves else ""
    beh = curves[0].behaviour if curves else ""
    for c in curves:
        for e, d, a in zip(c.effects, c.damages, c.alphas):
            by_alpha_e[a].append(e)
            by_alpha_d[a].append(d)
    alphas = sorted(by_alpha_e)
    eff = np.array([np.mean(by_alpha_e[a]) for a in alphas])
    dmg = np.array([np.mean(by_alpha_d[a]) for a in alphas])
    order = np.argsort(eff)
    return MethodCurve(beh, method, eff[order], dmg[order], np.array(alphas)[order])


# ── per-task curve extraction (consumes re-annotated records, no model) ───────

def per_task_curves(
    steered_results: list[dict],
    annotated_steered: list[dict],
    behaviour: str,
    method: str,
    target_behaviours: Optional[list[str]] = None,
    damage_fn: Callable[[dict], Optional[float]] = damage_scalar,
    correctness: Optional[dict] = None,
    vanilla_correctness: Optional[dict] = None,
) -> dict:
    """Build ``{task_id -> MethodCurve}`` for one (behaviour, method) for the
    paired bootstrap, from the SAME records the aggregator consumes.

    Uses the project's own per-record metrics so the bootstrap and the summary
    agree by construction: on-target fraction via ``behaviour_fraction`` (from
    ``src.evaluation``), repetition via ``repetition_rate``, degenerate flag via
    the token floor, off-target leakage via the same re-annotation. Effect is
    suppression vs the per-task VANILLA fraction (shared baseline expanded to
    this behaviour). Tasks lacking a vanilla fraction are skipped (no pairing
    baseline). This re-derives nothing the aggregator owns — it imports it.
    """
    from src.evaluation import (behaviour_fraction, repetition_rate,
                                DEGENERATE_TOKEN_FLOOR)
    if target_behaviours is None:
        from src.annotation import TARGET_BEHAVIOURS
        target_behaviours = TARGET_BEHAVIOURS
    SHARED = "shared"

    ann_index = {(r["task_id"], r["behaviour"], r["method"], r["alpha"]):
                 r.get("annotations", []) for r in annotated_steered}

    # Per-task vanilla on-target fraction (shared baseline → this behaviour).
    vanilla_frac: dict = {}
    for r in steered_results:
        if r["method"] != VANILLA:
            continue
        anns = ann_index.get((r["task_id"], r["behaviour"], r["method"], r["alpha"]))
        if not anns:
            continue
        vanilla_frac[r["task_id"]] = behaviour_fraction(anns, behaviour)

    rows: dict = defaultdict(lambda: {"alpha": [], "effect": [], "damage": []})
    for r in steered_results:
        if r["method"] != method or r["behaviour"] != behaviour:
            continue
        tid = r["task_id"]
        if tid not in vanilla_frac:
            continue
        anns = ann_index.get((tid, behaviour, method, r["alpha"]))
        if not anns:
            continue
        frac = behaviour_fraction(anns, behaviour)
        effect = vanilla_frac[tid] - frac
        cell = {
            "repetition_rate": repetition_rate(r.get("chain", "")),
            "degenerate_rate": float((r.get("n_tokens") or 0) < DEGENERATE_TOKEN_FLOOR),
            "leakage_mean": float(np.mean([behaviour_fraction(anns, o)
                                           for o in target_behaviours if o != behaviour])),
        }
        acc_cell = None
        if correctness is not None:
            c = correctness.get((tid, behaviour, method, r["alpha"]))
            vc = (vanilla_correctness or {}).get(tid)
            if c is not None and vc is not None:
                acc_cell = {"accuracy_drop_vs_vanilla": float(vc) - float(c)}
        d = (damage_fn(cell, accuracy_cell=acc_cell) if _accepts_accuracy(damage_fn)
             else damage_fn(cell))
        if d is None:
            continue
        rows[tid]["alpha"].append(float(r["alpha"]))
        rows[tid]["effect"].append(float(effect))
        rows[tid]["damage"].append(float(d))

    out: dict = {}
    for tid, rec in rows.items():
        e = np.asarray(rec["effect"]); d = np.asarray(rec["damage"]); a = np.asarray(rec["alpha"])
        order = np.argsort(e)
        out[tid] = MethodCurve(behaviour, method, e[order], d[order], a[order])
    return out


# ── (d) Holm–Bonferroni across the 4 behaviours ───────────────────────────────

def holm_bonferroni(pvalues: dict) -> dict:
    """Holm step-down over a ``{behaviour -> p}`` map; returns adjusted p's.

    Wraps the project's ``holm_correction`` so the four behaviour-level tests of
    the SINGLE pre-registered statistic share one family-wise error budget.
    """
    keys = list(pvalues.keys())
    adj = holm_correction([pvalues[k] for k in keys])
    return {k: float(p) for k, p in zip(keys, adj)}


def bootstrap_two_sided_p(boot_result: BootstrapResult,
                          boot_dist: Optional[np.ndarray] = None) -> float:
    """A bootstrap two-sided p for H0: matched-effect difference = 0.

    Without the stored distribution this uses the achieved-significance-level
    proxy ``2·min(Φ(0; θ̂, se), 1−Φ)`` derived from the BCa interval's implied
    SE (CI width / (2·z_{0.975})); with *boot_dist* it uses the empirical
    two-sided tail. Kept simple — the headline endpoint is the CI sign; this p
    only feeds the Holm correction.
    """
    if boot_dist is not None and np.asarray(boot_dist).size > 1:
        b = np.asarray(boot_dist, dtype=float)
        p_low = float(np.mean(b <= 0))
        p_high = float(np.mean(b >= 0))
        return float(min(1.0, 2.0 * min(p_low, p_high)))
    from scipy.stats import norm
    z = norm.ppf(0.975)
    se = (boot_result.ci_high - boot_result.ci_low) / (2.0 * z)
    if se <= 0:
        return 0.0 if boot_result.estimate != 0 else 1.0
    z_stat = abs(boot_result.estimate) / se
    return float(2.0 * (1.0 - norm.cdf(z_stat)))


# ── Top-level convenience: compare two methods across behaviours ──────────────

@dataclass
class BehaviourComparison:
    behaviour: str
    target_effect: float
    bootstrap: Optional[BootstrapResult]
    p_value: Optional[float]
    pareto_a_dominates: Optional[bool]


def compare_across_behaviours(
    steered_results: list[dict],
    annotated_steered: list[dict],
    summary: dict,
    method_a: str,
    method_b: str,
    target_behaviours: Optional[list[str]] = None,
    effect_quantile: float = 0.8,
    damage_fn: Callable[[dict], Optional[float]] = damage_scalar,
    accuracy_summary: Optional[dict] = None,
    correctness: Optional[dict] = None,
    vanilla_correctness: Optional[dict] = None,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict:
    """End-to-end matched-effect comparison of *method_a* vs *method_b* per
    behaviour, with Holm–Bonferroni across behaviours on the bootstrap p.

    For each behaviour:
      * build both methods' summary curves and pick a MATCHED target effect at
        ``effect_quantile`` of the two curves' overlap (a single pre-registered
        operating point per behaviour);
      * paired BCa bootstrap of damage_B − damage_A at that effect (positive ⇒ A
        less destructive at equal effect);
      * record whether A Pareto-dominates B on the overlap.
    Then Holm-correct the four bootstrap p's. Returns
    ``{"per_behaviour": {beh: BehaviourComparison-as-dict}, "holm": {beh: padj}}``.

    The bootstrap unit is the TASK (per-task curves from ``per_task_curves``), so
    equal α is never used; everything is read at matched on-target effect.
    """
    if target_behaviours is None:
        from src.annotation import TARGET_BEHAVIOURS
        target_behaviours = TARGET_BEHAVIOURS

    per_beh: dict = {}
    raw_p: dict = {}
    for beh in target_behaviours:
        ca = method_curve(summary, beh, method_a, damage_fn, accuracy_summary)
        cb = method_curve(summary, beh, method_b, damage_fn, accuracy_summary)
        grid = matched_effect_grid([ca, cb])
        if grid.size == 0:
            per_beh[beh] = BehaviourComparison(beh, float("nan"), None, None, None)
            continue
        target = float(np.quantile(grid, effect_quantile))
        ptc_a = per_task_curves(steered_results, annotated_steered, beh, method_a,
                                target_behaviours, damage_fn, correctness, vanilla_correctness)
        ptc_b = per_task_curves(steered_results, annotated_steered, beh, method_b,
                                target_behaviours, damage_fn, correctness, vanilla_correctness)
        boot = None
        shared = set(ptc_a) & set(ptc_b)
        if len(shared) >= 2:
            boot = paired_bootstrap_matched_effect(
                ptc_a, ptc_b, target, n_resamples=n_resamples, seed=seed)
        p = bootstrap_two_sided_p(boot) if boot is not None else None
        per_beh[beh] = BehaviourComparison(
            beh, target, boot, p, dominates(ca, cb))
        if p is not None:
            raw_p[beh] = p

    holm = holm_bonferroni(raw_p) if raw_p else {}
    return {
        "method_a": method_a, "method_b": method_b,
        "per_behaviour": {b: _comparison_to_dict(c) for b, c in per_beh.items()},
        "holm": holm,
    }


def _comparison_to_dict(c: BehaviourComparison) -> dict:
    b = c.bootstrap
    return {
        "behaviour": c.behaviour,
        "target_effect": c.target_effect,
        "p_value": c.p_value,
        "pareto_a_dominates": c.pareto_a_dominates,
        "bootstrap": None if b is None else {
            "estimate": b.estimate, "ci_low": b.ci_low, "ci_high": b.ci_high,
            "n_tasks": b.n_tasks, "n_resamples": b.n_resamples,
            "excludes_zero": b.excludes_zero(),
        },
    }


__all__ = [
    "on_target_effect", "damage_scalar", "MethodCurve", "method_curve",
    "collateral_at_matched_effect", "matched_effect_grid",
    "ParetoPoint", "pareto_frontier", "dominates",
    "BootstrapResult", "paired_bootstrap_matched_effect",
    "per_task_curves", "holm_bonferroni", "bootstrap_two_sided_p",
    "BehaviourComparison", "compare_across_behaviours",
]
