"""Phase-7 Δ_floor headline analysis — the de-confounded suppression test.

This is the PRIMARY endpoint for the steering experiment (E8). It answers: does a
behaviour's OWN direction suppress that behaviour *beyond a matched random floor*
at the same injected energy / dimension and the same injection schedule?

    suppression_X      = vanilla_fraction − steered_fraction_X
    Δ_floor(b, arm)    = suppression_arm − suppression_floor
                       = floor_fraction − arm_fraction          (vanilla CANCELS)

Δ_floor > 0  ⇔  the arm reduces the behaviour MORE than its matched floor ⇔ the
*direction* matters, not just the perturbation magnitude. Because the shared
vanilla baseline cancels in the difference, the headline does not depend on the
(noisy) vanilla fraction at all — it is a paired, per-task arm-vs-floor contrast.

Floor pairing (verified against ``src/steered_inference._build_arms``):
    single_direction  → energy_matched_random      (ENERGY-matched; the real floor)
    manifold_k{N}     → random_subspace_k{N}        (DIMENSION-matched only — disclose)
    manifold_auto     → random_subspace_kauto

Design rules this module enforces (from METHODOLOGY_REFINEMENT_2026-06-25.md):
  * Operating point = a SEALED, fixed per-behaviour α* (a fixed dose), NOT a point
    tuned on the eval curves. We never read the operating point off the data
    (that is the ``effect_quantile=0.8`` leak in ``steering_analysis``); the
    caller passes the α at which generation ran.
  * Resample unit = the TASK (paired BCa bootstrap), samples/subspace-replicates
    pooled within a task first (``#s{j}`` / ``#rs{rep}`` suffixes collapsed to the
    ``base_task_id``).
  * Acceptance band ``band_b`` = fraction-scale RMS of cross-annotator disagreement
    — NEVER κ-derived (κ is the wrong scale).
  * PASS (per behaviour×arm) = Δ_floor>0 AND Holm-corrected paired-BCa CI excludes
    0 AND Δ_floor > band_b. Single-annotator only ⇒ "preliminary, band-ungated".
  * McNemar's EXACT test + the sign test are reported corroborators, NOT gates.

Pure arithmetic over the records ``07_evaluate_steering.py`` writes
(``steering_results.json`` + ``annotated_steered.json``). No model, no I/O.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Reuse the project's BCa machinery + Holm so the headline and the existing
# matched-effect path share one statistical core (no re-derivation).
from src.steering_analysis import (
    BootstrapResult,
    _bca_interval,
    holm_bonferroni,
)
from src.evaluation import behaviour_fraction

logger = logging.getLogger(__name__)

VANILLA = "vanilla"
SHARED = "shared"  # = src.steered_inference.SHARED_BASELINE
_EPS = 1e-12


# ── arm → floor pairing ───────────────────────────────────────────────────────

def floor_for_arm(method: str) -> Optional[str]:
    """The matched-floor method for a headline *arm* (None if not a headline arm).

    single_direction → energy_matched_random (energy-matched);
    manifold_k{N}/manifold_auto → random_subspace_k{N}/random_subspace_kauto
    (dimension-matched). Returns None for vanilla, floors, or unknown labels.
    """
    if method == "single_direction":
        return "energy_matched_random"
    m = re.fullmatch(r"manifold_(auto|k\d+)", method)
    if m:
        suf = m.group(1)
        return "random_subspace_kauto" if suf == "auto" else f"random_subspace_{suf}"
    return None


# ── per-task fraction extraction (samples / subspace reps pooled per task) ─────

def _base_task(rec: dict) -> str:
    """The pairing unit: the true task id with ``#rs``/``#s`` suffixes removed."""
    bt = rec.get("base_task_id")
    if bt:
        return str(bt)
    return str(rec.get("task_id", "")).split("#")[0]


def _alpha_eq(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def per_task_fraction(
    steered_results: list[dict],
    annotated_steered: list[dict],
    behaviour: str,
    method: str,
    alpha: float,
) -> dict:
    """``{base_task_id -> mean behaviour_fraction}`` for one (behaviour, method, α).

    Pools all matching records (multiple samples ``#s{j}`` and subspace replicates
    ``#rs{rep}`` of the same base task) into the per-task mean fraction. The shared
    vanilla baseline is generated under behaviour ``"shared"`` but scored for the
    target behaviour, so vanilla matches purely on ``method == "vanilla"``.
    Missing or empty re-annotations are skipped (they become "unresolved" pairs
    downstream — never silently scored 0, which would flatter destructive arms).
    """
    ann_index = {(r["task_id"], r["behaviour"], r["method"], r["alpha"]):
                 r.get("annotations", []) for r in annotated_steered}
    acc: dict = defaultdict(list)
    for r in steered_results:
        if r["method"] != method:
            continue
        if method != VANILLA and r["behaviour"] != behaviour:
            continue
        if not _alpha_eq(r["alpha"], alpha):
            continue
        anns = ann_index.get((r["task_id"], r["behaviour"], r["method"], r["alpha"]))
        if not anns:  # None (missing) or [] (empty) → skip
            continue
        acc[_base_task(r)].append(behaviour_fraction(anns, behaviour))
    return {t: float(np.mean(v)) for t, v in acc.items() if v}


# ── paired BCa bootstrap of a per-task mean (reuses the project's BCa core) ────

def paired_bootstrap_mean(
    values: Sequence[float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    return_distribution: bool = False,
):
    """Paired BCa bootstrap of the MEAN of per-task values (resample unit = task).

    For Δ_floor the per-task value is ``floor_frac − arm_frac``; positive mean ⇒
    the arm suppresses more than its floor. BCa bias-correction + leave-one-task
    jackknife acceleration, falling back to the percentile interval when the BCa
    machinery degenerates (handled inside ``_bca_interval``). Raises on <2 tasks.

    With ``return_distribution`` also returns the bootstrap replicate array (so
    callers can compute the assumption-free EMPIRICAL two-sided p rather than the
    CI-width normal approximation).
    """
    vals = np.asarray(values, dtype=float)
    n = vals.size
    if n < 2:
        raise ValueError("paired bootstrap needs >= 2 tasks")
    theta = float(vals.mean())
    rng = np.random.default_rng(seed)
    boot = np.array([vals[rng.integers(0, n, size=n)].mean()
                     for _ in range(n_resamples)], dtype=float)
    jack = np.array([np.delete(vals, i).mean() for i in range(n)], dtype=float)
    if np.allclose(boot, boot[0]) or np.allclose(jack, jack[0]):
        lo, hi = float(np.quantile(boot, (1 - ci) / 2)), float(np.quantile(boot, 1 - (1 - ci) / 2))
    else:
        lo, hi = _bca_interval(theta, boot, jack, ci=ci)
    res = BootstrapResult(estimate=theta, ci_low=float(lo), ci_high=float(hi),
                          n_tasks=int(n), n_resamples=int(boot.size))
    res.sign_consistent = res.excludes_zero()
    return (res, boot) if return_distribution else res


def empirical_two_sided_p(boot_dist: Sequence[float], estimate: float) -> float:
    """Assumption-free two-sided bootstrap p with a +1 continuity floor.

    ASL = the +1-floored fraction of resamples on the OPPOSITE side of 0 from the
    point estimate, doubled (two-sided), capped at 1. The +1 floor makes the
    smallest reportable p 2/(B+1): a bootstrap with B resamples cannot evidence p
    below ~1/B, and — critically — a p of exactly 0 would make the Holm step-down
    collapse (0 × rank = 0), silently voiding the multiplicity correction. Counts
    one side only, so mass exactly at 0 is never double-counted.
    """
    b = np.asarray(boot_dist, dtype=float)
    B = b.size
    if B == 0:
        return 1.0
    opp = int(np.sum(b <= 0)) if estimate >= 0 else int(np.sum(b >= 0))
    return float(min(1.0, 2.0 * (1 + opp) / (1 + B)))


# ── per-task transitions + corroborator tests (reported, NOT gates) ───────────

def matched_pair_transitions(arm_fr: dict, floor_fr: dict,
                             margin: float = 0.0) -> dict:
    """Classify each base task by per-task Δ_floor = floor_frac − arm_frac.

    Improved  : Δ >  margin  (arm suppresses more than its floor on this task)
    Degraded  : Δ < −margin  (arm suppresses less)
    Preserved : |Δ| ≤ margin (a tie within the margin band)
    Unresolved: task missing from arm OR floor (skipped re-annotation)
    Returns the counts plus the list of resolved per-task diffs.
    """
    tasks = set(arm_fr) | set(floor_fr)
    improved = degraded = preserved = unresolved = 0
    diffs: list[float] = []
    for t in tasks:
        if t not in arm_fr or t not in floor_fr:
            unresolved += 1
            continue
        d = floor_fr[t] - arm_fr[t]
        diffs.append(d)
        if d > margin:
            improved += 1
        elif d < -margin:
            degraded += 1
        else:
            preserved += 1
    return {"improved": improved, "degraded": degraded, "preserved": preserved,
            "unresolved": unresolved, "n_resolved": len(diffs), "diffs": diffs}


def sign_test(diffs: Sequence[float], margin: float = 0.0) -> dict:
    """Exact sign test on per-task diffs (ties within ``margin`` dropped).

    Order-free corroborator of the BCa CI sign: are positive per-task diffs more
    frequent than negative under H0 p=0.5? Uses the exact binomial.
    """
    pos = int(sum(1 for d in diffs if d > margin))
    neg = int(sum(1 for d in diffs if d < -margin))
    n = pos + neg
    p = _exact_binom_two_sided(min(pos, neg), n)
    return {"n_pos": pos, "n_neg": neg, "n_used": n, "p_value": p}


def mcnemar_exact(arm_suppresses: Sequence[bool],
                  floor_suppresses: Sequence[bool]) -> dict:
    """Exact McNemar test on paired binary suppression (arm vs floor).

    Each task contributes whether the ARM suppressed the behaviour (vs vanilla)
    and whether the FLOOR did. Tests marginal homogeneity via the discordant
    pairs: ``arm_only`` (arm suppressed, floor didn't) vs ``floor_only``. Exact
    (binomial on the discordant count) — NOT the χ² approximation, which is
    invalid at the N≈10–50 hold-out. Distinct from the sign test: this thresholds
    each arm against vanilla, the sign test uses the continuous arm-vs-floor diff.
    """
    arm = list(arm_suppresses)
    flr = list(floor_suppresses)
    if len(arm) != len(flr):
        raise ValueError("mcnemar needs equal-length paired booleans")
    b = int(sum(1 for a, f in zip(arm, flr) if a and not f))   # arm only
    c = int(sum(1 for a, f in zip(arm, flr) if f and not a))   # floor only
    n_disc = b + c
    p = _exact_binom_two_sided(min(b, c), n_disc)
    return {"arm_only": b, "floor_only": c, "n_discordant": n_disc,
            "n_concordant": int(len(arm) - n_disc), "p_value": p}


def _exact_binom_two_sided(k: int, n: int) -> float:
    """Two-sided exact binomial p at p0=0.5 (1.0 when n==0)."""
    if n <= 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)


# ── acceptance noise band (fraction-scale RMS — NEVER κ-derived) ──────────────

def noise_band_fraction_rms(frac_a: Sequence[float],
                            frac_b: Sequence[float]) -> float:
    """``band_b`` = RMS of |frac_a − frac_b| over the SAME chains, two annotators.

    The acceptance band a Δ_floor must clear. Fraction-scale (same units as the
    effect), NOT chance-corrected agreement (κ): κ measures span-label agreement
    and is the wrong scale for a corpus-fraction effect. Inputs are paired
    per-chain (or per-task) behaviour fractions from the two annotators.
    """
    a = np.asarray(frac_a, dtype=float)
    b = np.asarray(frac_b, dtype=float)
    if a.size == 0 or a.shape != b.shape:
        raise ValueError("noise band needs equal-length non-empty fraction arrays")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("noise band inputs contain non-finite values")
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ── per (behaviour, arm) Δ_floor cell + the across-family headline ────────────

@dataclass
class DeltaFloorCell:
    behaviour: str
    arm: str
    floor: str
    alpha: float
    n_tasks: int
    delta_floor: Optional[float]                 # mean per-task (floor_frac − arm_frac)
    bootstrap: Optional[BootstrapResult]
    raw_p: Optional[float]
    holm_p: Optional[float] = None
    # Reported context (means over arm∩floor∩vanilla). NOTE: these need NOT
    # satisfy suppression_arm − suppression_floor == delta_floor — delta_floor
    # averages over arm∩floor, these over the (possibly smaller) set that also
    # has a vanilla baseline; they coincide only when every shared task has one.
    suppression_arm: Optional[float] = None      # mean (vanilla − arm)
    suppression_floor: Optional[float] = None
    transitions: Optional[dict] = None
    sign: Optional[dict] = None
    mcnemar: Optional[dict] = None
    band_b: Optional[float] = None
    status: str = "ok"

    def passes(self) -> Optional[bool]:
        """PASS = Δ_floor>0 AND Holm CI excludes 0 AND Δ_floor > band_b.

        None ⇒ cannot be adjudicated (insufficient tasks, or band_b absent ⇒
        "preliminary, band-ungated" — never a pass).
        """
        if self.bootstrap is None or self.holm_p is None or self.delta_floor is None:
            return None
        sig = self.bootstrap.excludes_zero() and self.delta_floor > 0 and self.holm_p < 0.05
        if self.band_b is None:
            return None  # band-ungated: report, don't pass
        return bool(sig and self.delta_floor > self.band_b)


def delta_floor_cell(
    steered_results: list[dict],
    annotated_steered: list[dict],
    behaviour: str,
    arm: str,
    alpha: float,
    floor: Optional[str] = None,
    margin: float = 0.0,
    band_b: Optional[float] = None,
    n_resamples: int = 2000,
    seed: int = 0,
) -> DeltaFloorCell:
    """Compute the full Δ_floor cell for one (behaviour, arm) at the sealed α."""
    floor = floor or floor_for_arm(arm)
    if floor is None:
        raise ValueError(f"no matched floor for arm {arm!r}")
    arm_fr = per_task_fraction(steered_results, annotated_steered, behaviour, arm, alpha)
    flr_fr = per_task_fraction(steered_results, annotated_steered, behaviour, floor, alpha)
    van_fr = per_task_fraction(steered_results, annotated_steered, behaviour, VANILLA, 0.0)

    shared = sorted(set(arm_fr) & set(flr_fr))
    transitions = matched_pair_transitions(arm_fr, flr_fr, margin=margin)
    base = DeltaFloorCell(behaviour=behaviour, arm=arm, floor=floor, alpha=float(alpha),
                          n_tasks=len(shared), delta_floor=None, bootstrap=None,
                          raw_p=None, transitions=transitions, band_b=band_b)
    if len(shared) < 2:
        base.status = "insufficient-tasks"
        return base

    d = np.array([flr_fr[t] - arm_fr[t] for t in shared], dtype=float)
    boot, boot_dist = paired_bootstrap_mean(
        d, n_resamples=n_resamples, seed=seed, return_distribution=True)
    base.delta_floor = float(d.mean())
    base.bootstrap = boot
    # Assumption-free empirical p (+1-floored so Holm never collapses); the CI
    # sign remains the headline endpoint.
    base.raw_p = empirical_two_sided_p(boot_dist, base.delta_floor)
    base.sign = sign_test(d.tolist(), margin=margin)

    # McNemar: suppress-vs-vanilla binaries on tasks with a vanilla baseline.
    mc_tasks = [t for t in shared if t in van_fr]
    if not van_fr:
        logger.warning(f"{behaviour}/{arm}: no vanilla baseline at α=0 → McNemar + "
                       f"suppression context unavailable (Δ_floor unaffected)")
    if mc_tasks:
        arm_supp = [(van_fr[t] - arm_fr[t]) > _EPS for t in mc_tasks]
        flr_supp = [(van_fr[t] - flr_fr[t]) > _EPS for t in mc_tasks]
        base.mcnemar = mcnemar_exact(arm_supp, flr_supp)
        base.suppression_arm = float(np.mean([van_fr[t] - arm_fr[t] for t in mc_tasks]))
        base.suppression_floor = float(np.mean([van_fr[t] - flr_fr[t] for t in mc_tasks]))
    return base


def delta_floor_headline(
    steered_results: list[dict],
    annotated_steered: list[dict],
    arms: Sequence[str],
    alpha_star: dict,
    behaviours: Optional[Sequence[str]] = None,
    band_b: Optional[dict] = None,
    margin: float = 0.0,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """The full headline: a Δ_floor cell per (behaviour, arm), Holm-corrected over
    the whole (behaviour × arm) family on the bootstrap p (one shared FWER budget).

    *alpha_star* maps ``behaviour -> sealed α`` (the fixed dose generation ran at).
    *band_b* optionally maps ``behaviour -> acceptance band`` (fraction-RMS). The
    Holm family is the set of adjudicable cells (those with a bootstrap) — frozen
    by the (behaviours × arms) you pass, NOT chosen post-hoc on eligibility.
    """
    if behaviours is None:
        from src.annotation import TARGET_BEHAVIOURS
        behaviours = TARGET_BEHAVIOURS
    band_b = band_b or {}

    cells: dict = {}
    raw_p: dict = {}
    cell_i = 0
    for beh in behaviours:
        alpha = alpha_star.get(beh)
        if alpha is None:
            logger.warning(f"no sealed alpha* for {beh!r}; skipping")
            continue
        for arm in arms:
            # Distinct bootstrap stream per cell (independent resamples across the
            # family) for a clean multiplicity story.
            cell = delta_floor_cell(
                steered_results, annotated_steered, beh, arm, alpha,
                margin=margin, band_b=band_b.get(beh),
                n_resamples=n_resamples, seed=seed + cell_i)
            cell_i += 1
            cells[(beh, arm)] = cell
            if cell.raw_p is not None:
                raw_p[(beh, arm)] = cell.raw_p

    holm = holm_bonferroni(raw_p) if raw_p else {}
    for key, padj in holm.items():
        cells[key].holm_p = float(padj)

    return {
        "alpha_star": dict(alpha_star),
        "arms": list(arms),
        "behaviours": list(behaviours),
        "cells": {f"{b}|{a}": _cell_to_dict(c) for (b, a), c in cells.items()},
        "holm": {f"{b}|{a}": float(p) for (b, a), p in holm.items()},
        "n_pass": sum(1 for c in cells.values() if c.passes() is True),
        "n_preliminary": sum(1 for c in cells.values() if c.passes() is None
                             and c.bootstrap is not None),
    }


def _cell_to_dict(c: DeltaFloorCell) -> dict:
    b = c.bootstrap
    return {
        "behaviour": c.behaviour, "arm": c.arm, "floor": c.floor, "alpha": c.alpha,
        "n_tasks": c.n_tasks, "delta_floor": c.delta_floor,
        "raw_p": c.raw_p, "holm_p": c.holm_p,
        "suppression_arm": c.suppression_arm, "suppression_floor": c.suppression_floor,
        "band_b": c.band_b, "passes": c.passes(), "status": c.status,
        "bootstrap": None if b is None else {
            "estimate": b.estimate, "ci_low": b.ci_low, "ci_high": b.ci_high,
            "n_tasks": b.n_tasks, "n_resamples": b.n_resamples,
            "excludes_zero": b.excludes_zero()},
        "transitions": None if c.transitions is None else
            {k: v for k, v in c.transitions.items() if k != "diffs"},
        "sign": c.sign, "mcnemar": c.mcnemar,
    }


__all__ = [
    "floor_for_arm", "per_task_fraction", "paired_bootstrap_mean",
    "empirical_two_sided_p", "matched_pair_transitions", "sign_test",
    "mcnemar_exact", "noise_band_fraction_rms", "DeltaFloorCell",
    "delta_floor_cell", "delta_floor_headline",
]
