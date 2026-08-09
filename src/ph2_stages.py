"""Phase-2 causal-transport stage library (PHASE2_TRANSPORT_PREREG_2026-08-08.md).

Everything the ph2_executor stages need, split so that the SEALED design facts
(sites, arms, doses, floors, thresholds, seeds) are importable constants and the
decision arithmetic is pure and unit-tested locally, while every model-touching
path lazy-imports torch/transformers (pod-only).

Layer-indexing convention (the one-block-off trap the E10.1 cis site-check
caught): the prereg names sites in HIDDEN-STATE indices — the grounded frame
lives at hs[17], neighbours hs[16]/hs[18]. The repo's extraction and steering
machinery ("layer L" in 04_extract_activations / ActivationCache / SteeredModel)
hooks the OUTPUT of ``model.model.layers[L]`` = hs[L+1]. So hs[K] is produced by
machinery layer K-1: the frame steers at SteeredModel layer 16, exactly as the
executed E10 P2 run did (23_p2_redesign.py DIRECTIONS["das"] = (…, 17, 16)).
Use ``machinery_layer(hs_index)`` everywhere; never hand-translate.

Intervention family (sealed §4): the E10 P2 bounded clamp generalised to a
width-k orthonormal frame U with class-mean coordinates c —

    h' = h + β · w · g · (c − Uᵀh) · Uᵀ        (β = clamp gain, sealed 1.0;
                                                w = mixing weight, 1.0 outside
                                                injection-recovery; g = energy
                                                scale, 1.0 except energy floor)

"induce" clamps toward c_on (source-class mean), "suppress" toward c_off — the
two signs of §4's ±. Floors run suppression-oriented (the primary estimand is
suppression-oriented Δ_floor, §5).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ── sealed sites (hidden-state indices; prereg §1/§4) ────────────────────────

HS_SITE = 17                    # the grounded frame's site (hs[17])
HS_NEIGHBOURS = (16, 18)        # fixed neighbouring-layer controls (prereg §4)
HS_ALL = (16, 17, 18)


def machinery_layer(hs_index: int) -> int:
    """hs[K] = output of model.model.layers[K-1] — the index every repo tool
    ("layer L" in extraction / SteeredModel / DAS window_logprob hooks on
    resid_pre[L] == hidden_states[L]) needs to touch hs[K].

    NOTE the DAS scripts' window_logprob uses a forward_PRE hook on layers[L]
    (= hs[L], resid_pre), while extraction/steering use a forward (post) hook
    on layers[L] (= hs[L+1]). This helper is for the POST-hook family; DAS
    pair-state helpers already take the hs index directly.
    """
    if hs_index < 1:
        raise ValueError(f"hs index must be >= 1 (hs[0] is the embedding); got {hs_index}")
    return hs_index - 1


# ── sealed checkpoints (prereg §2) ───────────────────────────────────────────

CHECKPOINTS = {
    "base": {"hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
             "revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"},
    "star1": {"hf_id": "UCSC-VLAA/STAR1-R1-Distill-1.5B",
              "revision": "f865d7ac5136370518986a5273f4d731d7a0f254"},
    "deepscaler": {"hf_id": "agentica-org/DeepScaleR-1.5B-Preview",
                   "revision": "e3f524ce413a296b4d388e7560dd5c82c1c56725"},
}
TARGET_ROLES = ("star1", "deepscaler")
ALL_ROLES = ("base",) + TARGET_ROLES

# ── sealed frame, doses, seeds, sizes ────────────────────────────────────────

FRAME_PATH = ROOT / "results/das/R1-1.5B/width/frame_k2.npy"   # width 2 (§1)
FRAME_WIDTH = 2
REFIT_WIDTHS = (1, 2)           # E10.2 sealed widths (§4 re-fit)
CLAMP_GAIN = 1.0                # β — the P2 full-clamp operating point
E8_ALPHA_STAR = 1.0             # E8 sealed α* (floors' projective continuity dose)
RUN_SEED = 20260808
B_BOOT = 10_000
N_NULL_FRAMES = 20              # 20 sham + 20 random-orthogonal per target (§4)
GATE_QUANTILE = 0.95            # pooled 40-frame target-null p95 (§4)
BASE_ABS_THRESHOLDS = {"coord_auc": 0.65, "state_dependence": 2.0}  # continuity only
INJECTION_FRACTIONS = (0.25, 0.5, 0.75)                  # §6; endpoints reuse battery
INJECTION_PASS_POWER = 0.80                               # ≥80% detection of f=0.5
INJECTION_ALPHA = 0.05                                    # one-sided
ATTENUATION_MARGIN = 0.5        # sealed retention margin: "retained" needs the
                                # attenuation CI to exclude f >= this (the f the
                                # injection gate certifies detectable at 80%)

# Sealed DAS builder budget (E10.1/E10.2 recipe; M5 — no target-specific tuning)
DAS_BUILD = {"bs": 16, "epochs": 60, "lr": 1e-2, "seed": 0, "window": 12,
             "ctx": 320, "min_ctx": 16, "n_pairs": 400}

BEHAVIOUR = "backtracking"

# Damage-gate thresholds (sealed, prereg §8-10): active arm vs matched floors —
# chain-level rep4>0.8 excess >10pp, truncation excess >10pp, or boxed-accuracy
# drop >15pp vs vanilla ⇒ primary interpretation stops for that arm.
DAMAGE_REP4_THRESH = 0.8
DAMAGE_REP4_EXCESS = 0.10
DAMAGE_TRUNC_EXCESS = 0.10
DAMAGE_BOXED_DROP = 0.15

# ── the sealed 13-cell battery (prereg §4; §11 sizes it at ~12 cells ≈ 13) ───

SIGNS = ("induce", "suppress")

def battery_arms() -> list[dict]:
    """The sealed per-model battery: 13 cells (§4), enumerated once.

    Each arm dict: {method, family, sign, frame, needs} where ``method`` is the
    record label downstream stats key on, ``frame`` names which frame artifact
    drives it, and ``needs`` lists the discovery statistics it consumes.
    Floors/controls run suppression-oriented (the primary is suppression-
    oriented Δ_floor, §5); the ± families run both signs.
    """
    arms: list[dict] = [
        {"method": "vanilla", "family": "vanilla", "sign": None,
         "frame": None, "needs": []},
    ]
    for sign in SIGNS:
        arms.append({"method": f"transported_raw_{sign}", "family": "transported_raw",
                     "sign": sign, "frame": "base", "needs": ["class_means_base"]})
    for sign in SIGNS:
        arms.append({"method": f"transported_norm_{sign}", "family": "transported_norm",
                     "sign": sign, "frame": "base",
                     "needs": ["class_means_base", "norm_gain"]})
    for sign in SIGNS:
        arms.append({"method": f"transported_whitened_{sign}",
                     "family": "transported_whitened", "sign": sign, "frame": "base",
                     "needs": ["class_means_base", "whitening"]})
    for sign in SIGNS:
        arms.append({"method": f"refit_{sign}", "family": "refit", "sign": sign,
                     "frame": "refit", "needs": ["class_means_refit"]})
    arms += [
        {"method": "sham_frame", "family": "control", "sign": "suppress",
         "frame": "sham0", "needs": ["class_means_sham0"]},
        {"method": "random_orthogonal", "family": "control", "sign": "suppress",
         "frame": "randorth0", "needs": ["class_means_randorth0"]},
        {"method": "count_matched_floor", "family": "floor", "sign": "suppress",
         "frame": "count_floor", "needs": ["class_means_count_floor"]},
        {"method": "energy_matched_floor", "family": "floor", "sign": "suppress",
         "frame": "energy_floor", "needs": ["class_means_energy_floor",
                                            "energy_scale"]},
    ]
    return arms


N_BATTERY_CELLS = 13  # locked by test_battery_is_thirteen_cells


def floor_for_battery_arm(method: str) -> Optional[str]:
    """Δ_floor pairing for Phase-2 arms (primary floor; §5 names sham as the
    second control — analyse computes both, energy-matched is primary, matching
    ``src.delta_floor``'s energy-matched philosophy for single_direction)."""
    active = re.fullmatch(
        r"(transported_(raw|norm|whitened)|refit)_(induce|suppress)", method)
    if active:
        return "energy_matched_floor"
    if method in ("sham_frame", "random_orthogonal", "count_matched_floor"):
        return "energy_matched_floor"   # controls contrasted against the same floor
    return None                          # vanilla / floors themselves


SECONDARY_FLOOR = "sham_frame"          # §5: "vs sham and energy-matched controls"


# ── cell enumeration + resume keys (pure) ────────────────────────────────────

def battery_cell_key(role: str, method: str, task_id: str) -> tuple:
    return (role, method, task_id)


def enumerate_battery_cells(tasks: Sequence[dict], roles: Sequence[str] = ALL_ROLES,
                            done: Optional[set] = None) -> list[dict]:
    """Every (role, arm, task) generation unit still pending. One sample per
    task per arm (sealed §4). ``done`` holds battery_cell_key tuples."""
    done = done or set()
    cells = []
    for role in roles:
        for arm in battery_arms():
            for t in tasks:
                key = battery_cell_key(role, arm["method"], t["id"])
                if key in done:
                    continue
                cells.append({"role": role, "arm": arm, "task": t, "key": key})
    return cells


def enumerate_injection_cells(tasks: Sequence[dict],
                              done: Optional[set] = None) -> list[dict]:
    """Injection-recovery units: BASE model only, f∈{0.25,0.5,0.75} × tasks
    (§6). The f=0 / f=1 endpoints are the main battery's transported_raw_suppress
    and sham_frame arms — never regenerated here."""
    done = done or set()
    cells = []
    for f in INJECTION_FRACTIONS:
        method = injection_method_label(f)
        for t in tasks:
            key = battery_cell_key("base", method, t["id"])
            if key in done:
                continue
            cells.append({"role": "base", "fraction": f, "method": method,
                          "task": t, "key": key})
    return cells


def injection_method_label(f: float) -> str:
    return f"injection_f{f:.2f}"


# ── frame construction + transport corrections (pure numpy) ──────────────────

def _seed_from(seed_key: str) -> int:
    return int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little")


def random_orthonormal_frame(d: int, k: int, seed_key: str) -> np.ndarray:
    """Seeded Haar-random (d, k) orthonormal frame — QR of a seeded Gaussian,
    the same construction as src.steered_inference.random_subspace_projection
    (deterministic per seed_key, distinct across keys)."""
    rng = np.random.default_rng(_seed_from(seed_key))
    G = rng.standard_normal((d, k))
    Q, _ = np.linalg.qr(G)
    return Q.astype(np.float32)


def class_mean_coords(H_on: np.ndarray, H_off: np.ndarray,
                      U: np.ndarray) -> dict[str, np.ndarray]:
    """c_on / c_off — the class-mean frame coordinates (the P2 clamp targets),
    computed from DISCOVERY prediction-position states only (§4: norm/whitening/
    gain statistics come from discovery anchors, never the evaluation battery).

    H_on: (n, d) source-class states (behaviour-onset predictors);
    H_off: (n, d) base-class states; U: (d, k) orthonormal frame.
    """
    U = np.asarray(U, dtype=np.float64)
    return {"c_on": (np.asarray(H_on, dtype=np.float64) @ U).mean(axis=0),
            "c_off": (np.asarray(H_off, dtype=np.float64) @ U).mean(axis=0)}


def norm_gain(base_states: np.ndarray, target_states: np.ndarray) -> float:
    """Scalar norm correction (§4 arm 2): per-model mean residual L2 ratio at
    the frame site, estimated on discovery spans only. Scales the transported
    clamp targets c by (target mean L2 / base mean L2)."""
    b = float(np.mean(np.linalg.norm(np.asarray(base_states, dtype=np.float64), axis=1)))
    t = float(np.mean(np.linalg.norm(np.asarray(target_states, dtype=np.float64), axis=1)))
    if b <= 0:
        return 1.0
    return t / b


def whitening_stats(states: np.ndarray) -> np.ndarray:
    """Per-dimension std over discovery states (§4 arm 3: DIAGONAL whitening
    only in this launch — no full covariance)."""
    s = np.asarray(states, dtype=np.float64).std(axis=0)
    return np.where(s < 1e-8, 1.0, s)


def whitened_frame(U: np.ndarray, base_std: np.ndarray,
                   target_std: np.ndarray) -> np.ndarray:
    """Diagonally-whitened transport: express the base frame in base-whitened
    coordinates, re-embed with the target's per-dimension scale, re-orthonormalise
    (QR). Columns keep orientation up to the QR sign convention."""
    U = np.asarray(U, dtype=np.float64)
    W = U * (np.asarray(target_std, dtype=np.float64)
             / np.asarray(base_std, dtype=np.float64))[:, None]
    Q, R = np.linalg.qr(W)
    # Fix QR sign indeterminacy so columns correlate positively with the input.
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return (Q * signs[None, :]).astype(np.float32)


def clamp_targets_for_arm(family: str, sign: str, class_means: dict,
                          gain: float = 1.0) -> np.ndarray:
    """The clamp constant c for one arm: c_on (induce) or c_off (suppress),
    optionally norm-gain-scaled (transported_norm)."""
    c = np.asarray(class_means["c_on" if sign == "induce" else "c_off"],
                   dtype=np.float64)
    return c * float(gain)


def mixing_weights(f: float) -> dict[str, float]:
    """Injection-recovery dose interpolation (§6): the applied perturbation is
    (1−f)·frame-clamp ⊕ f·sham-clamp, so f=0 reproduces the transported arm and
    f=1 the sham arm exactly (those endpoints come from the main battery)."""
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"attenuation fraction must be in [0,1], got {f}")
    return {"frame": 1.0 - float(f), "sham": float(f)}


# ── chain-level endpoint helpers (pure; A2 + damage gate) ────────────────────

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def boxed_answer(chain: str) -> Optional[str]:
    """Last \\boxed{...} content, whitespace-normalised (correctness = boxed
    exact-match, prereg §5)."""
    m = _BOXED_RE.findall(chain or "")
    if not m:
        return None
    return " ".join(m[-1].split())


def chain_stats(chain: str, n_tokens: int, cap: int,
                expected_answer: Optional[str] = None) -> dict:
    """Per-chain endpoint bundle: boxed answer/match, rep4 loop flag (E9
    convention), truncation (hit the cap), token count."""
    from src.evaluation import repetition_rate
    ans = boxed_answer(chain)
    correct = None
    if expected_answer is not None:
        correct = (ans is not None
                   and " ".join(str(expected_answer).split()) == ans)
    return {
        "boxed_answer": ans,
        "boxed_present": ans is not None,
        "boxed_correct": correct,
        "rep4": float(repetition_rate(chain, n=4)) if chain else 0.0,
        "looped": bool(chain) and repetition_rate(chain, n=4) > DAMAGE_REP4_THRESH,
        "truncated": int(n_tokens) >= int(cap),
        "n_tokens": int(n_tokens),
    }


def damage_gate(active: Sequence[dict], floor: Sequence[dict],
                vanilla: Sequence[dict]) -> dict:
    """Sealed damage gate for one (model, arm): active vs matched floor on
    loop/truncation excess, vs vanilla on boxed accuracy. Any breach ⇒
    damage_ok=False (primary interpretation stops for that arm — decide_outcome
    maps it to 'damage_stop')."""
    def rate(rows, key):
        vals = [bool(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def boxed_rate(rows):
        vals = [r["boxed_correct"] for r in rows if r.get("boxed_correct") is not None]
        if not vals:  # no ground-truth answers → fall back to boxed-present
            vals = [r["boxed_present"] for r in rows if "boxed_present" in r]
        return float(np.mean([bool(v) for v in vals])) if vals else None

    out = {"loop_excess": None, "trunc_excess": None, "boxed_drop": None,
           "damage_ok": True, "reasons": []}
    la, lf = rate(active, "looped"), rate(floor, "looped")
    ta, tf = rate(active, "truncated"), rate(floor, "truncated")
    ba, bv = boxed_rate(active), boxed_rate(vanilla)
    if la is not None and lf is not None:
        out["loop_excess"] = la - lf
        if out["loop_excess"] > DAMAGE_REP4_EXCESS:
            out["damage_ok"] = False
            out["reasons"].append(
                f"loop excess {out['loop_excess']:.3f} > {DAMAGE_REP4_EXCESS}")
    if ta is not None and tf is not None:
        out["trunc_excess"] = ta - tf
        if out["trunc_excess"] > DAMAGE_TRUNC_EXCESS:
            out["damage_ok"] = False
            out["reasons"].append(
                f"truncation excess {out['trunc_excess']:.3f} > {DAMAGE_TRUNC_EXCESS}")
    if ba is not None and bv is not None:
        out["boxed_drop"] = bv - ba
        if out["boxed_drop"] > DAMAGE_BOXED_DROP:
            out["damage_ok"] = False
            out["reasons"].append(
                f"boxed drop {out['boxed_drop']:.3f} > {DAMAGE_BOXED_DROP}")
    return out


# ── attenuation + primary test (pure) ────────────────────────────────────────

def attenuation_fraction(delta_base: float, delta_target: float,
                         min_denominator: float = 1e-6) -> Optional[float]:
    """f = 1 − Δ_target/Δ_base, reported ONLY with a stable sign/denominator
    (§5): Δ_base must be positive (the suppression-oriented effect exists in
    the base) and non-negligible. Returns None otherwise — callers must then
    report raw effects only, never a ratio."""
    if delta_base is None or delta_target is None:
        return None
    if delta_base <= min_denominator:
        return None
    return 1.0 - (delta_target / delta_base)


def paired_attenuation_test(per_task_base: dict, per_task_target: dict,
                            n_resamples: int = B_BOOT,
                            seed: int = RUN_SEED) -> dict:
    """The primary per-checkpoint cell: paired per-task Δ_floor difference
    (base − target) over the shared manifest tasks, cluster bootstrap by task
    (B=10,000, seed 20260808, same resampled indices across models by
    construction — the resample is over the shared task list).

    Inputs are per-task Δ_floor values (floor_frac − arm_frac, already pooled
    within task by src.delta_floor.per_task_fraction upstream).
    Returns raw effects, the paired difference CI, the empirical p, the guarded
    attenuation point estimate + bootstrap CI, and counts.
    """
    from src.delta_floor import paired_bootstrap_mean, empirical_two_sided_p
    shared = sorted(set(per_task_base) & set(per_task_target))
    out = {"n_shared_tasks": len(shared),
           "n_base_only": len(set(per_task_base) - set(per_task_target)),
           "n_target_only": len(set(per_task_target) - set(per_task_base))}
    if len(shared) < 2:
        out["status"] = "insufficient-tasks"
        return out
    b = np.array([per_task_base[t] for t in shared], dtype=float)
    t = np.array([per_task_target[t] for t in shared], dtype=float)
    out["delta_base"] = float(b.mean())
    out["delta_target"] = float(t.mean())
    boot, dist = paired_bootstrap_mean(b - t, n_resamples=n_resamples, seed=seed,
                                       return_distribution=True)
    out["paired_diff"] = {"estimate": boot.estimate, "ci_low": boot.ci_low,
                          "ci_high": boot.ci_high, "n_tasks": boot.n_tasks,
                          "excludes_zero": boot.excludes_zero()}
    out["raw_p"] = empirical_two_sided_p(dist, boot.estimate)
    out["attenuation"] = attenuation_fraction(out["delta_base"], out["delta_target"])
    # Bootstrap the attenuation ratio on the SAME resample stream (guarded per
    # replicate; replicates with unstable denominators are dropped and counted).
    rng = np.random.default_rng(seed)
    n = len(shared)
    fs = []
    n_invalid = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        f = attenuation_fraction(float(b[idx].mean()), float(t[idx].mean()))
        if f is None:
            n_invalid += 1
        else:
            fs.append(f)
    if fs and n_invalid / n_resamples < 0.05:
        out["attenuation_ci"] = [float(np.quantile(fs, 0.025)),
                                 float(np.quantile(fs, 0.975))]
    else:
        out["attenuation_ci"] = None
    out["attenuation_n_invalid_resamples"] = int(n_invalid)
    out["status"] = "ok"
    return out


def raw_retained(cell: dict, margin: float = ATTENUATION_MARGIN) -> bool:
    """'retained' needs the uncorrected transported effect to EXCLUDE
    attenuation >= the sealed margin: attenuation CI upper bound < margin, with
    a stable ratio. Unstable ratio or missing CI ⇒ not retained (raw effects
    still reported)."""
    ci = cell.get("attenuation_ci")
    if ci is None or cell.get("attenuation") is None:
        return False
    return ci[1] < margin


# ── injection-recovery power (pure; §6 gate) ─────────────────────────────────

def injection_detection_power(per_task_f0: dict, per_task_fx: dict,
                              alpha: float = INJECTION_ALPHA,
                              n_resamples: int = 2000,
                              n_power_reps: int = 500,
                              seed: int = RUN_SEED) -> dict:
    """Estimated power to DETECT attenuation f at one-sided alpha under the
    executed pipeline: per-task paired diffs d_t = Δ_floor_t(f=0) − Δ_floor_t(f),
    H0: mean ≤ 0. Power = fraction of task-resampled replicates whose one-sided
    bootstrap p < alpha (evaluated BEFORE any target outcome is inspected)."""
    shared = sorted(set(per_task_f0) & set(per_task_fx))
    if len(shared) < 2:
        return {"status": "insufficient-tasks", "n_shared": len(shared)}
    d = np.array([per_task_f0[t] - per_task_fx[t] for t in shared], dtype=float)
    rng = np.random.default_rng(seed)
    n = d.size
    rejections = 0
    for _ in range(n_power_reps):
        sample = d[rng.integers(0, n, size=n)]
        boots = np.array([sample[rng.integers(0, n, size=n)].mean()
                          for _ in range(n_resamples)])
        p_one_sided = (1 + int(np.sum(boots <= 0))) / (1 + n_resamples)
        if p_one_sided < alpha:
            rejections += 1
    power = rejections / n_power_reps
    return {"status": "ok", "n_shared": int(n), "mean_diff": float(d.mean()),
            "power": float(power), "alpha": alpha,
            "passes": bool(power >= INJECTION_PASS_POWER)}


# ── annotation glue helpers (pure) ───────────────────────────────────────────

ANNOTATION_DEDUP_KEYS = ("model_role", "behaviour", "method", "alpha", "task_id")


def proxy_chunk_budget_ok(safety_margin_tokens: int = 2300) -> dict:
    """Pre-spend check that the annotation pipeline's chunking keeps every
    single proxy call inside the 29-second AWS API-Gateway hard timeout.

    The annotated output echoes the input with labels, so output length ≈ input
    chunk length + label overhead. At ~80 tok/s the 29 s ceiling is ~2,300
    output tokens; the pipeline chunks above CHUNK_THRESHOLD_TOKENS to
    ~CHUNK_TARGET_TOKENS(+overlap) per call, which must sit well under that.
    """
    from src.annotation import (CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS,
                                CHUNK_THRESHOLD_TOKENS)
    worst_call = max(CHUNK_THRESHOLD_TOKENS,
                     CHUNK_TARGET_TOKENS + CHUNK_OVERLAP_TOKENS)
    overhead = 1.35  # label-delimiter echo overhead on the output side
    worst_output = int(worst_call * overhead)
    return {"worst_single_call_input_tokens": worst_call,
            "worst_output_tokens_est": worst_output,
            "budget_tokens": safety_margin_tokens,
            "ok": worst_output <= safety_margin_tokens}


def battery_record(role: str, method: str, task: dict, gen: dict,
                   sign: Optional[str], frame_name: Optional[str],
                   extra: Optional[dict] = None) -> dict:
    """One generation record in the E8-compatible schema so that
    ``src.delta_floor.per_task_fraction`` consumes Phase-2 files UNCHANGED
    (authoritative-pooling requirement — the red-team F-item): behaviour is
    "backtracking" for steered arms / "shared" for vanilla; ``method`` is the
    arm label; ``alpha`` is the clamp gain (0.0 for vanilla)."""
    is_vanilla = method == "vanilla"
    rec = {
        "behaviour": "shared" if is_vanilla else BEHAVIOUR,
        "method": method,
        "alpha": 0.0 if is_vanilla else CLAMP_GAIN,
        "task_id": task["id"],
        "base_task_id": task["id"],
        "model_role": role,
        "chain": gen["chain"],
        "n_tokens": gen["n_tokens"],
        "layer": None if is_vanilla else machinery_layer(HS_SITE),
        "hs_site": None if is_vanilla else HS_SITE,
        "sign": sign,
        "frame": frame_name,
        "temperature": gen.get("temperature", 0.0),
        "seed": gen.get("seed", RUN_SEED),
        "mean_abs_displacement": gen.get("mean_abs_displacement"),
    }
    if extra:
        rec.update(extra)
    return rec


# ── torch-lazy: frame-clamp steering engine (pod) ────────────────────────────

def frame_clamp_model(model, tokenizer, terms: list[dict], hs_site: int = HS_SITE):
    """A SteeredModel-compatible generator applying a SUM of frame-clamp terms

        h' = h + Σ_j β_j · w_j · g_j · (c_j − U_jᵀh) · U_jᵀ

    at hs[hs_site] (hook on machinery layer hs_site−1, matching the executed
    E10 P2 site convention). ``terms``: [{U (d,k) np, c (k,) np, beta, weight,
    energy_scale}]. One term = a battery arm; two terms with mixing_weights(f)
    = an injection-recovery arm. Inherits SteeredModel.generate/generate_batch
    (greedy default, config 8192 cap, OOM self-heal) — only the hook differs.

    Displacement accounting: mean |Δ(Uᵀh)| summed over frame coordinates —
    the E10 P2 matching variable, recorded per generation.
    """
    import torch
    from src.steered_inference import SteeredModel

    class _FrameClampModel(SteeredModel):
        def __init__(self):
            layer = machinery_layer(hs_site)
            d = model.config.hidden_size
            # Satisfy the parent ctor with an inert zero vector; the parent's
            # _r is unused because _hook_fn is overridden below.
            super().__init__(model, tokenizer, np.zeros(d, dtype=np.float32),
                             layer, alpha=0.0, mode="clamp")
            device = next(model.parameters()).device
            self._terms = []
            for t in terms:
                U = torch.tensor(np.asarray(t["U"], dtype=np.float32),
                                 device=device)
                if U.ndim == 1:
                    U = U[:, None]
                c = torch.tensor(np.asarray(t["c"], dtype=np.float32).reshape(-1),
                                 device=device)
                if U.shape[1] != c.shape[0]:
                    raise ValueError(f"frame width {U.shape[1]} != len(c) {c.shape[0]}")
                self._terms.append({
                    "U": U, "c": c,
                    "beta": float(t.get("beta", CLAMP_GAIN)),
                    "weight": float(t.get("weight", 1.0)),
                    "energy_scale": float(t.get("energy_scale", 1.0)),
                })

        def _hook_fn(self, module, input, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            h = hidden.float()                       # (b, s, d)
            delta = torch.zeros_like(h)
            for t in self._terms:
                U, c = t["U"], t["c"]                # (d, k), (k,)
                coords = torch.einsum("bsd,dk->bsk", h, U)
                gain = t["beta"] * t["weight"] * t["energy_scale"]
                cdelta = gain * (c.view(1, 1, -1) - coords)     # (b, s, k)
                self._disp_sum += float(cdelta.abs().sum().item())
                self._disp_count += int(cdelta.numel())
                delta = delta + torch.einsum("bsk,dk->bsd", cdelta, U)
            h = (h + delta).to(hidden.dtype)
            return ((h,) + output[1:]) if is_tuple else h

    return _FrameClampModel()


def measure_clamp_displacement(model, tokenizer, terms: list[dict],
                               tasks: Sequence[dict], hs_site: int = HS_SITE,
                               max_new_tokens: int = 64) -> float:
    """Mean realized |Δ(Uᵀh)| per coordinate for a term set on a few
    calibration tasks — the matching variable for the energy-matched floor
    (its energy_scale = active-arm displacement / floor displacement, the
    frame-clamp analogue of E8's energy_matched_scale)."""
    probe = frame_clamp_model(model, tokenizer, terms, hs_site)
    for task in tasks:
        probe.generate(task["prompt"], max_new_tokens=max_new_tokens)
    disp = probe.mean_abs_displacement()
    return float(disp) if disp is not None else 0.0


def load_checkpoint(role: str, local_dir: Optional[Path] = None,
                    dtype_str: str = "float16"):
    """Load a sealed checkpoint (revision-pinned) + the BASE R1 tokenizer
    (C1: byte-identical input ids come from the base tokenizer alias on every
    matched-input arm). Returns (tokenizer, model, resolved) where ``resolved``
    records exactly what was loaded for provenance."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = CHECKPOINTS[role]
    src = str(local_dir) if local_dir else spec["hf_id"]
    kwargs = {}
    if not local_dir:
        kwargs["revision"] = spec["revision"]
    dtype = getattr(torch, dtype_str)
    try:
        model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype, **kwargs)
    except TypeError:  # transformers 4.x spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, **kwargs)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(CHECKPOINTS["base"]["hf_id"],
                                        revision=CHECKPOINTS["base"]["revision"])
    resolved = {"role": role, "source": src,
                "revision": None if local_dir else spec["revision"],
                "hf_id": spec["hf_id"], "local": bool(local_dir),
                "tokenizer": "base-alias:" + CHECKPOINTS["base"]["hf_id"],
                "dtype": dtype_str, "device": device}
    return tok, model, resolved


def identity_reload_gate(model, tokenizer, tasks: Sequence[dict],
                         hs_site: int = HS_SITE) -> dict:
    """Draft §9 gate 1 component: a zero-term frame-clamp generation must be
    byte-identical to the plain unsteered generation (hook identity), greedy.
    Run on 2 short tasks; returns pass/fail + the compared token counts."""
    from src.chain_gen import generate_chain
    probe = frame_clamp_model(model, tokenizer, [], hs_site)
    results = []
    for task in tasks[:2]:
        a = probe.generate(task["prompt"], max_new_tokens=128)
        b = generate_chain(model, tokenizer, task["prompt"], max_new_tokens=128)
        results.append({"task_id": task.get("id"),
                        "identical": a["chain"] == b["chain"],
                        "n_tokens": [a["n_tokens"], b["n_tokens"]]})
    return {"pass": all(r["identical"] for r in results), "cases": results}


# ── A2 adjunct (pure; PHASE2_ADJUNCT_AMENDMENT_2026-08-08.md) ────────────────

A2_BEHAVIOURS = ("backtracking", "uncertainty-estimation", "example-testing",
                 "adding-knowledge")


def a2_adjunct_table(annotated_vanilla: dict[str, list[dict]],
                     cap: int, n_resamples: int = B_BOOT,
                     seed: int = RUN_SEED) -> dict:
    """The frozen six-endpoint observational table over the three vanilla arms
    (estimation ONLY — paired per-task differences target−base with cluster-
    bootstrap CIs; no significance verdicts, no causal language).

    ``annotated_vanilla``: role -> list of annotated vanilla records (each with
    task_id/chain/n_tokens/annotations). Missing/empty annotation rows are
    unresolved and pairwise-deleted per endpoint, with counts reported.
    """
    from src.evaluation import behaviour_fraction
    from src.delta_floor import paired_bootstrap_mean

    def per_task(role):
        rows = {}
        for r in annotated_vanilla.get(role, []):
            tid = r["task_id"]
            anns = r.get("annotations") or None
            st = chain_stats(r["chain"], r["n_tokens"], cap,
                             r.get("expected_answer"))
            ep = {"length": float(r["n_tokens"]),
                  "looped": float(st["looped"]),
                  "truncated": float(st["truncated"]),
                  "boxed": (None if st["boxed_correct"] is None
                            else float(st["boxed_correct"]))}
            if anns:
                for b in A2_BEHAVIOURS:
                    ep[f"prev_{b}"] = behaviour_fraction(anns, b)
                n_bt = sum(1 for a in anns if a.get("label") == "backtracking")
                ep["bt_per_1k"] = (1000.0 * n_bt / r["n_tokens"]
                                   if r["n_tokens"] else None)
            else:
                for b in A2_BEHAVIOURS:
                    ep[f"prev_{b}"] = None
                ep["bt_per_1k"] = None
            rows[tid] = ep
        return rows

    base = per_task("base")
    table: dict = {"endpoints": {}, "unit": "task",
                   "contrast": "target minus base, paired by task",
                   "estimation_only": True}
    endpoints = ([f"prev_{b}" for b in A2_BEHAVIOURS]
                 + ["bt_per_1k", "boxed", "length", "looped", "truncated"])
    for role in TARGET_ROLES:
        tgt = per_task(role)
        cells = {}
        for i, ep in enumerate(endpoints):
            paired = [(tgt[t][ep], base[t][ep]) for t in sorted(set(tgt) & set(base))
                      if tgt[t].get(ep) is not None and base[t].get(ep) is not None]
            n_unresolved = len(set(tgt) & set(base)) - len(paired)
            if len(paired) < 2:
                cells[ep] = {"status": "insufficient", "n": len(paired),
                             "n_unresolved": n_unresolved}
                continue
            d = [a - b for a, b in paired]
            boot = paired_bootstrap_mean(d, n_resamples=n_resamples,
                                         seed=seed + i)
            cells[ep] = {"diff_mean": boot.estimate, "ci_low": boot.ci_low,
                         "ci_high": boot.ci_high, "n": boot.n_tasks,
                         "n_unresolved": n_unresolved}
        table["endpoints"][role] = cells
    return table


# ── gate arithmetic (pure) ───────────────────────────────────────────────────

def target_gate_verdict(transported: dict, null_metrics: Sequence[dict],
                        quantile: float = GATE_QUANTILE) -> dict:
    """Gate (§4): transported frame must exceed the pooled null's p95 on BOTH
    coord-AUC and state-dependence. ``null_metrics`` pools the 20 sham + 20
    random-orthogonal frames. Base absolute thresholds reported for continuity
    only, never adjudication."""
    aucs = np.array([m["coord_auc"] for m in null_metrics], dtype=float)
    sds = np.array([m["state_dependence"] for m in null_metrics], dtype=float)
    thr_auc = float(np.quantile(aucs, quantile))
    thr_sd = float(np.quantile(sds, quantile))
    pass_auc = transported["coord_auc"] > thr_auc
    pass_sd = transported["state_dependence"] > thr_sd
    return {
        "n_null": len(null_metrics), "quantile": quantile,
        "null_p95_coord_auc": thr_auc, "null_p95_state_dependence": thr_sd,
        "transported_coord_auc": transported["coord_auc"],
        "transported_state_dependence": transported["state_dependence"],
        "pass_coord_auc": bool(pass_auc), "pass_state_dependence": bool(pass_sd),
        "gate_pass": bool(pass_auc and pass_sd),
        "base_absolute_thresholds_continuity": dict(BASE_ABS_THRESHOLDS),
    }


def refit_alignment(U_base: np.ndarray, U_refit: np.ndarray,
                    null_frames: Sequence[np.ndarray]) -> dict:
    """Principal-angle alignment of the re-fit frame vs the transported frame,
    adjudicated against the random-orthogonal null (§4: alignment null = the
    random-orthogonal frame set). Alignment statistic = mean cos of principal
    angles (1 = same subspace). refit_aligned ⇔ statistic > null p95;
    misaligned_beyond_null ⇔ statistic < null p5 — the in-between is neither."""
    def mean_cos(A, B):
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        if A.ndim == 1:
            A = A[:, None]
        if B.ndim == 1:
            B = B[:, None]
        s = np.linalg.svd(A.T @ B, compute_uv=False)
        return float(np.mean(np.clip(s, 0, 1)))

    stat = mean_cos(U_base, U_refit)
    null = np.array([mean_cos(U_base, N) for N in null_frames], dtype=float)
    hi = float(np.quantile(null, 0.95))
    lo = float(np.quantile(null, 0.05))
    return {"alignment_mean_cos": stat, "null_p95": hi, "null_p5": lo,
            "n_null": len(null_frames),
            "refit_aligned": bool(stat > hi),
            "refit_misaligned_beyond_null": bool(stat < lo)}


__all__ = [
    "HS_SITE", "HS_NEIGHBOURS", "HS_ALL", "machinery_layer", "CHECKPOINTS",
    "TARGET_ROLES", "ALL_ROLES", "FRAME_PATH", "FRAME_WIDTH", "REFIT_WIDTHS",
    "CLAMP_GAIN", "E8_ALPHA_STAR", "RUN_SEED", "B_BOOT", "N_NULL_FRAMES",
    "GATE_QUANTILE", "INJECTION_FRACTIONS", "INJECTION_PASS_POWER",
    "ATTENUATION_MARGIN", "DAS_BUILD", "BEHAVIOUR", "battery_arms",
    "N_BATTERY_CELLS", "floor_for_battery_arm", "SECONDARY_FLOOR",
    "battery_cell_key", "enumerate_battery_cells", "enumerate_injection_cells",
    "injection_method_label", "random_orthonormal_frame", "class_mean_coords",
    "norm_gain", "whitening_stats", "whitened_frame", "clamp_targets_for_arm",
    "mixing_weights", "boxed_answer", "chain_stats", "damage_gate",
    "attenuation_fraction", "paired_attenuation_test", "raw_retained",
    "injection_detection_power", "ANNOTATION_DEDUP_KEYS",
    "proxy_chunk_budget_ok", "battery_record", "frame_clamp_model",
    "measure_clamp_displacement", "load_checkpoint", "identity_reload_gate",
    "a2_adjunct_table", "target_gate_verdict", "refit_alignment",
]
