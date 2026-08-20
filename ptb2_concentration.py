#!/usr/bin/env python3
"""PT-B2 — does post-training move the behaviour-concentration map?

Sealed authority: results/prereg/PTB2_POSTTRAINING_CONCENTRATION_PREREG_2026-08-20.md

Estimand: fixed-top-ten variance concentration (the RQ1 statistic) per
(checkpoint, layer, behaviour). Primary = paired safety-minus-control
difference, within-session per seed, chain-bootstrapped. Secondary = the
per-checkpoint concentration map against a chain-stratified permutation null.

$0: local CPU on already-extracted activations. Deterministic given the seeds.

    python ptb2_concentration.py validate     # estimator agreement only
    python ptb2_concentration.py run          # full analysis
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import subprocess
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

# One BLAS thread per worker: with 8 processes each spawning a full
# thread pool the machine thrashes and runs SLOWER than serial.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "ptb2"
PREREG = ROOT / "results/prereg/PTB2_POSTTRAINING_CONCENTRATION_PREREG_2026-08-20.md"

BEHAVIOURS = ("backtracking", "uncertainty-estimation", "example-testing",
              "adding-knowledge")
LAYERS = (12, 16)
SEEDS = ("42", "43", "44")
TOP_K = 10
VALIDATION_TOL = 1e-6
B_BOOT = 2000
N_PERM_MAP = 1000
PTB2_SEED = 20260820
MAP_SEED = 42

BASE_DIR = "data/activations/R1-1.5B"


def arm_dir(recipe: str, seed: str) -> str:
    suffix = "" if seed == "42" else f"-s{seed}"
    return f"data/activations/R1-1.5B-lora-{recipe}1000{suffix}"


CHECKPOINTS = {"base": BASE_DIR}
for _r, _s in product(("safety", "control"), SEEDS):
    CHECKPOINTS[f"{_r}-s{_s}"] = arm_dir(_r, _s)


# ── estimator ────────────────────────────────────────────────────────────────

def top_k_ratio_fast(X: np.ndarray, k: int = TOP_K) -> float:
    """sum(top-k eigenvalues of the centred covariance) / trace(covariance).

    Identical estimand to src.nulls.top_k_variance_ratio; validated against it
    to <1e-6 on every cell before any result is produced (stage_validate).
    Only the top-k eigenvalues are requested (the trace supplies the
    denominator exactly), which is what makes the permutation nulls and the
    bootstrap affordable.
    """
    from scipy.linalg import eigh
    if X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    C = Xc.T @ Xc
    total = float(np.trace(C))
    if total <= 0:
        return float("nan")
    d = C.shape[0]
    kk = min(k, d)
    ev = eigh(C, eigvals_only=True, subset_by_index=[d - kk, d - 1])
    return float(ev.sum() / total)


# ── data access ──────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_chain_ids(act_dir: Path) -> dict:
    """behaviour -> np.array of chain_id per stored row (stored order)."""
    ri = json.loads((act_dir / "row_index.json").read_text())
    return {b: np.array([r["chain_id"] for r in rows])
            for b, rows in ri["rows"].items() if b in BEHAVIOURS}


def row_signature(act_dir: Path) -> str:
    ri = json.loads((act_dir / "row_index.json").read_text())
    payload = {b: [(r["chain_id"], r["annotation_index"]) for r in rows]
               for b, rows in sorted(ri["rows"].items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_cell(act_dir: Path, behaviour: str, layer: int) -> np.ndarray:
    return np.load(act_dir / f"{behaviour}_layer{layer}.npy").astype(np.float32)


# ── stages ───────────────────────────────────────────────────────────────────

def stage_validate(verbose: bool = True) -> dict:
    """Estimator agreement gate: fast path vs the reference implementation."""
    from src.nulls import top_k_variance_ratio
    deltas, worst = {}, 0.0
    for name, d in CHECKPOINTS.items():
        for layer, beh in product(LAYERS, BEHAVIOURS):
            X = load_cell(ROOT / d, beh, layer)
            ref = top_k_variance_ratio(X, k=TOP_K)
            fast = top_k_ratio_fast(X)
            delta = abs(ref - fast)
            worst = max(worst, delta)
            deltas[f"{name}|L{layer}|{beh}"] = {
                "reference": ref, "fast": fast, "abs_delta": delta}
            if delta > VALIDATION_TOL:
                raise SystemExit(
                    f"estimator validation FAILED for {name} L{layer} {beh}: "
                    f"|{ref} - {fast}| = {delta} > {VALIDATION_TOL}")
        if verbose:
            print(f"  validated {name}", flush=True)
    if verbose:
        print(f"estimator validation PASS on {len(deltas)} cells "
              f"(worst |delta| = {worst:.3e} < {VALIDATION_TOL})")
    return {"n_cells": len(deltas), "worst_abs_delta": worst,
            "tolerance": VALIDATION_TOL, "per_cell": deltas}


def chain_bootstrap_indices(chain_ids: np.ndarray, rng) -> np.ndarray:
    """Resample CHAINS with replacement; return the concatenated row indices."""
    uniq = np.unique(chain_ids)
    by_chain = {c: np.where(chain_ids == c)[0] for c in uniq}
    picked = rng.choice(uniq, size=uniq.size, replace=True)
    return np.concatenate([by_chain[c] for c in picked])


def bca_interval(theta: float, boot: np.ndarray, jack: np.ndarray,
                 ci: float = 0.95):
    from scipy.stats import norm
    if np.allclose(boot, boot[0]) or np.allclose(jack, jack[0]):
        a = (1 - ci) / 2
        return float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))
    z0 = norm.ppf(np.clip((boot < theta).mean(), 1e-9, 1 - 1e-9))
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    acc = 0.0 if den == 0 else num / den
    out = []
    for q in ((1 - ci) / 2, 1 - (1 - ci) / 2):
        z = norm.ppf(q)
        adj = z0 + (z0 + z) / (1 - acc * (z0 + z))
        out.append(float(np.quantile(boot, np.clip(norm.cdf(adj), 0, 1))))
    return out[0], out[1]


def seed_permutation_p(diffs: list) -> dict:
    """Exact sign-flip permutation over the three paired seed differences.

    A 3-vs-3 paired design has 2^3 = 8 sign assignments, so the two-sided
    floor is 2/8 = 0.25. Recorded as a stated power ceiling, not a result.
    """
    d = np.asarray(diffs, dtype=float)
    obs = d.mean()
    means = [np.mean(d * np.array(s)) for s in product((1, -1), repeat=len(d))]
    means = np.asarray(means)
    p = float((np.abs(means) >= abs(obs) - 1e-15).mean())
    return {"observed_mean": float(obs), "p_two_sided": p,
            "floor": 2.0 / len(means),
            "note": "exact sign-flip permutation; 3v3 paired floor is 0.25"}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


PARTS = OUT / "_parts"


def _part_path(kind: str, *key) -> Path:
    return PARTS / f"{kind}__{'__'.join(str(k) for k in key)}.json"


def _bootstrap_cell(task: tuple) -> tuple:
    """Worker: one primary cell (layer, behaviour). Module-level for pickling.

    Determinism: the RNG is constructed fresh from PTB2_SEED inside this
    function, so a cell's result is independent of which process runs it and
    of execution order. Parallel and serial execution agree byte-for-byte.
    """
    layer, beh = task
    part = _part_path("primary", layer, beh)
    if part.exists():
        return (layer, beh, json.loads(part.read_text()))

    ids = load_chain_ids(ROOT / BASE_DIR)[beh]
    arms = {}
    for recipe in ("safety", "control"):
        for s in SEEDS:
            arms[(recipe, s)] = load_cell(ROOT / CHECKPOINTS[f"{recipe}-s{s}"],
                                          beh, layer)

    def delta(idx=None):
        out = []
        for s in SEEDS:
            a = arms[("safety", s)]
            b = arms[("control", s)]
            if idx is not None:
                a, b = a[idx], b[idx]
            out.append(top_k_ratio_fast(a) - top_k_ratio_fast(b))
        return out

    per_seed = delta()
    theta = float(np.mean(per_seed))

    rng = np.random.default_rng(PTB2_SEED)
    boot = np.empty(B_BOOT, dtype=float)
    for i in range(B_BOOT):
        boot[i] = float(np.mean(delta(chain_bootstrap_indices(ids, rng))))

    uniq = np.unique(ids)
    by_chain = {c: np.where(ids == c)[0] for c in uniq}
    jack = np.empty(uniq.size, dtype=float)
    for j, c in enumerate(uniq):
        idx = np.concatenate([by_chain[x] for x in uniq if x != c])
        jack[j] = float(np.mean(delta(idx)))

    lo, hi = bca_interval(theta, boot, jack)
    p = float(min(1.0, 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())))
    p = max(p, 1.0 / (B_BOOT + 1))
    res = {
        "per_seed_differences": {s: per_seed[i] for i, s in enumerate(SEEDS)},
        "mean_difference": theta,
        "bootstrap": {"ci_low": lo, "ci_high": hi, "B": B_BOOT, "raw_p": p,
                      "kind": "paired chain BCa", "n_chains": int(uniq.size)},
        "seed_permutation": seed_permutation_p(per_seed),
    }
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(json.dumps(res, indent=1, sort_keys=True))
    return (layer, beh, res)


def _map_task(task: tuple) -> tuple:
    """Worker: the concentration map for one (checkpoint, layer)."""
    name, layer = task
    part = _part_path("map", name, layer)
    if part.exists():
        return (name, layer, json.loads(part.read_text()))

    chain_ids = load_chain_ids(ROOT / BASE_DIR)
    concat_ids = np.concatenate([chain_ids[b] for b in BEHAVIOURS])
    labels = np.concatenate([np.full(len(chain_ids[b]), b) for b in BEHAVIOURS])
    chain_to_idx = {c: np.where(concat_ids == c)[0]
                    for c in np.unique(concat_ids)}
    n_mixed = sum(1 for i in chain_to_idx.values()
                  if np.unique(labels[i]).size > 1)
    if n_mixed == 0:
        raise SystemExit("within-chain permutation is a NO-OP — chain-id "
                         "provenance is wrong; refusing a vacuous null")

    d = ROOT / CHECKPOINTS[name]
    X_all = np.concatenate([load_cell(d, b, layer) for b in BEHAVIOURS])
    real = {b: top_k_ratio_fast(X_all[labels == b]) for b in BEHAVIOURS}

    rng = np.random.default_rng(MAP_SEED)
    perm = labels.copy()
    draws = {b: np.empty(N_PERM_MAP) for b in BEHAVIOURS}
    for r in range(N_PERM_MAP):
        for idxs in chain_to_idx.values():
            perm[idxs] = rng.permutation(labels[idxs])
        for b in BEHAVIOURS:
            draws[b][r] = top_k_ratio_fast(X_all[perm == b])

    res = {}
    for b in BEHAVIOURS:
        nd = draws[b]
        res[f"L{layer}|{b}"] = {
            "real_value": real[b], "null_mean": float(nd.mean()),
            "null_p97_5": float(np.quantile(nd, 0.975)),
            "specificity_margin": real[b] - float(nd.mean()),
            "p_value": float((np.sum(nd >= real[b]) + 1) / (N_PERM_MAP + 1)),
            "n_resamples": N_PERM_MAP, "n_mixed_label_chains": n_mixed}
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(json.dumps(res, indent=1, sort_keys=True))
    return (name, layer, res)


def _pool(n_workers: int):
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    return ctx.Pool(n_workers)


def stage_run(n_workers: int = 8) -> None:
    validation = stage_validate(verbose=True)

    # Row-signature identity across checkpoints (paired-design precondition).
    sigs = {n: row_signature(ROOT / d) for n, d in CHECKPOINTS.items()}
    if len(set(sigs.values())) != 1:
        raise SystemExit(f"row indices differ across checkpoints: {sigs}")
    chain_ids = load_chain_ids(ROOT / BASE_DIR)

    analysis: dict = {
        "prereg": str(PREREG.relative_to(ROOT)),
        "estimand": "fixed-top-ten variance concentration (RQ1 statistic); "
                    "NOT correlation dimension, participation ratio, PCA "
                    "variance-threshold dimension, subspace, or manifold",
        "layers": list(LAYERS),
        "layer_note": "L12/L16 are the only adapter-extracted depths and do "
                      "NOT overlap RQ1's {11,14,17,20,27}; no PT-B2 cell is "
                      "numerically comparable to an RQ1 cell",
        "environment_rule": "primary safety-minus-control is WITHIN-session per "
                            "seed; base-referenced values are cross-session and "
                            "descriptive only (see PTB1_AMENDMENT_3)",
        "row_signature": next(iter(sigs.values())),
        "estimator_validation": {k: v for k, v in validation.items()
                                 if k != "per_cell"},
    }

    # ── point statistics: 7 checkpoints x 2 layers x 4 behaviours ───────────
    print("computing point statistics ...", flush=True)
    stats: dict = {}
    for name, d in CHECKPOINTS.items():
        for layer, beh in product(LAYERS, BEHAVIOURS):
            X = load_cell(ROOT / d, beh, layer)
            stats.setdefault(name, {})[f"L{layer}|{beh}"] = top_k_ratio_fast(X)
            del X
    analysis["point_statistics"] = stats

    # ── PRIMARY: paired safety - control, per (behaviour, layer) ────────────
    print(f"primary contrast: chain bootstrap ({n_workers} workers) ...",
          flush=True)
    primary: dict = {}
    raw_p: dict = {}
    tasks = list(product(LAYERS, BEHAVIOURS))
    with _pool(min(n_workers, len(tasks))) as pool:
        for layer, beh, res in pool.imap_unordered(_bootstrap_cell, tasks):
            key = f"L{layer}|{beh}"
            primary[key] = res
            raw_p[key] = res["bootstrap"]["raw_p"]
            b = res["bootstrap"]
            print(f"  {key}: delta={res['mean_difference']:+.5f} "
                  f"[{b['ci_low']:+.5f},{b['ci_high']:+.5f}] "
                  f"p={b['raw_p']:.4f}", flush=True)
    primary = {f"L{l}|{b}": primary[f"L{l}|{b}"]
               for l, b in product(LAYERS, BEHAVIOURS)}
    adj = holm(raw_p)
    for key in primary:
        primary[key]["holm_adjusted_p"] = adj[key]
        sig = adj[key] < 0.05
        primary[key]["verdict"] = (
            "resolved nonzero (Holm, chain bootstrap governs)" if sig
            else "not resolved — point estimate only")
    analysis["primary_safety_minus_control"] = primary

    # ── SECONDARY: the concentration map (chain-stratified null) ────────────
    print(f"secondary: concentration map ({n_workers} workers) ...", flush=True)
    cmap: dict = {}
    map_tasks = [(name, layer) for name in CHECKPOINTS for layer in LAYERS]
    with _pool(min(n_workers, len(map_tasks))) as pool:
        for name, layer, res in pool.imap_unordered(_map_task, map_tasks):
            cmap.setdefault(name, {}).update(res)
            print(f"  map {name} L{layer} done", flush=True)
    analysis["concentration_map"] = cmap
    analysis["map_note"] = ("descriptive map; no Holm claim attaches; smallest "
                            f"attainable p is 1/{N_PERM_MAP + 1}")

    # map agreement: does the pass/fail pattern differ by recipe class?
    agree: dict = {}
    for layer, beh in product(LAYERS, BEHAVIOURS):
        key = f"L{layer}|{beh}"
        agree[key] = {
            "base": cmap["base"][key]["p_value"] < 0.05,
            "safety_n_pass": sum(cmap[f"safety-s{s}"][key]["p_value"] < 0.05
                                 for s in SEEDS),
            "control_n_pass": sum(cmap[f"control-s{s}"][key]["p_value"] < 0.05
                                  for s in SEEDS)}
    analysis["map_pass_agreement"] = agree

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, capture_output=True, text=True).stdout.strip())
    inputs = {}
    for name, d in CHECKPOINTS.items():
        inputs[f"{d}/row_index.json"] = sha256(ROOT / d / "row_index.json")
        for layer, beh in product(LAYERS, BEHAVIOURS):
            rel = f"{d}/{beh}_layer{layer}.npy"
            inputs[rel] = sha256(ROOT / rel)
    prov = {
        "schema_version": "rom-result-provenance-v1",
        "stage": "ptb2-concentration",
        "prereg": {"path": str(PREREG.relative_to(ROOT)), "sha256": sha256(PREREG)},
        "git_commit": head, "git_dirty": dirty,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": {"bootstrap": PTB2_SEED, "map": MAP_SEED},
        "execution": {"parallel_workers": n_workers,
                      "determinism": "each cell/map task builds its own RNG from the fixed seed inside the worker, so results are independent of worker count and scheduling order; parallel and serial execution agree byte-for-byte (verified against the serial run for L12|backtracking and L12|uncertainty-estimation)"},
        "estimator_validation": validation,
        "inputs_sha256": inputs,
    }
    (OUT / "provenance").mkdir(parents=True, exist_ok=True)
    (OUT / "provenance" / "ptb2.json").write_text(json.dumps(prov, indent=1,
                                                             sort_keys=True))
    (OUT / "ptb2_analysis.json").write_text(json.dumps(analysis, indent=1,
                                                       sort_keys=True))
    write_report(analysis)
    print("\nPT-B2 complete.")


def write_report(a: dict) -> None:
    L = ["# PT-B2 — post-training and the behaviour-concentration map", "",
         f"Sealed authority: `{a['prereg']}`.", "",
         f"Estimand: {a['estimand']}.", "",
         f"**Layers.** {a['layer_note']}.", "",
         f"**Environment rule.** {a['environment_rule']}.", "",
         "## Primary: safety − control (paired within seed, Holm over 8)", "",
         "| Cell | Δ | 95% BCa | boot p | Holm | seed-perm p | Verdict |",
         "|---|---:|---|---:|---:|---:|---|"]
    for key, c in a["primary_safety_minus_control"].items():
        b = c["bootstrap"]
        L.append(f"| {key} | {c['mean_difference']:+.5f} | "
                 f"[{b['ci_low']:+.5f}, {b['ci_high']:+.5f}] | {b['raw_p']:.4f} | "
                 f"{c['holm_adjusted_p']:.4f} | "
                 f"{c['seed_permutation']['p_two_sided']:.2f} | {c['verdict']} |")
    L += ["", "Seed-permutation p has a hard floor of 0.25 in a 3-vs-3 paired "
          "design; the chain bootstrap governs every claim.", "",
          "## Secondary: concentration map (descriptive)", "",
          "Cells passing the chain-stratified null at p<.05, by recipe class "
          "(safety/control counts are out of 3 seeds):", "",
          "| Cell | base | safety | control |", "|---|---|---:|---:|"]
    for key, v in a["map_pass_agreement"].items():
        L.append(f"| {key} | {'pass' if v['base'] else 'fail'} | "
                 f"{v['safety_n_pass']}/3 | {v['control_n_pass']}/3 |")
    L += ["", f"{a['map_note']}.", "",
          "Point statistics, per-seed differences, specificity margins and "
          "full provenance: `ptb2_analysis.json`.", ""]
    (OUT / "PTB2_REPORT.md").write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("validate", "run"))
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel worker processes (results are identical for any value)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "validate":
        stage_validate()
    else:
        stage_run(n_workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
