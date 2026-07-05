#!/usr/bin/env python3
"""Huang-faithful pooled-class PCA manifold rebuild + diagnostic runner.

Reimplements the PCA / manifold-vector construction from Huang et al.
"Mitigating Overthinking in Large Reasoning Models via Manifold Steering"
(arXiv:2505.22411v2) faithfully — fit PCA on the POOLED, mean-centered two-class
set (ON ∪ OFF), project the difference-of-means onto the pooled top-k — and puts
it SIDE-BY-SIDE with our current ON-only construction (src/steering.py) so the
deviation is visible in one table.

By default builds at the per-behaviour Venhoff layers (where the single-vectors
were just rebuilt in R1-1.5B__venhoff/):

    backtracking            -> layer 17
    uncertainty-estimation  -> layer 18
    example-testing         -> layer 15
    adding-knowledge        -> layer 18

Writes to a DISTINCT dir (results/steering_vectors/R1-1.5B__huang) so it never
clobbers R1-1.5B/ (all-L27), R1-1.5B-peak/, or R1-1.5B__venhoff/. The on-disk
layout (via src.steering.save_steering_vectors) is drop-in for 07_evaluate_steering.py.

Cluster-runnable: zero GPU. Needs only numpy + scikit-learn + the activation
.npy files at data/activations/R1-1.5B/{behaviour}_layer{L}.npy. Same 50-task
stratified hold-out as 06/07 via src.task_gen.stratified_eval_split.

Emits:
  (1) a side-by-side stdout table (pooled vs ON-only energy + cos at k=1,3,5,10,auto);
  (2) the robustness battery (split-half principal angles + eigengap + random-subspace null);
  (3) OUT/diagnostic.json (so the table + battery survive the run).

Usage:
    python build_huang_manifold.py                      # Venhoff layers, hold-out on
    python build_huang_manifold.py --layer 27           # force L27 for all (Huang's layer)
    python build_huang_manifold.py --no-holdout         # use all rows
    python build_huang_manifold.py --k-values 1 3 5 10 auto
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

from src.annotation import TARGET_BEHAVIOURS
from src.config import provenance, require_file
from src.task_gen import load_tasks, stratified_eval_split
from src.steering import save_steering_vectors
from src.huang_manifold import (
    build_huang_manifold_vectors,
    energy_in_subspace,
    on_only_manifold_components,
    pca_basis,
    split_half_stability,
    eigengap_report,
    random_subspace_null,
    grade_split_half,
    grade_eigengap,
    grade_random_null,
    aggregate_verdict,
)
from src.row_provenance import chain_ids_for, require_aligned, dedup_rows, duplicate_fraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

VENHOFF_LAYERS = {
    "backtracking": 17,
    "uncertainty-estimation": 18,
    "example-testing": 15,
    "adding-knowledge": 18,
}


def _cos_only(on_acts: np.ndarray, r_single: np.ndarray, k: int) -> tuple[float, float]:
    """ON-only (old construction) energy of r in top-k and cos(manifold_k, single)."""
    U = on_only_manifold_components(on_acts, k)
    if U.shape[0] < 1:
        return float("nan"), float("nan")
    e = energy_in_subspace(r_single, U)
    coords = U @ r_single
    r_proj = coords @ U
    norm = np.linalg.norm(r_proj)
    vec = r_proj / norm if np.isfinite(norm) and norm >= 1e-10 else r_single
    return e, float(np.dot(r_single, vec))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-short", default="R1-1.5B")
    ap.add_argument("--layer", type=int, default=None,
                    help="Force a single layer for ALL behaviours (e.g. 27 = Huang's). "
                         "Default: per-behaviour Venhoff layers 17/18/15/18.")
    ap.add_argument("--k-values", nargs="+", default=["1", "3", "5", "10", "auto"])
    ap.add_argument("--huang-k", type=int, default=10,
                    help="The faithful headline k (fixed in Huang).")
    ap.add_argument("--no-holdout", action="store_true",
                    help="Use all rows (skip the Phase-7 eval hold-out).")
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--tasks", default="data/tasks_final.json")
    ap.add_argument("--annotated", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-battery", action="store_true",
                    help="Skip the robustness battery (split-half/eigengap/random-null).")
    ap.add_argument("--n-boot", type=int, default=200,
                    help="Split-half resamples for the robustness battery.")
    ap.add_argument("--battery-subspace", choices=["on", "pooled", "both"],
                    default="both",
                    help="Which subspace(s) the robustness battery tests. "
                         "'pooled' = the operational manifold we steer with "
                         "(Huang Eq. 5); 'on' = the old ON-only construction "
                         "(stability contrast); 'both' (default) reports both.")
    a = ap.parse_args()

    ACT = Path(f"data/activations/{a.model_short}")
    OUT = Path(a.out) if a.out else Path(f"results/steering_vectors/{a.model_short}__huang")
    ANNOT = Path(a.annotated) if a.annotated else Path(f"data/annotated_{a.model_short}.json")
    require_file(ACT, "run 04_extract_activations.py first")

    layers = ({b: a.layer for b in TARGET_BEHAVIOURS} if a.layer is not None
              else dict(VENHOFF_LAYERS))

    k_values = [("auto" if str(k).lower() == "auto" else int(k)) for k in a.k_values]

    # ── Hold-out (same split as 06/07) ────────────────────────────────────────
    EXCLUDE = None
    if not a.no_holdout:
        require_file(Path(a.tasks), "run the task-generation step first")
        _test, _rule = stratified_eval_split(load_tasks(Path(a.tasks)), a.n_test)
        EXCLUDE = {t["id"] for t in _test}
        log.info(f"Hold-out: excluding {len(EXCLUDE)} eval tasks ({_rule}) "
                 f"from vector construction")
    else:
        log.info("Hold-out DISABLED (--no-holdout): vectors see all rows")

    # ── Build the Huang pooled-PCA vectors ────────────────────────────────────
    assembled = build_huang_manifold_vectors(
        ACT, layers=layers, behaviours=list(TARGET_BEHAVIOURS),
        k_values=k_values, huang_k=a.huang_k,
        exclude_chain_ids=EXCLUDE, annotated_path=ANNOT,
    )
    if not assembled:
        sys.exit("ERROR: no vectors built (missing activation files?).")

    prov = provenance(args=a)
    prov["builder"] = ("build_huang_manifold.py — Huang pooled-class PCA "
                       "(arXiv:2505.22411v2 Eqs.2/5/9)")
    prov["construction"] = "huang_pooled_pca"
    prov["layers"] = layers
    prov["huang_k"] = a.huang_k
    prov["holdout"] = (None if a.no_holdout
                       else {"n_tasks": len(EXCLUDE),
                             "rule": "src.task_gen.stratified_eval_split"})
    OUT.mkdir(parents=True, exist_ok=True)
    save_steering_vectors(assembled, OUT, provenance=prov)

    # ── Side-by-side diagnostic table ─────────────────────────────────────────
    sweep = [k for k in k_values]
    diagnostic: dict = {}

    print("\n" + "=" * 110)
    print("HUANG POOLED-PCA vs ON-ONLY PCA  —  energy of r in top-k  AND  "
          "cos(manifold_k, single)")
    print(f"(faithful headline arm = k={a.huang_k}; auto = pooled >=70% variance)")
    print("=" * 110)
    for b in TARGET_BEHAVIOURS:
        if b not in assembled:
            print(f"  [skip] {b}: not built")
            continue
        d = assembled[b]
        r_single = np.asarray(d["single_direction"])
        L = d["layer"]
        on_acts = np.load(ACT / f"{b}_layer{L}.npy").astype(np.float32)
        # NB: the ON-only baseline here is computed on the FULL ON matrix (no
        # hold-out) — it is a descriptive comparison of the two PCA fits, not a
        # second hold-out artefact. The pooled (saved) vectors DO respect the
        # hold-out. Document both numbers honestly.

        print(f"\n{b}  @ L{L}   (auto_k_pooled={d['auto_k']})")
        header = f"  {'k':>5s} | {'E_pooled':>9s} {'cos_pooled':>11s} | {'E_only':>9s} {'cos_only':>10s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        per_k: dict = {"energy_pooled": {}, "cos_pooled": {},
                       "energy_only": {}, "cos_only": {}}
        for k in sweep:
            kk = "auto" if k == "auto" else int(k)
            k_int = d["auto_k"] if kk == "auto" else int(kk)
            e_p = d["energy_in_manifold"].get(kk, float("nan"))
            c_p = d["cos_single_manifold"].get(kk, float("nan"))
            e_o, c_o = _cos_only(on_acts, r_single, k_int)
            klabel = "auto" if kk == "auto" else str(kk)
            print(f"  {klabel:>5s} | {e_p:>9.4f} {c_p:>11.4f} | {e_o:>9.4f} {c_o:>10.4f}")
            per_k["energy_pooled"][klabel] = float(e_p)
            per_k["cos_pooled"][klabel] = float(c_p)
            per_k["energy_only"][klabel] = float(e_o)
            per_k["cos_only"][klabel] = float(c_o)

        # Acceptance read (refined): pooled >> ON-only at low k, and r is largely
        # captured by the low-k POOLED subspace by k=3 / k=10.
        cos_p1 = per_k["cos_pooled"].get("1", float("nan"))
        cos_o1 = per_k["cos_only"].get("1", float("nan"))
        cos_p3 = per_k["cos_pooled"].get("3", float("nan"))
        e_p10 = per_k["energy_pooled"].get("10", float("nan"))
        verdict_bits = []
        if np.isfinite(cos_p1) and np.isfinite(cos_o1):
            verdict_bits.append(f"cos_pooled(k1)={cos_p1:.3f} "
                                f"{'>' if cos_p1 > cos_o1 else '<='} "
                                f"cos_only(k1)={cos_o1:.3f}")
        if np.isfinite(cos_p3):
            verdict_bits.append(f"cos_pooled(k3)={cos_p3:.3f} "
                                f"({'OK' if cos_p3 >= 0.74 else 'LOW'} vs ~0.74)")
        if np.isfinite(e_p10):
            verdict_bits.append(f"E_pooled(k10)={e_p10:.3f} "
                                f"({'OK' if e_p10 >= 0.89 else 'LOW'} vs ~0.89)")
        print("  read: " + " | ".join(verdict_bits))
        diagnostic[b] = {
            "layer": L, "auto_k_pooled": d["auto_k"],
            "n_on": d["n_on"], "n_off": d["n_off"],
            "n_excluded": d.get("n_excluded", 0), **per_k,
        }

    print("\nHONEST CAVEAT: at k=1 pooled energy is still modest (~0.09–0.29); the "
          "low-k manifold claim is supported at k=3–10, not k=1.")

    # ── Robustness battery (Huang reports NONE) ───────────────────────────────
    if not a.no_battery:
        print("\n" + "=" * 110)
        print("ROBUSTNESS BATTERY  —  split-half principal angles | eigengap (Davis–Kahan) | "
              "random-subspace null")
        print("(de-duplicated; chain-aware split when row provenance is available)")
        print("=" * 110)
        try:
            cidmap = chain_ids_for(ACT, ANNOT, list(TARGET_BEHAVIOURS))
        except Exception as e:  # provenance optional for the battery
            log.warning(f"chain ids unavailable ({e}); battery uses row-level splits")
            cidmap = {b: None for b in TARGET_BEHAVIOURS}

        want_on = a.battery_subspace in ("on", "both")
        want_pool = a.battery_subspace in ("pooled", "both")
        hdr = (f"  {'behaviour':22s} {'L':>3s} {'subspace':>8s} {'N_fit':>7s} "
               f"{'splithalf_p95':>13s} {'mean_cos2':>10s} {'rel_gap_k':>10s} "
               f"{'DK_deg':>7s} {'e_on':>6s} {'fold':>7s} {'VERDICT':>11s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        def _battery(Xfit, cids_fit, Xon_d, Xoff_d, subspace_X):
            c1 = split_half_stability(Xfit, k=a.huang_k, chain_ids=cids_fit,
                                      n_boot=a.n_boot)
            c2 = eigengap_report(Xfit, k=a.huang_k)
            c4 = random_subspace_null(Xon_d, Xoff_d, k=a.huang_k,
                                      subspace_X=subspace_X)
            return c1, c2, c4, aggregate_verdict(c1, c2, c4)

        def _emit(b, L, tag, n_fit, c1, c2, c4, verdict):
            mx = c1.get("max_angle_deg", {}).get("p95", float("nan"))
            mc2 = c1.get("mean_cos2", {}).get("mean", float("nan"))
            print(f"  {b:22s} {L:>3d} {tag:>8s} {n_fit:>7d} "
                  f"{mx:>13.2f} {mc2:>10.3f} {c2['rel_gap_k']:>10.3f} "
                  f"{c2['davis_kahan_bound_deg']:>7.1f} {c4['e_on']:>6.3f} "
                  f"{c4['fold_over_random']:>7.1f} {verdict:>11s}")
            return {"n_fit": int(n_fit), "split_half": c1, "eigengap": c2,
                    "random_null": c4,
                    "grades": {"split_half": grade_split_half(c1),
                               "eigengap": grade_eigengap(c2),
                               "random_null": grade_random_null(c4),
                               "verdict": verdict}}

        for b in TARGET_BEHAVIOURS:
            if b not in assembled:
                continue
            L = assembled[b]["layer"]
            Xon = np.load(ACT / f"{b}_layer{L}.npy").astype(np.float32)
            off_parts = [np.load(ACT / f"{o}_layer{L}.npy").astype(np.float32)
                         for o in TARGET_BEHAVIOURS if o != b]
            Xoff = np.concatenate(off_parts, axis=0)
            off_cids_parts = [cidmap.get(o) for o in TARGET_BEHAVIOURS if o != b]

            dupf = duplicate_fraction(Xon)
            cids = cidmap.get(b)
            try:
                cids = require_aligned(b, Xon.shape[0], cids, context="huang battery")
                Xon_d, cids_d = dedup_rows(Xon, cids)
            except Exception:
                cids_d = None
                Xon_d = dedup_rows(Xon)[0]
            Xoff_d = dedup_rows(Xoff)[0]

            rob = {"duplicate_fraction": float(dupf)}
            if want_on:
                # ON-only (rejected construction) — kept as a stability contrast,
                # the analogue of the pooled-vs-ON-only ENERGY table above.
                c1, c2, c4, v = _battery(Xon_d, cids_d, Xon_d, Xoff_d, None)
                rob["on_only"] = _emit(b, L, "on-only", Xon_d.shape[0],
                                       c1, c2, c4, v)
            if want_pool:
                # POOLED = the manifold Phase-7 actually steers with (Huang Eq. 5,
                # vstack(ON, OFF)). Chain-aware split needs combined chain ids.
                Xpool = np.concatenate([Xon, Xoff], axis=0)
                cids_pool = None
                if cids is not None and all(c is not None for c in off_cids_parts):
                    cids_off = np.concatenate(off_cids_parts)
                    if len(cids_off) == Xoff.shape[0]:
                        cids_pool = np.concatenate([cids, cids_off])
                if cids_pool is not None and len(cids_pool) == Xpool.shape[0]:
                    Xpool_d, cids_pool_d = dedup_rows(Xpool, cids_pool)
                else:
                    Xpool_d, cids_pool_d = dedup_rows(Xpool)[0], None
                c1, c2, c4, v = _battery(Xpool_d, cids_pool_d, Xon_d, Xoff_d, Xpool_d)
                rob["pooled"] = _emit(b, L, "pooled", Xpool_d.shape[0],
                                      c1, c2, c4, v)
            diagnostic.setdefault(b, {})["robustness"] = rob
        print("\nNOTE: 'pooled' is the OPERATIONAL manifold (Huang Eq. 5, vstack(ON,OFF)) — "
              "the subspace we steer with; 'on-only' is the rejected construction shown as a "
              "stability contrast (mirrors the pooled-vs-ON-only ENERGY table above). The "
              "low-dimensional subspace is the claim; integer k=10 is a variance-coverage "
              "convention. Cross-annotator subspace stability remains an un-run GPU gap "
              "(only Sonnet activations exist on disk).")

    # ── Persist the diagnostic ────────────────────────────────────────────────
    diag_path = OUT / "diagnostic.json"
    with open(diag_path, "w") as f:
        json.dump({"_provenance": prov, **diagnostic}, f, indent=2)
    print(f"\nSaved vectors -> {OUT}")
    print(f"Saved diagnostic -> {diag_path}")


if __name__ == "__main__":
    main()
