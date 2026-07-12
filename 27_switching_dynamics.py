#!/usr/bin/env python3
"""27_switching_dynamics.py — A2: switching linear dynamics over reasoning steps (PG §12).

Fits a switching linear dynamical system to the pooled reasoning-step trajectories to
ask whether reasoning decomposes into discrete DYNAMICAL modes, and — the load-bearing
test per the pressure-test reframe (PG §12.1) — whether those modes recover the human
behaviour annotations WITHOUT labels (an annotator-circularity de-confound), rather than
merely rediscovering unlabeled regimes (which Carson & Reisizadeh 2506.04374 already did
with a plain SLDS).

Method (dependency-light; the `ssm`/Linderman variational rSLDS is the confirmatory
upgrade if A2 proves worth it — "decide later"):
  * PCA-reduce the pooled step states to d' dims (matches 2506.04374's rank-40 protocol;
    A_k in 1536-d would be hopelessly over-parameterised).
  * Hard-EM switching LINEAR dynamics: K modes, each a linear map A_k z_t + b_k → z_{t+1}
    (ridge). E-step assigns each transition to the mode with least prediction error;
    M-step refits A_k. This is a switching (not yet state-recurrent) LDS.

Reads (all local CPU, no torch/ssm):
  1. mode ↔ behaviour alignment: adjusted mutual information vs a shuffled-label null.
  2. ORDER control: refit on step-SHUFFLED trajectories — if modes need real temporal
     order (dynamics), AMI drops toward the null; if AMI survives, the modes are
     state-occupancy clusters (the pilot/A1/R3 verdict, tested a fourth way).
  3. per-mode spectral radius max|eig(A_k)|: |λ|>1 = an EXPANDING / unstable mode
     (a loop-attractor candidate).
  4. loop-mode enrichment: which modes are over-represented in collapse-bound chains.

  python3 27_switching_dynamics.py --layers 17 --modes 6
  python3 27_switching_dynamics.py --smoke 300 --layers 17

Output: results/predict/R1-1.5B/switching/switching_{tag}.json + .md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import backup_existing, provenance  # noqa: E402
from src.predict.collapse import collapse_labels  # noqa: E402
from src.predict.trajectory_dataset import build_step_datasets, step_shuffle_within_chain  # noqa: E402


def log(msg: str) -> None:
    print(f"[A2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gather_transitions(datasets, collapse_map: dict) -> dict:
    """All (z_t → z_{t+1}) transitions with source behaviour + loop flag."""
    Zt, Zt1, beh, cid, loop = [], [], [], [], []
    for ds in datasets:
        for t in range(ds.T - 1):
            Zt.append(ds.X[t])
            Zt1.append(ds.X[t + 1])
            beh.append(ds.behaviours[t] if ds.behaviours else "")
            cid.append(ds.chain_id)
            loop.append(bool(collapse_map.get(ds.chain_id, False)))
    return {
        "Zt": np.asarray(Zt, dtype=np.float32),
        "Zt1": np.asarray(Zt1, dtype=np.float32),
        "beh": np.asarray(beh, dtype=object),
        "cid": np.asarray(cid, dtype=object),
        "loop": np.asarray(loop, dtype=bool),
    }


def fit_switching(Zt, Zt1, K: int, *, iters: int = 20, alpha: float = 10.0, seed: int = 0):
    """Hard-EM switching linear dynamics. Returns (labels, A_list, intercepts)."""
    rng = np.random.default_rng(seed)
    # init by clustering the displacement direction
    disp = Zt1 - Zt
    labels = KMeans(K, random_state=seed, n_init=4).fit_predict(disp)
    N = Zt.shape[0]
    for _ in range(iters):
        A_list, b_list, valid = [], [], []
        for k in range(K):
            idx = labels == k
            if idx.sum() < Zt.shape[1] + 2:
                A_list.append(None); b_list.append(None); valid.append(False); continue
            reg = Ridge(alpha=alpha).fit(Zt[idx], Zt1[idx])
            A_list.append(reg.coef_); b_list.append(reg.intercept_); valid.append(True)
        err = np.full((N, K), np.inf)
        for k in range(K):
            if not valid[k]:
                continue
            pred = Zt @ A_list[k].T + b_list[k]
            err[:, k] = np.linalg.norm(Zt1 - pred, axis=1)
        new = err.argmin(1)
        # keep dead modes from swallowing everything: reseed an empty mode randomly
        for k in range(K):
            if (new == k).sum() == 0:
                new[rng.integers(0, N, size=max(2, N // (5 * K)))] = k
        if np.array_equal(new, labels):
            labels = new
            break
        labels = new
    return labels, A_list, b_list


def spectral_radius(A) -> float:
    if A is None:
        return float("nan")
    return float(np.abs(np.linalg.eigvals(A)).max())


def ami_vs_null(mode_labels, beh_labels, *, n_null: int = 200, seed: int = 0) -> dict:
    """AMI(mode, behaviour) with a shuffled-behaviour permutation null."""
    real = float(adjusted_mutual_info_score(beh_labels, mode_labels))
    rng = np.random.default_rng(seed)
    null = np.array([adjusted_mutual_info_score(rng.permutation(beh_labels), mode_labels)
                     for _ in range(n_null)])
    return {"ami": real, "null_mean": float(null.mean()), "null_p97_5": float(np.percentile(null, 97.5)),
            "z": float((real - null.mean()) / (null.std() + 1e-9)), "n_null": n_null}


def analyse_layer(datasets, collapse_map, layer, args) -> dict:
    tr = gather_transitions(datasets, collapse_map)
    n = tr["Zt"].shape[0]
    if n < 200:
        return {"layer": layer, "status": f"too few transitions ({n})"}

    pca = PCA(n_components=min(args.dim, tr["Zt"].shape[1])).fit(
        np.concatenate([tr["Zt"], tr["Zt1"]], axis=0))
    Ztp = pca.transform(tr["Zt"]).astype(np.float32)
    Zt1p = pca.transform(tr["Zt1"]).astype(np.float32)

    labels, A_list, _ = fit_switching(Ztp, Zt1p, args.modes, seed=0)

    out = {"layer": layer, "n_transitions": int(n), "dim": int(Ztp.shape[1]),
           "modes": args.modes, "pca_var_explained": float(pca.explained_variance_ratio_.sum())}

    # (1) mode↔behaviour alignment vs null
    out["mode_behaviour"] = ami_vs_null(labels, tr["beh"], seed=0)

    # (2) ORDER control: refit on step-shuffled trajectories
    shuffled = step_shuffle_within_chain(datasets, np.random.default_rng(0))
    trs = gather_transitions(shuffled, collapse_map)
    Ztp_s = pca.transform(trs["Zt"]).astype(np.float32)
    Zt1p_s = pca.transform(trs["Zt1"]).astype(np.float32)
    labels_s, _, _ = fit_switching(Ztp_s, Zt1p_s, args.modes, seed=0)
    out["mode_behaviour_shuffled_order"] = ami_vs_null(labels_s, trs["beh"], seed=0)
    out["order_sensitive"] = bool(
        out["mode_behaviour"]["ami"] - out["mode_behaviour_shuffled_order"]["ami"] >= 0.02)

    # (3) per-mode spectral radius + size + loop enrichment
    base_loop = float(tr["loop"].mean())
    modes = []
    for k in range(args.modes):
        idx = labels == k
        size = int(idx.sum())
        modes.append({
            "mode": k, "size": size, "frac": size / n,
            "spectral_radius": spectral_radius(A_list[k]),
            "loop_fraction": float(tr["loop"][idx].mean()) if size else float("nan"),
            "loop_enrichment": (float(tr["loop"][idx].mean() / base_loop)
                                if size and base_loop > 0 else float("nan")),
            "top_behaviour": (str(np.bincount(
                np.unique(tr["beh"][idx], return_inverse=True)[1]).argmax()) if size else ""),
        })
    out["mode_stats"] = modes
    out["base_loop_fraction"] = base_loop
    out["n_expanding_modes"] = int(sum(1 for m in modes
                                       if np.isfinite(m["spectral_radius"]) and m["spectral_radius"] > 1.0))
    return out


def write_md(path: Path, results, counts, smoke):
    L = ["# A2 — switching linear dynamics over reasoning steps", ""]
    if smoke:
        L.append("**SMOKE — first-N chains, random collapse labels; AMI≈0, no enrichment.**\n")
    L.append(f"Collapse classes: loop {counts.get('loop',0)} / clean {counts.get('clean',0)} "
             f"/ ambiguous {counts.get('ambiguous',0)}.\n")
    for r in results:
        L.append(f"## layer {r['layer']}")
        if "status" in r:
            L.append(f"- {r['status']}\n"); continue
        mb, mbs = r["mode_behaviour"], r["mode_behaviour_shuffled_order"]
        L += [f"- {r['n_transitions']} transitions, PCA d'={r['dim']} "
              f"({r['pca_var_explained']*100:.0f}% var), K={r['modes']} modes",
              f"- **mode ↔ behaviour AMI = {mb['ami']:.3f}** (null {mb['null_mean']:.3f}, "
              f"z={mb['z']:.1f}) — the annotation-validation read",
              f"- shuffled-ORDER control AMI = {mbs['ami']:.3f} ⇒ "
              f"**{'ORDER-SENSITIVE (dynamics)' if r['order_sensitive'] else 'order-free (occupancy modes)'}**",
              f"- expanding modes (|λ|>1, loop-attractor candidates): {r['n_expanding_modes']}/{r['modes']}",
              "", "| mode | size | |λ|max | loop frac | loop enrich |", "|---|---|---|---|---|"]
        for m in r["mode_stats"]:
            L.append(f"| {m['mode']} | {m['frac']*100:.0f}% | {m['spectral_radius']:.2f} | "
                     f"{m['loop_fraction']:.2f} | {m['loop_enrichment']:.2f}× |")
        L.append(f"\n- base loop fraction {r['base_loop_fraction']:.2f} "
                 f"(enrichment > 1 ⇒ mode over-represented in collapse-bound chains)\n")
    path.write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", default="data/activations/R1-1.5B")
    ap.add_argument("--chains", default="data/annotated_R1-1.5B.json")
    ap.add_argument("--layers", default="17")
    ap.add_argument("--modes", type=int, default=6, help="K (≈ corr-dim 6–8)")
    ap.add_argument("--dim", type=int, default=32, help="PCA dim for the dynamics (rank-40-style)")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default="results/predict/R1-1.5B/switching")
    args = ap.parse_args()

    chains = json.load(open(args.chains))
    smoke = bool(args.smoke)
    if smoke:
        chains = chains[: args.smoke]
    collapse_map, _onset, counts = collapse_labels(chains)
    if smoke:
        import random
        rng = random.Random(42)
        collapse_map = {cid: bool(rng.getrandbits(1)) for cid in collapse_map}
    log(f"collapse {counts}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for layer in [int(x) for x in args.layers.split(",")]:
        log(f"=== layer {layer} (K={args.modes}, PCA d'={args.dim}) ===")
        datasets = build_step_datasets(chains, args.activations, layer)
        r = analyse_layer(datasets, collapse_map, layer, args)
        results.append(r)
        if "status" in r:
            log(f"  {r['status']}")
        else:
            log(f"  AMI={r['mode_behaviour']['ami']:.3f} (null {r['mode_behaviour']['null_mean']:.3f}) "
                f"shuffled-order AMI={r['mode_behaviour_shuffled_order']['ami']:.3f} "
                f"| {'ORDER-SENSITIVE' if r['order_sensitive'] else 'occupancy'} "
                f"| expanding modes {r['n_expanding_modes']}/{r['modes']}")

    tag = "smoke" if smoke else "sparse"
    out_json = out_dir / f"switching_{tag}.json"
    backup_existing(out_json)
    json.dump({"results": results, "collapse_class_counts": counts, "smoke": smoke,
               "provenance": provenance(args, inputs=[args.chains])},
              open(out_json, "w"), indent=2, default=str)
    write_md(out_dir / f"switching_{tag}.md", results, counts, smoke)
    log(f"wrote {out_json}")


if __name__ == "__main__":
    main()
