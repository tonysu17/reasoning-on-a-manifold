#!/usr/bin/env python3
"""
Tier 0 geometry robustness suite — v3 (deduplicated, stable-estimator, with the
curvature same-N control). Runs on CPU from existing activation .npy files.

Closes the v2 loose end: for BOTH intrinsic dimension AND curvature we now compare
  full  vs  random-subsample(size = n_chains)  vs  chain-stratified(1 sentence/chain)
The random-subsample is a same-N control, so any difference between it and the
chain-stratified version isolates the CHAIN effect from the sparse-sampling effect.

Estimators: correlation dimension (PRIMARY, stable) + twoNN (secondary, noted unstable);
curvature = geodesic/Euclidean ratio + local-vs-global PCA dim ratio. Reuses Phase-5b code.

Outputs: results/robustness/geometry_robustness.json + geometry_robustness_summary.md
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
from src.intrinsic_dim import twoNN_estimate, correlation_dimension_estimate
from src.curvature import geodesic_euclidean_ratio, local_vs_global_dim_ratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ACT = Path("data/activations/R1-1.5B"); ANNOT = Path("data/annotated_R1-1.5B.json")
OUT = Path("results/robustness")
from src.config import PEAK_LAYERS as PEAK, SEED, provenance, require_file  # single source
TARGETS = list(PEAK)
B_DIM = 10          # resamples for the corr-dim keystone
B_CURV = 8          # resamples for the curvature control (slower)


# Sidecar-first row provenance + shared dedup (single source of truth).
from src.row_provenance import chain_ids_for, dedup_rows, require_aligned

def load_chain_ids(path, behs):
    return chain_ids_for(ACT, path, behs)

def dedup(X, cids):
    return dedup_rows(X, cids)

def pr_mp(X):
    N, d = X.shape
    C = np.cov(X.astype(np.float64), rowvar=False)
    ev = np.clip(np.sort(np.linalg.eigvalsh(C))[::-1], 0, None); evr = ev / ev.sum()
    pr = float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12))
    d70 = int(np.searchsorted(np.cumsum(evr), 0.70) + 1)
    R = np.corrcoef(X.astype(np.float64), rowvar=False)
    evz = np.clip(np.sort(np.linalg.eigvalsh(R))[::-1], 0, None)
    mp = (1 + np.sqrt(d / N)) ** 2
    return dict(pr=pr, d70_raw=d70, n_above_mp=int((evz > mp).sum()), mp_edge=float(mp))

def cdim(X, nb=6):
    return float(correlation_dimension_estimate(X.astype(np.float32), n_bootstrap=nb).estimate)

def geo(Y, nb=10, npairs=350):
    try: return float(geodesic_euclidean_ratio(Y.astype(np.float32), k=10, n_pairs=npairs, n_bootstrap=nb).mean)
    except Exception as e: log.warning(f"geo fail: {e}"); return None

def lgr(Y, nb=10, nanch=150):
    try: return float(local_vs_global_dim_ratio(Y.astype(np.float32), k=10, n_anchors=nanch, n_bootstrap=nb).mean)
    except Exception as e: log.warning(f"lg fail: {e}"); return None

def zscore(X):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
    return ((X - mu) / sd).astype(np.float32)

def pca_reduce(X, k=50):
    from sklearn.decomposition import PCA
    return PCA(n_components=min(k, X.shape[1], X.shape[0]-1)).fit_transform(X).astype(np.float32)

def ms(a):
    a = np.array([x for x in a if x is not None], float)
    return dict(mean=float(a.mean()), sd=float(a.std())) if a.size else None


def main():
    global ACT, ANNOT, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", default="R1-1.5B")
    ap.add_argument("--annotated", default=None)
    a = ap.parse_args()
    ACT = Path(f"data/activations/{a.model_short}")
    ANNOT = Path(a.annotated) if a.annotated else Path(f"data/annotated_{a.model_short}.json")
    OUT = Path(f"results/robustness/{a.model_short}")
    OUT.mkdir(parents=True, exist_ok=True)
    require_file(ANNOT, "run 03_annotate_chains.py first")
    require_file(ACT, "run 04_extract_activations.py first")
    cidmap = load_chain_ids(ANNOT, TARGETS); res = {}
    for b, L in PEAK.items():
        Xr = np.load(ACT / f"{b}_layer{L}.npy").astype(np.float32)
        # Hard error replaces the old arange fallback (which silently turned
        # the chain-stratified resample into a full-data resample).
        cids = require_aligned(b, Xr.shape[0], cidmap.get(b), context="robustness_geometry")
        dup_pct = 100.0 * (1 - len(np.unique(Xr, axis=0)) / Xr.shape[0])
        X, cu = dedup(Xr, cids); Nu = X.shape[0]
        obc = {}
        for i, c in enumerate(cu): obc.setdefault(c, []).append(i)
        uniq = list(obc); nc = len(uniq)
        log.info(f"=== {b} @ L{L}: N={Xr.shape[0]}, dup={dup_pct:.0f}%, N_unique={Nu}, chains={nc} ===")
        r = dict(behaviour=b, layer=L, N_raw=int(Xr.shape[0]), dup_pct=float(dup_pct),
                 N_unique=int(Nu), n_chains=int(nc), pr_mp=pr_mp(X))

        # ---- intrinsic-dim keystone (corr-dim): full vs randsub vs chainstrat ----
        cf = cdim(X, nb=B_DIM); rs, st = [], []
        # ---- curvature keystone (geo + lg): full vs randsub vs chainstrat ----
        gf, lf = geo(X), lgr(X)
        g_rs, g_st, l_rs, l_st = [], [], [], []
        for s in range(B_CURV):
            g = np.random.default_rng(SEED + s)
            ridx = g.choice(Nu, nc, replace=False)
            sidx = np.array([g.choice(obc[c]) for c in uniq])
            rs.append(cdim(X[ridx], nb=4)); st.append(cdim(X[sidx], nb=4))
            g_rs.append(geo(X[ridx], nb=6, npairs=250)); g_st.append(geo(X[sidx], nb=6, npairs=250))
            l_rs.append(lgr(X[ridx], nb=6, nanch=100)); l_st.append(lgr(X[sidx], nb=6, nanch=100))
        # a couple more corr-dim resamples are cheap; top up to B_DIM
        for s in range(B_CURV, B_DIM):
            g = np.random.default_rng(SEED + s)
            rs.append(cdim(X[g.choice(Nu, nc, replace=False)], nb=4))
            st.append(cdim(X[np.array([g.choice(obc[c]) for c in uniq])], nb=4))

        r["keystone_cdim"] = dict(full=cf, random_sub=ms(rs), chain_strat=ms(st), n=int(nc))
        r["keystone_geodesic"] = dict(full=gf, random_sub=ms(g_rs), chain_strat=ms(g_st), n=int(nc))
        r["keystone_local_global"] = dict(full=lf, random_sub=ms(l_rs), chain_strat=ms(l_st), n=int(nc))
        r["twoNN_dedup"] = float(twoNN_estimate(X, n_bootstrap=20).estimate)
        r["preprocessing_cdim"] = dict(raw=cf, zscore=cdim(zscore(X), nb=8), pca50=cdim(pca_reduce(X, 50), nb=8))

        log.info(f"  cdim: full={cf:.2f} randsub={r['keystone_cdim']['random_sub']['mean']:.2f} "
                 f"chainstrat={r['keystone_cdim']['chain_strat']['mean']:.2f}")
        log.info(f"  geo:  full={gf:.2f} randsub={ms(g_rs)['mean']:.2f} chainstrat={ms(g_st)['mean']:.2f}")
        res[b] = r
        json.dump(res, open(OUT / "geometry_robustness.json", "w"), indent=2)

    # Provenance stamp (git commit, seed, input hash) for traceability.
    json.dump(provenance(inputs=[str(ANNOT)]), open(OUT / "provenance.json", "w"), indent=2)

    # ---- markdown summary ----
    lines = ["# Geometry robustness (Tier 0) — deduplicated, stable estimator", "",
             "Intrinsic dim = correlation dimension (twoNN noted unstable). Curvature = geodesic/Euclidean ratio.",
             "`random_sub` and `chain_strat` are the SAME size (n_chains); their difference isolates the chain effect.", "",
             "| Behaviour | dup% | N_uniq | PR | n>MP | **cdim full** | cdim randsub | **cdim chainstrat** | geo full | geo randsub | geo chainstrat |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in TARGETS:
        r = res[b]; kc = r["keystone_cdim"]; kg = r["keystone_geodesic"]
        lines.append(f"| {b} | {r['dup_pct']:.0f}% | {r['N_unique']} | {r['pr_mp']['pr']:.0f} | "
                     f"{r['pr_mp']['n_above_mp']} | {kc['full']:.2f} | {kc['random_sub']['mean']:.2f} | "
                     f"{kc['chain_strat']['mean']:.2f} | {kg['full']:.2f} | {kg['random_sub']['mean']:.2f} | "
                     f"{kg['chain_strat']['mean']:.2f} |")
    lines += ["", "## Reading",
              "- **cdim chainstrat ≈ cdim randsub ≈ cdim full** → low intrinsic dimension is behaviour-intrinsic, NOT a chain confound (keystone PASS).",
              "- **geo randsub vs geo chainstrat** at equal N isolates real chain-trajectory curvature from sparse-graph effects.",
              "- 35–56% exact-duplicate pooled activations were removed before all estimates (fixed 1+10-token window on short repeated markers).",
              "- twoNN is duplicate/subsample-unstable here; correlation dimension is the reliable estimator."]
    (OUT / "geometry_robustness_summary.md").write_text("\n".join(lines))
    log.info(f"Saved -> {OUT/'geometry_robustness.json'} + summary.md")

if __name__ == "__main__":
    main()
