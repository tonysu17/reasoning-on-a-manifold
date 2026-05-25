#!/usr/bin/env python3
"""
Phase 5c - Cross-layer trajectory analysis.

Companion document Section 4 (revised): "Cross-layer trajectory and causal
localisation". Two cheap analyses that complement the curvature analysis
in Phase 5b and the activation patching in Phase 7b:

  (a) Layer-wise linear probing.
      Per-behaviour binary classifier (behaviour vs other) trained on
      activations at each saved layer; report accuracy-vs-depth.
      The depth at which behaviour becomes linearly decodable is informative
      about where in the computation the behaviour is encoded.

  (b) Non-adjacent layer-PCA principal-angle evolution.
      Compares PCA subspaces between layer L and layer L+k for k in {3,7,14}.
      This deliberately avoids adjacent-layer comparisons because the
      residual stream's x_{L+1} = x_L + f(x_L) structure makes adjacent
      subspaces near-identical by construction (Companion 4.2 caveat).

Outputs:
  results/cross_layer/<model>/probe_accuracy.json
  results/cross_layer/<model>/probe_curves.png
  results/cross_layer/<model>/subspace_angles.json
  results/cross_layer/<model>/subspace_angles_heatmap.png
  results/cross_layer/<model>/summary.md

Runtime: ~5-10 min on CPU.

Usage:
  python 05c_cross_layer_probing.py --model-short R1-1.5B
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


from src.annotation import TARGET_BEHAVIOURS
ALL_LABELS = TARGET_BEHAVIOURS + ["initializing", "deduction"]


# Layer-wise linear probing

def probe_accuracy_at_layer(
    behaviour_X:  np.ndarray,
    other_X:      np.ndarray,
    random_state: int = 42,
    test_size:    float = 0.3,
    C:            float = 1.0,
) -> dict:
    """Train a binary logistic regression (behaviour vs other) on activations
    at one layer; return train/test accuracy, ROC AUC, and class balance.

    Subsamples the larger class to balance, then splits train/test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    rng = np.random.default_rng(random_state)
    n_b = behaviour_X.shape[0]
    n_o = other_X.shape[0]
    n_min = min(n_b, n_o)
    if n_min < 10:
        return {"error": f"too few samples: n_behaviour={n_b}, n_other={n_o}"}
    # Balanced subsample
    b_idx = rng.choice(n_b, n_min, replace=False)
    o_idx = rng.choice(n_o, n_min, replace=False)
    X = np.vstack([behaviour_X[b_idx], other_X[o_idx]])
    y = np.concatenate([np.ones(n_min), np.zeros(n_min)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=random_state)
    clf.fit(Xtr, ytr)
    return {
        "train_acc": float(accuracy_score(ytr, clf.predict(Xtr))),
        "test_acc":  float(accuracy_score(yte, clf.predict(Xte))),
        "auc":       float(roc_auc_score(yte, clf.decision_function(Xte))),
        "n_per_class": int(n_min),
    }


def probe_all_layers_all_behaviours(
    act_dir: Path,
    layers:  list,
    behaviours: list,
    random_state: int = 42,
) -> dict:
    """Run probing for each (behaviour, layer) cell."""
    out = {}
    for beh in behaviours:
        out[beh] = {}
        # Build the "other" activations by concatenating all non-target labels
        other_files = [act_dir / f"{lbl}_layer{layers[0]}.npy"
                       for lbl in ALL_LABELS if lbl != beh]
        for L in layers:
            target_p = act_dir / f"{beh}_layer{L}.npy"
            if not target_p.exists():
                logger.warning(f"  missing {target_p}; skipping {beh}@L{L}")
                continue
            X_b = np.load(target_p)
            # Other = pool of non-target activations at this layer
            others = []
            for lbl in ALL_LABELS:
                if lbl == beh:
                    continue
                op = act_dir / f"{lbl}_layer{L}.npy"
                if op.exists():
                    others.append(np.load(op))
            if not others:
                continue
            X_o = np.vstack(others)
            logger.info(f"  {beh}@L{L}: n_target={X_b.shape[0]}, n_other={X_o.shape[0]}")
            try:
                r = probe_accuracy_at_layer(X_b, X_o, random_state=random_state)
            except Exception as e:
                logger.error(f"  probe failed: {type(e).__name__}: {e}")
                r = {"error": str(e)}
            out[beh][int(L)] = r
    return out


# Subspace angle evolution

def pca_subspace(X: np.ndarray, dim: int):
    centered = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:dim].T   # columns are basis vectors


def principal_angle_mean_deg(B1: np.ndarray, B2: np.ndarray) -> float:
    M = B1.T @ B2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.degrees(np.arccos(s).mean()))


def subspace_angle_evolution(
    act_dir: Path, behaviours: list, layers: list, top_k: int = 10,
) -> dict:
    """For each behaviour, compute principal angles between layer-L and
    layer-L+k subspaces for k = 3, 7, 14 (when present)."""
    out = {}
    for beh in behaviours:
        bases = {}
        for L in layers:
            p = act_dir / f"{beh}_layer{L}.npy"
            if not p.exists():
                continue
            X = np.load(p)
            if X.shape[0] < top_k:
                continue
            bases[L] = pca_subspace(X, top_k)
        if len(bases) < 2:
            out[beh] = {"error": "insufficient layers cached"}
            continue
        sorted_L = sorted(bases.keys())
        per_beh = {}
        for i, L in enumerate(sorted_L):
            for k_gap in (1, 3, 7, 14):
                target_L = L + k_gap
                # Find nearest available layer near L + k_gap
                candidates = [Lc for Lc in sorted_L if Lc > L]
                if not candidates:
                    continue
                Lc = min(candidates, key=lambda x: abs(x - target_L))
                actual_gap = Lc - L
                if actual_gap == 0:
                    continue
                key = f"L{L}_to_L{Lc}_gap{actual_gap}"
                per_beh[key] = principal_angle_mean_deg(bases[L], bases[Lc])
        out[beh] = per_beh
    return out


# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-short", default="R1-1.5B")
    parser.add_argument("--layers", nargs="+", type=int, default=[3, 7, 10, 14, 17, 21, 24, 27])
    parser.add_argument("--behaviours", nargs="+", default=TARGET_BEHAVIOURS)
    parser.add_argument("--top-k", type=int, default=10, help="PCA dim for subspace angle. Default 10.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"results/cross_layer/{args.model_short}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    act_dir = Path(f"data/activations/{args.model_short}")
    if not act_dir.exists():
        logger.error(f"Activations not found at {act_dir}")
        sys.exit(1)

    logger.info("=== Layer-wise linear probing ===")
    probe_results = probe_all_layers_all_behaviours(act_dir, args.layers, args.behaviours,
                                                      random_state=args.seed)
    (args.out_dir / "probe_accuracy.json").write_text(json.dumps(probe_results, indent=2))

    logger.info("=== Subspace angle evolution ===")
    angle_results = subspace_angle_evolution(act_dir, args.behaviours, args.layers,
                                              top_k=args.top_k)
    (args.out_dir / "subspace_angles.json").write_text(json.dumps(angle_results, indent=2))

    # Plot probe curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for beh, by_layer in probe_results.items():
            xs, ys, errs = [], [], []
            for L in sorted(by_layer.keys()):
                r = by_layer[L]
                if isinstance(r, dict) and "test_acc" in r:
                    xs.append(L); ys.append(r["test_acc"])
            ax.plot(xs, ys, marker="o", label=beh)
        ax.axhline(0.5, color="grey", linestyle="--", alpha=0.4, label="chance")
        ax.set(xlabel="Layer", ylabel="Binary probe test accuracy",
                title=f"Layer-wise behaviour decodability - {args.model_short}", ylim=(0.45, 1.02))
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(args.out_dir / "probe_curves.png", dpi=120)
        logger.info(f"Saved: {args.out_dir / 'probe_curves.png'}")

        # Heatmap of subspace angles
        if angle_results:
            from collections import defaultdict
            mat = defaultdict(dict)
            for beh, items in angle_results.items():
                if "error" in items:
                    continue
                for key, val in items.items():
                    mat[beh][key] = val
            if mat:
                behs = sorted(mat.keys())
                all_keys = sorted({k for v in mat.values() for k in v.keys()})
                M = np.full((len(behs), len(all_keys)), np.nan)
                for i, b in enumerate(behs):
                    for j, k in enumerate(all_keys):
                        M[i, j] = mat[b].get(k, np.nan)
                fig, ax = plt.subplots(figsize=(min(0.6 * len(all_keys) + 2, 16), 0.4 * len(behs) + 2))
                im = ax.imshow(M, aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(all_keys))); ax.set_xticklabels(all_keys, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(len(behs))); ax.set_yticklabels(behs)
                ax.set_title(f"Subspace principal angles (deg) - {args.model_short}")
                plt.colorbar(im, ax=ax, label="Mean principal angle (deg)")
                plt.tight_layout()
                plt.savefig(args.out_dir / "subspace_angles_heatmap.png", dpi=120)
                logger.info(f"Saved: {args.out_dir / 'subspace_angles_heatmap.png'}")
    except ImportError:
        logger.warning("matplotlib unavailable; skipping plots")

    # Summary
    md = ["# Phase 5c - Cross-layer trajectory", "", "## Probe accuracy by layer", ""]
    md.append("| Behaviour | " + " | ".join(f"L{L}" for L in args.layers) + " |")
    md.append("|-----------|" + "|".join(["---"] * len(args.layers)) + "|")
    for beh in args.behaviours:
        cells = []
        for L in args.layers:
            r = probe_results.get(beh, {}).get(L)
            if not r or "error" in (r or {}):
                cells.append("-")
            else:
                cells.append(f"{r['test_acc']:.2f}")
        md.append(f"| {beh} | " + " | ".join(cells) + " |")
    md += ["", "## Non-adjacent subspace angles (deg)", ""]
    for beh in args.behaviours:
        items = angle_results.get(beh)
        if not items or "error" in items:
            md.append(f"- **{beh}**: not available")
            continue
        md.append(f"- **{beh}**: " + ", ".join(f"{k}: {v:.1f}deg" for k, v in items.items()))

    (args.out_dir / "summary.md").write_text("\n".join(md))
    print(f"\nResults: {args.out_dir}")


if __name__ == "__main__":
    main()
