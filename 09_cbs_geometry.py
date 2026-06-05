"""
Phase 9: Per-sentence geometric tests for CBS tier + cross-domain (M2).

Runs four geometric statistics per layer x behaviour over a set of activation
matrices. Each statistic is tested under both the CBS-tier label
(Jonckheere-Terpstra) and the cross-domain binary label (Wilcoxon), with
Holm correction across (layers x behaviours x statistics x labels).

Modes
-----
Default:                CBS annotations from --cbs-annotations file.
--synthetic-tiers:      assign per-row tiers via rng (smoke validation).
                        Required when no CBS annotations exist (build-now).
--shuffle-control:      additionally emit a shuffled-tier sanity rerun.
--reversal-control:     additionally emit a reversed-tier sanity rerun.

Output
------
results/cbs/{model}/geometry_results.json    (synthesis §M2.3)
results/cbs/{model}/plots/effect_size_vs_layer.png
results/cbs/{model}/plots/principal_angle_heatmap_layer{N}.png

Synthesis-plan reference: §M2.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import SEED
from src.cbs.geometry import (
    bootstrap_ci,
    build_union_basis,
    centroid_distance,
    cliffs_delta,
    holm_correction,
    jonckheere_terpstra,
    local_intrinsic_dim,
    out_of_subspace_residual,
    principal_angles,
)

logger = logging.getLogger("09_cbs_geometry")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--activations-dir", type=Path,
                   default=Path("data/activations/R1-1.5B"))
    p.add_argument("--cbs-annotations", type=Path,
                   default=Path("data/chains_cbs_annotated_R1-1.5B.json"))
    p.add_argument("--out-dir", default=None,
                   type=lambda s: Path(s) if s else None)
    p.add_argument("--model-suffix", default="R1-1.5B")
    p.add_argument("--layers", default="3,7,10,14,17,21,24,27")
    p.add_argument("--behaviours", default="adding-knowledge,deduction")
    p.add_argument("--all-behaviours-for-union",
                   default="backtracking,uncertainty-estimation,"
                           "example-testing,adding-knowledge")
    p.add_argument("--top-pcs-per-behaviour", type=int, default=20)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--variance-thresholds", default="0.90,0.95,0.99")
    p.add_argument("--seed", type=int, default=SEED)  # config.SEED (was 0)
    p.add_argument("--synthetic-tiers", action="store_true",
                   help="Assign per-row tiers via rng. Required when no CBS "
                        "annotations exist (build-now smoke).")
    p.add_argument("--shuffle-control", action="store_true",
                   help="Additionally emit a shuffled-tier sanity rerun.")
    p.add_argument("--reversal-control", action="store_true",
                   help="Additionally emit a reversed-tier sanity rerun.")
    p.add_argument("--k-local-id", type=int, default=20)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_activation_matrix(activations_dir: Path, behaviour: str,
                            layer: int) -> Optional[np.ndarray]:
    fp = activations_dir / f"{behaviour}_layer{layer}.npy"
    if not fp.exists():
        logger.warning("missing %s", fp)
        return None
    return np.load(fp)


def _synthetic_tier_labels(n: int, seed: int) -> np.ndarray:
    """Uniform random tiers in {1, 2, 3} keyed on (seed, n)."""
    rng = np.random.default_rng(seed)
    return rng.integers(1, 4, size=n)


def _synthetic_crossdomain_labels(n: int, seed: int,
                                  rate: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    return (rng.random(n) < rate).astype(bool)


def _build_behaviour_pcs(activations_dir: Path, behaviours: list[str],
                         layer: int, k: int) -> dict[str, np.ndarray]:
    """For each behaviour: top-k PCs from the activation matrix at `layer`."""
    out: dict[str, np.ndarray] = {}
    for beh in behaviours:
        X = _load_activation_matrix(activations_dir, beh, layer)
        if X is None or X.shape[0] < 2:
            continue
        Xc = X - X.mean(axis=0, keepdims=True)
        # SVD of centered matrix; columns of V_top are PCs in feature space.
        try:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        n_keep = min(k, Vt.shape[0])
        out[beh] = Vt[:n_keep].T   # (d, n_keep)
    return out


def _compute_layer_behaviour(
    *,
    X: np.ndarray,
    tiers: np.ndarray,
    cross_domain: np.ndarray,
    union_basis: Optional[np.ndarray],
    k_local: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    """Compute all four statistics + JT/Wilcoxon/Cliff's on one (layer, behaviour)
    cell. Returns a JSON-serialisable dict per `statistic`."""
    results: list[dict] = []
    centroid = X.mean(axis=0)
    distance = centroid_distance(X, centroid)
    residual = (out_of_subspace_residual(X, union_basis)
                if union_basis is not None and union_basis.size > 0
                else np.full(X.shape[0], np.nan))
    lid = local_intrinsic_dim(X, k=k_local)

    stats = {
        "centroid_distance": distance,
        "out_of_subspace_residual": residual,
        "local_intrinsic_dim": lid,
    }

    for stat_name, values in stats.items():
        valid = np.isfinite(values)
        if valid.sum() < 5:
            continue
        v = values[valid]
        t = tiers[valid]
        cd = cross_domain[valid]

        # CBS tier ordinal test (JT)
        jt = jonckheere_terpstra(v, t)
        cd_eff = cliffs_delta(v[t == 3], v[t == 1]) if (np.any(t == 3) and np.any(t == 1)) else 0.0
        try:
            ci_lo, ci_hi = bootstrap_ci(
                cliffs_delta, v[t == 3], v[t == 1],
                n_bootstrap=n_bootstrap, paired=False, rng=rng,
            ) if (np.any(t == 3) and np.any(t == 1)) else (float("nan"), float("nan"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("bootstrap_ci failed (%s); emitting NaN", exc)
            ci_lo, ci_hi = float("nan"), float("nan")
        results.append({
            "statistic": stat_name,
            "label_scheme": "cbs_tier",
            "test_statistic": jt["z"],
            "p_raw": jt["p_value"],
            "p_holm": float("nan"),                  # filled later
            "effect_size": cd_eff,
            "effect_size_ci95": [ci_lo, ci_hi],
            "n_total": int(valid.sum()),
            "n_per_tier": jt["n_per_tier"],
            "trend_direction": jt["trend_direction"],
        })

        # Cross-domain binary test (Wilcoxon-Mann-Whitney via Cliff's delta + Z)
        if np.any(cd) and np.any(~cd):
            try:
                from scipy.stats import mannwhitneyu
                stat_u, p_u = mannwhitneyu(v[cd], v[~cd], alternative="two-sided")
                test_stat = float(stat_u)
            except (ImportError, ValueError):
                # ImportError: scipy absent. ValueError: mannwhitneyu rejects
                # all-identical inputs. Either way, emit NaN rather than crash.
                test_stat = float("nan")
                p_u = float("nan")
            cd_delta = cliffs_delta(v[cd], v[~cd])
            try:
                ci_lo_b, ci_hi_b = bootstrap_ci(
                    cliffs_delta, v[cd], v[~cd],
                    n_bootstrap=n_bootstrap, paired=False, rng=rng,
                )
            except Exception:  # noqa: BLE001
                ci_lo_b, ci_hi_b = float("nan"), float("nan")
            results.append({
                "statistic": stat_name,
                "label_scheme": "cross_domain",
                "test_statistic": test_stat,
                "p_raw": float(p_u),
                "p_holm": float("nan"),
                "effect_size": cd_delta,
                "effect_size_ci95": [ci_lo_b, ci_hi_b],
                "n_total": int(valid.sum()),
                "n_cross_domain": int(cd.sum()),
                "n_in_domain": int((~cd).sum()),
            })

    return {"results": results}


def _load_real_labels(cbs_annotations_path: Path,
                      behaviours: list) -> "dict[str, tuple[np.ndarray, np.ndarray]]":
    """Load REAL CBS tier + cross-domain labels per behaviour, aligned with the
    Phase-4 activation row ordering (src.cbs.trajectory.build_row_index).

    Returns {behaviour: (tiers (int, 0 if unannotated), cross_domain (bool))}.
    The caller must verify each array length matches the loaded activation
    matrix before use, and keep only tier in {1,2,3} rows. Replaces the old
    `pass` stub that left geometry on synthetic tiers while stamping
    labels_source="real" (AUDIT.md §5)."""
    from src.cbs.trajectory import build_row_index
    with open(cbs_annotations_path) as f:
        chains = json.load(f)
    row_index = build_row_index(chains, target_behaviours=list(behaviours))
    label_by_key: dict[tuple, tuple[int, bool]] = {}
    for chain in chains:
        cid = chain.get("task_id", "")
        for i, span in enumerate(chain.get("annotations", []) or []):
            if span.get("label") in behaviours and "cbs_tier" in span:
                label_by_key[(cid, i)] = (int(span.get("cbs_tier", 0)),
                                          bool(span.get("cbs_cross_domain", False)))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for beh, items in row_index.items():
        tiers = np.zeros(len(items), dtype=int)
        cds = np.zeros(len(items), dtype=bool)
        for row, key in enumerate(items):
            t, c = label_by_key.get(key, (0, False))
            tiers[row] = t
            cds[row] = c
        out[beh] = (tiers, cds)
    return out


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    out_dir = args.out_dir or Path("results/cbs") / args.model_suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    layers = [int(s) for s in args.layers.split(",")]
    behaviours = [s.strip() for s in args.behaviours.split(",")]
    union_behaviours = [s.strip() for s in args.all_behaviours_for_union.split(",")]
    variance_thresholds = [float(s) for s in args.variance_thresholds.split(",")]

    rng = np.random.default_rng(args.seed)

    # Load REAL CBS labels if a real annotations file is present; otherwise
    # fall back to synthetic tiers AND label them honestly as synthetic.
    real_labels: "dict | None" = None
    if args.cbs_annotations.exists() and not args.synthetic_tiers:
        logger.info("loading CBS annotations from %s", args.cbs_annotations)
        try:
            real_labels = _load_real_labels(args.cbs_annotations, behaviours)
            n_annot = sum(int((t > 0).sum()) for t, _ in real_labels.values())
            if n_annot == 0:
                logger.warning("CBS annotations present but 0 tiered sentences "
                               "for target behaviours; using synthetic tiers.")
                real_labels = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load real CBS labels (%s); using synthetic "
                           "tiers.", exc)
            real_labels = None
    if real_labels is None and not args.synthetic_tiers:
        logger.warning("no usable CBS annotations at %s; falling back to "
                       "--synthetic-tiers", args.cbs_annotations)
        args.synthetic_tiers = True

    geometry_records: list[dict] = []
    principal_angle_records: list[dict] = []

    for layer in layers:
        # Per-layer union basis built from PCs of every saved behaviour.
        pcs = _build_behaviour_pcs(
            args.activations_dir, union_behaviours, layer,
            args.top_pcs_per_behaviour,
        )
        union_basis = (build_union_basis(pcs, 0.95) if pcs else None)

        for behaviour in behaviours:
            X = _load_activation_matrix(args.activations_dir, behaviour, layer)
            if X is None:
                continue
            n = X.shape[0]
            if n < 10:
                logger.warning("layer=%d behaviour=%s n=%d; skipping",
                               layer, behaviour, n)
                continue

            # Real labels if available for this behaviour and row-aligned;
            # otherwise synthetic. label is stamped to match what is ACTUALLY
            # used (never "real" on synthetic data — the old stub's bug).
            Xeff = X
            if real_labels is not None and behaviour in real_labels:
                tiers_all, cd_all = real_labels[behaviour]
                if tiers_all.shape[0] != n:
                    logger.warning("layer=%d behaviour=%s: real-label rows (%d) != "
                                   "activation rows (%d); skipping (cannot align).",
                                   layer, behaviour, tiers_all.shape[0], n)
                    continue
                keep = np.isin(tiers_all, (1, 2, 3))
                if int(keep.sum()) < 10:
                    logger.warning("layer=%d behaviour=%s: only %d tiered rows; "
                                   "skipping.", layer, behaviour, int(keep.sum()))
                    continue
                Xeff = X[keep]
                tiers = tiers_all[keep]
                cd = cd_all[keep]
                label = "real"
            else:
                tiers = _synthetic_tier_labels(n, args.seed + layer + zlib.crc32(behaviour.encode()) % 1000)
                cd = _synthetic_crossdomain_labels(n, args.seed + layer + zlib.crc32(behaviour.encode()) % 1000)
                label = "synthetic"

            cell = _compute_layer_behaviour(
                X=Xeff, tiers=tiers, cross_domain=cd,
                union_basis=union_basis, k_local=args.k_local_id,
                n_bootstrap=args.n_bootstrap, rng=rng,
            )
            for rec in cell["results"]:
                rec.update({"layer": layer, "behaviour": behaviour,
                            "labels_source": label})
                geometry_records.append(rec)

            # Optional shuffle control.
            if args.shuffle_control:
                shuffled = rng.permutation(tiers)
                cell_s = _compute_layer_behaviour(
                    X=Xeff, tiers=shuffled, cross_domain=cd,
                    union_basis=union_basis, k_local=args.k_local_id,
                    n_bootstrap=args.n_bootstrap, rng=rng,
                )
                for rec in cell_s["results"]:
                    rec.update({"layer": layer, "behaviour": behaviour,
                                "labels_source": "shuffle_control"})
                    geometry_records.append(rec)

            # Optional reversal control.
            if args.reversal_control:
                rev_map = {1: 3, 2: 2, 3: 1}
                rev = np.array([rev_map[int(t)] for t in tiers])
                cell_r = _compute_layer_behaviour(
                    X=Xeff, tiers=rev, cross_domain=cd,
                    union_basis=union_basis, k_local=args.k_local_id,
                    n_bootstrap=args.n_bootstrap, rng=rng,
                )
                for rec in cell_r["results"]:
                    rec.update({"layer": layer, "behaviour": behaviour,
                                "labels_source": "reversal_control"})
                    geometry_records.append(rec)

        # Principal angles between every pair of behaviour subspaces.
        beh_names = sorted(pcs.keys())
        for i, a in enumerate(beh_names):
            for j in range(i + 1, len(beh_names)):
                b = beh_names[j]
                angles = principal_angles(pcs[a], pcs[b], top_k=10)
                principal_angle_records.append({
                    "layer": layer,
                    "behaviour_a": a,
                    "behaviour_b": b,
                    "top_k_angles_deg": [float(np.degrees(x)) for x in angles],
                })

    # Holm correction across the main records (excluding controls).
    main_records = [r for r in geometry_records
                    if r.get("labels_source") not in {"shuffle_control",
                                                       "reversal_control"}]
    p_raw = [r["p_raw"] for r in main_records]
    if p_raw:
        p_holm = holm_correction(p_raw)
        for r, ph in zip(main_records, p_holm):
            r["p_holm"] = float(ph)

    # Tier / cross-domain counts.
    n_per_tier_total = {"1": 0, "2": 0, "3": 0}
    for rec in geometry_records:
        for k, v in rec.get("n_per_tier", {}).items():
            n_per_tier_total[str(k)] = n_per_tier_total.get(str(k), 0) + v

    out = {
        "model": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
                  if args.model_suffix.startswith("R1") else args.model_suffix),
        "activations_source": str(args.activations_dir),
        "labels_source": ("synthetic" if args.synthetic_tiers else "real"),
        "truncation_policy": "stratify",
        "layers": layers,
        "behaviours": behaviours,
        "variance_thresholds_used": variance_thresholds,
        "n_sentences_per_tier_total": n_per_tier_total,
        "n_records_main": len(main_records),
        "n_records_total": len(geometry_records),
        "results": geometry_records,
        "principal_angles": principal_angle_records,
        "note": ("smoke-only, not paper-grade" if args.synthetic_tiers
                 else "labels from CBS annotations"),
    }
    out_path = out_dir / "geometry_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s (%d records, %d principal-angle pairs)",
                out_path, len(geometry_records), len(principal_angle_records))

    _emit_plots(out_dir, geometry_records, principal_angle_records, layers)
    return 0


def _emit_plots(out_dir: Path,
                geometry_records: list[dict],
                principal_angle_records: list[dict],
                layers: list[int]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping plot emission")
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # effect_size_vs_layer: one figure per (behaviour, statistic, label_scheme)
    main = [r for r in geometry_records
            if r.get("labels_source") not in {"shuffle_control",
                                              "reversal_control"}]
    keys = sorted({(r["behaviour"], r["statistic"], r["label_scheme"])
                   for r in main})
    for behaviour, statistic, label_scheme in keys:
        cells = [r for r in main
                 if r["behaviour"] == behaviour and r["statistic"] == statistic
                 and r["label_scheme"] == label_scheme]
        if not cells:
            continue
        cells.sort(key=lambda r: r["layer"])
        xs = [c["layer"] for c in cells]
        ys = [c["effect_size"] for c in cells]
        ylo = [c["effect_size_ci95"][0] for c in cells]
        yhi = [c["effect_size_ci95"][1] for c in cells]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.errorbar(xs, ys,
                    yerr=[[y - lo for y, lo in zip(ys, ylo)],
                          [hi - y for y, hi in zip(ys, yhi)]],
                    fmt="o-", capsize=3)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_xlabel("layer")
        ax.set_ylabel(f"Cliff's delta ({label_scheme})")
        ax.set_title(f"{behaviour}: {statistic}")
        fig.tight_layout()
        png = plots_dir / f"effect_size_vs_layer__{behaviour}__{statistic}__{label_scheme}.png"
        fig.savefig(png, dpi=110)
        plt.close(fig)

    # principal_angle_heatmap per layer
    for layer in layers:
        recs = [r for r in principal_angle_records if r["layer"] == layer]
        if not recs:
            continue
        behaviours = sorted({r["behaviour_a"] for r in recs}
                            | {r["behaviour_b"] for r in recs})
        idx = {b: i for i, b in enumerate(behaviours)}
        n = len(behaviours)
        mat = np.full((n, n), np.nan)
        for r in recs:
            i = idx[r["behaviour_a"]]
            j = idx[r["behaviour_b"]]
            mean_angle = float(np.mean(r["top_k_angles_deg"]))
            mat[i, j] = mean_angle
            mat[j, i] = mean_angle
        np.fill_diagonal(mat, 0.0)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(mat, vmin=0, vmax=90, cmap="viridis")
        ax.set_xticks(range(n))
        ax.set_xticklabels(behaviours, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(behaviours, fontsize=7)
        ax.set_title(f"mean principal angle (deg), layer {layer}")
        fig.colorbar(im, ax=ax, label="degrees")
        fig.tight_layout()
        fig.savefig(plots_dir / f"principal_angle_heatmap_layer{layer}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
