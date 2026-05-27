"""
Phase 11: Trajectory analysis — group comparisons + matched pairs + verification
gradient (M4).

Reads trajectory summaries (parquet/json) from Phase 10 + CBS-annotated chains
+ optional multi-seed chains; runs:

  1. Group comparisons (residualised on chain length T) on the
     trajectory summary parquet:
        * truncated vs not-truncated
        * high-CBS (>=2 tier-3) vs low-CBS (0 tier-3)
        * long vs short chains as a positional control
  2. Per-sentence regression long-format DataFrame (curvature ~ tier).
  3. Matched-pair tier-3 success-vs-failure analysis (Jaccard >= 0.6).
     **Blocked on Phase 7 answer-checker + multi-seed re-gen.**
  4. Verification-gradient 5-fold CV probe.
     **Blocked on Phase 7 answer-checker.**

Synthesis-plan reference: §M4.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.cbs.trajectory import compare_groups, PHASE_4_BEHAVIOURS

logger = logging.getLogger("11_trajectory_analysis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--trajectory-dir", type=Path, default=None,
                   help="defaults to results/trajectory/{model-suffix}")
    p.add_argument("--cbs-annotations", type=Path,
                   default=Path("data/chains_cbs_annotated_R1-1.5B.json"))
    p.add_argument("--multi-seed-chains", type=Path,
                   default=Path("data/chains_R1-1.5B_multiseed.json"),
                   help="Output of multi-seed re-generation (synthesis §M4.5). "
                        "Required for matched-pair analysis.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="defaults to --trajectory-dir")
    p.add_argument("--model-suffix", default="R1-1.5B")
    p.add_argument("--layers", default="17,27")
    p.add_argument("--jaccard-threshold", type=float, default=0.6)
    p.add_argument("--skip-matched-pair", action="store_true")
    p.add_argument("--skip-verification-gradient", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_summary_df(traj_dir: Path, layer: int):
    import pandas as pd
    p_parquet = traj_dir / f"layer{layer}_summary.parquet"
    p_json = traj_dir / f"layer{layer}_summary.json"
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_json.exists():
        return pd.read_json(p_json)
    return None


def _group_comparison_records(df, model_suffix: str, layer: int) -> dict:
    """Run group comparisons on every group_col present."""
    out: dict = {"layer": int(layer), "n_chains": int(len(df)),
                 "comparisons": []}
    stat_cols = [c for c in ["arc_length", "mean_curvature",
                              "max_curvature", "cone_angle",
                              "return_rate", "n_transitions"]
                 if c in df.columns]

    # truncated vs not (the P0.4 stratification — primary group comparison)
    if "truncated" in df.columns and df["truncated"].nunique() >= 2:
        gc = compare_groups(df, group_col="truncated",
                            stat_cols=stat_cols,
                            residualise_on=["T"])
        gc["group_col"] = "truncated"
        out["comparisons"].append({
            "group_col": "truncated",
            "rationale": ("P0.4 stratification; primary cohort split for "
                          "build-now smoke (no success/failure labels yet)."),
            "rows": gc.to_dict(orient="records"),
        })

    # high-CBS vs low-CBS — synthesis §M4.1 (1) (proxy when annotations
    # available).
    if "n_tier3_sentences" in df.columns and df["n_tier3_sentences"].sum() > 0:
        df["high_cbs"] = (df["n_tier3_sentences"] >= 2).astype(int)
        if df["high_cbs"].nunique() >= 2:
            gc = compare_groups(df, group_col="high_cbs",
                                stat_cols=stat_cols,
                                residualise_on=["T"])
            out["comparisons"].append({
                "group_col": "high_cbs",
                "rationale": ">=2 tier-3 sentences vs zero (synthesis §M4.1.1)",
                "rows": gc.to_dict(orient="records"),
            })

    # long vs short (positional control: median split on T)
    if "T" in df.columns and df["T"].nunique() > 2:
        median_T = float(df["T"].median())
        df["long_chain"] = (df["T"] > median_T).astype(int)
        if df["long_chain"].nunique() >= 2:
            gc = compare_groups(df, group_col="long_chain",
                                stat_cols=[c for c in stat_cols if c != "T"],
                                residualise_on=[])
            out["comparisons"].append({
                "group_col": "long_chain",
                "rationale": (f"median-split on T (median={median_T}); "
                              "positional control."),
                "rows": gc.to_dict(orient="records"),
            })

    return out


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    traj_dir = args.trajectory_dir or (Path("results/trajectory") / args.model_suffix)
    out_dir = args.out_dir or traj_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    layers = [int(s) for s in args.layers.split(",")]

    # ── (1, 3) Group comparisons + per-sentence regression ────────────────
    all_group_records: list[dict] = []
    for layer in layers:
        df = _load_summary_df(traj_dir, layer)
        if df is None:
            logger.warning("no summary file for layer %d at %s", layer, traj_dir)
            continue
        rec = _group_comparison_records(df, args.model_suffix, layer)
        all_group_records.append(rec)

    group_path = out_dir / "group_comparisons.json"
    group_path.write_text(json.dumps({
        "model_suffix": args.model_suffix,
        "trajectory_dir": str(traj_dir),
        "layers": layers,
        "records": all_group_records,
        "note": ("smoke-only, not paper-grade" if "smoke" in args.model_suffix
                 else "real data"),
    }, indent=2, default=str))
    logger.info("wrote %s", group_path)

    # ── (2) Matched-pair analysis — gated on multi-seed re-gen + Phase 7 ──
    matched_pair_path = out_dir / "matched_pair_results.json"
    if (args.skip_matched_pair or not args.multi_seed_chains.exists()
            or not args.cbs_annotations.exists()):
        matched_pair_path.write_text(json.dumps({
            "status": "blocked",
            "blockers": [
                ("multi_seed_chains missing: "
                 f"{args.multi_seed_chains}; generate via "
                 "02_generate_chains.py --temperature 0.7 --seeds 0,...,19 "
                 "after M4.5 chain_gen change lands."),
                ("cbs_annotations missing: "
                 f"{args.cbs_annotations}; run 08_annotate_cbs.py with the "
                 "locked anchor block after P0.2."),
                "Phase 7 answer-checker labels (success/failure) required.",
            ],
            "skip_matched_pair_flag": bool(args.skip_matched_pair),
            "synthesis_reference": "§M4.4 / §M4.5",
            "what_will_run_when_unblocked": [
                ("build_matched_pairs(success_chains, failure_chains, "
                 "cbs_tier_filter=3, similarity_threshold=0.6)"),
                ("paired_geometric_tests on centroid_distance, OOS residual, "
                 "deduction-subspace projection (per layer)."),
                "Sensitivity sweep at Jaccard >= {0.5, 0.6, 0.7}.",
            ],
        }, indent=2))
        logger.info("matched-pair analysis blocked; wrote stub to %s",
                    matched_pair_path)
    else:
        # Full code path would load chains + activations and call
        # build_matched_pairs + paired_geometric_tests; deferred to run phase.
        matched_pair_path.write_text(json.dumps({
            "status": "ready",
            "synthesis_reference": "§M4.4 / §M4.5",
            "note": "implementation deferred to run phase",
        }, indent=2))

    # ── (4) Verification gradient — gated on Phase 7 answer-checker ───────
    vg_path = out_dir / "verification_gradient.json"
    if args.skip_verification_gradient or not args.cbs_annotations.exists():
        vg_path.write_text(json.dumps({
            "status": "blocked",
            "blockers": [
                "Phase 7 answer-checker labels (correct/incorrect) required.",
                ("cbs_annotations missing: "
                 f"{args.cbs_annotations}."),
            ],
            "synthesis_reference": "§M4.2",
            "what_will_run_when_unblocked": [
                ("verification_gradient(correct_acts, incorrect_acts, "
                 "cv_folds=5, seed=0)"),
                ("cosine(probe_weights, v_CBS) and "
                 "cosine(probe_weights, v_adding_knowledge_centroid) "
                 "filled in by the M5 ablation step."),
            ],
        }, indent=2))
        logger.info("verification-gradient blocked; wrote stub to %s", vg_path)

    # ── UMAP 2D projection of trajectories (smoke-runnable) ───────────────
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from umap import UMAP
        for layer in layers:
            df = _load_summary_df(traj_dir, layer)
            if df is None or len(df) < 4:
                continue
            # Project chain summary stats (a tiny "trajectory profile" vector
            # per chain) into 2D for visual inspection. Real run uses
            # per-sentence trajectory points; this is build-now-grade.
            feats = ["T", "arc_length", "mean_curvature", "max_curvature",
                     "cone_angle", "return_rate", "n_transitions"]
            feats = [c for c in feats if c in df.columns]
            if len(feats) < 2:
                continue
            X = df[feats].astype(float).fillna(0.0).values
            n = X.shape[0]
            n_neighbors = min(15, max(2, n - 1))
            umap = UMAP(n_components=2, random_state=args.seed,
                        n_neighbors=n_neighbors, min_dist=0.1)
            try:
                emb = umap.fit_transform(X)
            except Exception as exc:  # noqa: BLE001
                logger.warning("UMAP failed on layer %d (%s); skipping plot",
                               layer, exc)
                continue
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = (df["truncated"].astype(int).values
                      if "truncated" in df.columns else np.zeros(n))
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=colors, cmap="coolwarm",
                            s=18, edgecolor="k", linewidth=0.4)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title(f"trajectory summary UMAP, layer {layer}")
            cbar = fig.colorbar(sc, ax=ax, fraction=0.05)
            cbar.set_label("truncated")
            fig.tight_layout()
            fig.savefig(plots_dir / f"umap_trajectories_layer{layer}.png",
                        dpi=120)
            plt.close(fig)
    except ImportError as exc:
        logger.warning("UMAP plot skipped (%s)", exc)

    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
