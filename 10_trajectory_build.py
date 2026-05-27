"""
Phase 10: Per-chain trajectory construction at chosen layers (M3).

For each chain in the post-truncation cohort, assemble a ChainTrajectory at
each --layers entry; compute arc length, curvature (arc-length-reparameterised
Frenet), subspace dynamics, cone angle; save per-chain JSON; aggregate to a
per-layer parquet.

Synthesis-plan reference: §M3.

CLI
---
  --activations-dir   default data/activations/R1-1.5B
  --cbs-annotations   default data/chains_cbs_annotated_R1-1.5B.json
  --raw-chains        default data/annotated_R1-1.5B.json  (fallback when
                       CBS annotations are not yet available, build-now)
  --layers            default 17,27
  --out-dir           default results/trajectory/{model-suffix}
  --model-suffix      default R1-1.5B
  --truncation-policy default stratify   (one of: regenerate, stratify, filter)
  --max-chains        limit for smoke runs (default: 0 = all)
  --target-behaviours default backtracking,uncertainty-estimation,
                             example-testing,adding-knowledge
  --skip-subspace-visits  do not compute subspace dynamics (faster smoke)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.cbs.trajectory import (
    PHASE_4_BEHAVIOURS,
    arc_length_sequence,
    build_row_index,
    build_trajectory,
    cross_subspace_returns,
    curvature_sequence,
    load_layer_activations,
    total_arc_length,
    trajectory_cone_angle,
)

logger = logging.getLogger("10_trajectory_build")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--activations-dir",
                   default=Path("data/activations/R1-1.5B"), type=Path)
    p.add_argument("--cbs-annotations",
                   default=Path("data/chains_cbs_annotated_R1-1.5B.json"),
                   type=Path)
    p.add_argument("--raw-chains",
                   default=Path("data/annotated_R1-1.5B.json"), type=Path,
                   help="Phase 3 annotated chains; used when CBS annotations "
                        "absent (build-now smoke).")
    p.add_argument("--layers", default="17,27")
    p.add_argument("--out-dir", default=None,
                   type=lambda s: Path(s) if s else None)
    p.add_argument("--model-suffix", default="R1-1.5B")
    p.add_argument("--truncation-policy", default="stratify",
                   choices=["regenerate", "stratify", "filter"])
    p.add_argument("--target-behaviours",
                   default=",".join(PHASE_4_BEHAVIOURS))
    p.add_argument("--max-chains", type=int, default=0)
    p.add_argument("--skip-subspace-visits", action="store_true")
    p.add_argument("--top-pcs-per-behaviour", type=int, default=10)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _build_per_behaviour_subspaces(
    activations: dict[str, np.ndarray],
    k: int = 10,
) -> dict[str, np.ndarray]:
    """Top-k PC subspaces per behaviour from the per-layer activation matrices."""
    out: dict[str, np.ndarray] = {}
    for beh, X in activations.items():
        if X.shape[0] < 2:
            continue
        Xc = X - X.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        n_keep = min(k, Vt.shape[0])
        out[beh] = Vt[:n_keep].T          # (d, n_keep)
    return out


def _summarise_trajectory(traj, *, layer: int,
                           subspaces: dict[str, np.ndarray],
                           skip_subspace: bool) -> dict:
    if traj.T == 0:
        return {
            "chain_id": traj.chain_id, "layer": layer, "T": 0,
            "arc_length": 0.0, "mean_curvature": None, "max_curvature": None,
            "return_rate": None, "n_transitions": None,
            "cone_angle": None,
            "n_tier3_sentences": 0, "n_cross_domain_sentences": 0,
            "truncated": traj.truncated,
            "n_dropped_sentences": "see _build_trajectory log",
        }
    kappa = curvature_sequence(traj)
    arc = total_arc_length(traj)
    cone = trajectory_cone_angle(traj)
    visit = (cross_subspace_returns(traj, subspaces)
             if not skip_subspace and subspaces else
             {"n_transitions": None, "return_rate": None,
              "visit_sequence": [], "transition_matrix": {}})
    n_tier3 = int(sum(1 for t in traj.cbs_tiers if t == 3))
    n_cd = int(sum(1 for c in traj.cross_domain if c is True))
    return {
        "chain_id": traj.chain_id,
        "layer": layer,
        "T": traj.T,
        "arc_length": float(arc),
        "mean_curvature": (float(np.nanmean(kappa)) if traj.T >= 3 else None),
        "max_curvature": (float(np.nanmax(kappa)) if traj.T >= 3 else None),
        "n_transitions": visit["n_transitions"],
        "return_rate": visit["return_rate"],
        "cone_angle": float(cone),
        "n_tier3_sentences": n_tier3,
        "n_cross_domain_sentences": n_cd,
        "truncated": bool(traj.truncated),
        "behaviours_visited": list(set(traj.behaviours)),
    }


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    out_dir = args.out_dir or Path("results/trajectory") / args.model_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = [int(s) for s in args.layers.split(",")]
    target_behaviours = tuple(s.strip()
                              for s in args.target_behaviours.split(","))

    # Prefer CBS-annotated chains; fall back to raw Phase 3 chains if absent.
    chains_path = args.cbs_annotations if args.cbs_annotations.exists() \
                  else args.raw_chains
    if not chains_path.exists():
        logger.error("no chain source found at %s or %s",
                     args.cbs_annotations, args.raw_chains)
        return 2
    logger.info("loading chains from %s", chains_path)
    with open(chains_path) as f:
        chains = json.load(f)
    if args.max_chains > 0:
        chains = chains[: args.max_chains]

    row_index = build_row_index(chains, target_behaviours=target_behaviours)
    logger.info("row index sizes: %s",
                {b: len(v) for b, v in row_index.items()})

    all_summaries: list[dict] = []
    for layer in layers:
        layer_dir = out_dir / f"layer{layer}"
        layer_dir.mkdir(exist_ok=True)
        activations = load_layer_activations(args.activations_dir, layer,
                                             behaviours=target_behaviours)
        if not activations:
            logger.warning("no activations found at layer %d; skipping", layer)
            continue
        subspaces = _build_per_behaviour_subspaces(
            activations, k=args.top_pcs_per_behaviour,
        )

        n_processed = 0
        for chain in chains:
            traj = build_trajectory(
                chain, activations_dir=args.activations_dir, layer=layer,
                row_index=row_index, activations=activations,
                target_behaviours=target_behaviours,
            )
            if traj.T == 0:
                continue
            n_processed += 1

            # Per-chain JSON
            summary = _summarise_trajectory(
                traj, layer=layer, subspaces=subspaces,
                skip_subspace=args.skip_subspace_visits,
            )
            per_chain = layer_dir / f"{traj.chain_id}.json"
            per_chain.write_text(json.dumps({
                **summary,
                "sentence_ids": traj.sentence_ids,
                "behaviours": traj.behaviours,
                "cbs_tiers": traj.cbs_tiers,
                "cross_domain": traj.cross_domain,
                "arc_length_sequence": arc_length_sequence(traj).tolist(),
                "curvature_sequence": [
                    None if np.isnan(v) else float(v)
                    for v in curvature_sequence(traj)
                ],
            }, indent=2))
            all_summaries.append(summary)

        logger.info("layer=%d: built %d trajectories",
                    layer, n_processed)

    # Aggregate parquet per layer.
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas unavailable; skipping parquet aggregation")
        return 0

    df = pd.DataFrame(all_summaries)
    df["truncation_policy"] = args.truncation_policy
    for layer in layers:
        sub = df[df["layer"] == layer]
        if sub.empty:
            continue
        parquet = out_dir / f"layer{layer}_summary.parquet"
        try:
            sub.to_parquet(parquet, index=False)
            logger.info("wrote %s (%d rows)", parquet, len(sub))
        except (ImportError, ValueError) as exc:
            # Fall back to JSON if pyarrow is unavailable.
            json_alt = out_dir / f"layer{layer}_summary.json"
            sub.to_json(json_alt, orient="records", indent=2)
            logger.warning("parquet write failed (%s); wrote %s instead",
                           exc, json_alt)

    # Write run metadata
    (out_dir / "run_metadata.json").write_text(json.dumps({
        "activations_dir": str(args.activations_dir),
        "chains_source": str(chains_path),
        "layers": layers,
        "target_behaviours": list(target_behaviours),
        "truncation_policy": args.truncation_policy,
        "n_chains_input": len(chains),
        "n_trajectories_built": len(all_summaries),
        "row_index_sizes": {b: len(v) for b, v in row_index.items()},
        "note": ("smoke-only, not paper-grade — labels from synthetic / Phase 3 "
                 "only, CBS fields absent until P0.2 lock"),
    }, indent=2))
    logger.info("done — total trajectories: %d", len(all_summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
