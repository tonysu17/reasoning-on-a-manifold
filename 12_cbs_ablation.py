"""
Phase 12: CBS steering ablation — causal experiment (M5).

Builds v_CBS from M1's CBS-tier annotations at the steering layer, validates
it (HARD FAIL-STOP — synthesis §M5.3), constructs textbook-solvable vs
bridge-required task sets, then generates ablated chains across conditions
{baseline, v_cbs, v_random, v_adding_knowledge} and ablation strengths
{0, 0.5, 1.0, 2.0}. Re-annotates the ablated chains via the M1 pipeline;
reports tier distribution + accuracy + selectivity ratio.

Synthesis-plan reference: §M5.

CLI matches synthesis §M5.2 verbatim. The intervention loop (model.generate
+ annotation + tabulation) lives at run-phase; this runner emits a CLI ready
to drive that loop and a `dry-run` validator path for build-now.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.config import SEED
from src.cbs.ablation import (
    FAILSTOP_COS_MAX,
    FAILSTOP_PROBE_ACC_MIN,
    FAILSTOP_PROBE_STD_MAX,
    build_v_cbs,
    construct_task_sets,
    selectivity_ratio,
    validate_v_cbs,
)

logger = logging.getLogger("12_cbs_ablation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--cbs-annotations", type=Path,
                   default=Path("data/chains_cbs_annotated_R1-1.5B.json"))
    p.add_argument("--activations-dir", type=Path,
                   default=Path("data/activations/R1-1.5B"))
    p.add_argument("--steering-layer", type=int, default=27,
                   help="Layer for v_CBS construction (27 default; 17 also "
                        "supported via this flag).")
    p.add_argument("--v-cbs-source", type=Path, default=None,
                   help="Optional precomputed v_cbs .npy. If absent, computed "
                        "in this run and saved to "
                        "results/cbs/{model-suffix}/v_cbs_layer{N}.npy.")
    p.add_argument("--validation-output", type=Path, default=None,
                   help="Path to write v_cbs validation JSON. Defaults to "
                        "results/cbs/{model-suffix}/v_cbs_validation.json.")
    p.add_argument("--ablation-strengths", default="0,0.5,1.0,2.0")
    p.add_argument("--conditions",
                   default="baseline,v_cbs,v_random,v_adding_knowledge")
    p.add_argument("--seeds-per-task", type=int, default=5)
    p.add_argument("--seed", type=int, default=SEED,
                   help="RNG seed for the v_cbs validation CV probe (config.SEED).")
    p.add_argument("--model-suffix", default="R1-1.5B")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--dry-run-validate-only", action="store_true",
                   help="Run only the v_cbs construction + validation step "
                        "(build-now). Skips actual generation.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_tier_acts(cbs_annotations_path: Path, activations_dir: Path,
                    layer: int) -> tuple:
    """Pull tier-3 and tier-1 activation rows from per-behaviour matrices,
    using the M3 row-index reconstruction (only target behaviours).

    Returns (t3, t1, t3_chain_ids, t1_chain_ids). The chain ids are passed as
    CV groups to validate_v_cbs so the hard fail-stop probe does not leak
    same-chain sentences across folds (see src/cbs/matching.cv_probe)."""
    from src.cbs.trajectory import (PHASE_4_BEHAVIOURS, build_row_index,
                                     load_layer_activations)
    with open(cbs_annotations_path) as f:
        chains = json.load(f)
    row_index = build_row_index(chains, target_behaviours=PHASE_4_BEHAVIOURS)
    activations = load_layer_activations(activations_dir, layer,
                                          behaviours=PHASE_4_BEHAVIOURS)
    t3_rows: list[np.ndarray] = []
    t1_rows: list[np.ndarray] = []
    t3_groups: list = []
    t1_groups: list = []
    for beh, items in row_index.items():
        # Look up per-row tier from the chains source.
        mat = activations.get(beh)
        if mat is None:
            continue
        # Build a (chain_id, span_idx) -> tier table.
        tier_by_key = {}
        for chain in chains:
            cid = chain.get("task_id", "")
            for i, span in enumerate(chain.get("annotations", []) or []):
                if span.get("label") == beh:
                    tier_by_key[(cid, i)] = int(span.get("cbs_tier", 0))
        for row, key in enumerate(items):
            tier = tier_by_key.get(key, 0)
            cid = key[0] if isinstance(key, (tuple, list)) else key
            if tier == 3 and row < mat.shape[0]:
                t3_rows.append(mat[row]); t3_groups.append(cid)
            elif tier == 1 and row < mat.shape[0]:
                t1_rows.append(mat[row]); t1_groups.append(cid)
    t3 = np.stack(t3_rows) if t3_rows else np.zeros((0, 0))
    t1 = np.stack(t1_rows) if t1_rows else np.zeros((0, 0))
    return t3, t1, np.array(t3_groups), np.array(t1_groups)


def _load_adding_knowledge_centroid(activations_dir: Path, layer: int) -> np.ndarray:
    fp = activations_dir / f"adding-knowledge_layer{layer}.npy"
    if not fp.exists():
        raise FileNotFoundError(
            f"adding-knowledge centroid requires {fp}; load_layer_activations "
            f"path failed. Re-run Phase 4 with adding-knowledge behaviour."
        )
    X = np.load(fp)
    centroid = X.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    return centroid


def _failstop_template(report: dict, layer: int) -> str:
    return (
        "# FAILSTOP — M5 v_CBS validation\n\n"
        f"Synthesis-plan reference: §M5.3 hard fail-stop.\n\n"
        f"## Failing values\n\n"
        f"- |cos(v_cbs, v_adding_knowledge_centroid)| = "
        f"{abs(report['cosine_sim_with_knowledge_centroid']):.3f} "
        f"(threshold < {FAILSTOP_COS_MAX})\n"
        f"- cv_probe_accuracy_mean = {report['cv_probe_accuracy_mean']:.3f} "
        f"(threshold >= {FAILSTOP_PROBE_ACC_MIN})\n"
        f"- cv_probe_accuracy_std = {report['cv_probe_accuracy_std']:.3f} "
        f"(threshold <= {FAILSTOP_PROBE_STD_MAX})\n"
        f"- n_tier3 = {report['n_tier3']}, n_tier1 = {report['n_tier1']}\n"
        f"- layer = {layer}\n\n"
        f"## Failures\n\n"
        + "\n".join(f"- {f}" for f in report["failures"]) + "\n\n"
        f"## Three options for the human\n\n"
        "1. Re-curate CBS anchors (P0.2) and re-annotate; this is the most "
        "common cause of low probe accuracy.\n"
        "2. Switch the steering layer (try layer 17 if 27 failed, or vice "
        "versa); subspace correlations differ across depth.\n"
        "3. Declare v_CBS unsuited at the 1.5B scale and either (a) scale up "
        "to R1-Distill-7B for the ablation step alone, or (b) drop M5 from "
        "the paper and report the failure as a result.\n\n"
        "## Raw report\n\n"
        f"```json\n{json.dumps(report, indent=2)}\n```\n"
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    out_dir = args.out_dir or Path("results/cbs") / args.model_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    layer = args.steering_layer
    v_path = (args.v_cbs_source
              or out_dir / f"v_cbs_layer{layer}.npy")
    val_path = (args.validation_output
                or out_dir / f"v_cbs_validation_layer{layer}.json")

    # ── Step 1: build v_cbs (or load) ──────────────────────────────────
    try:
        if v_path.exists() and args.v_cbs_source is not None:
            v_cbs_preloaded: np.ndarray | None = np.load(v_path)
            logger.info("loaded v_cbs from %s (shape %s)",
                        v_path, v_cbs_preloaded.shape)
        else:
            v_cbs_preloaded = None
        t3, t1, t3_groups, t1_groups = _load_tier_acts(args.cbs_annotations,
                                                        args.activations_dir, layer)
    except FileNotFoundError as exc:
        logger.warning("v_cbs construction blocked: %s", exc)
        (out_dir / "v_cbs_construction_blocked.json").write_text(json.dumps({
            "status": "blocked",
            "blockers": [
                f"CBS annotations missing or sparse at "
                f"{args.cbs_annotations}",
                "P0.2 anchor lock + 08_annotate_cbs full run required",
                f"or: missing activation file ({exc})",
            ],
            "synthesis_reference": "§M5.2",
        }, indent=2))
        return 0

    if v_cbs_preloaded is None:
        if t3.shape[0] < 5 or t1.shape[0] < 5:
            logger.warning("not enough tier-3 (%d) or tier-1 (%d) activations; "
                           "needs CBS-annotated full corpus.",
                           t3.shape[0], t1.shape[0])
            (out_dir / "v_cbs_construction_blocked.json").write_text(json.dumps({
                "status": "blocked",
                "n_tier3": int(t3.shape[0]),
                "n_tier1": int(t1.shape[0]),
                "blockers": [
                    f"CBS annotations missing or sparse at "
                    f"{args.cbs_annotations}",
                    "P0.2 anchor lock + 08_annotate_cbs full run required",
                ],
                "synthesis_reference": "§M5.2",
            }, indent=2))
            return 0
        v_cbs = build_v_cbs(t3, t1)
        np.save(v_path, v_cbs)
        logger.info("wrote v_cbs to %s", v_path)
    else:
        v_cbs = v_cbs_preloaded

    # ── Step 2: load adding-knowledge centroid for validation ──────────
    try:
        centroid = _load_adding_knowledge_centroid(args.activations_dir, layer)
    except FileNotFoundError as exc:
        logger.error("validation aborted: %s", exc)
        return 2

    # ── Step 3: validate (HARD FAIL-STOP) ──────────────────────────────
    report = validate_v_cbs(v_cbs, centroid, t3, t1, cv_folds=5, seed=args.seed,
                            tier3_groups=t3_groups, tier1_groups=t1_groups)
    val_path.write_text(json.dumps(report, indent=2,
                                    default=lambda o: o.tolist()
                                    if hasattr(o, "tolist") else str(o)))
    logger.info("validation: cos=%.3f cv_mean=%.3f cv_std=%.3f passes=%s",
                report["cosine_sim_with_knowledge_centroid"],
                report["cv_probe_accuracy_mean"],
                report["cv_probe_accuracy_std"], report["passes"])

    if not report["passes"]:
        failstop_path = out_dir / "FAILSTOP_M5.md"
        failstop_path.write_text(_failstop_template(report, layer))
        logger.error("HARD FAIL-STOP — wrote %s. Halting; do not proceed.",
                     failstop_path)
        return 1

    if args.dry_run_validate_only:
        logger.info("dry-run: validation passed; intervention step skipped.")
        return 0

    # ── Step 4: construct task sets ────────────────────────────────────
    with open(args.cbs_annotations) as f:
        chains = json.load(f)
    try:
        sets = construct_task_sets(chains)
    except RuntimeError as exc:
        logger.error("construct_task_sets failed: %s", exc)
        (out_dir / "task_sets_blocked.json").write_text(json.dumps({
            "status": "blocked", "error": str(exc),
            "synthesis_reference": "§M5.2",
        }, indent=2))
        return 1
    logger.info("task sets ready: A=%d, B=%d",
                len(sets["set_a_textbook"]), len(sets["set_b_bridge"]))

    # ── Step 5: generate + annotate ablated chains ────────────────────
    # The intervention loop loads the HF model, applies CBSAblationModel
    # per condition+alpha+seed, annotates the result via the M1 pipeline,
    # and tabulates tier distribution + accuracy + selectivity ratio. This
    # is a cluster-GPU step (~25h) and stays in run-phase.
    (out_dir / "ablation_run_pending.json").write_text(json.dumps({
        "status": "validation_passed_intervention_pending",
        "v_cbs_path": str(v_path),
        "validation_path": str(val_path),
        "set_a_size": len(sets["set_a_textbook"]),
        "set_b_size": len(sets["set_b_bridge"]),
        "conditions": args.conditions.split(","),
        "ablation_strengths": [float(s)
                                for s in args.ablation_strengths.split(",")],
        "seeds_per_task": int(args.seeds_per_task),
        "synthesis_reference": "§M5.2 step 4-5",
        "next_command": (
            "python -c 'from src.cbs.ablation import CBSAblationModel; "
            "...intervention loop here, ~25h cluster GPU...'"
        ),
    }, indent=2))
    logger.info("validation passed; intervention loop stays in run-phase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
