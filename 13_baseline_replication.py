"""
Phase 13: Baseline-model replication of M1-M4 on Qwen-2.5-Math-1.5B (M6).

Orchestrates the CBS pipeline against the baseline model's activations, then
computes the cross-model comparison tables: per-statistic deltas + bootstrap
p, trajectory Wasserstein, and cross-model classifier transfer accuracy.

Synthesis-plan reference: §M6.

Prerequisites — Extension A pipeline on QwenMath-1.5B:
  Phase 2b chain generation (02b_generate_baseline_chains.py),
  Phase 3 annotation (03_annotate_chains.py --model-short QwenMath-1.5B),
  Phase 4 extraction (04_extract_activations.py --model qwen-math-1.5b),
  M1 CBS annotation (08_annotate_cbs.py --model-suffix QwenMath-1.5B),
  M2 geometry (09_cbs_geometry.py --model-suffix QwenMath-1.5B),
  M3 trajectory (10_trajectory_build.py --model-suffix QwenMath-1.5B),
  M4 group comparisons (11_trajectory_analysis.py).

No smoke run at build time — emits a blocked summary when any prerequisite
is missing, so the build phase completes cleanly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import SEED
from src.cbs.comparison import (
    cross_model_classifier,
    cross_model_compare,
    trajectory_wasserstein,
)

logger = logging.getLogger("13_baseline_replication")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--r1-results-dir", type=Path,
                   default=Path("results/cbs/R1-1.5B"))
    p.add_argument("--base-results-dir", type=Path,
                   default=Path("results/cbs/QwenMath-1.5B"))
    p.add_argument("--r1-trajectory-dir", type=Path,
                   default=Path("results/trajectory/R1-1.5B"))
    p.add_argument("--base-trajectory-dir", type=Path,
                   default=Path("results/trajectory/QwenMath-1.5B"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/cbs/cross_model"))
    p.add_argument("--tier3-rate-floor", type=float, default=0.03,
                   help="Below this baseline tier-3 rate, report a "
                        "structured null per synthesis §M6.4.")
    p.add_argument("--seed", type=int, default=SEED)  # config.SEED (was 0)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _check_prerequisites(args) -> list[str]:
    blockers: list[str] = []
    for p, name in [
        (args.r1_results_dir / "geometry_results.json",
         "R1 geometry_results.json"),
        (args.base_results_dir / "geometry_results.json",
         "Baseline geometry_results.json — Extension A pipeline blocker"),
    ]:
        if not p.exists():
            blockers.append(f"missing {name}: {p}")
    return blockers


def _baseline_tier3_rate(base_results: dict) -> float:
    n_per_tier = base_results.get("n_sentences_per_tier_total", {})
    total = sum(int(v) for v in n_per_tier.values())
    if total == 0:
        return 0.0
    return float(n_per_tier.get("3", 0)) / float(total)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    blockers = _check_prerequisites(args)
    if blockers:
        out = {
            "status": "blocked",
            "blockers": blockers,
            "synthesis_reference": "§M6.2",
            "what_will_run_when_unblocked": [
                "cross_model_compare on geometry_results.json pairs.",
                "trajectory_wasserstein on (success, failure) flag distributions.",
                ("cross_model_classifier on R1 success/failure features "
                 "tested against the baseline."),
                ("structured-null report if baseline tier-3 rate < "
                 f"{args.tier3_rate_floor:.0%}."),
            ],
            "needed_prereqs": [
                "Phase 2b: 02b_generate_baseline_chains.py",
                "Phase 3:  03_annotate_chains.py --model-short QwenMath-1.5B",
                "Phase 4:  04_extract_activations.py --model qwen-math-1.5b",
                "M1:       08_annotate_cbs.py --model-suffix QwenMath-1.5B",
                "M2:       09_cbs_geometry.py --model-suffix QwenMath-1.5B",
                "M3:       10_trajectory_build.py --model-suffix QwenMath-1.5B",
                "M4:       11_trajectory_analysis.py --model-suffix QwenMath-1.5B",
            ],
        }
        (args.out_dir / "cross_model_blocked.json").write_text(
            json.dumps(out, indent=2),
        )
        logger.warning("blocked; wrote %s",
                       args.out_dir / "cross_model_blocked.json")
        return 0

    # ── Cross-model geometry comparison ────────────────────────────────
    r1_geom = json.loads((args.r1_results_dir / "geometry_results.json").read_text())
    base_geom = json.loads((args.base_results_dir / "geometry_results.json").read_text())
    cmp_out = cross_model_compare(r1_geom, base_geom, seed=args.seed)
    (args.out_dir / "cross_model_geometry.json").write_text(
        json.dumps(cmp_out, indent=2),
    )
    logger.info("cross_model_compare: compared=%d missing=%d",
                cmp_out["n_compared"], cmp_out["n_missing"])

    # ── Structured-null check on baseline tier-3 rate ─────────────────
    base_tier3_rate = _baseline_tier3_rate(base_geom)
    if base_tier3_rate < args.tier3_rate_floor:
        (args.out_dir / "structured_null_baseline.json").write_text(json.dumps({
            "finding": "baseline tier-3 rate below floor",
            "baseline_tier3_rate": base_tier3_rate,
            "floor": args.tier3_rate_floor,
            "interpretation": (
                "Distillation appears to add tier-3 capacity; cross-model "
                "comparisons should report this as the result rather than "
                "force parallel statistical tests on incommensurable sample "
                "sizes (synthesis §M6.4)."
            ),
        }, indent=2))
        logger.info("structured-null: baseline tier-3 rate=%.3f < %.3f",
                    base_tier3_rate, args.tier3_rate_floor)

    # ── Trajectory Wasserstein (stub for build-now — needs (success, ─────
    # ── failure) split, which depends on Phase 7 answer-checker labels) ──
    (args.out_dir / "trajectory_wasserstein_pending.json").write_text(json.dumps({
        "status": "pending",
        "blockers": [
            "Needs (success, failure) split on each model.",
            "Phase 7 answer-checker output required.",
        ],
        "synthesis_reference": "§M6.3",
        "what_will_run_when_unblocked": (
            "trajectory_wasserstein(r1_success, r1_failure, "
            "base_success, base_failure)"
        ),
    }, indent=2))

    # ── Cross-model classifier (same blocker) ─────────────────────────
    (args.out_dir / "cross_model_classifier_pending.json").write_text(json.dumps({
        "status": "pending",
        "blockers": [
            "Needs (success, failure) labels for R1 train + Baseline test.",
            "Phase 7 answer-checker output required.",
        ],
        "synthesis_reference": "§M6.3",
    }, indent=2))

    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
