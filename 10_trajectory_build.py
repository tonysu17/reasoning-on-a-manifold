"""
Phase 10: Per-chain trajectory construction at chosen layers (M3).

For each chain in the post-truncation cohort, assemble a ChainTrajectory at
each --layers entry; compute arc-length, curvature (arc-length-reparameterised
Frenet), subspace visits, cone angle; save per-chain JSON; aggregate per-
layer parquet.

Synthesis-plan reference: §M3.

CLI
---
  --activations-dir   default data/activations/R1-1.5B
  --cbs-annotations   default data/chains_cbs_annotated_R1-1.5B.json
  --layers            default 17,27
  --out-dir           default results/trajectory/{model-suffix}
  --model-suffix      default R1-1.5B
  --truncation-policy default stratify  (one of: regenerate, stratify, filter)

Output:
  results/trajectory/{model}/layer{N}/{chain_id}.json   (per-chain)
  results/trajectory/{model}/layer{N}_summary.parquet   (aggregate)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--activations-dir",
                   default="data/activations/R1-1.5B", type=Path)
    p.add_argument("--cbs-annotations",
                   default="data/chains_cbs_annotated_R1-1.5B.json", type=Path)
    p.add_argument("--layers", default="17,27")
    p.add_argument("--out-dir", default=None,
                   type=lambda s: Path(s) if s else None,
                   help="defaults to results/trajectory/{model-suffix}")
    p.add_argument("--model-suffix", default="R1-1.5B")
    p.add_argument("--truncation-policy", default="stratify",
                   choices=["regenerate", "stratify", "filter"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = args.out_dir or Path("results/trajectory") / args.model_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "10_trajectory_build is implemented at M3 (synthesis §M3.2). "
        "Scaffolded at P0.3."
    )


if __name__ == "__main__":
    sys.exit(main())
