"""
Phase 11: Trajectory analysis - group comparisons + matched pairs + verification
gradient (M4).

Reads layer-summary parquets from Phase 10 + CBS-annotated chains; runs:
  1. Group comparisons (residualised on chain length).
  2. Matched-pair tier-3 success-vs-failure analysis (Jaccard >= 0.6).
  3. Verification-gradient probe (5-fold CV).
  4. UMAP 2D trajectory projection.

Synthesis-plan reference: §M4.

CLI
---
  --trajectory-summary   default results/trajectory/{model}/layer{N}_summary.parquet
  --cbs-annotations      default data/chains_cbs_annotated_R1-1.5B.json
  --multi-seed-chains    default data/chains_R1-1.5B_multiseed.json
  --out-dir              default results/trajectory/{model-suffix}
  --model-suffix         default R1-1.5B
  --jaccard-threshold    default 0.6  (sensitivity sweep adds 0.5, 0.7)
  --skip-matched-pair    skip matched-pair analysis if multi-seed chains absent
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--trajectory-summary", default=None,
                   type=lambda s: Path(s) if s else None,
                   help="defaults to results/trajectory/{model}/layer{N}_summary.parquet")
    p.add_argument("--cbs-annotations",
                   default="data/chains_cbs_annotated_R1-1.5B.json", type=Path)
    p.add_argument("--multi-seed-chains",
                   default="data/chains_R1-1.5B_multiseed.json", type=Path)
    p.add_argument("--out-dir", default=None,
                   type=lambda s: Path(s) if s else None)
    p.add_argument("--model-suffix", default="R1-1.5B")
    p.add_argument("--jaccard-threshold", type=float, default=0.6)
    p.add_argument("--skip-matched-pair", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = args.out_dir or Path("results/trajectory") / args.model_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "11_trajectory_analysis is implemented at M4 (synthesis §M4.2). "
        "Scaffolded at P0.3."
    )


if __name__ == "__main__":
    sys.exit(main())
