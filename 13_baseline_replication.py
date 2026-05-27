"""
Phase 13: Baseline-model replication of M1-M4 on Qwen-2.5-Math-1.5B (M6).

Orchestrates the CBS pipeline against the baseline model's activations, then
computes the cross-model comparison tables (deltas, bootstrap CIs,
trajectory Wasserstein, cross-model transfer accuracy).

Synthesis-plan reference: §M6. Prerequisites: Extension A pipeline
(Phase 2b chain generation, Phase 3 annotation, Phase 4 extraction on the
baseline). No smoke run at build time.

CLI
---
  --r1-results-dir      default results/cbs/R1-1.5B
  --base-results-dir    default results/cbs/QwenMath-1.5B
  --out-dir             default results/cbs/cross_model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--r1-results-dir",
                   default="results/cbs/R1-1.5B", type=Path)
    p.add_argument("--base-results-dir",
                   default="results/cbs/QwenMath-1.5B", type=Path)
    p.add_argument("--out-dir",
                   default="results/cbs/cross_model", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "13_baseline_replication is implemented at M6 (synthesis §M6.3). "
        "Scaffolded at P0.3."
    )


if __name__ == "__main__":
    sys.exit(main())
