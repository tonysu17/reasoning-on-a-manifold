"""
Phase 9: Per-sentence geometric tests for CBS tier + cross-domain (M2).

Runs the four geometric statistics (centroid distance, OOS residual, local
intrinsic dim, principal angles) per layer x behaviour over a set of
activation matrices. Each statistic is tested under both the CBS-tier label
(Jonckheere-Terpstra) and the cross-domain binary label (Wilcoxon), with
Holm correction across (layers x behaviours x statistics x labels).

Synthesis-plan reference: §M2.

CLI
---
  --activations-dir       default data/activations/R1-1.5B
  --cbs-annotations       default data/chains_cbs_annotated_R1-1.5B.json
  --out-dir               default results/cbs/{model-suffix}
  --model-suffix          default R1-1.5B; smoke runs use R1-1.5B-smoke
  --layers                default 3,7,10,14,17,21,24,27
  --behaviours            default adding-knowledge,deduction
  --n-bootstrap           default 1000
  --variance-thresholds   default 0.90,0.95,0.99

Output:
  results/cbs/{model}/geometry_results.json
  results/cbs/{model}/plots/effect_size_vs_layer.png
  results/cbs/{model}/plots/principal_angle_heatmap_layer{N}.png
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
    p.add_argument("--out-dir", default=None,
                   type=lambda s: Path(s) if s else None,
                   help="defaults to results/cbs/{model-suffix}")
    p.add_argument("--model-suffix", default="R1-1.5B")
    p.add_argument("--layers", default="3,7,10,14,17,21,24,27")
    p.add_argument("--behaviours", default="adding-knowledge,deduction")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--variance-thresholds", default="0.90,0.95,0.99")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = args.out_dir or Path("results/cbs") / args.model_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "09_cbs_geometry is implemented at M2 (synthesis §M2.2). "
        "Scaffolded at P0.3."
    )


if __name__ == "__main__":
    sys.exit(main())
