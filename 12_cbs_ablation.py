"""
Phase 12: CBS steering ablation - causal experiment (M5).

Builds v_CBS from M1 outputs at the steering layer, validates it (hard
fail-stop), constructs textbook-solvable vs bridge-required task sets, then
generates ablated chains across conditions (baseline, v_cbs, v_random,
v_adding_knowledge) and ablation strengths. Annotates the ablated chains via
the M1 pipeline; reports tier distribution + accuracy + selectivity ratio.

Synthesis-plan reference: §M5.

CLI
---
  --v-cbs-source         default results/cbs/{model}/v_cbs.npy
  --validation-output    default results/cbs/{model}/v_cbs_validation.json
  --steering-layer       default 27 (also supports 17)
  --ablation-strengths   default 0,0.5,1.0,2.0
  --conditions           default baseline,v_cbs,v_random,v_adding_knowledge
  --seeds-per-task       default 5  (synthesis recommends trimming to 3)
  --model-suffix         default R1-1.5B
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--v-cbs-source", default=None,
                   type=lambda s: Path(s) if s else None)
    p.add_argument("--validation-output", default=None,
                   type=lambda s: Path(s) if s else None)
    p.add_argument("--steering-layer", type=int, default=27)
    p.add_argument("--ablation-strengths", default="0,0.5,1.0,2.0")
    p.add_argument("--conditions",
                   default="baseline,v_cbs,v_random,v_adding_knowledge")
    p.add_argument("--seeds-per-task", type=int, default=5)
    p.add_argument("--model-suffix", default="R1-1.5B")
    p.add_argument("--out-dir", default=None,
                   type=lambda s: Path(s) if s else None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    raise NotImplementedError(
        "12_cbs_ablation is implemented at M5 (synthesis §M5.2). "
        "Scaffolded at P0.3. Gated on Tony's go-ahead before implementation."
    )


if __name__ == "__main__":
    sys.exit(main())
