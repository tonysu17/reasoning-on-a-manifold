"""
Phase 8: CBS-tier + cross-domain annotation (M1).

Second-pass annotator over Phase-3 behaviour-labelled chains. Adds
`cbs_tier`, `cbs_knowledge_domain`, `cbs_cross_domain`, `cbs_rationale`,
`cbs_confidence` fields to every sentence whose behaviour is in
{adding-knowledge, deduction}.

Synthesis-plan reference: §M1.3.

CLI
---
  --in                   default data/annotated_R1-1.5B.json
  --out                  default data/chains_cbs_annotated_R1-1.5B.json
  --seed                 default 0
  --dual-seed-kappa      run at seed=0 AND seed=1, write both, compute kappa
  --pilot                stratified 100-sentence pilot only
  --max-workers          default 8
  --out-dir              results/cbs/{model}/ — for kappa + pilot artefacts
  --model-suffix         default R1-1.5B; used to build results dir name

Output (full run):
  --out JSON: original + cbs_* fields per sentence.

Output (--dual-seed-kappa):
  results/cbs/{model}/kappa_run1_run2.json

Output (--pilot):
  results/cbs/{model}/pilot_for_human_review.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--in", dest="in_path",
                   default="data/annotated_R1-1.5B.json", type=Path)
    p.add_argument("--out",
                   default="data/chains_cbs_annotated_R1-1.5B.json", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dual-seed-kappa", action="store_true")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--model-suffix", default="R1-1.5B")
    p.add_argument("--out-dir", default=None, type=lambda s: Path(s) if s else None,
                   help="defaults to results/cbs/{model-suffix}")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = args.out_dir or Path("results/cbs") / args.model_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "08_annotate_cbs is implemented at M1 (synthesis §M1.3). "
        "Scaffolded at P0.3."
    )


if __name__ == "__main__":
    sys.exit(main())
