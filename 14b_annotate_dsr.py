#!/usr/bin/env python3
"""Phase 14b — Multi-annotator DSR annotation + agreement (red-team F4).

Annotates the analysis-channel chains of a safety run for Deliberative Safety
Reasoning (spec_citation / adjudication / decision / harm_recognition) with a
panel of >= 3 LLM judges on the lab proxy, aggregates a character-level consensus,
and reports per-label inter-annotator agreement so the F4 gates can be applied:

    kappa < 0.40  -> that label's geometry is UNINTERPRETABLE (report negative only)
    0.40–0.60     -> geometry must REPLICATE across annotators before it is citable
    kappa >= 0.60 -> CITABLE

This is Gate B in safety_reasoning_extension.md §14.3: it runs on the existing API
budget, before any GPU is spent standing up gpt-oss, and decides which DSR labels
are admissible as a dependent variable at all.

Inputs:
    --chains data/chains_<safety-run>.json   (chain_gen format: {task_id, chain, ...})
Output:
    --out results/safety/dsr_annotated.json  (records + consensus + per-judge raw)
    --agreement-out results/safety/dsr_agreement.json  (per-label kappa + gate)

Build-now-run-later: with no proxy credentials this still imports and --dry-run
prints the panel and the chain count without calling out.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.chain_gen import load_chains
from src.config import provenance, backup_existing
from src.safety.annotate import (
    Judge, agreement_report, annotate_chains_dsr, default_judges,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def _parse_judges(spec: "list[str] | None") -> list[Judge]:
    """``--judges Name:model-id ...`` → panel; default panel if omitted."""
    if not spec:
        return default_judges()
    out = []
    for s in spec:
        if ":" not in s:
            raise SystemExit(f"--judges entry {s!r} must be NAME:MODEL_ID")
        name, model = s.split(":", 1)
        out.append(Judge(name.strip(), model.strip()))
    return out


def main():
    p = argparse.ArgumentParser(description="Phase 14b: multi-annotator DSR annotation")
    p.add_argument("--chains", required=True, help="chains JSON (chain_gen format)")
    p.add_argument("--judges", nargs="+", default=None,
                   help="NAME:MODEL_ID per judge (>=3 recommended); default panel if omitted")
    p.add_argument("--policy", default=None,
                   help="optional path to a safety-policy excerpt to ground the judges")
    p.add_argument("--out", default="results/safety/dsr_annotated.json")
    p.add_argument("--agreement-out", default="results/safety/dsr_agreement.json")
    p.add_argument("--limit", type=int, default=None, help="annotate at most N chains")
    p.add_argument("--dry-run", action="store_true",
                   help="print panel + chain count and exit (no proxy calls)")
    args = p.parse_args()

    judges = _parse_judges(args.judges)
    chains = load_chains(Path(args.chains))
    if args.limit:
        chains = chains[:args.limit]

    logger.info(f"DSR panel: {[j.name for j in judges]}")
    logger.info(f"Chains to annotate: {len(chains)}")
    if len(judges) < 3:
        logger.warning("fewer than 3 judges — F4 recommends >=3 to decorrelate annotator error")

    if args.dry_run:
        logger.info("dry-run: not calling the proxy")
        return

    policy_excerpt = Path(args.policy).read_text() if args.policy else None

    out = Path(args.out)
    annotated = annotate_chains_dsr(
        chains, judges, save_path=out, policy_excerpt=policy_excerpt,
    )

    agreement = agreement_report(annotated)
    report = {
        "agreement": agreement,
        "n_chains": len(annotated),
        "n_complete": sum(1 for a in annotated if a.get("dsr_complete")),
        "judges": [j.name for j in judges],
        "provenance": provenance(args=args),
    }
    agr_out = Path(args.agreement_out)
    agr_out.parent.mkdir(parents=True, exist_ok=True)
    backup_existing(agr_out)
    agr_out.write_text(json.dumps(report, indent=2))

    logger.info(f"Wrote {out} and {agr_out}")
    logger.info("Per-label DSR agreement (F4 gate):")
    for label, a in agreement.items():
        logger.info(f"  {label:<16s} kappa={a['kappa']}  gate={a['gate']}  "
                    f"(n_chars={a['n_chars']})")


if __name__ == "__main__":
    main()
