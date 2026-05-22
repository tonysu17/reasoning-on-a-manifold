#!/usr/bin/env python3
"""
Phase 3 — Behavioural annotation via Claude Sonnet 4.5 (AWS proxy).

Annotates every sentence in every reasoning chain using the verbatim Venhoff
et al. prompt (arXiv:2506.18167 Appendix A).

Note: Venhoff used GPT-4o; GPT-4o-2024-11-20 is unavailable on the AWS proxy
used in this project. Claude Sonnet 4.5 is used instead (noted in methods).

Supports checkpointing — safe to interrupt and resume by re-running the same
command.

Requirements:
    export CLAUDE_PROXY_URL=https://...
    export CLAUDE_PROXY_KEY=rp_...
    Input: data/chains_pilot.json  (--pilot) or data/chains_<model>.json

Runtime:  ~1–2 min/chain  |  pilot 20 chains ~30 min  |  full 1000 ~25 h
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.annotation import (
    VENHOFF_FRACTIONS,
    annotate_chains,
    behaviour_counts,
    load_annotated,
)
from src.chain_gen import load_chains

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_LABELS = {"backtracking", "uncertainty-estimation",
                 "example-testing", "adding-knowledge"}


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Behavioural annotation")
    parser.add_argument("--model-short", default="R1-1.5B",
                        help="Model short name matching chains file (default: R1-1.5B)")
    parser.add_argument("--pilot", action="store_true",
                        help="Annotate pilot chains (data/chains_pilot.json)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: annotate first 3 chains only")
    parser.add_argument("--kill-after", type=int, default=None, metavar="N",
                        help="Exit after annotating N chains (for resume-logic smoke test)")
    args = parser.parse_args()

    proxy_url = os.environ.get("CLAUDE_PROXY_URL")
    proxy_key = os.environ.get("CLAUDE_PROXY_KEY")
    if not proxy_url or not proxy_key:
        logger.error(
            "Missing environment variables.\n"
            "  export CLAUDE_PROXY_URL=https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke\n"
            "  export CLAUDE_PROXY_KEY=rp_..."
        )
        sys.exit(1)

    if args.pilot:
        chains_path = Path("data/chains_pilot.json")
        out_path    = Path("data/annotated_pilot.json")
    else:
        chains_path = Path(f"data/chains_{args.model_short}.json")
        out_path    = Path(f"data/annotated_{args.model_short}.json")

    if not chains_path.exists():
        logger.error(f"Chains not found at {chains_path}. Run Phase 2 first.")
        sys.exit(1)

    chains = load_chains(chains_path)

    if args.smoke:
        chains = chains[:3]
        logger.info(f"SMOKE TEST: annotating {len(chains)} chains")
    elif args.kill_after:
        logger.info(f"KILL-AFTER={args.kill_after}: resume smoke test mode")
    elif args.pilot:
        logger.info(f"PILOT: annotating {len(chains)} chains → {out_path}")
    else:
        logger.info(f"Annotating {len(chains)} chains → {out_path}")

    # Early-exit check — all chains fully complete?
    if out_path.exists():
        existing = load_annotated(out_path)
        n_complete = sum(1 for c in existing if c.get("annotation_complete", False))
        if n_complete >= len(chains):
            logger.info(f"Already complete at {out_path}")
            _print_report(existing, out_path, pilot=args.pilot)
            return

    annotated = annotate_chains(
        chains,
        save_path=out_path,
        checkpoint_every=5 if (args.pilot or args.smoke) else 25,
        proxy_url=proxy_url,
        proxy_key=proxy_key,
        kill_after=args.kill_after,
    )

    _print_report(annotated, out_path, pilot=args.pilot)


def _print_report(annotated: list[dict], out_path: Path, pilot: bool = False) -> None:
    counts = behaviour_counts(annotated)
    total  = sum(counts.values())

    print(f"\n{'='*70}")
    print(f"Annotation complete: {len(annotated)} chains, {total} sentences")
    print(f"Saved → {out_path}")
    print(f"\n{'Label':<28s} {'Count':>6s} {'Observed':>10s} {'Venhoff':>9s} {'Ratio':>7s} {'Status':>8s}")
    print("-" * 70)

    order = ["backtracking", "uncertainty-estimation", "example-testing",
             "adding-knowledge", "initializing", "deduction"]

    all_ok = True
    for label in order:
        n        = counts.get(label, 0)
        obs      = n / total if total else 0
        expected = VENHOFF_FRACTIONS.get(label)
        is_tgt   = label in TARGET_LABELS

        if expected and pilot:
            ratio   = obs / expected
            within  = 0.5 <= ratio <= 1.5
            status  = "ok" if within else "!!"
            ratio_s = f"{ratio:.2f}x"
            if not within and is_tgt:
                all_ok = False
        elif expected:
            diff    = abs(obs - expected)
            within  = diff <= 0.10
            status  = "ok" if within else "!!"
            ratio_s = f"{obs - expected:+.1%}"
            if not within and is_tgt:
                all_ok = False
        else:
            status  = "  —"
            ratio_s = "  —"

        exp_s = f"{expected:.0%}" if expected else "  —"
        print(f"{label:<28s} {n:>6d} {obs:>9.1%} {exp_s:>9s} {ratio_s:>7s} {status:>8s}")

    print()
    if pilot:
        print("Pilot tolerance: ±50% of Venhoff value (small N expected).")
        if all_ok:
            print("Pilot annotation PASS ✓ — fractions consistent with Venhoff Figure 2.")
            print("Next: python 03_annotate_chains.py  (full 1000-chain run)")
        else:
            print("WARNING: some fractions outside ±50% — check prompt fidelity before scaling.")
    else:
        below = [l for l in TARGET_LABELS if counts.get(l, 0) < 100]
        if below:
            print(f"WARNING: {below} have fewer than 100 instances.")
        else:
            print("All target behaviours ≥ 100 instances. ✓")
            print("Next: python 04_extract_activations.py")


if __name__ == "__main__":
    main()
