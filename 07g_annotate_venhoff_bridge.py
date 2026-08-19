#!/usr/bin/env python3
"""Guarded annotation and analysis for ``07f_venhoff_fixed_bridge.py``.

This is intentionally a separate stage from generation.  Preparing and
analysing are API-free.  The paid subcommand requires all of:

* the literal ``annotate`` subcommand;
* the exact SHA-256 of the immutable guard manifest; and
* ``--confirm-spend-ceiling-usd 15``.

Examples (run from the repository root)::

    python 07g_annotate_venhoff_bridge.py prepare \
      --eval-dir results/eval/R1-1.5B__venhoff_constant_all4_user_publishednorm

    python 07g_annotate_venhoff_bridge.py annotate \
      --eval-dir results/eval/R1-1.5B__venhoff_constant_all4_user_publishednorm \
      --guard-sha256 <digest printed by prepare> \
      --confirm-spend-ceiling-usd 15

    python 07g_annotate_venhoff_bridge.py analyse \
      --eval-dir results/eval/R1-1.5B__venhoff_constant_all4_user_publishednorm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.annotation_budget import AnnotationAttemptLimitError
from src.venhoff_bridge_eval import (
    SPEND_CEILING_USD,
    format_report,
    prepare_annotation_manifest,
    run_analysis,
    run_annotation,
)


ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("venhoff_bridge_annotation")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guarded all-four annotation/evaluation for the Venhoff fixed bridge"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser(
        "prepare", help="validate 250 generations and create an API-free spend manifest"
    )
    prepare.add_argument("--eval-dir", type=Path, required=True)

    annotate = sub.add_parser(
        "annotate", help=(
            "run/resume paid annotation under the repository-global immutable USD 15 guard"
        )
    )
    annotate.add_argument("--eval-dir", type=Path, required=True)
    annotate.add_argument(
        "--guard-sha256",
        required=True,
        help="exact digest printed by the prepare subcommand",
    )
    annotate.add_argument(
        "--confirm-spend-ceiling-usd",
        type=float,
        required=True,
        help="must be exactly 15; prevents an accidental paid invocation",
    )

    analyse = sub.add_parser(
        "analyse", help="API-free token scoring, pairing, bootstrap CIs, and Holm tests"
    )
    analyse.add_argument("--eval-dir", type=Path, required=True)
    analyse.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="pinned Qwen tokenizer snapshot (normally read from generation provenance)",
    )
    analyse.add_argument("--n-resamples", type=int, default=10_000)
    analyse.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    eval_dir = args.eval_dir
    if not eval_dir.is_absolute():
        eval_dir = ROOT / eval_dir

    if args.command == "prepare":
        prepared = prepare_annotation_manifest(eval_dir, root=ROOT)
        print(json.dumps(prepared, indent=2))
        print(
            "\nNo API call was made. To run the paid stage, pass this exact "
            f"manifest SHA-256 and confirm the ${SPEND_CEILING_USD:.0f} ceiling."
        )
        return 0

    if args.command == "annotate":
        if abs(args.confirm_spend_ceiling_usd - SPEND_CEILING_USD) > 1e-9:
            raise SystemExit(
                "--confirm-spend-ceiling-usd must be exactly "
                f"{SPEND_CEILING_USD:g}; no API call made"
            )
        if not (os.environ.get("CLAUDE_PROXY_URL") and os.environ.get("CLAUDE_PROXY_KEY")):
            raise SystemExit(
                "CLAUDE_PROXY_URL/CLAUDE_PROXY_KEY are not set; no API call made"
            )
        try:
            status = run_annotation(
                eval_dir,
                root=ROOT,
                manifest_sha256=args.guard_sha256,
            )
        except AnnotationAttemptLimitError as exc:
            raise SystemExit(f"annotation spend guard refused execution: {exc}") from exc
        print(json.dumps(status, indent=2))
        return 0

    report = run_analysis(
        eval_dir,
        tokenizer_path=args.tokenizer_path,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    print(format_report(report))
    print(f"\nReport -> {eval_dir / 'venhoff_bridge_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
