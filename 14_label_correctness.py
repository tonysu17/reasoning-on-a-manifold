#!/usr/bin/env python3
"""
14_label_correctness.py — Generate reasoning-correctness labels (LLM judge).

Selects a category/difficulty-balanced pilot of chains and judges each one's
final answer with Claude Sonnet on the AWS Bedrock proxy. This is the only step
that spends API budget; it reads CLAUDE_PROXY_URL / CLAUDE_PROXY_KEY from the
environment (the same vars Phase 1/3 use). Use --dry-run to inspect the
selection and a sample judge prompt WITHOUT calling the API.

Examples
--------
  # inspect the pilot + a sample prompt, no API:
  python3 14_label_correctness.py --dry-run

  # run the ~200-chain pilot (needs CLAUDE_PROXY_URL / CLAUDE_PROXY_KEY exported):
  python3 14_label_correctness.py --pilot 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import backup_existing, provenance  # noqa: E402
from src.predict.labels import (  # noqa: E402
    build_judge_prompt,
    generate_correctness_labels,
    select_balanced_pilot,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chains", default="data/annotated_R1-1.5B.json")
    ap.add_argument("--tasks", default="data/tasks_final.json")
    ap.add_argument("--pilot", type=int, default=200, help="target #chains to label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/correctness_R1-1.5B_pilot.json")
    ap.add_argument("--allow-truncated", action="store_true",
                    help="do not prefer complete chains in selection")
    ap.add_argument("--max-chains", type=int, default=None,
                    help="hard cap on NEW judge calls this run (resume-safe)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print selection + a sample prompt; no API calls")
    args = ap.parse_args()

    chains = json.load(open(args.chains))
    tasks = json.load(open(args.tasks))
    selected, meta = select_balanced_pilot(
        chains, tasks, n=args.pilot, prefer_complete=not args.allow_truncated,
        seed=args.seed,
    )

    by_cat = Counter(m["category"] for m in meta.values())
    by_diff = Counter(m["difficulty"] for m in meta.values())
    n_trunc = sum(1 for m in meta.values() if m["truncated"])
    print(f"selected {len(selected)} chains")
    print(f"  by difficulty: {dict(by_diff)}")
    print(f"  by category  : {dict(by_cat)}")
    print(f"  truncated    : {n_trunc}/{len(selected)}")

    if args.dry_run:
        print("\n--- sample judge prompt (first selected chain) ---\n")
        print(build_judge_prompt(selected[0])[:2000])
        sel_path = Path(args.out).with_suffix(".selection.json")
        sel_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"chain_ids": [c["task_id"] for c in selected], "meta": meta},
                  open(sel_path, "w"), indent=2)
        print(f"\n[dry-run] wrote selection to {sel_path}; no API calls made.")
        return

    out = Path(args.out)
    backup_existing(out)
    labels = generate_correctness_labels(
        selected, save_path=out, meta=meta, max_chains=args.max_chains,
    )
    n_corr = sum(1 for r in labels.values() if r.get("correct") is True)
    n_inc = sum(1 for r in labels.values() if r.get("correct") is False)
    n_unc = sum(1 for r in labels.values() if r.get("correct") is None)
    print(f"\njudged {len(labels)} chains -> correct={n_corr} incorrect={n_inc} "
          f"uncertain(excluded)={n_unc}")
    prov = provenance(args, inputs=[args.chains, args.tasks])
    json.dump(prov, open(out.with_suffix(".provenance.json"), "w"), indent=2)
    print(f"labels -> {out}")


if __name__ == "__main__":
    main()
