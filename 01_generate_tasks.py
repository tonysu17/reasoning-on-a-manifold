#!/usr/bin/env python3
"""
Phase 1 — Task generation.

Generates 1000 diverse reasoning tasks across 10 categories using Claude
via the proxy endpoint.
Output: data/tasks.json

Requirements:
  pip install requests                  (no GPU needed)
  export CLAUDE_PROXY_URL=https://...
  export CLAUDE_PROXY_KEY=rp_...

Runtime: ~5–10 minutes  |  Cost: ~$2 from proxy budget
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.task_gen import CATEGORIES, generate_tasks, load_tasks, task_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Task generation via Claude proxy")
    parser.add_argument("--n", type=int, default=100,
                        help="Tasks per category (default: 100 → 1000 total)")
    parser.add_argument("--out", default="data/tasks.json",
                        help="Output path (default: data/tasks.json)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: generate only 2 tasks per category (20 total)")
    args = parser.parse_args()

    if not os.environ.get("CLAUDE_PROXY_URL") or not os.environ.get("CLAUDE_PROXY_KEY"):
        logger.error(
            "Missing environment variables.\n"
            "  export CLAUDE_PROXY_URL=https://...\n"
            "  export CLAUDE_PROXY_KEY=rp_..."
        )
        sys.exit(1)

    out_path = Path(args.out)

    if out_path.exists():
        logger.info(f"Output already exists at {out_path} — loading existing tasks.")
        tasks = load_tasks(out_path)
        task_summary(tasks)
        return

    n = 2 if args.smoke else args.n
    if args.smoke:
        logger.info("SMOKE TEST: generating 2 tasks per category (20 total)")

    logger.info(f"Generating {n} tasks per category × {len(CATEGORIES)} categories …")

    tasks = generate_tasks(
        n_per_category=n,
        save_path=out_path,
    )

    print("\n" + "="*60)
    task_summary(tasks)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
