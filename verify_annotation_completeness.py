#!/usr/bin/env python3
"""
Verify annotation completeness and resume-logic correctness.

Checks:
  1. All expected chains are present (no missing task_ids)
  2. No duplicate task_ids
  3. No chain has empty annotations
  4. All chains are marked annotation_complete=True (no partial records)
  5. Label vocabulary matches VALID_LABELS

Usage:
    python verify_annotation_completeness.py --pilot
    python verify_annotation_completeness.py --model-short R1-1.5B
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.annotation import VALID_LABELS
from src.chain_gen import load_chains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--model-short", default="R1-1.5B")
    args = parser.parse_args()

    if args.pilot:
        chains_path = Path("data/chains_pilot.json")
        ann_path    = Path("data/annotated_pilot.json")
    else:
        chains_path = Path(f"data/chains_{args.model_short}.json")
        ann_path    = Path(f"data/annotated_{args.model_short}.json")

    if not chains_path.exists():
        print(f"FAIL: chains not found at {chains_path}")
        sys.exit(1)
    if not ann_path.exists():
        print(f"FAIL: annotations not found at {ann_path}")
        sys.exit(1)

    chains = load_chains(chains_path)
    with open(ann_path) as f:
        annotated = json.load(f)

    expected_ids = {c["task_id"] for c in chains}
    annotated_ids = [a["task_id"] for a in annotated]

    fails = []

    # 1. No missing
    missing = expected_ids - set(annotated_ids)
    if missing:
        fails.append(f"Missing {len(missing)} chains: {sorted(missing)}")

    # 2. No duplicates
    from collections import Counter
    dups = [tid for tid, n in Counter(annotated_ids).items() if n > 1]
    if dups:
        fails.append(f"Duplicate task_ids: {dups}")

    # 3. No empty annotations
    empty = [a["task_id"] for a in annotated if not a.get("annotations")]
    if empty:
        fails.append(f"{len(empty)} chains have empty annotations: {empty}")

    # 4. All complete
    partial = [a["task_id"] for a in annotated if not a.get("annotation_complete", False)]
    if partial:
        fails.append(f"{len(partial)} chains not marked annotation_complete: {partial}")

    # 5. Valid labels
    bad_labels = set()
    for a in annotated:
        for span in a.get("annotations", []):
            if span["label"] not in VALID_LABELS:
                bad_labels.add(span["label"])
    if bad_labels:
        fails.append(f"Unknown labels found: {bad_labels}")

    if fails:
        print("VERIFICATION FAILED:")
        for f in fails:
            print(f"  ✗ {f}")
        sys.exit(1)
    else:
        total_spans = sum(len(a.get("annotations", [])) for a in annotated)
        print(f"VERIFICATION PASSED ✓")
        print(f"  {len(annotated)} chains, {total_spans} spans, all complete, all labels valid")


if __name__ == "__main__":
    main()
