#!/usr/bin/env python3
"""Recover the handful of chains that never completed annotation during the main
6-shard run (they hit transient 503 / empty-response from the proxy under the
sustained parallel load). Strategy: re-annotate ONLY the still-incomplete chains
SEQUENTIALLY (one stream → no server overload), resuming into a recovery file,
then fold the newly-complete ones back into annotated_steered.json.

Idempotent + resumable: re-run to keep chipping at any remaining stragglers.
Prints the straggler count remaining (0 == fully recovered)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.annotation import annotate_chains  # noqa: E402

EVAL = Path("results/eval/R1-1.5B__E1")
DEDUP = ("task_id", "behaviour", "method", "alpha")
MODEL = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"


def key(r):
    return tuple(r.get(k) for k in DEDUP)


def main():
    ann = json.loads((EVAL / "annotated_steered.json").read_text())
    complete = {key(r) for r in ann if r.get("annotation_complete")}
    allc = json.loads((EVAL / "steering_results.json").read_text())
    stragglers = [c for c in allc if key(c) not in complete]
    print(f"{len(stragglers)} stragglers to (re)annotate sequentially")
    if not stragglers:
        return

    rec = annotate_chains(
        stragglers, save_path=EVAL / "annotated_steered.recovery.json",
        checkpoint_every=1, dedup_keys=DEDUP, model=MODEL,
    )
    newly = [r for r in rec if r.get("annotation_complete")]
    print(f"recovery file now holds {newly.__len__()}/{len(stragglers)} complete")

    # Fold recovered-complete back in (disjoint keys; dedup as belt-and-braces).
    seen, final = set(), []
    for r in ann + newly:
        if not r.get("annotation_complete"):
            continue
        k = key(r)
        if k in seen:
            continue
        seen.add(k)
        final.append(r)
    tmp = EVAL / "annotated_steered.json.tmp"
    tmp.write_text(json.dumps(final, indent=2, ensure_ascii=False))
    tmp.replace(EVAL / "annotated_steered.json")

    remaining = len(allc) - len(final)
    print(f"annotated_steered.json: {len(final)} complete | {remaining} stragglers remain")


if __name__ == "__main__":
    main()
