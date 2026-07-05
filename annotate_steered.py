#!/usr/bin/env python3
"""Annotate already-generated steered chains via the proxy — API only, NO GPU
model load (unlike 07's coupled annotation). Saves after EVERY chain
(checkpoint_every=1) so a credit-exhaustion mid-run loses ≤1 chain and re-running
resumes exactly. Use distinct --out files for distinct annotators (headline +
the cross-annotator noise band) so each resumes independently.

    python annotate_steered.py --eval-dir results/eval/R1-1.5B__L27 \
        --annotator amazon.nova-pro-v1:0 --out annotated_steered.json
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
from src.annotation import annotate_chains

ap = argparse.ArgumentParser()
ap.add_argument("--eval-dir", required=True)
ap.add_argument("--annotator", required=True, help="proxy model id (non-builder for the headline)")
ap.add_argument("--out", required=True, help="save filename (under eval-dir) or absolute path")
ap.add_argument("--limit", type=int, default=None, help="annotate at most N new chains (smoke)")
a = ap.parse_args()

ed = Path(a.eval_dir)
results = json.loads((ed / "steering_results.json").read_text())
save = Path(a.out) if Path(a.out).is_absolute() else ed / a.out
print(f"annotating {len(results)} chains with {a.annotator} -> {save} "
      f"(checkpoint every chain; resumable)")

annotate_chains(
    results,
    save_path=save,
    checkpoint_every=1,                       # save after EVERY chain
    dedup_keys=("task_id", "behaviour", "method", "alpha"),
    model=a.annotator,
    kill_after=a.limit,
)
done = json.loads(save.read_text())
ne = sum(1 for r in done if r.get("annotations"))
comp = sum(1 for r in done if r.get("annotation_complete"))
print(f"done: {len(done)} records | non-empty {ne} | complete {comp} -> {save}")
