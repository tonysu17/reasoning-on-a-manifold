#!/usr/bin/env python3
"""One-off: time the annotation stage on the 12 cached smoke chains + a
before/after peek. Annotation-only (NO regeneration). Resume-safe."""
import json, time, os, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, ".")

print("CREDS_URL_SET:", bool(os.environ.get("CLAUDE_PROXY_URL")),
      "| KEY_SET:", bool(os.environ.get("CLAUDE_PROXY_KEY")), flush=True)

from src.annotation import annotate_chains, ANNOTATION_MODEL

r = json.load(open("results/eval/R1-1.5B/steering_results.json"))
print(f"loaded {len(r)} cached chains; annotating with {ANNOTATION_MODEL} ...", flush=True)

t0 = time.time()
annotated = annotate_chains(
    r,
    save_path=Path("results/eval/R1-1.5B/trial_annotated.json"),
    dedup_keys=("task_id", "behaviour", "method", "alpha"),
)
dt = time.time() - t0
n = len(annotated)
nc = sum(1 for a in annotated if a.get("annotation_complete"))
print(f"\nTRIAL_RESULT: {n} chains in {dt:.1f}s = {dt/max(n,1):.1f} s/chain | {nc}/{n} complete", flush=True)

# before (vanilla) / after (steered) backtracking fraction — indicative only (n=3, builder annotator)
fr = defaultdict(list)
for a in annotated:
    anns = a.get("annotations", []) or []
    if not anns:
        continue
    f = sum(1 for s in anns if s.get("label") == "backtracking") / len(anns)
    fr[(a["behaviour"], a["method"])].append(f)
print("BEFORE/AFTER backtracking fraction (indicative; Sonnet; n=3):", flush=True)
for k in sorted(fr):
    v = fr[k]
    print(f"  {k[0]:10} {k[1]:20} {sum(v)/len(v):.3f}  (n={len(v)})", flush=True)
print("DONE", flush=True)
