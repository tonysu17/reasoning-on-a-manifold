#!/usr/bin/env python3
"""Reliability test for the E8 annotation step on REAL LONG chains (the 8192-token,
chunked path the smoke didn't cover). Annotates a few already-generated long chains
with Sonnet to a TEST file (does NOT touch annotated_steered.json), and reports
whether each completed with valid parsed spans. Also re-runs to confirm resume
(skips completed chains). Read-only w.r.t. the live run."""
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, ".")
from src.annotation import annotate_chains

ANNOTATOR = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
SRC = Path("results/eval/R1-1.5B__E1/steering_results.json")
OUT = Path("results/eval/_annot_selftest.json")

d = json.loads(SRC.read_text())
# pick long chains (chunked) across a couple of arm types + a vanilla
longs = [r for r in d if r.get("n_tokens", 0) >= 8000]
sample = longs[:5]
print(f"testing {len(sample)} LONG chains, n_tokens={[r['n_tokens'] for r in sample]}")
print(f"  (each chunks into ~{[math.ceil(r['n_tokens']/1200) for r in sample]} pieces)\n")

OUT.unlink(missing_ok=True)
out = annotate_chains(sample, save_path=OUT, checkpoint_every=1,
                      dedup_keys=("task_id", "behaviour", "method", "alpha"),
                      model=ANNOTATOR)

ok = 0
for r in out:
    anns = r.get("annotations", [])
    labels = [a.get("label") for a in anns][:6]
    complete = r.get("annotation_complete")
    valid = bool(anns) and complete
    ok += valid
    print(f"  {r['method']:22s} n_tok={r['n_tokens']:<5d} complete={complete} "
          f"spans={len(anns):<3d} {'OK' if valid else 'FAIL'}  labels={labels}")

print(f"\n{ok}/{len(out)} long chains annotated COMPLETE with valid spans")

# Resume check: re-run on the SAME save file → should skip all (already complete), 0 new API calls.
print("\n--- resume check (re-run; should skip all completed) ---")
out2 = annotate_chains(sample, save_path=OUT, checkpoint_every=1,
                       dedup_keys=("task_id", "behaviour", "method", "alpha"),
                       model=ANNOTATOR)
print(f"after resume: {sum(1 for r in out2 if r.get('annotation_complete'))}/{len(out2)} complete")
print("\n" + ("FLAWLESS — long-chain chunking + parsing + resume all work"
              if ok == len(out) else "ISSUE — investigate before the real run"))
