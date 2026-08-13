"""A5 smoke run: annotate the first N eligible rows and stop.

Replicates ph2_executor.stage_annotate's preparation byte-for-byte (window,
chain_full, annotated_tokens) and honours the same guard contract, so the rows
it writes are identical to what the full stage would produce and the full run
resumes cleanly from the same journal.
"""
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
ROOT = Path("/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold")
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import ph2_executor as px
from src.annotation import (ANNOTATION_MODEL, annotate_chains,
                            annotation_initial_request_count, max_prompt_chars)
from src.annotation_budget import AnnotationAttemptGuard
from src.annotation_coverage import COVERAGE_RULE_VERSION
from src.ph2_stages import (ALL_ROLES, ANNOTATION_DEDUP_KEYS,
                            ANNOTATION_INCLUDE_POST_THINK,
                            ANNOTATION_WINDOW_TOKENS, annotation_window,
                            proxy_chunk_budget_ok)

KILL_AFTER = None
MANIFEST = ROOT / ".codex/out/PH2_ANNOTATION_GUARD_MANIFEST_A5R10_2026-08-13.json"
MANIFEST_SHA = "f997e95103702fcfd309247ad6d91df3b1ce9de614a8bbfdc5104cf080ce4557"
ANN = ROOT / "results/ph2/annotation"

budget = proxy_chunk_budget_ok()
assert budget["ok"], budget

bound_paths = (
    "ph2_executor.py", "src/annotation.py", "src/annotation_budget.py",
    "src/annotation_coverage.py", "src/annotation_quarantine.py",
    "src/ph2_stages.py", "src/delta_floor.py",
    "results/prereg/PHASE2_TRANSPORT_PREREG_2026-08-08.md",
    "results/prereg/PHASE2_ADJUNCT_AMENDMENT_2026-08-08.md",
    "results/prereg/phase2_task_manifest.json",
)
guard = AnnotationAttemptGuard.from_manifest(
    MANIFEST, MANIFEST_SHA, ANN / "api_attempt_journal.jsonl",
    root=ROOT, expected_model=ANNOTATION_MODEL,
    expected_annotation_window_tokens=ANNOTATION_WINDOW_TOKENS,
    expected_bound_paths=bound_paths,
    expected_max_output_tokens=px.ANNOTATION_MAX_TOKENS,
    expected_max_prompt_chars=max_prompt_chars(),
    expected_coverage_rule_version=COVERAGE_RULE_VERSION,
)

shards = [(ROOT / "results/ph2/battery" / f"{r}.json", ANN / f"{r}.json")
          for r in ALL_ROLES]
shards.append((ROOT / "results/ph2/injection/base_injection.json",
               ANN / "base_injection.json"))

prepared, observed_sources, observed_requests = [], [], 0
for src_path, dst_path in shards:
    if not src_path.exists():
        continue
    observed_sources.append(str(src_path.relative_to(ROOT)))
    rows = []
    for r in json.loads(src_path.read_text()):
        win, est, truncated = annotation_window(r["chain"])
        rows.append({**r, "chain": win, "chain_full": r["chain"],
                     "annotated_tokens": est,
                     "annotation_window_tokens": ANNOTATION_WINDOW_TOKENS,
                     "annotation_window_truncated": truncated})
        from src.annotation_coverage import region_source_text
        observed_requests += annotation_initial_request_count(
            region_source_text(win, include_post_think=ANNOTATION_INCLUDE_POST_THINK))
    prepared.append((src_path, dst_path, rows))

assert tuple(sorted(observed_sources)) == guard.policy.source_paths, "source-set mismatch"
guard.assert_planned_initial_requests(observed_requests)
print(f"[smoke] plan verified: {observed_requests} initial requests; "
      f"worst-case/call ${guard.policy.worst_case_attempt_cost_usd:.6f}")
print("[full] releasing the full run over all shards")

for src_path, dst_path, rows in prepared:
    print(f"[full] shard {dst_path.name}: {len(rows)} rows")
    annotate_chains(
        rows, save_path=dst_path, dedup_keys=ANNOTATION_DEDUP_KEYS,
        model=ANNOTATION_MODEL, max_tokens=px.ANNOTATION_MAX_TOKENS,
        shrink_on_retry=True, max_retries=3, attempt_guard=guard,
        coverage_validation=True, include_post_think=ANNOTATION_INCLUDE_POST_THINK,
    )
print("[full] all shards done")
print(json.dumps(guard.summary(), indent=2))
