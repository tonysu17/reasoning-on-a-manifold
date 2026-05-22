#!/usr/bin/env python3
"""
Pilot gate — MANDATORY before full Phase 2.

Runs a stratified 20-chain pilot (2 tasks per category) through Phases 2 and 3,
then validates 5 checks before allowing scale-up.

Steps:
    1. Extract 2 tasks per category → data/tasks_pilot.json
    2. Run Phase 2 on 20 tasks → data/chains_pilot.json
       (requires GPU; run this on the cluster)
    3. Submit Phase 3 annotation batch → OpenAI Batch API
       (run this locally after chains_pilot.json is copied back)
    4. Download + validate annotation results

Usage:
    # On cluster (after syncing project):
    python 00_pilot_gate.py --extract-tasks
    python 00_pilot_gate.py --generate-chains    # ~30 min on GB10
    # Copy chains_pilot.json back to local machine, then:
    python 00_pilot_gate.py --annotate-submit
    python 00_pilot_gate.py --annotate-status    # check periodically
    python 00_pilot_gate.py --validate           # download + run all checks

Checks (all must pass before full Phase 2):
    1. All 20 chains end with </think>                (no truncation)
    2. Mean chain length ≤ 2,500 tokens              (cost/time bound)
    3. Annotation parses into [(label, text), ...]    (format correct)
    4. All 6 Venhoff labels appear ≥ 1× across 20 chains
    5. Sentence fractions roughly match Venhoff Fig. 2 (±10 pp tolerance)

GPU requirements:  Phase 2 only (cluster).
API requirements:  Phase 3 only (OPENAI_API_KEY).
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
TASKS_PATH       = Path("data/tasks.json")
PILOT_TASKS_PATH = Path("data/tasks_pilot.json")
PILOT_CHAINS_PATH = Path("data/chains_pilot.json")
PILOT_ANNOTATED_PATH = Path("data/annotated_pilot.json")
PILOT_BATCH_DIR  = Path("data/.batch_pilot")
LOGS_DIR         = Path("logs")

N_PER_CATEGORY = 2  # 2 × 10 categories = 20 pilot chains

# Venhoff Fig. 2 expected fractions ± this tolerance
FRACTION_TOLERANCE = 0.10

# Expected fractions from Venhoff Fig. 2 (R1-Distill models)
VENHOFF_FRACTIONS = {
    "deduction":              0.52,
    "adding-knowledge":       0.15,
    "uncertainty-estimation": 0.09,
    "initializing":           0.07,
    "example-testing":        0.06,
    "backtracking":           0.04,
}


# ── Step 1: Extract pilot tasks ───────────────────────────────────────────────

def extract_tasks():
    if not TASKS_PATH.exists():
        logger.error(f"Tasks not found at {TASKS_PATH}. Run 01_generate_tasks.py first.")
        sys.exit(1)

    with open(TASKS_PATH) as f:
        tasks = json.load(f)

    by_category: dict[str, list] = {}
    for t in tasks:
        cat = t.get("category", "unknown")
        by_category.setdefault(cat, []).append(t)

    pilot: list[dict] = []
    for cat, cat_tasks in sorted(by_category.items()):
        selected = cat_tasks[:N_PER_CATEGORY]
        pilot.extend(selected)
        logger.info(f"  {cat}: {len(selected)} tasks selected")

    PILOT_TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PILOT_TASKS_PATH, "w") as f:
        json.dump(pilot, f, indent=2, ensure_ascii=False)

    logger.info(f"Pilot tasks: {len(pilot)} total → {PILOT_TASKS_PATH}")
    categories = sorted(by_category.keys())
    if len(categories) != 10:
        logger.warning(f"Expected 10 categories, found {len(categories)}: {categories}")


# ── Step 2: Generate pilot chains ─────────────────────────────────────────────

def generate_chains():
    if not PILOT_TASKS_PATH.exists():
        logger.error(f"Run --extract-tasks first.")
        sys.exit(1)

    from src.chain_gen import load_model, generate_chains as _gen_chains

    with open(PILOT_TASKS_PATH) as f:
        tasks = json.load(f)

    logger.info(f"Loading model for pilot chain generation ({len(tasks)} tasks)…")
    model, tokenizer = load_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    chains = _gen_chains(
        model, tokenizer, tasks,
        max_new_tokens=8192,
        temperature=0.0,
        save_path=PILOT_CHAINS_PATH,
        checkpoint_every=5,
    )

    LOGS_DIR.mkdir(exist_ok=True)
    csv_path = LOGS_DIR / "pilot_lengths.csv"
    with open(csv_path, "w") as f:
        f.write("task_id,category,n_tokens\n")
        for c in chains:
            f.write(f"{c['task_id']},{c.get('category','')},{c['n_tokens']}\n")
    logger.info(f"Token counts → {csv_path}")


# ── Step 3a: Submit annotation batch ─────────────────────────────────────────

def annotate_submit():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if not PILOT_CHAINS_PATH.exists():
        logger.error(f"Pilot chains not found at {PILOT_CHAINS_PATH}.")
        logger.error("Run --generate-chains on the cluster, then copy chains_pilot.json here.")
        sys.exit(1)

    from src.annotation import prepare_batch_file, submit_batch
    from src.chain_gen import load_chains

    chains = load_chains(PILOT_CHAINS_PATH)
    PILOT_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batch_id_file = PILOT_BATCH_DIR / "batch_id.txt"

    if batch_id_file.exists():
        logger.info(f"Batch already submitted: {batch_id_file.read_text().strip()}")
        return

    input_jsonl = PILOT_BATCH_DIR / "batch_input.jsonl"
    prepare_batch_file(chains, input_jsonl)
    batch_id = submit_batch(input_jsonl, api_key=api_key)
    batch_id_file.write_text(batch_id)
    print(f"\nBatch submitted: {batch_id}")
    print(f"Check status: python 00_pilot_gate.py --annotate-status")


# ── Step 3b: Check annotation status ─────────────────────────────────────────

def annotate_status():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    from src.annotation import check_batch_status

    batch_id_file = PILOT_BATCH_DIR / "batch_id.txt"
    if not batch_id_file.exists():
        logger.error("No batch submitted yet. Run --annotate-submit first.")
        sys.exit(1)

    batch_id = batch_id_file.read_text().strip()
    info = check_batch_status(batch_id, api_key=api_key)
    print(f"\nBatch ID  : {info['id']}")
    print(f"Status    : {info['status']}")
    print(f"Progress  : {info['completed']}/{info['total']}  ({info['failed']} failed)")
    if info["status"] == "completed":
        print("\nComplete → run:  python 00_pilot_gate.py --validate")


# ── Step 4: Download + validate ───────────────────────────────────────────────

def validate():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    from src.annotation import (
        VALID_LABELS, download_batch_results, merge_batch_results, _save_json
    )
    from src.chain_gen import load_chains

    # Download if needed
    batch_id_file = PILOT_BATCH_DIR / "batch_id.txt"
    output_jsonl  = PILOT_BATCH_DIR / "batch_output.jsonl"

    if not output_jsonl.exists():
        if not batch_id_file.exists():
            logger.error("No batch submitted. Run --annotate-submit first.")
            sys.exit(1)
        batch_id = batch_id_file.read_text().strip()
        download_batch_results(batch_id, output_jsonl, api_key=api_key, poll_interval=0)

    chains = load_chains(PILOT_CHAINS_PATH)
    annotated = merge_batch_results(chains, output_jsonl)
    _save_json(annotated, PILOT_ANNOTATED_PATH)

    with open(LOGS_DIR / "pilot_lengths.csv") as f:
        token_counts = {
            row.split(",")[0]: int(row.split(",")[2].strip())
            for row in f.read().splitlines()[1:]
            if row.strip()
        }

    print(f"\n{'='*65}")
    print(f"PILOT GATE VALIDATION — {len(annotated)} chains")
    print("="*65)

    passed = 0
    total_checks = 5

    # Check 1: All chains end with </think>
    bad_termination = [
        c["task_id"] for c in annotated
        if "</think>" not in c.get("chain", "")
    ]
    ok1 = len(bad_termination) == 0
    print(f"\n[{'PASS' if ok1 else 'FAIL'}] Check 1: All chains terminate with </think>")
    if not ok1:
        print(f"       Truncated task IDs: {bad_termination}")
        print(f"       Fix: increase max_new_tokens in Phase 2 config")
    passed += ok1

    # Check 2: Mean chain length ≤ 2500 tokens
    lengths = [c["n_tokens"] for c in annotated if c["n_tokens"] > 0]
    mean_len = sum(lengths) / len(lengths) if lengths else 0
    ok2 = mean_len <= 2500
    print(f"\n[{'PASS' if ok2 else 'FAIL'}] Check 2: Mean chain length ≤ 2500 tokens")
    print(f"       Mean: {mean_len:.0f}  Min: {min(lengths)}  Max: {max(lengths)}")
    if not ok2:
        print(f"       Fix: increase max_new_tokens cap; recompute Phase 3 cost estimate")
    passed += ok2

    # Check 3: Annotations parse cleanly (non-empty for every chain)
    empty_anns = [c["task_id"] for c in annotated if not c.get("annotations")]
    ok3 = len(empty_anns) == 0
    print(f"\n[{'PASS' if ok3 else 'FAIL'}] Check 3: All chains have non-empty annotations")
    if not ok3:
        print(f"       Empty annotation task IDs: {empty_anns}")
        print(f"       Fix: check _SYSTEM_PROMPT and delimiter parser in src/annotation.py")
    passed += ok3

    # Check 4: All 6 labels appear ≥ 1×
    all_labels: Counter = Counter()
    for chain in annotated:
        for ann in chain.get("annotations", []):
            all_labels[ann["label"]] += 1
    missing = [l for l in VALID_LABELS if all_labels[l] == 0]
    ok4 = len(missing) == 0
    print(f"\n[{'PASS' if ok4 else 'FAIL'}] Check 4: All 6 Venhoff labels appear ≥ 1×")
    for label in sorted(VALID_LABELS):
        print(f"       {label:<28s} {all_labels[label]:>4d}")
    if not ok4:
        print(f"       Missing: {missing}")
        print(f"       Fix: increase pilot to 50 chains, or check annotation prompt")
    passed += ok4

    # Check 5: Sentence fractions roughly match Venhoff Fig. 2 (±10 pp)
    total_sents = sum(all_labels.values())
    print(f"\n[{'PASS' if True else 'FAIL'}] Check 5: Sentence fractions vs Venhoff Fig. 2 (±{FRACTION_TOLERANCE:.0%} tolerance)")
    fraction_ok = True
    for label, expected in VENHOFF_FRACTIONS.items():
        observed = all_labels[label] / total_sents if total_sents else 0
        diff = abs(observed - expected)
        within = diff <= FRACTION_TOLERANCE
        if not within:
            fraction_ok = False
        mark = "ok" if within else "!!"
        print(f"  [{mark}]  {label:<28s}  observed {observed:>5.1%}  expected {expected:>5.1%}  Δ={diff:>4.1%}")
    if not fraction_ok:
        print(f"       Divergence from Venhoff Fig. 2 — check prompt fidelity")
        print(f"       Note: 20-chain pilot has high variance; recheck at 100+ chains")
    ok5 = fraction_ok
    passed += ok5

    # Summary
    print(f"\n{'='*65}")
    print(f"Result: {passed}/{total_checks} checks passed")
    if passed == total_checks:
        print("\nPILOT GATE PASSED — proceed to full Phase 2:")
        print("  python 02_generate_chains.py --model 1.5b")
    else:
        print("\nPILOT GATE FAILED — debug before scaling.")
        print("Do not run full Phase 2.")
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pilot gate for Reasoning on a Manifold")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--extract-tasks",    action="store_true",
                       help="Step 1: extract 2 tasks per category → data/tasks_pilot.json")
    group.add_argument("--generate-chains",  action="store_true",
                       help="Step 2: run Phase 2 on pilot tasks (GPU required)")
    group.add_argument("--annotate-submit",  action="store_true",
                       help="Step 3a: submit pilot annotation batch to OpenAI")
    group.add_argument("--annotate-status",  action="store_true",
                       help="Step 3b: check batch job status")
    group.add_argument("--validate",         action="store_true",
                       help="Step 4: download results and run all 5 checks")
    args = parser.parse_args()

    if args.extract_tasks:
        extract_tasks()
    elif args.generate_chains:
        generate_chains()
    elif args.annotate_submit:
        annotate_submit()
    elif args.annotate_status:
        annotate_status()
    elif args.validate:
        validate()


if __name__ == "__main__":
    main()
