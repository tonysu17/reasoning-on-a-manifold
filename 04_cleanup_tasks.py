#!/usr/bin/env python3
"""
Phase 1.5 — Tasks dataset cleanup.

Steps:
  1. Load data/tasks_deduped.json (901 tasks after earlier dedup).
  2. Drop all lateral_thinking tasks (61/67 are classic-puzzle variants).
  3. Regenerate lateral_thinking from scratch (100 tasks) with a blocklist of
     known classic puzzles so the model can't recycle them.
  4. Top up every other category from its current count to TARGET_PER_CATEGORY.
  5. Run a final 150-char prefix dedup over the combined set.
  6. Save → data/tasks_final.json and print verification.

Environment variables required (same as Phase 1):
    CLAUDE_PROXY_URL
    CLAUDE_PROXY_KEY

Usage:
    python 04_cleanup_tasks.py
    python 04_cleanup_tasks.py --smoke      # 5 lateral + 2 topup tasks only
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.task_gen import CATEGORIES, _call_api, load_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TARGET_PER_CATEGORY = 100
BATCH_SIZE = 5
DEDUP_PREFIX_LEN = 150

# Classic puzzles to block when regenerating lateral_thinking.
# The model must not produce any variant of these themes.
LATERAL_THINKING_BLOCKLIST = [
    "elevator riddle: person takes stairs down because they cannot reach the upper buttons (short person)",
    "three light switches in one room controlling a bulb in another — identify switch without re-entering",
    "surgeon says 'I cannot operate on this boy — he is my son': surgeon turns out to be the mother",
    "man found dead on floor with puddle of water and open window — stood on block of ice",
    "two guards (or sentinels) at two doors: one always lies, one always tells truth — find door to freedom",
    "two identical doors, one leads to freedom/treasure, one leads to death/punishment — figure out which",
    "pedestrians walking toward oncoming traffic in the road — it is a one-way street",
    "woman at funeral meets attractive stranger; later murders her own sister — to meet stranger again",
    "barber paradox: barber shaves exactly those who do not shave themselves",
    "room with dead man and 53 bicycles (playing cards) on the floor",
    "man hanging from rafter in empty room with a puddle of water — stood on ice block",
    "dead man in a field, parachute next to him, never opened",
    "man in car crash, father is dead, boy says 'that's my father' — second parent surprise",
    "two coins totalling 30 cents, one is not a nickel (it's the other one)",
    "nine dots puzzle — connect all nine with four straight lines without lifting pen",
    "pills A and B look identical; one is poison; you must swallow one — dissolve one in water",
    "three gods: Random, True, False — identify them with three yes/no questions",
    "a man walks into a restaurant, eats albatross soup, goes home and kills himself",
    "identical twins one always lying, one always telling truth",
    "prisoner hat colour deduction (red/blue hats, can see others not own)",
    "man shot dead in car in locked garage — suicide ruled out — murderer got in and out",
    "woman pushes car to hotel and loses all her money — Monopoly board",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def dedup_tasks(tasks: list[dict]) -> tuple[list[dict], list[dict]]:
    """150-char prefix dedup (same logic as the earlier dedup step)."""
    seen: set[str] = set()
    kept, removed = [], []
    for task in tasks:
        key = task["prompt"].strip()[:DEDUP_PREFIX_LEN].lower()
        if key not in seen:
            seen.add(key)
            kept.append(task)
        else:
            removed.append(task)
    return kept, removed


def _next_id_start(existing_tasks: list[dict]) -> int:
    """Return the next available numeric ID suffix (max existing + 1)."""
    if not existing_tasks:
        return 0
    nums = []
    for t in existing_tasks:
        parts = t.get("id", "").split("_")
        if len(parts) == 2 and parts[1].isdigit():
            nums.append(int(parts[1]))
    return max(nums) + 1 if nums else 0


def generate_with_context(
    category: str,
    n_needed: int,
    start_id: int,
    context_summaries: list[str],
    smoke: bool = False,
) -> list[dict]:
    """
    Generate `n_needed` tasks for `category`, passing `context_summaries` to
    each batch call so the model avoids repeating known tasks/themes.
    """
    if smoke:
        n_needed = min(n_needed, 5)

    prefix = category[:4].upper()
    description = CATEGORIES[category]
    new_tasks: list[dict] = []
    n_batches = -(-n_needed // BATCH_SIZE)

    for b in range(n_batches):
        n_this = min(BATCH_SIZE, n_needed - len(new_tasks))
        if n_this <= 0:
            break
        batch_start = start_id + len(new_tasks)

        # Always include the running list of newly-generated tasks as context
        # so consecutive batches don't repeat each other either.
        all_context = context_summaries + [t["prompt"][:120] for t in new_tasks]

        batch = _call_api(
            category, description, prefix, batch_start, n_this,
            context_summaries=all_context,
        )
        new_tasks.extend(batch)
        logger.info(f"  [{category}] batch {b + 1}/{n_batches}: {len(batch)} tasks")
        time.sleep(0.5)

    return new_tasks[:n_needed]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1.5: Tasks dataset cleanup")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: generate only 5 lateral + 2 topup tasks")
    args = parser.parse_args()

    # ── 1. Load existing deduped tasks ────────────────────────────────────────
    tasks_path = Path("data/tasks_deduped.json")
    if not tasks_path.exists():
        logger.error(f"Not found: {tasks_path}. Run dedup step first.")
        sys.exit(1)

    tasks = load_tasks(tasks_path)
    logger.info(f"Loaded {len(tasks)} tasks from {tasks_path}")

    by_cat: dict[str, list[dict]] = {}
    for t in tasks:
        by_cat.setdefault(t["category"], []).append(t)

    # ── 2. Drop all lateral_thinking tasks ────────────────────────────────────
    lateral_existing = by_cat.pop("lateral_thinking", [])
    logger.info(f"Dropping all {len(lateral_existing)} lateral_thinking tasks — regenerating from scratch")

    # ── 3. Regenerate lateral_thinking with blocklist ─────────────────────────
    logger.info("Regenerating lateral_thinking (100 tasks) with classic-puzzle blocklist …")
    n_lateral = 5 if args.smoke else TARGET_PER_CATEGORY
    lateral_new = generate_with_context(
        category="lateral_thinking",
        n_needed=n_lateral,
        start_id=0,
        context_summaries=LATERAL_THINKING_BLOCKLIST,
        smoke=args.smoke,
    )
    logger.info(f"  lateral_thinking: {len(lateral_new)} new tasks generated")
    by_cat["lateral_thinking"] = lateral_new

    # ── 4. Top up every other deficient category ──────────────────────────────
    topup_smoke_budget = 2  # tasks to generate per category in smoke mode
    for cat in sorted(CATEGORIES):
        if cat == "lateral_thinking":
            continue
        existing = by_cat.get(cat, [])
        n_needed = TARGET_PER_CATEGORY - len(existing)
        if n_needed <= 0:
            continue
        if args.smoke:
            n_needed = min(n_needed, topup_smoke_budget)
            topup_smoke_budget = 0  # only top up first deficient category in smoke

        next_id = _next_id_start(existing)
        logger.info(f"Topping up {cat}: {len(existing)} → {len(existing) + n_needed} (+{n_needed}, IDs from {next_id})")
        context = [t["prompt"][:120] for t in existing]
        new_tasks = generate_with_context(
            category=cat,
            n_needed=n_needed,
            start_id=next_id,
            context_summaries=context,
            smoke=False,  # n_needed already capped for smoke above
        )
        by_cat[cat] = existing + new_tasks
        logger.info(f"  {cat}: now {len(by_cat[cat])} tasks")

    # ── 5. Combine + final dedup ──────────────────────────────────────────────
    all_tasks: list[dict] = []
    for cat_tasks in by_cat.values():
        all_tasks.extend(cat_tasks)

    all_tasks, removed_post = dedup_tasks(all_tasks)
    if removed_post:
        logger.info(f"Post-generation dedup removed {len(removed_post)} duplicate tasks")

    # ── 6. Verify ─────────────────────────────────────────────────────────────
    cats = Counter(t["category"] for t in all_tasks)
    logger.info("\nFinal task counts:")
    any_low = False
    for cat in sorted(CATEGORIES):
        n = cats.get(cat, 0)
        flag = "  ✗ BELOW TARGET" if n < TARGET_PER_CATEGORY else ""
        logger.info(f"  {cat:<35s} {n:>3d}{flag}")
        if n < TARGET_PER_CATEGORY:
            any_low = True
    logger.info(f"  {'TOTAL':<35s} {len(all_tasks):>3d}")

    if any_low and not args.smoke:
        logger.warning("Some categories are below target — check for API failures above.")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    out_path = Path("data/tasks_final.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(all_tasks, f, indent=2, ensure_ascii=False)
    tmp.rename(out_path)
    logger.info(f"\nSaved → {out_path}  ({len(all_tasks)} tasks)")

    if args.smoke:
        logger.info("SMOKE TEST complete — tasks_final.json written for inspection only; "
                    "re-run without --smoke to generate the full dataset.")


if __name__ == "__main__":
    main()
