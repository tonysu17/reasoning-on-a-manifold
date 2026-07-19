#!/usr/bin/env python3
"""pt11b: assemble the harmful/benign prompt pool for the GRPO refusal-format arm.

HARMFUL side: STAR-1 prompts (the same distribution the SFT/DPO safety arms train
on, so the recipe comparison holds the prompt distribution fixed), stratified
round-robin across categories, EXCLUDING any record touching 'Self-Harm': under
the binary refusal-format rubric a refusal is the wrong target behaviour for a
person in distress, so rewarding refusal there would be reward misspecification,
not safety. Multi-label records are excluded if ANY label is Self-Harm.

BENIGN side: tasks_final.json, stratified equally across its ten reasoning
categories. Declared caveat (in the output metadata): benign prompts are all
reasoning-task style, so 'benign' is partially confounded with task-style surface
features; the reward keys on the harmful flag, but what the policy learns to
condition on inherits this confound.

Run:  python3 pt11b_build_refusal_pool.py [--n-harmful 250] [--n-benign 250]
Out:  data/grpo_refusal_prompts.json  (pt11 --reward refusal-format --prompts input)
"""
import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("pt11b")

EXCLUDED_CATEGORIES = {"Self-Harm"}


def build_pool(star1_path: str, tasks_path: str, n_harmful: int, n_benign: int, seed: int) -> dict:
    rng = random.Random(seed)

    star = json.load(open(star1_path))
    by_cat = defaultdict(list)
    n_excluded = 0
    seen = set()
    for r in star:
        cats = r.get("category") or ["uncategorised"]
        if any(c in EXCLUDED_CATEGORIES for c in cats):
            n_excluded += 1
            continue
        key = " ".join(r["prompt"].split()).lower()
        if key in seen:
            continue
        seen.add(key)
        by_cat[cats[0]].append(r)
    log.info("STAR-1 eligible: %d records over %d categories (%d Self-Harm-touching excluded)",
             sum(len(v) for v in by_cat.values()), len(by_cat), n_excluded)

    # round-robin over categories so no single category dominates
    for v in by_cat.values():
        rng.shuffle(v)
    order = sorted(by_cat)
    harmful, idx = [], 0
    while len(harmful) < n_harmful and any(by_cat[c] for c in order):
        c = order[idx % len(order)]
        idx += 1
        if by_cat[c]:
            r = by_cat[c].pop()
            harmful.append({"id": f"harmful_{len(harmful):03d}", "prompt": r["prompt"],
                            "harmful": True, "source": "STAR-1", "category": c})

    tasks = json.load(open(tasks_path))
    t_by_cat = defaultdict(list)
    for t in tasks:
        t_by_cat[t["category"]].append(t)
    per_cat = n_benign // len(t_by_cat)
    benign = []
    for c in sorted(t_by_cat):
        pool = sorted(t_by_cat[c], key=lambda x: x["id"])
        rng.shuffle(pool)
        for t in pool[:per_cat]:
            benign.append({"id": f"benign_{len(benign):03d}", "prompt": t["prompt"],
                           "harmful": False, "source": "tasks_final", "category": c})

    records = harmful + benign
    rng.shuffle(records)
    return {
        "meta": {
            "seed": seed,
            "n_harmful": len(harmful),
            "n_benign": len(benign),
            "excluded_categories": sorted(EXCLUDED_CATEGORIES),
            "exclusion_rationale": "refusal is the wrong target behaviour for self-harm disclosures under a binary refusal rubric",
            "benign_caveat": "benign pool is reasoning-task style only; benign/harmful partially confounded with task-style surface features",
        },
        "records": records,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--star1", default="data/safety_star1_sft.json")
    ap.add_argument("--tasks", default="data/tasks_final.json")
    ap.add_argument("--n-harmful", type=int, default=250)
    ap.add_argument("--n-benign", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/grpo_refusal_prompts.json")
    args = ap.parse_args(argv)

    pool = build_pool(args.star1, args.tasks, args.n_harmful, args.n_benign, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # pt11's prompt loader expects a bare list (a top-level dict is treated as
    # {id: row} pairs); metadata goes to a sidecar.
    json.dump(pool["records"], open(args.out, "w"), indent=1)
    json.dump(pool["meta"], open(str(args.out) + ".meta.json", "w"), indent=1)
    from collections import Counter
    hc = Counter(r["category"] for r in pool["records"] if r["harmful"])
    log.info("harmful by category: %s", dict(hc))
    log.info("wrote %d records (%d harmful / %d benign) -> %s",
             len(pool["records"]), pool["meta"]["n_harmful"], pool["meta"]["n_benign"], args.out)


if __name__ == "__main__":
    main()
