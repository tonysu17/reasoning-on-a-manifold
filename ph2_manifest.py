#!/usr/bin/env python3
"""ph2 — Phase-2 evaluation task manifest (prereg §3 as amended by AMENDMENT A1).

The original draw rule was unsatisfiable (the annotation corpus covers all 1000 pool ids);
A1 replaces it with FRESH generation: 100 tasks, 10/category, via src.task_gen.generate_tasks
(same categories/prompt schema, Sonnet via the lab proxy), re-numbered to ids _102+ per
category — disjoint from the corpus (000-099) and E8 (097-101) by construction, then verified
programmatically. Drawn ONCE, hashed, committed; this tool refuses to overwrite.

Usage:
  python3 ph2_manifest.py --generate     # needs CLAUDE_PROXY_URL/KEY; ~$0.5; Tony-gated
  python3 ph2_manifest.py --verify       # re-check an existing manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
MANIFEST = ROOT / "results/prereg/phase2_task_manifest.json"
SEED = 20260808
N_PER_CAT = 10
# A1 correction (2026-08-08, pre-draw): corpus ids reach _116 in some categories, so the
# fresh-id start is DYNAMIC = max excluded suffix + 1, computed at draw time and recorded in
# the manifest. ID_START below is only the static floor used by unit tests.
ID_START = 102


def dynamic_id_start(exclusions: set[str]) -> int:
    return max(int(i.rsplit("_", 1)[1]) for i in exclusions) + 1


def _prefix(cat: str) -> str:
    return cat[:4].upper()


def build_exclusions(root: Path = ROOT) -> set[str]:
    corpus = {r["task_id"] for r in
              json.loads((root / "data/annotated_R1-1.5B.json").read_text())}
    e8 = set(json.loads(
        (root / "results/eval/R1-1.5B__E1/eval_task_ids.json").read_text())["task_ids"])
    return corpus | e8


def assign_ids(tasks: list[dict], id_start: int = ID_START) -> list[dict]:
    """Deterministically renumber fresh tasks to {PREFIX}_{id_start+i} per category."""
    out, counters = [], {}
    for t in tasks:
        cat = t["category"]
        i = counters.get(cat, 0)
        counters[cat] = i + 1
        out.append({**t, "id": f"{_prefix(cat)}_{id_start + i:03d}"})
    return out


def verify_disjoint(tasks: list[dict], exclusions: set[str]) -> None:
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in manifest")
    clash = set(ids) & exclusions
    if clash:
        raise ValueError(f"manifest ids collide with exclusion set: {sorted(clash)[:5]}")


def manifest_hash(tasks: list[dict]) -> str:
    payload = json.dumps(sorted([t["id"] for t in tasks])).encode()
    return hashlib.sha256(payload).hexdigest()


def write_manifest(tasks: list[dict], exclusions: set[str], gen_meta: dict,
                   path: Path = MANIFEST) -> dict:
    if path.exists():
        raise SystemExit(f"REFUSING to overwrite existing manifest: {path} "
                         "(immutable once drawn — prereg A1)")
    verify_disjoint(tasks, exclusions)
    cats: dict[str, int] = {}
    for t in tasks:
        cats[t["category"]] = cats.get(t["category"], 0) + 1
    if sorted(cats.values()) != [N_PER_CAT] * len(cats):
        raise ValueError(f"stratification violated: {cats}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip()
    doc = {"amendment": "A1", "seed": SEED, "n": len(tasks),
           "id_start": ID_START, "per_category": cats,
           "n_excluded_pool": len(exclusions),
           "ids_sha256": manifest_hash(tasks),
           "generator": gen_meta, "git_commit": commit,
           "tasks": tasks}
    path.write_text(json.dumps(doc, indent=1))
    return doc


def cmd_generate() -> None:
    import os
    if not (os.environ.get("CLAUDE_PROXY_URL") and os.environ.get("CLAUDE_PROXY_KEY")):
        raise SystemExit("CLAUDE_PROXY_URL/KEY not set — manifest generation is proxy-gated "
                         "(~$0.5; needs Tony's go).")
    sys.path.insert(0, str(ROOT))
    from src.task_gen import generate_tasks, CATEGORIES  # noqa: E402
    raw = generate_tasks(categories=CATEGORIES, n_per_category=N_PER_CAT, batch_size=5)
    if len(raw) < N_PER_CAT * len(CATEGORIES):
        raise SystemExit(f"generator returned {len(raw)} tasks — incomplete; not committing")
    exclusions = build_exclusions()
    start = dynamic_id_start(exclusions)
    tasks = assign_ids(raw, id_start=start)
    doc = write_manifest(tasks, exclusions,
                         {"module": "src.task_gen.generate_tasks",
                          "n_per_category": N_PER_CAT, "batch_size": 5, "id_start": start,
                          "note": "fresh draw per AMENDMENT A1 + same-day correction: "
                                  "id_start = max excluded suffix + 1 (dynamic)"})
    print(json.dumps({k: doc[k] for k in ("n", "per_category", "ids_sha256")}, indent=1))


def cmd_verify() -> None:
    doc = json.loads(MANIFEST.read_text())
    verify_disjoint(doc["tasks"], build_exclusions())
    ok = manifest_hash(doc["tasks"]) == doc["ids_sha256"]
    print(json.dumps({"hash_ok": ok, "n": doc["n"], "per_category": doc["per_category"]}))
    if not ok:
        raise SystemExit("HASH MISMATCH — manifest was modified after drawing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.generate:
        cmd_generate()
    elif a.verify:
        cmd_verify()
    else:
        ap.print_help()
