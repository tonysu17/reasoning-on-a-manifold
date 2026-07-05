#!/usr/bin/env python3
"""LOCAL (laptop, API-only — no GPU) annotation of the E8 steered chains.

Reuses the PROVEN annotate_chains (identical code path to 03_annotate_chains.py /
the pod runner). Index-sharded so several shards run in parallel for speed, each
to its OWN file — shards are a disjoint round-robin partition of the chains
(idx % shards == shard-id), so there are no overlapping keys and the merge is a
plain concat. Every mode is fully resumable (checkpoint after each chain); re-run
to continue exactly where it stopped.

Modes:
  --shard-id I --shards N   annotate the slice {idx % N == I} of steering_results.json
                            -> annotated_steered.shardI.json  (resumable)
  --merge --shards N        concat the N shard files -> annotated_steered.json,
                            and write _incomplete_count (how many records still
                            have annotation_complete=False) for the wrapper loop.

Used by run_local_annotation.sh. Safe to run by hand too.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.annotation import annotate_chains  # noqa: E402

ANNOTATOR_DEFAULT = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"  # Sonnet (per Tony)
# Each steered variant is its own annotation unit (one chain per task×behaviour×
# method×alpha) — same dedup tuple the pod runner uses.
DEDUP = ("task_id", "behaviour", "method", "alpha")


def shard_path(eval_dir: Path, i: int) -> Path:
    return eval_dir / f"annotated_steered.shard{i}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True,
                    help="dir holding steering_results.json (the synced gen output)")
    ap.add_argument("--annotator", default=ANNOTATOR_DEFAULT)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--merge", action="store_true",
                    help="concat shard files -> annotated_steered.json and report incompletes")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    src = eval_dir / "steering_results.json"
    if not src.exists():
        sys.exit(f"missing {src} — sync the generated chains from the pod first")
    chains = json.loads(src.read_text())

    if args.merge:
        all_recs: list[dict] = []
        seen: set = set()
        for i in range(args.shards):
            p = shard_path(eval_dir, i)
            if not p.exists():
                print(f"  WARN: {p.name} missing (shard {i} produced nothing yet)")
                continue
            for r in json.loads(p.read_text()):
                key = tuple(r.get(k) for k in DEDUP)
                if key in seen:        # belt-and-braces: shards are disjoint, but never double-count
                    continue
                seen.add(key)
                all_recs.append(r)
        # The analysis (08 / delta_floor) does NOT filter on annotation_complete and
        # would otherwise compute a behaviour fraction over a chain's PARTIAL spans.
        # So write ONLY fully-complete records to annotated_steered.json; the missing
        # (still-incomplete) ones are simply skipped as unresolved pairs (N drops a
        # little, and is reported per cell) until the self-healing loop finishes them.
        complete = [r for r in all_recs if r.get("annotation_complete")]
        incomplete = len(all_recs) - len(complete)
        out = eval_dir / "annotated_steered.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(complete, indent=2, ensure_ascii=False))
        tmp.replace(out)
        # incomplete = (chains not yet annotated at all) + (chains with a failed chunk)
        not_started = len(chains) - len(all_recs)
        (eval_dir / "_incomplete_count").write_text(str(incomplete + not_started))
        print(f"merged: {len(complete)}/{len(chains)} complete -> {out.name}  "
              f"| partial={incomplete}  not-yet-started={not_started}")
        return

    # ── shard mode ──────────────────────────────────────────────────────────
    mine = [c for idx, c in enumerate(chains) if idx % args.shards == args.shard_id]
    out = shard_path(eval_dir, args.shard_id)
    print(f"shard {args.shard_id}/{args.shards}: {len(mine)} chains -> {out.name} "
          f"(annotator={args.annotator})")
    annotate_chains(
        mine, save_path=out, checkpoint_every=1,
        dedup_keys=DEDUP, model=args.annotator,
    )


if __name__ == "__main__":
    main()
