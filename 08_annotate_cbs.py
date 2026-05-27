"""
Phase 8: CBS-tier + cross-domain annotation runner (M1).

Adds `cbs_*` fields to every adding-knowledge / deduction sentence in a
Phase-3 annotated chain file.

Modes
-----
  default            Full annotation, single seed.
  --pilot            Stratified 100-sentence pilot (dual seed + kappa).
                     Used to gate the full run (synthesis §P0.2).
  --dual-seed-kappa  Full corpus dual seed + kappa report.
  --build-anchors    Emit results/cbs/anchor_candidates.csv for P0.2 curation.

Outputs
-------
  --out JSON: Phase 3 + cbs_* fields per sentence.
  results/cbs/{model}/kappa_run1_run2.json     (--dual-seed-kappa, --pilot)
  results/cbs/{model}/pilot_for_human_review.csv (--pilot)
  results/cbs/anchor_candidates.csv              (--build-anchors)

Synthesis-plan reference: §M1.3.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

from src.cbs.annotation import (
    ProxyClient,
    TARGETED_BEHAVIOURS,
    annotate_chains_cbs,
    annotate_sentence_cbs,
    annotate_task_domain,
    build_anchor_candidates_csv,
    cohen_kappa_three_tier,
)

logger = logging.getLogger("08_annotate_cbs")


# Pass criteria from synthesis §P0.2.
PILOT_KAPPA_FLOOR = 0.5
PILOT_TIER3_RATE_FLOOR = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--in", dest="in_path",
                   default="data/annotated_R1-1.5B.json", type=Path)
    p.add_argument("--out",
                   default="data/chains_cbs_annotated_R1-1.5B.json", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dual-seed-kappa", action="store_true")
    p.add_argument("--pilot", action="store_true",
                   help="Stratified pilot run (synthesis §P0.2).")
    p.add_argument("--pilot-size", type=int, default=100)
    p.add_argument("--build-anchors", action="store_true",
                   help="Emit anchor_candidates.csv for P0.2 curation; "
                        "does not run the annotator.")
    p.add_argument("--anchors-per-category", type=int, default=6)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--model-suffix", default="R1-1.5B")
    p.add_argument("--out-dir", default=None,
                   type=lambda s: Path(s) if s else None,
                   help="default: results/cbs/{model-suffix}")
    p.add_argument("--anchor-block-path", default=None, type=Path,
                   help="If set, replaces PLACEHOLDER_ANCHOR_BLOCK with the "
                        "text contents of this file. Locked after P0.2.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_anchor_block(path: Path | None) -> str:
    from src.cbs.annotation import PLACEHOLDER_ANCHOR_BLOCK
    if path is None:
        return PLACEHOLDER_ANCHOR_BLOCK
    return path.read_text()


def _sample_pilot_sentences(
    chains: list[dict],
    *,
    n_target: int,
    behaviours: tuple[str, ...] = TARGETED_BEHAVIOURS,
    seed: int = 0,
) -> list[dict]:
    """Sample sentences stratified by chain.category, picking ceil(n/k) per
    category where k = number of categories present. Returns enriched dicts
    with the context needed to call `annotate_sentence_cbs`."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for chain in chains:
        cat = chain.get("category", "unknown")
        spans = chain.get("annotations", []) or []
        for i, span in enumerate(spans):
            if span.get("label") in behaviours:
                by_cat[cat].append({
                    "task_id": chain.get("task_id", ""),
                    "category": cat,
                    "sentence_idx": i,
                    "sentence": span.get("text", ""),
                    "behaviour": span.get("label", ""),
                    "task_prompt": chain.get("instruction", ""),
                    "prev_sentences": [s.get("text", "") for s in spans[max(0, i - 3):i]],
                })
    rng = random.Random(seed)
    cats = sorted(by_cat.keys())
    if not cats:
        return []
    per_cat = max(1, (n_target + len(cats) - 1) // len(cats))
    out: list[dict] = []
    for cat in cats:
        bucket = by_cat[cat]
        k = min(per_cat, len(bucket))
        out.extend(rng.sample(bucket, k))
    rng.shuffle(out)
    return out[:n_target]


def run_pilot(args: argparse.Namespace, out_dir: Path) -> int:
    with open(args.in_path) as f:
        chains = json.load(f)
    sentences = _sample_pilot_sentences(
        chains, n_target=args.pilot_size, seed=args.seed,
    )
    if not sentences:
        logger.error("no targeted sentences found in %s", args.in_path)
        return 2
    logger.info("pilot: sampled %d sentences across %d categories",
                len(sentences), len({s["category"] for s in sentences}))

    client = ProxyClient()
    anchor_block = _load_anchor_block(args.anchor_block_path)

    task_domain_cache: dict[str, str] = {}

    def _classify(seed_value: int) -> list[dict]:
        from tqdm import tqdm
        results = []
        for s in tqdm(sentences, desc=f"pilot seed={seed_value}"):
            tid = s["task_id"]
            if tid not in task_domain_cache:
                task_domain_cache[tid] = annotate_task_domain(
                    {"task_id": tid, "instruction": s["task_prompt"]},
                    client,
                )
            try:
                cbs = annotate_sentence_cbs(
                    sentence=s["sentence"], behaviour=s["behaviour"],
                    task_domain=task_domain_cache[tid],
                    task_prompt=s["task_prompt"],
                    prev_sentences=s["prev_sentences"],
                    client=client, seed=seed_value,
                    anchor_block=anchor_block,
                )
                results.append({**s, "tier": cbs.tier,
                                "knowledge_domain": cbs.knowledge_domain,
                                "cross_domain": cbs.cross_domain,
                                "rationale": cbs.rationale,
                                "confidence": cbs.confidence,
                                "error": None})
            except Exception as exc:  # noqa: BLE001
                logger.warning("pilot annotation failed: %s", exc)
                results.append({**s, "tier": None, "error": str(exc)})
        return results

    run1 = _classify(0)
    run2 = _classify(1)

    # Filter to pairs where both seeds returned a tier; kappa is only meaningful
    # over the intersection.
    pairs = [(r1, r2) for r1, r2 in zip(run1, run2)
             if r1["tier"] is not None and r2["tier"] is not None]
    if not pairs:
        logger.error("pilot produced zero usable tier pairs (all failures)")
        return 2

    a = [r1["tier"] for r1, _ in pairs]
    b = [r2["tier"] for _, r2 in pairs]
    kappa = cohen_kappa_three_tier(a, b)

    from collections import Counter
    dist1 = dict(Counter(a))
    dist2 = dict(Counter(b))
    tier3_rate1 = dist1.get(3, 0) / len(a)
    tier3_rate2 = dist2.get(3, 0) / len(b)

    passes_kappa = kappa >= PILOT_KAPPA_FLOOR
    passes_tier3 = max(tier3_rate1, tier3_rate2) >= PILOT_TIER3_RATE_FLOOR

    report = {
        "kappa": kappa,
        "n_pairs": len(pairs),
        "n_sampled": len(sentences),
        "n_failed_run1": sum(1 for r in run1 if r["tier"] is None),
        "n_failed_run2": sum(1 for r in run2 if r["tier"] is None),
        "tier_distribution_run1": dist1,
        "tier_distribution_run2": dist2,
        "tier3_rate_run1": tier3_rate1,
        "tier3_rate_run2": tier3_rate2,
        "anchor_block_locked": args.anchor_block_path is not None,
        "pass_criteria": {
            "kappa_floor": PILOT_KAPPA_FLOOR,
            "tier3_rate_floor": PILOT_TIER3_RATE_FLOOR,
        },
        "passes_kappa": passes_kappa,
        "passes_tier3_rate": passes_tier3,
        "passes_all": passes_kappa and passes_tier3,
        "note": ("pilot-only" if not args.anchor_block_path
                 else "anchors-locked"),
    }
    (out_dir / "kappa_run1_run2.json").write_text(json.dumps(report, indent=2))

    # Emit pilot CSV for human review on 50 sentences (synthesis §P0.2).
    pilot_csv = out_dir / "pilot_for_human_review.csv"
    fieldnames = ["task_id", "sentence_idx", "category", "behaviour",
                  "context", "sentence", "tier_run1", "tier_run2",
                  "rationale_run1", "human_label"]
    with open(pilot_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r1, r2 in pairs:
            w.writerow({
                "task_id": r1["task_id"],
                "sentence_idx": r1["sentence_idx"],
                "category": r1["category"],
                "behaviour": r1["behaviour"],
                "context": "\n".join(r1["prev_sentences"])[:1000],
                "sentence": r1["sentence"],
                "tier_run1": r1["tier"],
                "tier_run2": r2["tier"],
                "rationale_run1": r1.get("rationale", "")[:200],
                "human_label": "",
            })
    logger.info("pilot kappa=%.3f tier3=%.3f (run1) | %.3f (run2)  ->  %s",
                kappa, tier3_rate1, tier3_rate2,
                "PASS" if report["passes_all"] else "FAIL")

    if not report["passes_all"]:
        failstop_path = out_dir / "FAILSTOP_M1.md"
        failstop_path.write_text(_failstop_template(report))
        logger.error("pilot FAILED - wrote %s", failstop_path)
        return 1
    return 0


def _failstop_template(report: dict) -> str:
    return (
        "# FAILSTOP — M1 CBS pilot\n\n"
        f"Synthesis-plan reference: §P0.2 pass criteria.\n\n"
        f"## Failing values\n\n"
        f"- kappa = **{report['kappa']:.3f}** (floor {PILOT_KAPPA_FLOOR})\n"
        f"- tier-3 rate run1 = {report['tier3_rate_run1']:.3f}\n"
        f"- tier-3 rate run2 = {report['tier3_rate_run2']:.3f} "
        f"(floor {PILOT_TIER3_RATE_FLOOR})\n"
        f"- n_pairs = {report['n_pairs']} | failures: "
        f"{report['n_failed_run1']} (run1), {report['n_failed_run2']} (run2)\n\n"
        f"## Three options for the human\n\n"
        "1. Refine the CBS prompt and re-pilot (cheap; ~$2 + Tony's time).\n"
        "2. Widen the corpus by sampling 500 sentences and re-pilot.\n"
        "3. Declare the 1.5B model unsuited for the CBS-tier framing and "
        "switch to the binary cross-domain framing instead (the cross_domain "
        "flag is annotated in parallel and remains usable).\n\n"
        "## Data sample\n\n"
        f"```json\n{json.dumps(report, indent=2)}\n```\n"
    )


def run_dual_seed_full(args: argparse.Namespace, out_dir: Path) -> int:
    with open(args.in_path) as f:
        chains = json.load(f)
    client = ProxyClient()
    anchor_block = _load_anchor_block(args.anchor_block_path)

    out1 = annotate_chains_cbs(chains, client, seed=0,
                               max_workers=args.max_workers,
                               anchor_block=anchor_block)
    out2 = annotate_chains_cbs(chains, client, seed=1,
                               max_workers=args.max_workers,
                               anchor_block=anchor_block)
    # Save both runs side-by-side and compute kappa over the intersection.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out1, indent=2, ensure_ascii=False))
    side = args.out.with_suffix(".seed1.json")
    side.write_text(json.dumps(out2, indent=2, ensure_ascii=False))

    pairs = []
    for c1, c2 in zip(out1, out2):
        for s1, s2 in zip(c1.get("annotations", []), c2.get("annotations", [])):
            if "cbs_tier" in s1 and "cbs_tier" in s2:
                pairs.append((s1["cbs_tier"], s2["cbs_tier"]))
    kappa = cohen_kappa_three_tier([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else 0.0
    (out_dir / "kappa_run1_run2.json").write_text(json.dumps({
        "kappa": kappa, "n_pairs": len(pairs),
        "out_run1": str(args.out), "out_run2": str(side),
        "anchor_block_locked": args.anchor_block_path is not None,
    }, indent=2))
    logger.info("full dual-seed kappa=%.3f (n=%d)", kappa, len(pairs))
    return 0


def run_full_single_seed(args: argparse.Namespace) -> int:
    with open(args.in_path) as f:
        chains = json.load(f)
    client = ProxyClient()
    anchor_block = _load_anchor_block(args.anchor_block_path)

    out = annotate_chains_cbs(chains, client, seed=args.seed,
                              max_workers=args.max_workers,
                              anchor_block=anchor_block)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("wrote %s (%d chains)", args.out, len(out))
    return 0


def run_build_anchors(args: argparse.Namespace) -> int:
    # Anchor candidates live at results/cbs/anchor_candidates.csv (model-agnostic
    # — Tony curates once, the locked block is shared across models).
    out_csv = Path("results/cbs/anchor_candidates.csv")
    client = None
    try:
        client = ProxyClient()
    except RuntimeError as exc:
        logger.warning("ProxyClient unavailable (%s) - falling back to "
                       "candidates without tier estimates.", exc)
    build_anchor_candidates_csv(
        annotated_chains_path=args.in_path,
        out_csv=out_csv,
        client=client,
        n_per_category=args.anchors_per_category,
        seed=args.seed,
        rank_with_pilot_tier=client is not None,
    )
    print(f"\n>> Anchor candidates written to {out_csv}")
    print(">> NEXT STEP (human task): Tony picks 15 anchors (5 per tier) and "
          "edits them into a locked anchor-block text file. Re-run "
          "08_annotate_cbs.py with --anchor-block-path <file> for the pilot.")
    return 0


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    out_dir = args.out_dir or Path("results/cbs") / args.model_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.build_anchors:
        return run_build_anchors(args)
    if args.pilot:
        return run_pilot(args, out_dir)
    if args.dual_seed_kappa:
        return run_dual_seed_full(args, out_dir)
    return run_full_single_seed(args)


if __name__ == "__main__":
    sys.exit(main())
