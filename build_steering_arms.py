#!/usr/bin/env python3
"""Build LINEAR PCA-projected steering vectors for every annotator arm, and
report cross-arm replication.

Multi-annotator robustness design: the SAME base model (R1-1.5B) is labelled by
three annotators (Sonnet-4.5 / Qwen3-235B / Nova-Pro). Each arm has its own
behaviour spans -> its own activations -> its own steering vectors. The headline
robustness check is whether the steering DIRECTIONS replicate across arms (high
cosine) despite differing label distributions.

Steering is purely LINEAR here:
  single_direction   = unit(mean(on) - mean(off))            (Venhoff)
  manifold_projected = project single_direction onto the top-k PCA subspace of
                       the behaviour's own activations, renormalised           (Huang)
No curvature / geodesic content — that is separate diagnostic work, deferred.

CPU-only: operates on the saved per-behaviour .npy activation matrices, so it
runs locally without the model. Arms whose activations are not yet on disk are
reported as PENDING and skipped (the other session's run_multiannotator_pipeline.sh
produces them).

Usage:
  python build_steering_arms.py                 # layer 27 (Huang), all arms found
  python build_steering_arms.py --layer 27 --k 1 3 5 10 auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.steering import build_steering_vectors, save_steering_vectors
from src.annotation import TARGET_BEHAVIOURS
from src.config import provenance

# arm tag -> SHORT name. MUST match run_multiannotator_pipeline.sh
# (Sonnet uses the canonical SHORT; the others are R1-1.5B__<tag>).
ARMS = {
    "sonnet-4.5": "R1-1.5B",
    "qwen3-235b": "R1-1.5B__qwen3-235b",
    "nova-pro":   "R1-1.5B__nova-pro",
}


def _arm_complete(act_dir: Path, layer: int) -> bool:
    return all((act_dir / f"{b}_layer{layer}.npy").exists() for b in TARGET_BEHAVIOURS)


def _print_arm_table(tag: str, vecs: dict, k_values: list) -> None:
    print(f"\n  {tag}: cos(single_direction, manifold_projected_k)")
    hdr = "  ".join(f"k={k}" for k in k_values)
    print(f"    {'behaviour':<24}{'n_on':>5}{'auto_k':>7}   {hdr}")
    for beh, d in vecs.items():
        s = d["single_direction"]
        row = "  ".join(f"{float(s @ d['manifold_projected'][k]):+.3f}" for k in k_values)
        print(f"    {beh:<24}{d['n_on']:>5}{d['auto_k']:>7}   {row}")


def _cross_arm_replication(built: dict, layer: int) -> None:
    """cos between arms' single-direction vectors, per behaviour. High => the
    steering direction is annotator-robust (the multi-annotator headline)."""
    tags = list(built)
    print("\n=== Cross-arm replication: cos(single_direction) between arms ===")
    pairs = [(a, b) for i, a in enumerate(tags) for b in tags[i + 1:]]
    print(f"  {'behaviour':<24}" + "".join(f"{a}~{b}".rjust(20) for a, b in pairs))
    for beh in TARGET_BEHAVIOURS:
        cells = []
        for a, b in pairs:
            va = built[a].get(beh, {}).get("single_direction")
            vb = built[b].get(beh, {}).get("single_direction")
            cells.append(f"{float(va @ vb):+.3f}".rjust(20) if va is not None and vb is not None
                         else "n/a".rjust(20))
        print(f"  {beh:<24}" + "".join(cells))
    print("  (cos→1 = the behaviour's steering direction is the same regardless of annotator)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=27,
                    help="Steering layer (default 27, Huang's choice for R1-1.5B).")
    ap.add_argument("--k", nargs="+", default=["1", "3", "5", "10", "auto"],
                    help="manifold-projection k values.")
    ap.add_argument("--variance-threshold", type=float, default=0.70)
    args = ap.parse_args()
    k_values = [int(k) if k != "auto" else "auto" for k in args.k]

    built: dict[str, dict] = {}
    print(f"Building LINEAR steering vectors at layer {args.layer} for {len(ARMS)} arms\n")
    for tag, short in ARMS.items():
        act_dir = Path(f"data/activations/{short}")
        if not _arm_complete(act_dir, args.layer):
            print(f"[PENDING] {tag:12s} ({short}): activations not on disk — skip "
                  f"(run_multiannotator_pipeline.sh produces this arm)")
            continue
        vecs = build_steering_vectors(act_dir, layer=args.layer, k_values=k_values,
                                      variance_threshold=args.variance_threshold)
        out_dir = Path(f"results/steering_vectors/{short}")
        save_steering_vectors(vecs, out_dir,
                              provenance=provenance(args, inputs=[str(act_dir)]))
        built[tag] = vecs
        print(f"[BUILT]   {tag:12s} ({short}) -> {out_dir}")
        _print_arm_table(tag, vecs, k_values)

    print(f"\nBuilt {len(built)}/{len(ARMS)} arms.")
    if len(built) >= 2:
        _cross_arm_replication(built, args.layer)
    else:
        missing = [t for t in ARMS if t not in built]
        print(f"Cross-arm replication needs >=2 arms; still pending: {', '.join(missing)} "
              f"(being generated in the multi-annotator session).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
