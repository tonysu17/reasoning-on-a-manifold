#!/usr/bin/env python3
"""
Verify the base model of DeepSeek-R1-Distill-Qwen-1.5B.

Companion document Extension 1 (Sec 3.1) assumes the base is
Qwen-2.5-Math-1.5B-Instruct. DeepSeek's own technical report does not
explicitly name the 1.5B-distill base variant. This script settles the
question empirically by comparing weight tensors between R1-Distill-Qwen-1.5B
and three candidate bases:
  - Qwen/Qwen2.5-Math-1.5B-Instruct   (current assumption)
  - Qwen/Qwen2.5-Math-1.5B            (math, no instruct tuning)
  - Qwen/Qwen2.5-1.5B                 (no math specialization)

Methodology:
  (1) Embedding matrix cosine similarity. Token embeddings are among the
      most-preserved tensors through distillation. cosine_sim > 0.999 means
      same base; ~0.7 means different base.
  (2) LM-head cosine similarity. Often tied to embed_tokens but verified
      independently.
  (3) Per-layer Frobenius-delta curves for attention.q_proj and mlp.gate_proj.
      The fractional change induced by distillation should be smallest and
      smoothest for the true base.

Output:
  results/base_model_verification/
    similarity_table.json
    per_layer_delta.png
    verdict.md

CPU-only; ~10-20 min wall-clock once models are cached locally.

Usage:
  python verify_base_model.py
  python verify_base_model.py --candidates qwen-math-instruct qwen-math
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Candidate registry

DISTILLED_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

CANDIDATES = {
    "qwen-math-instruct": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "qwen-math":          "Qwen/Qwen2.5-Math-1.5B",
    "qwen-base":          "Qwen/Qwen2.5-1.5B",
}


# Weight loading

def load_state_dict(model_id: str, device: str = "cpu") -> dict:
    """Load weights only; never instantiate the model graph beyond what's needed."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError(
            "transformers package required. Install with: pip install transformers"
        ) from e

    import torch

    logger.info(f"Loading weights: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=device,
    )
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    del model
    return sd


# Similarity primitives

def cosine_similarity_matrices(A: np.ndarray, B: np.ndarray) -> float:
    """Flatten and compute scalar cosine similarity. Handles shape mismatch by
    restricting to the common minimum along each axis (correct for embed_tokens
    when one model has a slightly different vocabulary size)."""
    if A.shape != B.shape:
        min_shape = tuple(min(a, b) for a, b in zip(A.shape, B.shape))
        A = A[tuple(slice(0, s) for s in min_shape)]
        B = B[tuple(slice(0, s) for s in min_shape)]
    a = A.flatten().astype(np.float64)
    b = B.flatten().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def fractional_delta(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius(A - B) / Frobenius(B). How much A diverges from B, relative
    to B's scale."""
    if A.shape != B.shape:
        return float("nan")
    num = np.linalg.norm(A - B)
    den = np.linalg.norm(B)
    if den == 0:
        return float("nan")
    return float(num / den)


# Layered comparison

def compare_top_level(distilled: dict, candidate: dict) -> dict:
    """Embedding + LM-head cosine similarities."""
    out = {}
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        if key in distilled and key in candidate:
            out[key] = cosine_similarity_matrices(distilled[key], candidate[key])
        else:
            out[key] = None
    return out


def per_layer_deltas(distilled: dict, candidate: dict) -> list:
    """For each transformer block, compute fractional delta on q_proj + gate_proj.

    Qwen2.5 architecture:
      model.layers.<i>.self_attn.q_proj.weight
      model.layers.<i>.mlp.gate_proj.weight
    """
    results = []
    layer_idx = 0
    while True:
        q_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        m_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        if q_key not in distilled or q_key not in candidate:
            break
        results.append({
            "layer": layer_idx,
            "q_proj_delta":    fractional_delta(distilled[q_key], candidate[q_key]),
            "gate_proj_delta": fractional_delta(distilled[m_key], candidate[m_key]),
        })
        layer_idx += 1
    return results


# Verdict logic

def rank_candidates(results: dict) -> list:
    """Lower aggregate delta = better base candidate.

    Aggregate score: mean of (1 - embed_cos, 1 - head_cos, mean(q_deltas),
    mean(gate_deltas)).
    """
    scores = []
    for cand_key, payload in results.items():
        if payload.get("error"):
            continue
        top = payload["top_level"]
        embed_cos = top.get("model.embed_tokens.weight")
        head_cos  = top.get("lm_head.weight")
        embed_dist = (1 - embed_cos) if embed_cos is not None else 1.0
        head_dist  = (1 - head_cos)  if head_cos  is not None else 1.0
        layer_q = np.nanmean([d["q_proj_delta"]    for d in payload["per_layer"]])
        layer_m = np.nanmean([d["gate_proj_delta"] for d in payload["per_layer"]])
        agg = float(np.mean([embed_dist, head_dist, layer_q, layer_m]))
        scores.append((cand_key, agg))
    return sorted(scores, key=lambda x: x[1])


def write_verdict(results: dict, ranking: list, out_path: Path) -> None:
    lines = [
        "# Base-model verification - DeepSeek-R1-Distill-Qwen-1.5B",
        "",
        f"Distilled model: `{DISTILLED_MODEL}`",
        "",
        "## Aggregate ranking (lower = closer base)",
        "",
        "| Rank | Candidate | HF ID | Aggregate delta |",
        "|------|-----------|-------|------------------|",
    ]
    for i, (key, score) in enumerate(ranking, 1):
        lines.append(f"| {i} | `{key}` | `{CANDIDATES[key]}` | {score:.6f} |")

    lines += [
        "",
        "## Top-level cosine similarity",
        "",
        "| Candidate | embed_tokens cos | lm_head cos |",
        "|-----------|------------------|-------------|",
    ]
    for cand_key, payload in results.items():
        if payload.get("error"):
            lines.append(f"| `{cand_key}` | ERROR: {payload['error']} | - |")
            continue
        e = payload["top_level"].get("model.embed_tokens.weight")
        h = payload["top_level"].get("lm_head.weight")
        e_s = f"{e:.6f}" if e is not None else "-"
        h_s = f"{h:.6f}" if h is not None else "-"
        lines.append(f"| `{cand_key}` | {e_s} | {h_s} |")

    lines += [
        "",
        "## Verdict",
        "",
    ]
    if ranking:
        winner = ranking[0][0]
        winner_score = ranking[0][1]
        runner_up_score = ranking[1][1] if len(ranking) > 1 else float("inf")
        margin = runner_up_score - winner_score
        if winner_score < 0.05 and margin > 0.05:
            lines.append(f"**Confident:** `{winner}` (`{CANDIDATES[winner]}`) is the base model of R1-Distill-Qwen-1.5B.")
        elif winner_score < 0.10:
            lines.append(f"**Likely:** `{winner}` (`{CANDIDATES[winner]}`) is the base; margin over runner-up is {margin:.4f}.")
        else:
            lines.append(f"**Inconclusive:** lowest aggregate delta is {winner_score:.4f}; no candidate is an obvious base. Possible: (i) distillation modified weights more than expected, (ii) the true base is not in the candidate list, (iii) tokenizer or architectural changes. Recommend manual inspection.")

    out_path.write_text("\n".join(lines))


# Main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES.keys()),
                        choices=list(CANDIDATES.keys()),
                        help="Candidate keys to compare against. Default: all three.")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/base_model_verification"),
                        help="Output directory.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the matplotlib plot.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    distilled = load_state_dict(DISTILLED_MODEL)

    results = {}
    for cand_key in args.candidates:
        cand_id = CANDIDATES[cand_key]
        logger.info(f"Comparing against candidate: {cand_key} ({cand_id})")
        try:
            cand = load_state_dict(cand_id)
            top   = compare_top_level(distilled, cand)
            layer = per_layer_deltas(distilled, cand)
            results[cand_key] = {"top_level": top, "per_layer": layer}
            del cand
        except Exception as e:
            logger.error(f"Failed to compare {cand_id}: {e}")
            results[cand_key] = {"error": str(e)}

    (args.out_dir / "similarity_table.json").write_text(json.dumps(results, indent=2))

    ranking = rank_candidates(results)
    write_verdict(results, ranking, args.out_dir / "verdict.md")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
            for cand_key, payload in results.items():
                if payload.get("error"):
                    continue
                layers = [d["layer"] for d in payload["per_layer"]]
                q = [d["q_proj_delta"]    for d in payload["per_layer"]]
                m = [d["gate_proj_delta"] for d in payload["per_layer"]]
                axes[0].plot(layers, q, marker="o", label=cand_key)
                axes[1].plot(layers, m, marker="o", label=cand_key)
            axes[0].set(title="attn.q_proj fractional delta vs layer", xlabel="layer",
                        ylabel="||distilled - candidate||F / ||candidate||F")
            axes[1].set(title="mlp.gate_proj fractional delta vs layer", xlabel="layer")
            axes[0].legend(); axes[1].legend()
            axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.out_dir / "per_layer_delta.png", dpi=120)
            logger.info(f"Plot saved: {args.out_dir / 'per_layer_delta.png'}")
        except ImportError:
            logger.warning("matplotlib not installed - skipping plot")

    print("\n" + "=" * 60)
    print("BASE MODEL VERIFICATION - RANKING (lowest = best base)")
    print("=" * 60)
    for i, (key, score) in enumerate(ranking, 1):
        print(f"  {i}. {key:25s}  aggregate delta = {score:.6f}   ({CANDIDATES[key]})")
    print()
    print(f"Full verdict: {args.out_dir / 'verdict.md'}")
    print(f"Raw data:     {args.out_dir / 'similarity_table.json'}")


if __name__ == "__main__":
    main()
