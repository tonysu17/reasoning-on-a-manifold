"""Safety-reasoning extension (flagship arm). See ../../safety_reasoning_extension.md.

The S4 "post-training fingerprint" engine lives in ``refusal_direction.py``:
extract the refusal direction per recipe, measure how sharply it separates
harmful from harmless (the per-recipe fingerprint), and compare directions /
subspaces across recipes — deliberative-alignment (gpt-oss-20b) vs RLHF
(Qwen-Instruct) vs reasoning-distillation (R1-Distill) vs base. Same-ambient
recipes use cosine + principal angles; cross-architecture pairs of different
width (gpt-oss 2880-d vs R1 1536-d) use linear CKA on paired prompts.

Built and tested on synthetic geometry (tests/test_safety_refusal.py).

TODO (needs real activations / the Spark):
  - DSR (deliberative safety reasoning) span annotation schema (§6 of the plan);
  - early-token KL shallowness (Qi 2024) for S4;
  - a ``14_safety_geometry.py`` runner wiring this over extracted activations;
  - S2 benign/malicious off-manifold and S3 forged-vs-genuine probe.
"""

from src.safety.refusal_direction import (
    refusal_direction, project, directional_ablation,
    separation, recipe_fingerprint,
    recipe_direction_cosine, cross_recipe_cosines,
    category_refusal_subspace, recipe_principal_angles,
    linear_cka, effort_engagement,
)

__all__ = [
    "refusal_direction", "project", "directional_ablation",
    "separation", "recipe_fingerprint",
    "recipe_direction_cosine", "cross_recipe_cosines",
    "category_refusal_subspace", "recipe_principal_angles",
    "linear_cka", "effort_engagement",
]
