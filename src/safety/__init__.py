"""Safety-reasoning extension (flagship arm). See ../../safety_reasoning_extension.md.

The S4 "post-training fingerprint" engine lives in ``refusal_direction.py``:
extract the refusal direction per recipe, measure how sharply it separates
harmful from harmless (the per-recipe fingerprint), and compare directions /
subspaces across recipes — deliberative-alignment (gpt-oss-20b) vs RLHF
(Qwen-Instruct) vs reasoning-distillation (R1-Distill) vs base. Same-ambient
recipes use cosine + principal angles; cross-architecture pairs of different
width (gpt-oss 2880-d vs R1 1536-d) use linear CKA on paired prompts.

Modules:
  - refusal_direction : the S4 fingerprint engine (directions, separation, CKA).
  - geometry_analysis : analyze_recipes / build_per_model — the runner core.
  - stimuli           : harmful/benign/matched stimulus loaders (real sets loaded
                        at run time; placeholder builtin for offline tests).
  - deliberation      : the DSR (deliberative safety reasoning) annotation schema
                        + heuristic detector + LLM-judge prompt.
  - shallowness       : early-token KL (Qi 2024) — first-token-reflex discriminator.
Runner: ``14_safety_geometry.py``.

Built and tested on synthetic geometry (tests/test_safety_refusal.py,
tests/test_safety_pipeline.py).

TODO (needs real activations / the Spark):
  - a safety extraction pass writing harmful_layer{L}.npy / harmless_layer{L}.npy;
  - the multi-annotator LLM-judge DSR pass (deliberation.build_dsr_judge_prompt);
  - S2 benign/malicious off-manifold and S3 forged-vs-genuine probe.
"""

from src.safety.refusal_direction import (
    refusal_direction, project, directional_ablation,
    separation, recipe_fingerprint,
    recipe_direction_cosine, cross_recipe_cosines,
    category_refusal_subspace, recipe_principal_angles,
    linear_cka, effort_engagement,
)
from src.safety.geometry_analysis import (
    analyze_recipes, summarise, build_per_model, load_safety_activations,
)
from src.safety import stimuli, deliberation, shallowness

__all__ = [
    "refusal_direction", "project", "directional_ablation",
    "separation", "recipe_fingerprint",
    "recipe_direction_cosine", "cross_recipe_cosines",
    "category_refusal_subspace", "recipe_principal_angles",
    "linear_cka", "effort_engagement",
    "analyze_recipes", "summarise", "build_per_model", "load_safety_activations",
    "stimuli", "deliberation", "shallowness",
]
