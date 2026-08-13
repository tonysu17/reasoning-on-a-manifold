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
  - fingerprint       : de-confounded scoring (held-out grouped d, permutation
                        null, bootstrap CI, pre-registered layer) + F6 length
                        controls. Use these, not analyze_recipes' in-sample argmax.
  - stimuli           : harmful/benign/matched stimulus loaders (real sets loaded
                        at run time; placeholder builtin for offline tests).
  - deliberation      : the DSR (deliberative safety reasoning) annotation schema
                        + heuristic detector + LLM-judge prompt.
  - annotate          : multi-annotator DSR pass (>=3 judges) + consensus.
  - agreement         : per-label inter-annotator kappa + F4 reliability gates.
  - cot_extraction    : CoT-spanning extraction — pool activations per DSR span of
                        the GENERATED chain (the H1/H2/H3 object), not the prompt
                        token (which extraction.py pools = refusal OUTPUT only).
  - capability        : the mandatory H1 difficulty/capability control (CF-11).
  - forgery           : forged-vs-genuine policy-deliberation dataset builder (S3).
  - shallowness       : early-token KL (Qi 2024) — first-token-reflex discriminator.
Runners: ``14_safety_geometry.py`` (S4 geometry), ``14b_annotate_dsr.py`` (DSR).

Built and tested on synthetic geometry / mocked judges (tests/test_safety_*.py).

Honest status of the package (red-team 2026-06-13): ``extraction.py`` pools the
last PROMPT token, i.e. it measures refusal OUTPUT geometry (Arditi), not safety
reasoning. ``cot_extraction.py`` is the corrected reasoning-geometry path. The
remaining run-time work needs the Spark: generating + DSR-annotating real safety
chains, then the F1 extraction; S2 off-manifold; S3 probe + patching.
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
from src.safety import (
    stimuli, deliberation, shallowness, extraction,
    annotate, agreement, cot_extraction, capability, forgery, fingerprint,
)

__all__ = [
    "refusal_direction", "project", "directional_ablation",
    "separation", "recipe_fingerprint",
    "recipe_direction_cosine", "cross_recipe_cosines",
    "category_refusal_subspace", "recipe_principal_angles",
    "linear_cka", "effort_engagement",
    "analyze_recipes", "summarise", "build_per_model", "load_safety_activations",
    "stimuli", "deliberation", "shallowness",
    "annotate", "agreement", "cot_extraction", "capability", "forgery", "fingerprint",
]
