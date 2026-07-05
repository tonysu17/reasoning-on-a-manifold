"""Safety post-training as an intervention on generic reasoning geometry.

This package implements the *post-training spillover* extension (see
``../post_training_spillover_extension.md``): apply a safety post-training step
to a reasoning model (R1-1.5B) and measure how the geometry of its **generic,
non-safety** reasoning behaviours shifts.

Modules
-------
- ``contrastive`` : build an LLM-generated **contrastive dataset** of harmful /
  non-harmful prompts (+ safe target responses) via the Bedrock proxy, and
  format it for supervised fine-tuning. A deterministic offline ``mock_*`` path
  lets the pipeline be exercised without proxy credentials.
- ``sft``         : LoRA supervised fine-tuning of a reasoning model on the
  contrastive data (prompt-masked completion-only loss), with dose-response and
  a size-matched non-safety control. Heavy deps (torch/peft) are imported lazily
  inside functions so the module is cheap to import in CPU/annotation contexts.
- ``spillover``   : numpy-only before/after geometry diff (principal angles,
  effective dimensionality, centroid/mean-direction drift) consumed by the
  measurement runner.

The runners ``pt01_generate_contrastive.py`` / ``pt02_train_safety_lora.py`` /
``pt03_measure_spillover.py`` at the repo root orchestrate these.

DEFENSIVE-USE NOTE: the contrastive dataset pairs harmful *requests* (the things
a safety-aligned model should refuse) with *refusals* — it never contains
operational harmful content. Its purpose is safety alignment + the study of
safety-reasoning geometry.
"""

__all__ = ["contrastive", "sft", "spillover"]
