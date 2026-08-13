# Approved P5 powered-generic design freeze

**Status:** design approved; non-executable and Phase-2/spend-gated  
**Internal SHA-256:** `0bf279cbac7a7fae51c89642af9b1dd068274a2acb517622b30efa6221664e2a`  
**JSON file SHA-256:** `e36d44c20c6bde9dc197d0d2bccb6479c7134b654d8c29175f418a26375e2b6f`  
**Calls/spend authorized by this freeze:** none

## Frozen primary design

- Contrast: owned matched full-FT control seed 42 versus owned full-FT safety seed 42.
- Endpoints: backtracking, uncertainty estimation, example testing, and adding knowledge.
- Smallest absolute effect: 0.05 sentence fraction.
- Power: 80%.
- Planning: two-sided Bonferroni alpha 0.0125 per endpoint.
- Analysis: Holm across the four primary endpoints.
- Tasks: 100, balanced across ten categories.
- Analytic prefix: first 4,096 generated token IDs or earlier EOS.

Arm-blinded conservative task requirements are backtracking 80, uncertainty estimation 60, example testing 50, and adding knowledge 50. All fit the 100-task design. These are planning quantities, not outcome findings.

Public STAR1 versus base R1 and every DeepScaleR comparison are secondary. The design does not claim equal five-point power for every secondary cell. The owned comparison has one training seed and remains checkpoint-bounded rather than recipe-level replication.

## Remaining blockers

The Phase-2 shared-vanilla hashes, P5-owned generation manifest and spend authorization, prospective generic-annotation validation, and final prefix/analysis hash bindings are still absent. This document cannot launch generation, annotation, inference, API calls, pod work, or human annotation.
