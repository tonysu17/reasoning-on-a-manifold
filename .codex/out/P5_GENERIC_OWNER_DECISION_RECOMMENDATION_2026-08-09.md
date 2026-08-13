# P5 powered-generic owner-decision recommendation

**Status:** recommendation only; not yet frozen or executable  
**Pilot outcomes used:** arm-blinded unsigned variance and completeness only; no signed or arm-labelled effect inspected

## Recommended freeze

1. **Primary contrast:** owned full-FT safety seed 42 versus owned matched-control seed 42.
2. **Primary endpoint family:** the four reasoning-behaviour sentence fractions.
3. **Smallest effect of interest:** absolute change of 0.05 in sentence fraction.
4. **Target power:** 80%.
5. **Planning alpha:** conservative Bonferroni bound `0.05/4 = 0.0125`; final analysis uses Holm across the four primary endpoints.
6. **Task count:** retain the 100-task, ten-category Phase-2 manifest as the hard maximum for the shared generic study.
7. **Analytic prefix:** first 4,096 generated token IDs or earlier EOS.
8. **Secondary comparisons:** public STAR1 versus base R1, and every DeepScaleR comparison. Report effect estimates and intervals, but do not claim that the 100-task design was powered for every five-point secondary effect.

## Why this is the best cost-aware choice

The owned safety/control pair is the cleanest available checkpoint attribution contrast because the training comparison is matched and locally owned. Under the pilot's conservative unsigned paired-variance proxy, 80% power, a five-point absolute effect, and four-test Bonferroni planning, the category-balanced requirements are:

- backtracking: 80 tasks;
- uncertainty estimation: 60 tasks;
- example testing: 50 tasks;
- adding knowledge: 50 tasks.

Thus the 100-task manifest covers all four primary cells while retaining a margin for incomplete pairs. These are conservative planning calculations, not guarantees; the pilot contains only 20 independent tasks and the annotation validation must still pass prospectively.

Keeping both checkpoint contrasts primary would force an unattractive choice between much more data and a larger detectable effect. Under the existing conservative eight-test calculation, a five-point effect for public STAR1 versus base uncertainty estimation requires 300 category-balanced tasks, while the other seven cells require 20–100. Expanding the shared manifest solely for that one noisy cell would increase generation and annotation cost substantially.

The 4,096-token prefix is recommended because the pilot variance is directly calibrated to that cap and 34/80 generic pilot responses hit it. A 2,048-token primary prefix would be cheaper to annotate, but it would define a different estimand and require new blinded variance evidence. Phase 2 may retain its sealed longer raw generation; P5 can apply the 4,096-token prefix analytically without regenerating the shared vanilla rows.

## Scientific boundary

This choice makes the owned safety/control comparison the primary test of whether safety post-training changes fine-grained reasoning behaviour. Public STAR1 and DeepScaleR remain useful checkpoint case studies, but not equally strong attribution contrasts. One owned training seed still cannot establish recipe-level replication.

## Authorization effect

If Tony approves this recommendation, Codex can populate and hash a versioned powered-generic protocol manifest as soon as Claude supplies the Phase-2 shared-artefact hashes. Approval does not itself authorize generation, annotation, model inference, API spend, or pod spend.
