# P5 powered-generic primary analysis contract

**Status:** prospective; no powered-study outcomes exist  
**Independent unit:** task/prompt  
**Primary contrast:** owned full-FT safety seed 42 minus owned matched-control seed 42  
**Primary family:** four reasoning-behaviour sentence fractions

## Estimation

For each endpoint, form one paired difference per task with both checkpoint annotations present. The point estimand is the mean task-level safety-minus-control difference over complete pairs. Checkpoint rows and sentence units are nested observations and are never resampled independently.

Report the number of planned tasks, complete pairs, missing rows by arm, complete pairs per category, per-arm task means, the paired mean difference, and a 95% category-stratified paired-bootstrap interval using 10,000 draws and seed 20260808. Missing endpoints remain missing; endpoint-specific complete-pair denominators may differ.

## Testing and multiplicity

Use a two-sided task-level sign-flip test of the mean paired difference. With 100 tasks, use 100,000 deterministic Monte Carlo sign-flip draws and seed 20260808; include the observed configuration and calculate `(extreme + 1)/(draws + 1)`. Apply Holm correction once across the four primary endpoint p-values.

The sign-flip p-value is a paired null diagnostic under its symmetry/exchangeability assumption. The bootstrap interval and effect magnitude remain primary reporting objects. Do not infer a post-training recipe effect from one owned training seed.

## Secondary analyses

Public STAR1 versus base R1 and all DeepScaleR comparisons are secondary. They use the same task-paired estimand and missingness rules but remain outside the primary four-test family and must be labelled as secondary checkpoint comparisons. Their power is not represented as equal to the owned primary contrast.

## Guardrails

- Validate every input against generation-row, prefix-row, source-unit, annotation, and scorer hashes before analysis.
- Require category identity to agree across checkpoint roles for each task.
- Reject conflicting duplicate task-role rows.
- Do not pool endpoints, roles, sentences, or geometry statistics into a composite.
- Do not inspect or use the P5 pilot's signed arm effects in the powered analysis plan.
- Representation displacement and behavioural change are separate estimands; any later association analysis is secondary and cannot identify a circuit or mechanism.
