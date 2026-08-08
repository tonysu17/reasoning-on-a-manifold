# DRAFT — Phase 2 causal transport of the grounded backtracking frame

**DRAFT — Tony seals; this document authorises no run, spend, result regeneration, or thesis
edit.**  
**Draft date:** 2026-08-02  
**Parent plan:** `post_training_geometry_unified_plan_2026-08-02.md`, Phase 2 and amendments
A2, A3, A4, A6, A8, A10.  
**Phase-0 dependency:** pod s3 remains **AWAITING RESULT**. No Phase-2 launch until its
full-sequence state extraction establishes the disposition of the L17 contraction-calibration
cell and the Phase-0 closure gate is signed off.

## 1. Question and claim boundary

For the existing base-model backtracking frame grounded at L17, width 2, what happens across a
verified post-training boundary: is the tested coordinate retained, rescaled, rotated,
disabled, or decoupled from behaviour?

The frame is one grounded backtracking coordinate. E10.3 did not ground the other behaviour
transfers, so no result here generalises to uncertainty-estimation, example-testing,
adding-knowledge, “reasoning geometry” globally, another layer family, or another intervention
class. “Disabled” is always bounded to the tested target model, layers, widths, task battery,
search family, floors, and detectable effect.

## 2. Checkpoints and prerequisites

Proposed primary boundaries:

1. `UCSC-VLAA/STAR1-R1-Distill-1.5B` (safety full-SFT descendant), compared with the
   full-fine-tune non-safety control where a regime-matched descriptive comparator is needed.
2. `agentica-org/DeepScaleR-1.5B-Preview` (RLVR case study with the strongest current public
   provenance).

A tool-use descendant is not in this launch. It may be added only after the Phase-4A replay
gate and a separate amendment. Public checkpoints without training seeds remain checkpoint
case studies, not recipe-level replications.

Hard prerequisites:

- immutable checkpoint commits and metadata hashes recorded; weight lineage checked wherever
  direct coordinates are assumed;
- byte-identical input IDs supplied from the R1 tokenizer alias for every matched-input arm;
- identical layer hook location and precision/reload controls;
- pod-s3 status and any permitted use of the contraction mapping recorded here:
  **[AWAITING POD-S3 — insert raw estimand, family-membership verdict, and provenance]**;
- target-model generation smoke test establishes storage, throughput, syntax integrity, and
  floor execution without inspecting the primary behavioural contrast;
- task IDs, discovery/evaluation split, prompts, seeds, annotator, frame-search space, doses,
  and multiplicity family sealed in a signed copy of this draft.

## 3. Units, splits, and pairing

- Independent unit: the problem/task. Tokens, sentences, layers, intervention positions, and
  generated samples are nested observations.
- Use the same problem IDs in base and target batteries. All base/target contrasts are paired
  by problem; samples and floor replicates are pooled within problem before resampling.
- Discovery data are used for target-model re-fitting, whitening/norm-scale estimation, and
  any permitted gain calibration. The held-out evaluation battery is not used for those
  choices.
- The evaluation battery must remain disjoint from the original frame-builder data. Any reuse
  of the earlier E8 held-out tasks is stated explicitly; it does not create a second
  independent replication.
- Bootstrap by problem, B >= 2,000. Use the same resampled problem indices across base and
  target for the paired contrast. Pool samples/subspace replicates within problem first.
- If task pairing is broken for any checkpoint, do not substitute an unpaired analysis under
  the paired sample size; route to the unpaired MDE contingency in section 6.

Exact task IDs and counts: **[AWAITING TONY SEAL]**.

## 4. Frames, interventions, and controls

### 4.1 Transported frame

Apply the sealed base R1 L17 width-2 backtracking frame without changing its orientation,
width, layer, or search hyperparameters. Report raw-coordinate, activation-norm-corrected, and
whitened transport as distinct predeclared arms. Norm/whitening statistics come from discovery
anchors only.

### 4.2 Target-model grounding gate

Re-derive the grounding-gate null distribution separately in each target model. The base
model's sealed absolute thresholds are reported for continuity but are not the sole target
verdict. Each target distribution includes:

- sham frames constructed under the same builder/search budget;
- random orthogonal frames with the same width and norm;
- neighbouring-layer controls fixed before the run;
- identity/reload and precision controls.

The target gate reports coordinate discrimination and state dependence against target-model
nulls. Decodability or target-gate passage alone is not the causal endpoint.

Target null replicate count and gate quantiles: **[AWAITING TONY SEAL]**.

### 4.3 Re-fitted frame

Re-fit a target-model frame on discovery data under the original sealed search rules. Do not
inherit a target-specific layer, width, dose, or early-stop decision from evaluation outcomes.
Compare transported and re-fitted frames on held-out aligned anchors, with whitening
sensitivity and random-alignment nulls. Frame angle is secondary; it cannot establish causal
retention by itself.

### 4.4 Causal intervention battery

Run both signs and the sealed dose grid for:

- transported base frame, raw coordinates;
- transported frame with the predeclared norm/gain correction;
- transported frame with the predeclared whitening correction;
- target re-fitted frame;
- sham frame;
- random orthogonal frame;
- count-matched floor;
- energy-matched floor.

An on-manifold floor motivated by arXiv:2602.02315 is optional. It enters only if its
construction, curvature estimand, hyperparameters, and comparison family are sealed before
generation; it is never added after seeing damage from a linear intervention.

The primary behavioural measure is the held-out sentence-fraction `Delta_floor` for
backtracking, suppression-oriented as matched floor minus active arm, pooled within task.
Per-1k backtracking-labelled sentences is a secondary variance-sensitivity estimand and cannot
replace the primary after outcome inspection. Report task performance and generation damage
(degeneracy, repetition, truncation, token count) alongside every intervention effect.

Annotator and correctness/verdict protocol: **[AWAITING TONY SEAL; non-builder required]**.

## 5. Primary and secondary estimands

Primary estimand for each target checkpoint:

> Paired change in held-out `Delta_floor` of the transported frame from R1 to the target,
> relative to the predeclared sham and matched-energy controls.

Let `Delta_R1` and `Delta_target` be the task-aggregated causal effects. Define attenuation
fraction `f = 1 - Delta_target / Delta_R1` only when the sign and denominator are stable under
the sealed base battery. Always report both raw effects and their intervals; do not report only
the ratio.

Secondary estimands:

- corrected-dose transported-frame effect;
- re-fitted-frame effect;
- transported/re-fitted held-out alignment angle against its null;
- target grounding-gate statistics against target nulls;
- task performance and damage contrasts;
- per-1k version of `Delta_floor`;
- descriptive norm/whitening gain and support-shift diagnostics.

Any trajectory-language claim reports a step/order-shuffle null. Without it, ordered hidden
states are described only as state occupancy.

## 6. MDE and battery consequence

The Phase-2 simulator (`.codex/out/ph2_mde_sim.json`, seed 20260802, 5,000 trials/cell) exactly
reproduces the Phase-0 legacy first cut at f*=1.293 for n=49 under that first cut's additional
arm/floor-unpairing assumption.

For the primary sentence-fraction estimand under the simulator's hierarchical model:

- task-paired across models: interpolated f*(50) = 0.435; first tested n with f* <= 0.5 is
  **50**;
- unpaired across models: f*(200) = 0.502 and f*(300) = 0.421; first tested n with f* <= 0.5
  is **300**.

These are sizing-model outputs, not achieved sensitivity. The paired model assigns 86.5% of
observed task-delta variance to a shared task component and uses an independent-sentence
binomial floor for arm noise. Sentence labels within a chain need not be independent, so the
paired n=50 result may be optimistic.

Proposed sealing rule:

1. Candidate primary battery: **n=50 paired tasks**, only if pre-outcome injection-recovery at
   this exact battery achieves >=80% power for f=0.5 under the executed annotation and floor
   pipeline.
2. If that gate fails, enlarge along the predeclared grid 100 -> 150 -> 200 -> 300 -> 400,
   stopping at the first count that passes, subject to Tony's spend/annotation approval.
3. If cross-model task pairing is unavailable, use **n=300** as the current tested-grid
   fallback candidate and rerun injection-recovery; do not assume the simulation guarantee.
4. If no approved battery passes, remove “retained” as a verdict. Report a positive surviving
   effect as **not disabled under the tested intervention**, with its detectable-attenuation
   bound; report an unresolved null as inconclusive rather than “disabled.”

Final sealed n, simulation version/hash, injection-recovery repetitions, and achieved power:
**[AWAITING TONY SEAL / PRE-OUTCOME CALIBRATION]**.

Injection-recovery mixes the grounded frame with an energy-matched sham frame at attenuation
fractions {0.1, 0.2, ..., 1.0}. It is executed without target-outcome labels. Each “retained”
claim states the achieved detectable attenuation bound.

## 7. Multiplicity and intervals

Proposed primary family: the two target checkpoints x the transported-frame primary
attenuation test, Holm-corrected across the two checkpoint cells. This family is
**[AWAITING TONY SEAL]**.

Separate predeclared families:

- target grounding gates: checkpoint x sealed target-null gate component;
- re-fit recovery: checkpoint x transported/corrected/re-fitted frame arm;
- damage/performance: checkpoint x predeclared damage metric;
- exploratory layer neighbours and optional on-manifold floor.

Do not move a test between families after observing its p-value. Report task-bootstrap
intervals and raw/preregistered adjusted p-values. Threshold crossing without effect magnitude
and sensitivity does not change the outcome label.

## 8. Outcome decision table

Apply this hierarchy per checkpoint after all gates and sensitivity checks:

| outcome | transported representation | re-fitted representation | causal battery | bounded interpretation |
|---|---|---|---|---|
| retained | target gate passes | aligned within null-calibrated bound | uncorrected transported effect passes floors and excludes attenuation >= sealed margin | same tested coordinate remains causally coupled within detectable attenuation |
| rescaled | raw transport misses retention bound; predeclared gain/whitening arm passes | aligned up to gain | corrected dose recovers the effect without excess damage | compatible coordinate with changed gain/coupling scale; correction is reported, never silent |
| rotated | transported frame fails/weakens | target re-fit passes and is misaligned beyond null | re-fitted intervention recovers against floors | tested variable remains accessible through a changed coordinate under the sealed re-fit family |
| decoupled | transported or re-fitted signal remains above target null | may align | causal effect falls below the sealed retention bound while syntax/damage controls pass | information remains recoverable but tested behavioural coupling weakens |
| disabled | transported and sealed re-fit gates/effects fail | fails within sealed search | neither intervention beats floors, and the battery had adequate sensitivity | implementation inaccessible under tested model/layers/widths/interventions; never global absence |

If more than one row appears applicable, use the first satisfied row in the table except that
decoupled takes precedence over disabled whenever representational signal remains. If
sensitivity is inadequate, use the section-6 downgraded vocabulary, not a five-way verdict.

## 9. Gates and stop rules

1. **Provenance/alignment gate:** immutable metadata, tensor lineage where required,
   byte-identical input IDs, layer-hook identity, and identity reload pass.
2. **Execution gate:** prompts, counts, energies, signs, and generation settings match the
   sealed manifest; missing/empty annotations are unresolved, never zero.
3. **Damage gate:** primary interpretation stops if the active arm causes predeclared excess
   degeneration, repetition, truncation, or task-performance loss relative to matched floors.
4. **Target-null gate:** target grounding is adjudicated against target sham/random-orthogonal
   distributions.
5. **MDE gate:** executed injection-recovery reaches the sealed f=0.5 sensitivity or the
   verdict vocabulary is downgraded before target outcomes are inspected.
6. **Stop rule inherited from Phase 0:** any gate failing twice after one good-faith fix stops
   that line. Report the bound; do not iterate layers, widths, doses, or prompts toward a
   positive result.

Missing checkpoints, an unmatched tokenizer, a broken task pair, or an unexecuted control are
not negative scientific results. They are execution/provenance failures and are labelled as
such.

## 10. Required provenance payload

Record for every artefact:

- script and config hashes, git commit and dirty state;
- exact checkpoint ID, immutable commit, local metadata/weight hash status;
- input task-list hash and split manifest;
- tokenizer source and byte-identical input-ID gate result;
- frame source, layer, width, search/dose seal, whitening/gain fit split;
- task, sample, sham, orthogonal, count-floor, and energy-floor counts;
- annotator model/commit, prompt, temperature, resume state, and missing-label counts;
- bootstrap seed/repetitions, multiplicity family, simulation version, and achieved MDE;
- explicit `amended` marker for any executed deviation, kept separate from evidence status.

## 11. Items Tony must seal

- [ ] primary checkpoints and immutable revisions;
- [ ] exact discovery/evaluation task IDs and final n;
- [ ] n=50 paired route, larger-grid contingency, or verdict downgrade;
- [ ] injection-recovery implementation and pass threshold;
- [ ] target-null replicate count and gate quantiles;
- [ ] norm, whitening, and gain-correction definitions;
- [ ] frame re-fit search space and held-out alignment null;
- [ ] signs, doses, samples, seeds, generation cap, and damage thresholds;
- [ ] non-builder annotator and task-performance verdict protocol;
- [ ] optional on-manifold floor included or excluded;
- [ ] multiplicity families;
- [ ] pod-s3 Phase-0 closure disposition;
- [ ] spend, annotation, and stop authority.
