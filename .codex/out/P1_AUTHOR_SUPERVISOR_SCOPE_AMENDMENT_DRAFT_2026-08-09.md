# P1 author–supervisor thesis and experimental scope amendment

**Prepared:** 9 August 2026  
**Status:** **DRAFT FOR JOINT SEAL — not yet authoritative; not a preregistration, result, or authority to spend**  
**Applies to:** UCL MSc dissertation and thesis-minimum post-training programme  
**Prior authority:** `thesis/_planning/THESIS_REVISION_CHARTER_2026-07-26.md`

Tony has recorded in chat that his supervisor agrees with the narrative pivot and proposed
programme. That records intent, but it is not the joint, dated written decision required by
the 26 July charter. This amendment takes effect only when Tony and the supervisor complete
the decision and seal lines below.

## 1. Decision requested

On joint seal, approve the following changes to the dissertation scope.

### Title

**Intervening on Machine Reasoning**  
*Residual-Stream Steering, Post-Training, and Behaviour-Indexed Representation Change*

### Narrative and contribution boundary

The thesis is a bounded study of intervention on reasoning at two timescales:

1. **Inference-time intervention:** selected residual-stream directions are modified while
   model weights remain fixed.
2. **Training-time intervention:** safety and matched non-safety post-training change model
   parameters, with persistent representational effects and potentially behavioural effects.
3. **Bridge:** causal transport tests whether the already specified backtracking handle
   remains accessible and causally coupled across selected post-training boundaries.
4. **Role of geometry:** geometry is representation-level measurement and analysis of what
   is available for intervention and what changes under it. It is not the thesis's focal
   object, a universal reasoning manifold, or a complete mechanistic explanation.

The work is described as **representation-level and mechanistically motivated**, not as
circuit identification or full mechanistic interpretability.

### Research questions

Approve three RQs, replacing the charter's four-RQ structure:

- **RQ1:** What measurable structure do annotation-indexed reasoning activations exhibit,
  and which properties are behaviour-specific under chain-aware controls?
- **RQ2:** How do estimated residual-stream directions change target behaviour relative to
  paired unsteered generation, and how specific and robust are those changes in the executed
  intervention?
- **RQ3 (prospective expanded wording):** How does post-training change generated reasoning
  behaviour and generic-reasoning representations, and does the tested backtracking
  coordinate remain causally coupled across the selected post-training boundaries?

RQ3 states a question, not an outcome. It does not presume retained, rescaled, rotated,
decoupled, inaccessible, or disabled transport. If neither new study becomes admissible
before the evidence cutoff, the submitted thesis retains the narrower current RQ3—
representation change under safety versus matched non-safety post-training—and the explicit
behavioural and causal gap.

## 2. Evidence and presentation rules

This amendment changes scope and narrative hierarchy; it does not strengthen existing
evidence.

- The geometry evidence remains bounded and mixed. Correlation dimension, participation
  ratio, PCA variance-threshold dimension, fixed-top-ten variance concentration, and
  curvature remain separate estimands. A low estimate is not a subspace or manifold claim.
- The steering chapter presents the **raw task-paired steered-versus-unsteered results as the
  primary result**. Learned-operator versus matched-perturbation comparisons remain one
  compact robustness paragraph in that chapter. The robustness verdict may be summarised
  elsewhere only to preserve the stronger claim boundary.
- The existing steering evidence remains amended and provisional. The missing accuracy
  guard, single dose, reduced ranks, within-annotator scoring, and unresolved execution
  lineage remain visible.
- Fixed-input teacher-forced representation change, on-policy generated behaviour, and
  causal transport are three different evidence planes. None substitutes for another.
- Public-checkpoint differences remain distinct from the owned safety-versus-matched-control
  contrast. A one-seed owned comparison is checkpoint-bounded, not recipe-level replication.
- Empirical evidence status, provenance status, protocol marker, and thesis disposition
  remain separate. `Amended` is not an evidence status; prospective/unrun work is not a
  result; failed and unrun are not interchangeable. Unresolved provenance travels with every
  use of a number.
- No new value enters the thesis before a final machine-readable artefact and provenance
  record, analysis-ledger and methodology entries, refreshed thesis evidence snapshot and
  hashes, and a claim-ledger row exist.

## 3. Prospective thesis evidence authorised in scope

### Phase 2: causal transport

Phase 2 is approved as **thesis-relevant prospective evidence**, governed by the sealed
`results/prereg/PHASE2_TRANSPORT_PREREG_2026-08-08.md` and its dated amendments.

The sealed 100-task manifest and A2 observational vanilla adjunct remain in force. Amendment
A3 makes Sonnet 4.5 the annotator and requires the qualifier **builder-annotator scored** for
behavioural-rate endpoints.

All provenance, input-alignment, target-null, injection-recovery, achieved-sensitivity,
damage, multiplicity, stop, and outcome-hierarchy gates continue to control. This scope
approval does not authorize a positive claim or a launch.

### P5: controlled post-training behavioural evaluation

P5 is approved as **thesis-relevant prospective evidence** to evaluate tangible behavioural
effects of post-training:

- generic reasoning: the four annotated behaviours, correctness coverage, response length,
  repetition, and truncation;
- evaluated safety: harmful-request refusal and benign-lookalike compliance with degeneration
  guards;
- primary controlled contrast: the owned seed-42 safety full-FT checkpoint versus its
  matched non-safety full-FT checkpoint;
- separate public case study: STAR1 versus base R1;
- DeepScaleR only in the generic-reasoning and transport comparison, not in safety-recipe
  attribution.

The prompt or problem is the scientific unit; sentence labels are nested observations.

The existing P5 pilot is pipeline and power-planning work and never becomes a thesis finding.
Its separate chat authorization does not authorize the powered study. The five-percentage-
point benign-compliance and damage margins require supervisor confirmation before arm-labelled
pilot outcomes are used.

The powered sample, final held-out safety manifest, scorer reliability gates, and exact budget
must be frozen before execution.

### Excluded or deferred

Method-plane generalisations, broad RLVR sweeps, the alpha-dial, pt21, agentic or order-effect
work, geometry-aware training, and Phase 3 behaviour-targeted KTO are outside the
thesis-minimum scope.

KTO is a follow-on-paper study and must not delay submission. The absent pt02 LoRA arms remain
retrain-conditional and do not enter by prose or activation-directory name.

## 4. Resource and launch authority

Scope approval and spend approval are separate. Unchecked spend is not authorized.

| Workstream | Scope disposition | Spend and launch gate |
|---|---|---|
| P5 pilot | Separately authorized in chat; pipeline-only and excluded from claims | Existing hard ceiling only: 176 generations, 212 scorer calls, `$3` generation plus `$15` scoring. No expansion without a new decision. |
| P5 powered benchmark | Approve as prospective thesis scope | **Not authorized by scope alone.** After blinded pilot sizing and protocol freeze, Tony must approve the exact generation, scoring, and any retraining ceiling. |
| Phase 2 | Approve as prospective thesis scope | **Not authorized by scope alone.** Launch requires completed sealed execution gates and Tony's explicit spend sign-off. Current planning envelope: approximately `$20–40` 4090 compute, `$5–10` teacher-forced extraction, and `$100–250` annotation. Because A3 replaced Nova-Pro with Sonnet, the annotation ceiling must be re-costed and signed before launch. |
| Phase-2 enlargement | Conditional only under the sealed achieved-sensitivity rule | Each `100 → 150 → 200 → 300` step requires a new spend approval before execution. |
| Optional LoRA retrain | Outside the core four-checkpoint study unless separately admitted | Separate immutable manifests and separate approximately `$1–3` approval required. |
| Phase 3 KTO and deferred work | Follow-on paper; outside thesis minimum | No thesis schedule or spend authority. |

The Phase-0 closure and committed Phase-2 task manifest satisfy two prior prerequisites. They
do not waive unfinished executor gates, the P1 seal, or spend approval.

The sealed Phase-2 vanilla artefact is generated once and consumed read-only by P5. P5 must
not regenerate overlapping rows or inspect causal-family outcomes before the
injection-recovery gate.

## 5. Thesis integration and appendix disposition

- The exploratory appendix added after the 26 July charter remains an authoritative included
  chapter. It is retained under this amendment. Any later cut, relocation, or material
  expansion requires an explicit author-supervisor decision rather than an editorial
  assumption.
- The 63-page ceiling remains binding. New evidence is integrated by replacement and
  compression, not simple addition.
- Do not cut the raw steering table, its robustness paragraph, failed or unrun controls,
  provenance disclosures, scientific-unit statements, or estimand distinctions to make room.
- If Phase 2 and P5 both land, the thesis may relate fixed-input representation change,
  natural generated behaviour, and causal coupling while keeping their estimands separate.
- If only one lands, only that evidence plane is added.
- If neither lands, retain the present bounded result and explicit behavioural and causal gap.

## 6. Decisions to complete at seal

### Joint scope decision

- [ ] **Approve as written:** title and subtitle, intervention-first narrative, three-RQ
  structure, Phase 2 and P5 as prospective thesis evidence, appendix retention, and the
  evidence boundaries above.
- [ ] Approve with the attached written amendments.
- [ ] Do not approve; retain the 26 July charter pending a replacement decision.

### Resource decision

- [ ] Phase-2 scope approved; spend deferred to a separate Tony sign-off.
- [ ] Phase-2 launch spend approved up to: compute `$_____`; extraction `$_____`; Sonnet
  annotation `$_____`; total hard ceiling `$_____`.
- [ ] P5 powered scope approved; spend deferred until the blinded powered-design freeze.
- [ ] P5 powered spend approved after freeze up to a hard ceiling of `$_____`.
- [ ] Optional LoRA retraining remains excluded.
- [ ] Optional LoRA retraining may be proposed later under a separate manifest and spend
  decision.

### Schedule and design choices

- Thesis evidence cutoff: `____________________`.
- Latest Phase-2 completion date for thesis admission: `____________________`.
- Latest P5 powered-result completion date for thesis admission: `____________________`.
- Supervisor decision on proposed five-point benign-compliance and damage margins:
  `[ approve / revise to _____ / defer ]`.
- If a deadline is missed:
  `[ use the bounded fallback specified above / other: ____________________ ]`.

## 7. Joint seal

By signing, both parties confirm that they have reviewed the changed scope, evidence basis,
resource authority, schedule, and fallback rules.

The seal supersedes the 26 July charter only on those recorded points. All unmodified hedge,
evidence-status, provenance, and claim-admission rules remain in force.

**Author — Tony Su**  
Decision: `approve / approve with amendments / do not approve`  
Signature or dated written name: `____________________________`  
Date: `________________`

**Academic supervisor — Dr Paolo Barucca**  
Decision: `approve / approve with amendments / do not approve`  
Signature or dated written name: `____________________________`  
Date: `________________`

**Linked amendment or email/chat record, if used:**  
`Chat record 2026-08-09 (Claude session): Tony — "we approve the P1 joint seal."
Recorded same day in RESULTS_LEDGER.md and session memory. Scope decision =
approve as written. The signature/date lines above remain to be completed by
both parties; the resource-decision checkboxes (§6) remain OPEN — Phase-2 spend
ceilings still require Tony's separate sign-off (annotation line re-costed
2026-08-09 under prereg Amendment A4: ≈$275 windowed vs ≈$582 full-chain).`

## Source basis reviewed

- `INTEGRATED_THESIS_POSTTRAIN_PLAN_2026-08-08.md`;
- current thesis title, abstract, RQs, steering, post-training, and conclusion;
- `SESSION_HANDOFF_2026-08-08_EVENING.md` and the post-closure Claude/Codex handoffs;
- sealed Phase-2 preregistration and A2/A3 amendments;
- P5 behavioural specification, pilot and pre-spend QA, protocol validation, shared-vanilla
  contract, and thesis-integration crosswalk;
- thesis charter, claim/status vocabulary, evidence-snapshot, and page-budget contracts.
