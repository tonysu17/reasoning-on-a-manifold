# P5 protocol validation report

## Overall assessment: Ready for explicit pilot spend authorisation; not executed

The controlled behavioural design answers the intended question and preserves the thesis's estimand, unit, and provenance boundaries. The zero-cost implementation safeguards are sound. The received handoff removes Phase-2 completion as a prerequisite for the disjoint four-checkpoint pilot; execution still requires Tony's explicit `$18`-ceiling spend authorisation.

## Methodology review

- **Question framing:** Correctly separates natural free-generation behaviour from teacher-forced representation transport and from steering intervention effects.
- **Population/data selection:** Generic and safety strata are separate. The planned union-disjoint safety manifest is necessary because the existing harmful pool is training data for both STAR1 safety checkpoints.
- **Scientific unit:** Prompt/task is primary; sentence labels are nested; training seed is higher-level only when independent checkpoints exist.
- **Baseline/comparison:** The owned safety-versus-matched-control seed-42 full-FT pair is the strongest available attribution contrast. Public STAR1-versus-base remains a checkpoint case study. DeepScaleR is excluded from safety attribution.
- **Metrics:** Four sentence-fraction behaviours, bt/1k, correctness coverage, length, repetition, truncation, harmful refusal, benign compliance, and degeneration are separately defined. Boxed presence is not conflated with correctness; missing annotations are not zero.
- **Uncertainty:** Paired prompt bootstrap and category stratification match the data grain. The one-seed owned comparison remains checkpoint-bounded, not recipe-level replication.

## Issues found

1. **High for the powered study — final held-out safety population absent:** The 24-prompt pilot manifest is now frozen, schema-valid, and union-train-disjoint. The final powered-study safety manifest does not yet exist. All 250 harmful prompts in `grpo_refusal_prompts.json` exactly match STAR-1 training prompts and remain inadmissible for held-out claims.
2. **High for the final generic study — Phase-2 manifest SHA pending:** Reuse is prescribed, but the manifest has not yet been generated. P5 must wait for its `ids_sha256` and shared vanilla artefact before final generic execution, not for `analyse.done`.
3. **Medium — limited replication:** The owned full-FT safety/control comparison has one seed. Prompt resampling estimates prompt uncertainty but cannot estimate recipe-to-recipe training variability.
4. **Medium — margins await confirmation:** The proposed five-percentage-point benign-compliance and damage margins need supervisor confirmation before pilot arm outcomes are inspected.
5. **Medium — scorer performance unknown:** Safety-rubric parse rate, binary agreement, kappa, and uncertain-label rate require the spend-gated pilot.
6. **Low — correctness coverage unknown:** The fresh generic manifest may contain tasks without deterministic frozen gold answers. Correctness coverage must be quantified; missing golds cannot be scored wrong.

## Calculation and integrity spot-checks

- **Checkpoint identity:** Verified full weight hashes for the base, public STAR1, owned safety full-FT, and owned control full-FT checkpoints.
- **Train overlap:** Verified 250/250 harmful evaluation prompts overlap STAR-1 training; 0/500 prompts overlap the MetaMathQA control training manifest under the frozen exact/near lexical rules.
- **Steering arithmetic:** Recomputed all twelve raw steered-versus-unsteered cells from the present source arrays; values match thesis rounding. Historical execution lineage remains unresolved.
- **Metric fixtures:** Behaviour fraction uses sentence count; bt/1k uses whitespace-token denominator; four-gram repetition matches the house definition; missing/empty values remain null.
- **Schema/idempotence:** Successful, failed, resumed, duplicated, and conflicting generation records are covered.
- **Tokenizer/template identity:** The four checkpoints contain three distinct tokenizer-file hashes but one common chat-template hash. The common base-tokenizer alias gate now passes for all 44 pilot prompts across all four checkpoint roles with zero mismatches.
- **Safety pilot manifest:** 24/24 rows pass schema validation; all 12 pair IDs contain exactly one harmful and one benign prompt; zero exact/near overlaps occur across the public/owned STAR-1 and owned-control training roles.
- **Scorer integrity:** The frozen `p5-safety-rubric-1` file and six local parser/endpoint fixtures are hash-recorded and actively checked by readiness gate A9. Successful and failed scored rows have a provenance-complete schema; resume accepts byte-identical rows and rejects conflicting duplicates.
- **Behaviour-annotation integrity:** Successful generic annotations use the frozen six-label ontology, retain ordered sentence text, bind to the generation-record hash, and recompute all four target fractions. Empty sentence arrays remain unresolved rather than becoming zero; identical resumes are idempotent and conflicting duplicates fail.
- **Estimand implementation:** The local estimator computes mean paired arm differences over complete prompts, stratifies bootstrap resampling by frozen category, reports per-arm and joint missingness, fixes the draw count/seed, and implements Holm step-down adjustment for the four-behaviour family.
- **Automated suite:** 48/48 P5 preflight tests pass, including deterministic expected input-ID hashing, cross-role mismatch detection, scorer artefact integrity/fail-closed behaviour, annotation/score derived-endpoint validation, paired-bootstrap/missingness checks, Holm adjustment, and resume conflict detection.
- **Spend boundary:** Every named generation/scoring/retraining stage refuses without explicit authorisation and a valid frozen-manifest hash. The harness contains no execution implementation.

## Visualisation review

No P5 result visualisations exist because no pilot or powered outcome has been run. Future plots must keep representation displacement, rotation, and behavioural deltas in separate panels/scales and label the arm count prominently.

## Required improvements before execution

1. Execute the 176-generation/212-score pilot only after Tony explicitly approves the `$3 GPU + $15 scorer` ceiling; require every generation record to match the frozen input-ID hashes.
2. Retain the frozen, audited pilot manifest; create and freeze the larger final harmful/benign-lookalike manifest only after the pilot sizing freeze, then rerun the union-of-training-manifests overlap audit.
3. Confirm substantive non-inferiority/damage margins before viewing pilot arm differences.
4. Wait for the Phase-2 manifest SHA and shared vanilla artefact before final generic execution; do not inspect its causal-family outcomes ahead of the injection-recovery gate.

## Required caveats

- P5 is prospective and unrun.
- The existing harmful GRPO prompts are not held out for STAR1-trained checkpoints.
- One full-FT seed supports only a bounded checkpoint comparison.
- LoRA arms are absent and remain recovery/retrain-conditional.
- A behavioural difference would not itself identify a representation-level mechanism or a safety circuit.
