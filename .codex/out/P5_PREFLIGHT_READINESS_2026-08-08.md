# P5 preflight readiness

**Status:** `ready_for_spend_gated_pilot`  
**Model/API calls:** none  
**Spend authorised:** no

| Gate | Status | Blocking | Detail |
|---|---|---:|---|
| P2 | deferred | no | Phase 2 is not complete, but the received handoff explicitly removes analyse.done as a P5 pilot prerequisite; final shared-generic execution still waits for the manifest SHA (transport=sealed, adjunct_file=draft_for_seal) |
| A1 | pass | yes | four core checkpoints present with verified weight hashes |
| A2 | blocked | no | pt02 safety-SFT LoRA adapters are definitively absent locally and on the searched volume; extension remains provenance-clean retrain-conditional |
| A5 | pass | yes | generic_pilot: /Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold/data/tasks_pilot.json: 20 rows; 0 errors |
| A3 | blocked | no | generic_final: missing manifest: /Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold/results/prereg/phase2_task_manifest.json |
| A5 | pass | yes | safety_pilot: /Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold/.codex/out/P5_SAFETY_PILOT_MANIFEST_2026-08-08.json: 24 rows; 0 errors |
| A4 | blocked | no | safety_final: missing manifest: /Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold/.codex/out/P5_SAFETY_FINAL_MANIFEST_2026-08-08.json |
| A4P | pass | yes | safety_pilot_overlap: zero exact/near overlap across 3 training roles |
| A4 | blocked | no | safety_final_overlap: missing overlap audit: /Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold/.codex/out/P5_SAFETY_FINAL_OVERLAP_AUDIT_2026-08-08.json |
| A6 | pass | yes | offline base-tokenizer alias gate passed for 44 prompts across 4 checkpoint roles |
| A7 | pass | yes | generation and six-label behaviour-annotation schemas are implemented and unit-tested |
| A8 | pass | yes | metric fixtures, prompt-paired category-stratified bootstrap, and Holm adjustment are unit-tested |
| A9 | pass | yes | frozen rubric and 6 local fixtures validate |
| A10 | pass | yes | provenance-complete mock generation row is implemented and unit-tested |
| A11 | pass | yes | all named spend stages require authorisation and a valid manifest hash |
| A12 | pass | yes | generation, behaviour-annotation, and safety-score resume merges accept identical rows and reject conflicts |

## Interpretation

The core checkpoint inventory and local metric/schema safeguards are ready. Current blocking gates: none. The existing GRPO harmful prompts remain rejected because they are exact STAR-1 training records.
