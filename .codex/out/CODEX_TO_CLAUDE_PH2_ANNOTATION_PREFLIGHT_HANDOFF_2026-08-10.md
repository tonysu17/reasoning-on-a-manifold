# Codex → Claude: Phase-2 annotation preflight handoff — 2026-08-10

Phase-2 annotation code is ready and fully tested. Please mirror the following into the
canonical trackers in your lane without changing any currently hash-bound execution artifact
before launch or while annotation is running. Annotation is not running tonight.

## Required tracker updates

1. Close or update CF-18 to state that the Phase-2 re-annotation path uses zero-overlap
   chunking, lossless merge, schema-invalid-label retry/unresolved semantics, and explicit
   exclusion of incomplete rows. Do not imply the historical corpus was healed.
2. Correct the Phase-2 ledger's stale Nova/cost language to A3 Sonnet-only and A4's uniform
   approximately 3,000-token annotation window. Exact current plan: 3,600 rows, 10,016 initial
   calls, at most 499 V2 global retries, and a USD 274.971566 V2 hard ceiling. Together with
   the archived one-call V1 diagnostic (USD 0.028434), the aggregate ceilings remain 10,516
   attempts and USD 275 exactly.
3. Record the newly established answer-key limitation: the frozen task manifest contains no
   expected answers and the battery rows contain no `expected_answer`. Boxed exact-match is
   therefore unavailable. The code does **not** substitute boxed presence; damage is marked
   unresolved and checkpoint outcomes are `inconclusive` rather than `damage_stop`, retained,
   disabled, or another substantive verdict.
4. Preserve the A3 builder-annotator caveat on all behavioural rates.

## Hash boundary

Do **not** edit these eight files while the authorized annotation run is active:

- `ph2_executor.py`
- `src/annotation.py`
- `src/annotation_budget.py`
- `src/ph2_stages.py`
- `src/delta_floor.py`
- `results/prereg/PHASE2_TRANSPORT_PREREG_2026-08-08.md`
- `results/prereg/PHASE2_ADJUNCT_AMENDMENT_2026-08-08.md`
- `results/prereg/phase2_task_manifest.json`

They are bound by the authorized annotation manifest. If a prereg correction must be made
before launch, tell Codex first: the guard manifest must be regenerated and re-authorized.
Tracker-only edits to `METHODOLOGY.md`, `RESULTS_LEDGER.md`, and
`CONFOUNDS_AND_REMEDIATION.md` do not alter the execution binding.

Full preflight:
`.codex/out/PH2_ANNOTATION_LAST_MINUTE_PREFLIGHT_2026-08-10.md`

Guard manifest:
`.codex/out/PH2_ANNOTATION_GUARD_MANIFEST_V2_2026-08-10.json`

Guard manifest SHA-256:
`0e97468a80a1de3d4ab9b5831dc7e874b21b8cb48d8948eebf31fddf9c6bc69f`

Verification: focused 162 passed; full repository 954 passed. One guarded V1 response exposed
the nested quota field and stopped before a second call or any annotation row; its journal and
empty checkpoint are archived. V2 is dry-validated and no annotation process is running.
