# AMENDMENT 2 — operator-managed pod termination

**Date:** 2026-08-18. **Status:** AMENDED before execution. At the time of this
amendment no 7B fit, score, or model-dependent result existed.

Parent sheet: `JSPACE_7B_PAIR_SHEET_2026-08-17.md`. Parent execution/evaluator
amendment: `JSPACE_7B_PAIR_AMENDMENT_1_2026-08-18.md`.

## Lifecycle-only change

The replacement pod was provisioned manually through the RunPod console and
does not expose a platform-level automatic-termination setting. The operator
explicitly authorized immediate launch and will terminate the pod manually when
notified that the job has completed and all artifacts have been synchronized
and verified locally.

The local watcher retains its 10-hour deadline, but because it deliberately
preserves a pod on failure, that deadline is not a hard billing stop. The
registered 10-hour/$10 limit is therefore operator-managed for this execution,
not mechanically enforced. This limitation travels with the execution record.

## Unchanged scientific and provenance contract

All model revisions, fit prompts and hash, estimator, OOM ladder, numerical
environment, readout/scoring inputs, layer bands, registered questions,
evaluator corrections, payload hashes, run identity, success conditions, and
artifact verification gates remain exactly as frozen in the sheet and
Amendment 1. The pod must remain fresh and pass the full launcher preflight.
