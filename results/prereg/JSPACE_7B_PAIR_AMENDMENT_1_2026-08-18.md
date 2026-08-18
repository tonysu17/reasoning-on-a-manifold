# AMENDMENT 1 — 7B pair execution hardening and evaluator correction

**Date:** 2026-08-18. **Status:** AMENDED before execution. At the time of this
amendment no 7B fit, score, or model-dependent result existed. Three earlier
infrastructure attempts never reached a fit. This amendment is frozen by the
analysis-repository commit that first contains this file; that full commit is
embedded in the transmitted payload and every produced manifest.

Parent sheet: `JSPACE_7B_PAIR_SHEET_2026-08-17.md`.

## Reason

A pre-launch code audit found execution/provenance defects in the lightweight
payload and unattended watcher, plus three defects in the registered evaluator:
the existing 7B-Instruct anchor was loaded from the wrong diagnostics stage;
E5 assumed shared token IDs without retaining each model's eligibility rows;
and E2/E3 silently introduced numerical cutoffs absent from the sealed sheet.
These findings precede every 7B model observation.

## Execution and provenance repairs

1. The launcher refuses a dirty analysis repository and a non-fresh pod. It
   records a unique run ID, the full source commit, and SHA256 for every payload
   file. The common provenance helper uses that explicit commit when `.git` is
   intentionally absent.
2. The installer pins pip, PyTorch 2.6.0 from the CUDA 12.4 index, the committed
   requirements lock, and J-lens commit `581d398…` installed without dependency
   re-resolution. `pip check` is a pre-fit gate. The resolved `pip freeze`, GPU
   report, disk layout, payload hashes, run ID, and source commit are retained
   with the lenses.
3. The launcher and runner assert Python 3.11, exactly one L40S/A6000-class GPU
   with at least 45,000 MiB, BF16 support, at least 100 GB allocated container
   disk, 28 layers, `d_model=3584`, and head vocabulary 152,064. A success marker
   is written only after both model fits, both scores, structural outputs,
   packaging, and checksums succeed. Failure has a distinct status and non-zero
   exit.
4. The watcher uses the run-specific PID/status rather than `pgrep -f`, checks
   all four stage-success markers, verifies byte hashes and exact source commit,
   refuses destination overwrite, and binds any termination call to the pod
   owning the synchronized host/port. Its local deadline is 10 hours. This local
   deadline is secondary: deployment is permitted only with RunPod's independent
   platform-level automatic termination set no later than 9 h 30 min. At the
   selected ceiling of $0.99/GPU-h plus the displayed container-disk charge,
   this is strictly inside both registered limits (10 h and $10).

## Evaluator amendments

1. Each scoring bundle retains exact per-model item order and eligible token
   IDs. E5 is computed only on common-eligible item names, using each model's
   own token IDs. The synchronizer validates internal row/count consistency but
   does not require the two models to have identical eligibility counts.
2. The evaluator loads the existing 7B-Instruct anchor explicitly from D2.
3. E2 reports the continuous `mid_max` and `late_max` values only; no Boolean or
   cutoff operationalises “confined to late” or “approximately zero.” E3 reports
   continuous J-minus-logit deltas and references; it does not apply the
   unregistered `0.03` categorical cutoff. These are corrections to the
   registered evaluator, while the sealed directional questions remain fixed.

## Correction to the sheet's background description

The sheet's claim that both 1.5B checkpoints were exactly zero throughout the
mid-stack is inaccurate. The base checkpoint's multihop layer profile has
sparse non-zero values at L13–L16 (maximum 0.0247); both checkpoints are zero at
the preregistered L17 endpoint. This correction does not use a 7B observation.
Accordingly, E2 will be reported as the full fixed-band maxima and layer
profiles, without treating exact mid-stack zero as the historical baseline.

## Unchanged scientific contract

Models and revisions, 100 fit prompts and their hash, J-lens estimator, OOM
ladder, BF16/eager model execution, source/target layers, evaluation files,
token eligibility rule, top-25 scorer, permutations and seed, fixed layer bands,
E1 direction, E2/E3/E4 directional questions, E5 qualitative status, typo
control, descriptive evidence status, and all claim boundaries are unchanged.
The evaluator corrections above change how those questions are computed or
summarised; they do not introduce a model-result threshold after inspection.
