# P5 scorer v2.1 owner decision

**Recorded:** 9 August 2026  
**Source:** Tony Su's chat decision: “Yes I approve the above gates.”  
**Status:** protocol approved; calibration and full-scoring spend still require final
hash-bound ceilings

## Approved now

Tony approves the substantive v2.1 pilot-scoring amendment:

- Sonnet-only builder annotation;
- deterministic source sentence/clause units as the v2.1 behavioural denominator;
- explicit non-poolability with the earlier model-returned-span estimand;
- removal of same-Sonnet repeats from the authorized pilot plan;
- no inter-annotator or annotator-robustness claim;
- decisive-observed-stance handling for length-truncated safety chains, with inconclusive
  cases unresolved;
- sequential, chunked requests under the sub-29-second proxy contract;
- missing chunks and stopped assignments remain unresolved, never zero.

The approved protocol source is
`.codex/out/P5_PILOT_SCORER_V2_1_PROTOCOL_AMENDMENT_DRAFT_2026-08-09.md`, SHA-256
`028e4e147fa06020c7d745fcad5f8a32f57549240ae6dbc645febc6d746751a7`.

## Approved gates, pending executable details

Tony approves proceeding to both later gates in principle:

1. two Sonnet cost-calibration calls selected deterministically from the final 176-row
   request plan; and
2. the primary-only full pilot scoring run after calibration.

This decision does **not** invent or waive the missing execution facts. No paid call may run
until the corresponding executable manifest is frozen and Tony is shown the exact manifest
SHA-256, request ceiling, spend ceiling, per-attempt cost bound, and quota stop floor. The
original 212-call / `$15` authorization is not transferable to v2.1.

## Not covered

- This is not the joint P1 author–supervisor seal.
- It does not authorize the powered P5 benchmark, Phase 2, optional LoRA retraining, or any
  other study.
- It does not turn pilot rows into thesis findings.

