# P5 v2.1 pilot integrity and blinded sizing

**Integrity:** `pass`  
**Powered-size gate:** `not_ready_for_powered_size_freeze`  
**Use:** pipeline pilot only; no arm-labelled effect differences computed or reported  
**Machine-readable report SHA-256:** `b40735b0fe331c7f82975aa89fb21cf6394260540b457a1a43d15bb98aa20939`

## Execution integrity

- Assignments: 176/176, primary only; repeat rows: 0
- Generation lineage errors: 0
- Successful source reassembly errors: 0
- Requests: 385 completed; 0 retries; 4 non-success responses
- Reported cost: $3.189870; conservative authorization commitment: $9.625
- Maximum reported call cost: $0.016419; minimum remaining quota: $132.45072

## Pipeline QA

- Overall scorer parse: 115/176 (65.3%); frozen gate: 98%
- Generic: 78/80 (97.5%)
- Harmful safety: 23/48 (47.9%)
- Benign safety: 14/48 (29.2%)
- Length-cap hits: 67/176 (38.1%); the 5% trigger is exceeded

Transport mostly succeeded, but strict per-chunk safety schema/evidence validation caused high unresolvedness. Missing rows remain unresolved, never zero. Error-class-only terminal logging prevents a finer retrospective ValueError diagnosis.

## Blinded sizing disposition

The JSON report contains pooled squared-difference and discordance curves across the two pre-specified contrast families without arm-labelled means or directions. They are diagnostics only: safety resolution and complete-pair rates are too low to freeze a powered sample size. The proposed five-point margins still require supervisor confirmation.

## Required decision

Do not authorize the powered run from this pilot. Approve a versioned safety-scoring repair plus a new disjoint validation/rescoring budget, preserve current unresolved rows as missing, and retain the 6,144-token cap trigger for every P5-owned full-run arm.
