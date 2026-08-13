# P5 safety v2.2 validation — integrity and blinded sizing readiness

**Integrity:** `pass`  
**Frozen validation gate:** `fail`  
**Held-out scoring:** `not executed`  
**Arm-labelled effect differences:** `not computed or reported`  
**Machine-readable report SHA-256:** `0645e560083012a940e57a556cfc5e045763430c4bcc1d977f34adf80a221f44`

## Validation result

- Atomic terminal persistence: 104/104 (100.0%)
- Network success: 90/104 (86.5%); required ≥98%
- Complete parser plus deterministic evidence: 90/90 (100.0%); required ≥98%
- Network failures: 14 `ProxyProtocolError` chunks

The repair eliminated observed parser/evidence failures among successful responses, but the frozen gate failed on proxy/network completeness. Missing chunks remain unresolved. The held-out 48 assignments were correctly not started.

## Integrity and accounting

- Validation assignments: 48/48
- Terminal chunks: 104/104; lineage, chunk-plan, evidence, and aggregation recomputation errors: 0
- Attempts: 104; retries: 0
- Reported successful-response cost: $0.488511
- Responses without cost accounting: 14; total billed cost is therefore not known exactly
- Conservative authorized commitment: $2.600
- Maximum reported call: $0.011778; minimum/final reported quota: $131.877204
- All hard guards were respected; same-Sonnet repeats were not run

## Blinded sizing disposition

Only missingness was used: harmful-refusal complete pairs 8/12 (66.7%); benign-compliance complete pairs 9/12 (75.0%). No endpoint values, arm-labelled means, directions, or differences were used for sizing.

A powered sample size cannot be frozen from a failed validation half. These completion figures are operational diagnostics only.

## Required decision

Do not run the held-out stage from the current manifest. Investigate the 14 ProxyProtocolError responses and, if recovery is desired, create and separately authorize a new hash-bound validation-recovery protocol and budget. Do not reinterpret missing chunks as negative outcomes.
