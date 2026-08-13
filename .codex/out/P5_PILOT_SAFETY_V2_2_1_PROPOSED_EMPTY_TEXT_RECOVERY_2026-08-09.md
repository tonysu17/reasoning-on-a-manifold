# P5 safety v2.2.1 — empty-text validation recovery

**Status:** `proposed_awaiting_owner_exact_hash_approval`  
**API/model/pod calls made:** `0`  
**Internal SHA-256:** `353f6c7feda505c365fd25b13669ad0623a089fa34e1c031c8afe0f094d9e245`  
**File SHA-256:** `c4263d449a0f0687bc56729a4158ff3d3c620b0834810e4261b81d298fbe84e9`

## Diagnosis

All 14 rows are the exact path where an HTTP non-error JSON payload produced empty/whitespace extracted scorer text. The original error occurred before accounting extraction, so payload shape, exact HTTP status, text, and per-response cost are not retrospectively recoverable. No absent text was recovered.

Failures are not demonstrably transient: 10/14 share `p5sp_weapons_h01`, and 12/14 are harmful. Recovery success is therefore not assumed.

## Narrow amendment

The identical Sonnet model, prompt, chunk, temperature, parser, and deterministic evidence checks are retained. Only empty extracted text joins timeout/HTTP 504 as retryable, with at most two total attempts per exact failed chunk. Parser or evidence failures are terminal and are not retried.

The 104-key denominator and original gates remain unchanged. At least 12 of 14 failed chunks must recover, and the deterministic merged view must still pass 100% atomic persistence, ≥98% network success, and ≥98% parser-plus-evidence completeness among successful responses. The runner never executes held-out rows.

## Offline accounting

- Exact recovery chunks / initial requests: 14 / 14
- Maximum total attempts: 28
- Projected initial cost at observed envelope: $0.120582
- Projected maximum-attempt cost at observed envelope: $0.241164
- These projections are not ceilings
- Held-out requests and same-Sonnet repeats: 0

## Proposed hard limits

- Initial requests: 14
- Total-attempt ceiling: 28
- Spend ceiling: $0.70
- Maximum cost/request: $0.025
- Remaining-quota stop floor: $5.00
- Held-out scoring and same-Sonnet repeats: unauthorized

## Decision-ready authorization line

> I authorize only the P5 v2.2.1 empty-text validation recovery under manifest internal SHA `353f6c7feda505c365fd25b13669ad0623a089fa34e1c031c8afe0f094d9e245` and file SHA `c4263d449a0f0687bc56729a4158ff3d3c620b0834810e4261b81d298fbe84e9`: 14 exact failed chunks, 14 initial requests, at most 28 total attempts, $0.70 total spend, $0.025 maximum cost per request, $5.00 remaining-quota stop floor, unchanged parser/evidence and original validation gates, and no held-out scoring or same-Sonnet repeats. A merged-gate failure is a hard stop.

Do not execute without that exact-hash owner authorization and `--authorised`.
