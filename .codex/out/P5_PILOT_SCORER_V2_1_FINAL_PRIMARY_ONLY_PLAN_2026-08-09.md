# P5 pilot scorer v2.1 — dry-run budget report

**Status:** `dry_run_not_authorized`  
**Proxy calls made:** `0`  
**Generation snapshot:** `176/176` rows  
**Model:** `anthropic.claude-sonnet-4-5-20250929-v1:0` (builder annotator; Sonnet only)

## Exact accounting on the captured completed generations

- Logical primary assignments: 176
- Optional same-annotator repeat assignments available: 36
- Initial network requests in selected `drop` plan: 490
  - Primary-pass requests: 490
  - Optional same-annotator repeat requests, not included when dropped: 108
- Worst case with three retries per initial request: 1960
- Difference from the old 212-request interpretation: +278
- Exact full-176 primary-only count: 490 initial requests

All 176 generation texts are present, so the primary-only initial request count is exactly 490. Cost is still not estimable from the handoff: no proxy call was made, and there is no auditable per-request Sonnet cost bound. Therefore compatibility with the prior $15 authorization is **not proven**.

## Reliability/status replacement

A3 makes the original non-builder double-score gate unsatisfiable. The frozen 36-key subset is optional and should be dropped from the authorized pilot. If ever run separately, it is only a same-Sonnet repeatability diagnostic: it cannot estimate inter-annotator agreement and cannot support an annotator-robustness claim. Behavioural outputs must be described as **builder-annotator scored**.

## Protocol compatibility

Stable-unit chunking uses one most-salient label per deterministic source sentence/clause. This does not preserve v1's model-returned span denominator, and safety uses a new deterministic all-chunk aggregation. V2.1 output therefore cannot be pooled with v1 and requires a versioned pilot protocol amendment before execution or analysis.

## Execution blocker

This manifest is deliberately non-executable. Scoring requires a separately approved v2.1 manifest with a network-request ceiling, a total spend ceiling, an approved maximum cost per request, a quota floor, and its exact manifest hash.
