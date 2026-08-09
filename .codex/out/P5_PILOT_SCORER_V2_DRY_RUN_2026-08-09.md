# P5 pilot scorer v2 — dry-run budget report

**Status:** `dry_run_not_authorized`  
**Proxy calls made:** `0`  
**Generation snapshot:** `131/176` rows  
**Model:** `anthropic.claude-sonnet-4-5-20250929-v1:0` (builder annotator; Sonnet only)

## Exact accounting on the captured completed generations

- Logical primary assignments: 131
- Same-annotator repeat assignments: 27
- Initial network requests after chunking: 417
  - Primary-pass requests: 341
  - Same-annotator repeat requests: 76
- Worst case with three retries per initial request: 1668
- Difference from the old 212-request interpretation: +205
- Full-success-grid lower bound from this snapshot: 471 initial requests

The final exact full-run request count is not available until every generation text exists; the lower bound assumes each remaining output is non-empty and needs at least one chunk. Cost is also not estimable from the handoff: no proxy call was made, and there is no auditable per-request Sonnet cost bound. Therefore compatibility with the prior $15 authorization is **not proven**.

## Reliability/status replacement

A3 makes the original non-builder double-score gate unsatisfiable. The frozen 36-key subset is retained only as a same-Sonnet repeatability diagnostic. It cannot estimate inter-annotator agreement and cannot support an annotator-robustness claim. Behavioural outputs must be described as **builder-annotator scored**.

## Protocol compatibility

Stable-unit chunking uses one most-salient label per deterministic source sentence/clause. This does not preserve v1's model-returned span denominator, and safety uses a new deterministic all-chunk aggregation. V2 output therefore cannot be pooled with v1 and requires a versioned pilot protocol amendment before execution or analysis.

## Execution blocker

This manifest is deliberately non-executable. Scoring requires a separately approved v2 manifest with a network-request ceiling, a total spend ceiling, an approved maximum cost per request, a quota floor, and its exact manifest hash.
