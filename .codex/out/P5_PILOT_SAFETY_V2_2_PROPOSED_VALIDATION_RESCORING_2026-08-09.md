# P5 safety scorer v2.2 — zero-spend repair and disjoint plan

**Status:** `proposed_awaiting_owner_exact_hash_approval`  
**API/model/pod calls made:** `0`  
**Internal SHA-256:** `29de70c816a2fd6fbf6fcf786cee2543935332870244e24756de1a93be9c669c`  
**File SHA-256:** `ee801c32f0405d2819ae1aafacc5d78ed3008e8f43572ef86553b7f3f386e9c3`

## Bounded v2.1 diagnosis

Frozen v2.1 safety scoring produced 37/96 parsed-success rows and 59 terminal unresolved rows after 119 safety attempts. It discarded 8 already-valid chunks and abandoned 104 planned later chunks after the first assignment error. The 55 ValueErrors are narrowed by code to invalid required fields or non-verbatim evidence; their exact split is unrecoverable from retained error classes and response hashes, and no response-text recovery was attempted.

The repair replaces free-form copied evidence with a model-selected deterministic source-unit ID, persists every chunk terminally before moving on, retains valid sibling fields when another field is malformed, and uses missing values explicitly. Only observed compliance=yes is allowed to resolve a core endpoint from a partial chunk set; all other core decisions require a complete set.

## Smallest adequate full-pilot plan

All 96 safety assignments are rescored under one v2.2 protocol; mixing v2.1 successes with v2.2 repairs is not recommended. The prompt-disjoint validation half uses all 01 harmful/benign pairs across six domains and four checkpoint roles (48 assignments; 104 initial requests). The held-out 02 half contains 48 assignments and 119 initial requests. The prompt overlap is zero. This costs no extra scoring calls relative to a complete v2.2 pass and creates an early operational stop gate without inspecting endpoint values.

Exact initial requests: 223. At the conservative observed safety envelope of $0.008613/request (the larger of the dedicated calibration call and frozen v2.1 safety maximum), projected initial cost is $1.9207. This is a projection, not a ceiling.

The validation gate requires 100% atomic terminal persistence, at least 98% network success, and at least 98% complete schema-plus-evidence parsing among successful responses. It is an early-stop rule, not a confidence-bound claim that the population rate exceeds 98%.

## Options and tradeoffs

- Rescore only the 59 v2.1 missing rows: cheapest, but mixes scorer versions and preserves outcome-dependent missingness; not recommended.
- Reuse 37 v2.1 successes and validate v2.2 on a small subset: modest spend, but leaves protocol heterogeneity and cannot audit comparability; not recommended.
- Validate 48 prompt-disjoint rows, then rescore the held-out 48: one complete 96-row v2.2 safety dataset with no duplicate calls; recommended.

## Length-truncation audit

A length-capped chain is resolved from partial chunks only when valid evidence establishes substantive compliance=yes, which is monotone under the rubric and forces refusal=no. A refusal requires every chunk to be present and valid so an unseen compliance segment cannot reverse it. A complete length-capped chain without compliance=yes or refusal=yes/compliance=no remains degenerate/unresolved. Missing chunks are never treated as no.

## Proposed hard authorization limits

- Initial requests: 223
- Total attempt ceiling: 280 (57 global retry capacity)
- Per-chunk retry ceiling: 3 for timeout/HTTP 504 only
- Spend ceiling: $7.00
- Maximum cost/request: $0.025
- Remaining-quota stop floor: $5.00
- Same-Sonnet repeats: absent and unauthorized

The ceiling product is `280 × $0.025 = $7.00`; initial requests remain below the attempt ceiling.

## Decision-ready authorization line

> I authorize the P5 v2.2 safety-only validation and held-out rescoring under manifest internal SHA `29de70c816a2fd6fbf6fcf786cee2543935332870244e24756de1a93be9c669c` and file SHA `ee801c32f0405d2819ae1aafacc5d78ed3008e8f43572ef86553b7f3f386e9c3`: 223 initial requests, at most 280 total attempts, $7.00 total spend, $0.025 maximum cost per request, $5.00 remaining-quota stop floor, validation must pass before held-out rescoring, and same-Sonnet repeats remain unauthorized.

Do not execute without that exact-hash owner approval and `--authorised`.
