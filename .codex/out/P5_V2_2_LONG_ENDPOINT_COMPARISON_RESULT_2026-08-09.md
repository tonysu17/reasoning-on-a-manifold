# P5 v2.2 official long-endpoint comparison — 2026-08-09

## Authorized scope and execution

Tony directly authorized one deterministic, diagnostic-only comparison call:

> One official long-endpoint comparison. Send the identical diagnostic once
> through the 120-second endpoint supplied. Expected cost about $0.006; hard
> ceiling $0.025; diagnostic-only.

- Authorized manifest internal SHA-256:
  `d4d14169e34c840cc16146226c9d45024531fc5190396f814097b33155bb67f0`
- Authorized manifest file SHA-256:
  `a6681b29a111f2c79a98b8698e1c00932b9163a734ed94f90e65f67ed8e6fdca`
- Identical request-body SHA-256:
  `8c075e5defcebb0eba555aab4a0241a3883c20394a195906876649d0c0dcf0be`
- Long endpoint URL SHA-256:
  `2e9bb6979f6ac76b368a67332ede97eab840112146ec0904b1c74e34c847b7a8`
- Network/model attempts: **1**
- Retries: **0**
- Model / temperature / maximum output: exact frozen Sonnet / 0 / 400 tokens
- Client timeout: **25 seconds**, unchanged from the standard diagnostic and
  below the long endpoint's 120-second service limit
- Total/per-call authorization ceiling: **$0.025**
- Quota stop floor: **$5.00**
- Hard-guard events recorded by the runner: none

The full P5 suite passed before execution: **151 tests**.

## Sanitized result

Runner classification: **`empty_scorer_text_without_filter_marker`**.

- HTTP status class: `2xx`
- Normalized content type: `application/json`
- JSON parse: successful; top-level type `object`
- Top-level keys only: `choices`, `created`, `id`, `model`, `object`, `usage`
- Expected top-level `content`: absent
- Text blocks visible to the frozen proxy parser: **0**
- Stop field present: yes
- Filter, guardrail, and safety marker fields present: no
- Sanitized total-token count: 1,985
- Raw payload size: 297 bytes
- Raw payload SHA-256:
  `87501357327432846b6911181fc3f245616257499f29b732ff3cb51f6d49ef70`
- Hashed provider request ID available: yes

The long endpoint returned an OpenAI-style top-level response shape rather than
the standard proxy wrapper.  In particular, it returned `choices` and omitted
the top-level `content`, `metadata.remaining_quota`, and `usage.cost` fields
required by the frozen proxy parser/accounting contract.

No raw response or nested `choices` content was persisted, and it must not be
recovered retrospectively.  Therefore this result does **not** establish that
the underlying long-endpoint model response contained no text.  It establishes
that the long endpoint's returned schema is incompatible with the frozen
standard-endpoint parser and accounting contract.

## Cost and quota qualification

The long response did not expose parseable `usage.cost` or remaining-quota
metadata.  Consequently:

- exact observed cost: **unavailable from the retained response**;
- quota after the call: **unavailable from the retained response**;
- the pre-request one-call commitment was capped by authorization at $0.025,
  and no subsequent call occurred;
- approximately $0.006 is an expectation based on Tony's authorization and the
  identical standard request's observed $0.005967 cost, not an observed long-
  endpoint accounting value.

This accounting absence is itself a response-schema failure.  The per-call
cost ceiling and quota floor could not be verified ex post, so the result must
not be reported as having passed those observed-accounting guards.

## Shape comparison

| Field | Standard endpoint | Official long endpoint |
|---|---|---|
| HTTP / JSON | `2xx`, parsed object | `2xx`, parsed object |
| Top-level response family | proxy wrapper | OpenAI-style `choices` |
| Top-level `content` | empty array | absent |
| Stop field | absent | present |
| Explicit filter marker | absent | absent |
| Cost/quota accounting | parseable | unavailable |
| Frozen text extractor result | empty | empty because expected field is absent |

The outcomes share the same parser-level label but are **not the same response
shape**.  The comparison is inconclusive about underlying scorer text and
decisive about endpoint schema incompatibility.

## Persisted artifacts

- `p5_runs/p5-pilot-20260808/safety_v2_2_long_endpoint_shape_diagnostic.json`
  — SHA-256
  `87e6e8eb95d51553193d624f928bb29cd56a0106a64efa7ef45a02f76f5dbbc0`
- `p5_runs/p5-pilot-20260808/safety_v2_2_long_endpoint_shape_diagnostic_journal.jsonl`
  — SHA-256
  `d837534f3ee0201054ce032841c9cc6de6ec4851160cd3b979077fba2726921f`

The journal contains exactly one reservation and one completion event.

## Leak and immutability audit

No long endpoint URL, proxy key, user prompt, scorer prompt, source chunk, raw
response payload, generated text, or raw header/provider identifier was
persisted.  An exact-value and prohibited-field audit found no leaks.

The standard diagnostic, generation snapshot, frozen v2.2 chunk/assignment
scores, v2.2 journal, and validation gate retained their prior hashes.  No
annotation, gate, held-out, or recovery output was modified.

## Disposition

1. Do **not** use the long endpoint with the frozen standard proxy parser or
   accounting guard, and do not run recovery or held-out scoring.
2. Ask the operator to confirm the official long endpoint's intended response
   envelope and whether it should provide the same proxy wrapper, cost, and
   quota metadata as the standard endpoint.
3. Any prospective adapter must parse the documented `choices` schema,
   re-establish cost/quota accounting, retain the same non-disclosure rules,
   and pass separate offline tests and disjoint validation before scoring.
4. Until then, preserve the original v2.2 missing chunks as unresolved and the
   validation gate as failed.
