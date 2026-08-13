# P5 v2.2 proxy metadata read-only check — 2026-08-09

## Outcome

No exact-call proxy/operator metadata was obtained for either representative
failed v2.2 call ID.  The retained scorer documentation describes only the
model-invocation `POST` contract; it does not document a call-inspection or
request-metadata route.  Read-only discovery and exact-ID probes found a model
catalogue but no accessible call metadata API.

The check made **zero model invocation calls**, sent **no `POST` requests**,
and incurred **zero model-scoring spend**.  It did not print or persist the
configured proxy URL, API key, response bodies, or credentials.  No state was
changed.

Representative retained call IDs checked:

- `12c12c00-5440-405a-8d90-87fd91bfd848`
- `e1bab460-79da-4acb-96ca-ddf3bbce7593`

Both IDs are present in the retained v2.2 journal as completed attempts with
`ProxyProtocolError` and null usage/cost accounting.

## Local documentation/configuration check

- The previously working scorer execution context was available.  Presence
  was checked without printing values.
- The retained proxy contract documents the configured route as an invocation
  `POST` with authenticated request headers and response cost/quota fields.
- No retained document or configuration entry names a read-only call lookup,
  request lookup, operator log, or metadata endpoint.
- No credentials were searched for or extracted from unrelated locations.

## Sanitized read-only route results

Only `OPTIONS` and `GET` were used.  Hostnames, stage names, credential values,
and response values are intentionally omitted.

| Sanitized route | Method | Result | Returned schema / observation |
|---|---:|---:|---|
| configured invocation route | `OPTIONS` | 204 | Empty; no `Allow` header |
| configured invocation route | `GET` | 401 | JSON object with only `message` |
| origin `/openapi.json` | `GET` | 401 | JSON object with only `message` |
| origin `/docs` | `GET` | 401 | JSON object with only `message` |
| origin `/health` | `GET` | 401 | JSON object with only `message` |
| origin `/models` | `GET` | 401 | JSON object with only `message` |
| stage `/openapi.json` | `GET` | 401 | JSON object with only `message` |
| stage `/docs` | `GET` | 401 | JSON object with only `message` |
| stage `/health` | `GET` | 401 | JSON object with only `message` |
| stage `/models` | `GET` | 200 | JSON list, 57 entries; item keys: `category`, `inputPrice`, `modelId`, `name`, `outputPrice`, `provider` |

No OpenAPI document exposing read paths was returned.

Each of the following exact-ID candidates was tried for each representative
call ID:

| Sanitized exact-ID route | Method | Result for both IDs | Returned schema |
|---|---:|---:|---|
| stage `/calls/{call_id}` | `GET` | 401 | JSON object with only `message` |
| stage `/requests/{call_id}` | `GET` | 401 | JSON object with only `message` |
| stage `/invocations/{call_id}` | `GET` | 401 | JSON object with only `message` |
| stage `/metadata/{call_id}` | `GET` | 401 | JSON object with only `message` |
| stage `/usage/{call_id}` | `GET` | 401 | JSON object with only `message` |

The error-body value was not printed or retained.  No usage, cost, quota,
content-block, stop-reason, filtering, provider request ID, or upstream status
metadata was returned by these probes.

## Scientific/operational implication

The local evidence still establishes only that the HTTP response passed the
existing status and JSON checks but yielded no extractable scorer text.  It
does not establish whether the upstream payload contained an empty string, an
empty content list, non-text content, a filter marker, or a different response
shape.  The proposed v2.2.1 retry should therefore remain unexecuted pending an
operator lookup or separately authorized diagnostic.

The preferred next action is a zero-model-cost operator log lookup for the two
call IDs above (and, if needed, the remaining twelve retained IDs), requesting
only response-shape, stop/filter classification, usage/cost, and upstream
status metadata—not prompt or response text.

## Smallest prospective instrumented diagnostic — not run

If operator metadata is unavailable, the smallest useful prospective test is
one separately authorized diagnostic invocation of one retained failed chunk,
outside annotation outputs and outside the recovery manifest.  It should have:

- exactly one initial attempt and no automatic retry;
- the frozen Sonnet model, 25-second timeout, and a reduced output budget no
  greater than 400 tokens;
- a $0.025 per-call and total-spend ceiling, plus the existing $5 quota floor;
- response-shape telemetry captured before text extraction: HTTP status class,
  content type, top-level key names, content-container type/count, block type
  names, text-block count and byte lengths, whitespace-only flags, presence of
  stop/filter/guardrail fields, usage/cost/quota fields, and an in-memory raw
  payload SHA-256;
- no persisted raw payload, prompt, generated text, credential, URL, or raw
  provider request header; any provider request identifier should be hashed;
- output explicitly excluded from annotations and validation-gate counts.

A non-empty diagnostic response would not by itself prove the original errors
were transient and would not relax the original 98% gate.  Any broader retry
or recovery plan would still require its own exact-hash authorization.
