# Claude handoff — long-request proxy endpoint (2026-08-09)

Tony supplied a long-request proxy endpoint described as having the same API
key, request schema, response schema, models, and pricing as the standard
endpoint, but with a maximum request duration of 120 seconds rather than the API
gateway's 30-second cap:

`https://q7s6v6seerne7eyh5ttsovjjcu0hxbou.lambda-url.eu-west-2.on.aws/v1/chat/completions`

## Operational routing

### Correction after the authorized live comparison

**Do not switch Phase-2 to this endpoint yet.** One authorized diagnostic call
on 2026-08-09 showed that the returned response envelope is not the same as the
standard proxy wrapper. The long endpoint returned top-level keys
`choices`, `created`, `id`, `model`, `object`, and `usage`, omitted top-level
`content`, and did not expose the standard `usage.cost` or remaining-quota
fields. This is an OpenAI-style response family and is incompatible with the
current Phase-2/P5 text extractor and accounting guard.

The operator must confirm or normalize the long endpoint's response contract,
or a prospective adapter must parse the documented nested `choices` schema and
restore enforceable cost/quota accounting with offline tests. Until then, use
the standard endpoint and existing request-size constraints; do not treat the
long endpoint as a drop-in host substitution.

- Keep the standard endpoint for ordinary short requests.
- Once its response and accounting contract is validated, use the long-request
  endpoint when a request may exceed 30 seconds, including large prompts, long
  requested outputs, or slow reasoning-model calls.
- Phase-2 annotation calls with large A4-window inputs and/or large output
  budgets may benefit from the long endpoint only after that validation.
- Preserve the existing API key, payload schema, response parsing, cost
  accounting, retry limits, provenance, and spend guards; only the base URL
  changes.

## P5 v2.2 implication

This endpoint information does **not** explain the existing P5 v2.2 empty-text
failures. Those requests used a 25-second client timeout and 400-token output
budget, and the failed calls returned HTTP non-error JSON after approximately
1.6--1.9 seconds. They did not approach the standard endpoint's 30-second
gateway cap. The frozen one-call P5 response-shape diagnostic should therefore
remain on the standard endpoint to reproduce the original path. Switching its
endpoint would confound the diagnostic.

If a later, separately authorized endpoint-comparison diagnostic is required,
the same exact request can be sent once through each endpoint with sanitized
response-shape telemetry and kept outside all annotation and validation
outputs.
