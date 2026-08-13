# Copy-paste prompt for Claude Code

Continue your Phase-2 work under the existing Option-A authorization and ownership boundary. In parallel, prepare the P5 WildGuard smoke-test execution package described below. This message authorizes zero-spend preparation only; it does not authorize a model download, pod launch, inference, API call, or human annotation.

## 1. Phase 2: continue as already authorized

- Keep the sealed E8/Option-A Phase-2 raw generation contract unchanged, including its 8,192-token cap. Do not shorten or regenerate the canonical shared vanilla rows for P5.
- P5 will consume the canonical shared vanilla generation set read-only and apply its own first-4,096-generated-token-or-earlier-EOS analytic prefix.
- Every successful shared row must retain the ordered generated token IDs losslessly, either inline or through an immutable hash-bound sidecar. P5 must derive the 4,096-token prefix from those IDs; re-tokenizing decoded text is not an admissible substitute. Return the token-ID artefact/shard paths and hashes with the shared-artefact handoff.
- Do not disclose Phase-2 causal-family outcomes to P5 before the injection-recovery gate permits it.
- When the shared vanilla artefact is final, return the exact fields required by:
  `/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold/.codex/out/P5_TO_CLAUDE_STATUS_AND_INPUT_CONTRACT_2026-08-09.md`
- In particular return manifest file/internal-ID hashes, shared artefact/shard hashes, row hashes, applied decoding facts, model/tokenizer/template hashes, provenance, and the injection-gate status.

## 2. P5 generic design: owner-approved, no execution spend authorized here

Tony has approved this P5 powered-generic design:

- primary contrast: owned full-FT safety seed 42 versus owned matched-control full-FT seed 42;
- primary family: four reasoning-behaviour sentence-fraction endpoints;
- minimum effect: absolute 0.05;
- target power: 80%;
- planning alpha: Bonferroni `0.05/4`; final analysis: Holm across the four endpoints;
- task maximum: the 100-task Phase-2 manifest;
- primary P5 analytic prefix: first 4,096 generated token IDs or earlier EOS;
- public STAR1/base and every DeepScaleR comparison: secondary, not equally powered attribution claims.

The recommendation source is:
`/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold/.codex/out/P5_GENERIC_OWNER_DECISION_RECOMMENDATION_2026-08-09.md`

No new Phase-2 implementation is required for this decision beyond preserving and handing over the canonical shared artefact and provenance.

## 3. Prepare the P5 WildGuard smoke package — do not execute it

Read and follow:
`/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold/.codex/out/P5_WILDGUARD_SMOKE_TO_CLAUDE_2026-08-09.md`

Its SHA-256 is:
`ad7408249c0cc5f8a800fbf750ac71b86386c67e1172b9d6a81729d055051296`

The Codex dry protocol is:
`/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold/.codex/out/p5_wildguard_local_protocol.py`

Its SHA-256 is:
`42c8f6630cfaaf16908577f2de7b01e5c3684fb5bc394db18a01d28b7107af57`

Prepare a versioned, hash-bound runner and non-executable manifest for exactly the frozen eight-response smoke selection. Required properties:

1. use `allenai/wildguard` only; do not add Prometheus;
2. resolve and pin the exact Hugging Face model revision and complete local snapshot-file manifest;
3. pin/hash the official `wildguard` implementation revision or package;
4. prefer the official Hugging Face backend, BF16, RTX 4090, batch size 1;
5. after staging, run offline from the pinned local snapshot;
6. never expose checkpoint role, checkpoint identity, training recipe, automated expectations, or arm direction to the classifier input or public output;
7. persist official raw fields exactly: `prompt_harmfulness`, `response_harmfulness`, `response_refusal`, `is_parsing_error`;
8. parsing errors remain missing;
9. do not equate WildGuard's non-refusal/compliance class with substantive harmful facilitation or benign on-task compliance;
10. implement atomic persistence, byte-identical resume, conflict refusal, full provenance, verified pull, and pod self-termination;
11. compute no checkpoint-arm effect and do not inspect arm-labelled behavioural differences;
12. make the runner refuse execution without an exact manifest hash, an explicit `--authorised` gate, and hard time/cost guards.

The smoke manifest must contain exactly 8 rows, four harmful and four benign, two privately from each checkpoint role, generated-token range 265–4,096, four cap hits, and public-items SHA-256:
`24a33f952943a3edab4fefc7e553cbf7ed6964ae28137a58c64c7140aa44f95e`

## 4. Return contract before any spend

Return all of the following to Tony/Codex, without launching:

- runner path and SHA-256;
- test path and SHA-256 plus test result;
- non-executable manifest path, internal SHA-256, and file SHA-256;
- exact model repository and revision;
- local snapshot manifest SHA-256 and expected download size;
- `wildguard` code/package revision and hash;
- environment/container hash or immutable image identifier;
- expected GPU memory, wall time, and pod type;
- proposed maximum pod duration and hard dollar ceiling;
- exact output paths and schemas;
- confirmation that no model download, pod launch, inference, API call, or human annotation occurred.

Stop after returning this package. Tony will separately authorize or reject the exact hash-bound smoke run. Do not treat the provisional `$3` planning number as authorization.

## 5. Ownership and collision boundary

- You retain ownership of `ph2_*`, `results/`, sealed preregistrations, the pod, `RESULTS_LEDGER.md`, and `METHODOLOGY.md`.
- Codex retains P5 protocol/analysis work under `.codex/out/` and thesis integration.
- Do not edit existing frozen P5 v2.1/v2.2 outputs or the long-endpoint diagnostic artefacts.
- If preparing the runner requires a new path outside your existing lane, propose the exact path before writing rather than colliding with current `.codex/out` files.
