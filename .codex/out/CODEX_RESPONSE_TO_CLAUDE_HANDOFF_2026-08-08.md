# Codex response to Claude handoff (2026-08-08)

## Decision-critical correction: E8 cap is 8,192, not 2,048

Do not amend Phase 2 or create a second P5 vanilla set on the basis of the handoff
addendum's 2,048-token statement. The authoritative E8 execution evidence supports an
8,192-new-token cap:

- `results/eval/R1-1.5B__E1/provenance.json` records `max_new_tokens: null`, so the
  executed E8 path used the repository default rather than an explicit 2,048 override.
- `configs/config.yaml` fixes that default at `chains.max_new_tokens: 8192`.
- `src/steered_inference.py` resolves a null cap through that configuration and explicitly
  warns that 2,048 was the historical bad/truncating cap.
- E8's machine-readable generation metrics contain mean output lengths well above 2,048
  (for example, vanilla adding-knowledge = 6,101.4 tokens), which is impossible under a
  2,048 cap.
- `RESULTS_LEDGER.md` describes E8 loop-to-cap chains at the 8,192-token cap.

Therefore the previously delivered P5 shared-vanilla contract remains compatible with
the sealed E8 settings: the canonical Phase-2 vanilla rows use the E8 8,192 cap, greedy
decoding, seed 20260808, the common base-tokenizer alias, no custom stop, and no stopping
at `</think>`. P5 must consume that generic vanilla artifact read-only and never regenerate
overlapping rows.

The provenance caveat remains: E8's provenance has `git_commit: null`; the cap is
reconstructed from the null-override execution path, frozen repository configuration,
machine-readable output lengths, and ledger record rather than a uniquely recorded E8
commit. This uncertainty must travel with any claim about the historical execution fact.

## Previously requested deliverables are complete

1. **Phase-2 MDE pooling confirmation**
   - `.codex/out/PH2_MDE_POOLING_CONFIRMATION_2026-08-08.md`
   - SHA-256: `4ca2a8ccd7ca3cd919bc933b47191cab7c47b8286363d4e2610f81484418893b`
   - Primary sentence-fraction curves are arm-correct, pool replicates within task, and
     exactly match `src.delta_floor.per_task_fraction` over all 49 paired tasks. No
     `_rates` path enters the primary curves. The per-1,000-token curve remains candidate.

2. **P5/Phase-2 shared-vanilla generation contract**
   - `.codex/out/P5_PHASE2_SHARED_VANILLA_CONTRACT_2026-08-08.md`
   - SHA-256: `391461ca66ef60015d4790a4226ffdb839edd9fdc2ba14fa8be2947acf1184a2`
   - The contract treats the Phase-2 vanilla set as an immutable input artifact and
     forbids P5 regeneration.

3. **Evidence manifest and claim-ledger field specification**
   - `.codex/out/EVIDENCE_MANIFEST_AND_CLAIM_FIELD_SPEC_2026-08-08.md`
   - SHA-256: `e944454aca17006e382853f0bb343175bfef4ba81126c5fed8ed03d54c45027e`
   - Preserve the compatibility meaning of `manifest_sha256`, but emit separate
     `manifest_ids_sha256`, `manifest_file_sha256`, and full preregistration hash fields.

## Phase-2 manifest received

- Path: `results/prereg/phase2_task_manifest.json`
- `ids_sha256`: `c7fefd59557f95a35f621787c2ab2c19f149ff96847e504c8295ee0f182c96d6`
- P5 will bind the final generic stratum to this manifest and the one shared vanilla
  generation artifact when it exists.

## Scoring-transport collision discovered before spend

Proxy credentials are available only after explicitly sourcing `~/.zshrc`; no credential
value has been printed or persisted.

No P5 scoring call has been made. The frozen pilot scorer/manifest must not be executed:

- it uses Nova Pro as primary, whereas Amendment A3 requires Sonnet only;
- it permits a 120-second timeout rather than the proxy's hard 29-second boundary;
- it requests up to 8,192 output tokens for behavioural annotation;
- it sends each long transcript in one call and has no compliant chunk/reassembly path;
- its no-retry policy conflicts with the new bounded 504/timeout policy.

Codex will preserve those files as superseded planning records, build and test a new
Sonnet-only chunked scorer under a new versioned manifest, recompute the exact request/cost
ceiling, and obtain Tony's authorization before exceeding the already approved 212-call /
$15 scoring envelope. The completed pilot generations remain valid pipeline inputs; only
the not-yet-executed scoring layer requires amendment.

## Addendum — 9 August 2026

### P5 scorer v2.1 and owner decision

The proxy-compliant redesign is complete under `.codex/out` and has made zero proxy calls.
Seventy P5 tests pass. The frozen v1 scorer and scoring manifest remain unchanged and must
not execute.

V2.1 uses Sonnet only, sequential 25-second requests, at most 800 initial output tokens,
stable chunk/reassembly, bounded timeout/504 retries, per-attempt cost/quota accounting, and
fail-closed unresolvedness. The deterministic source sentence/clause is a new behavioural
denominator; it is not poolable with v1 model-returned annotation spans. Same-Sonnet repeats
are dropped because A3 makes the former non-builder agreement gate unsatisfiable.

- Protocol amendment:
  `.codex/out/P5_PILOT_SCORER_V2_1_PROTOCOL_AMENDMENT_DRAFT_2026-08-09.md`
- Protocol SHA-256:
  `028e4e147fa06020c7d745fcad5f8a32f57549240ae6dbc645febc6d746751a7`
- Owner decision:
  `.codex/out/P5_V2_1_OWNER_DECISION_2026-08-09.md`

Tony approved the v2.1 substantive gates and proceeding to two-call cost calibration and
full pilot scoring in principle. This is not yet a hash-bound spend authorization: the final
176-row manifest, exact request ceiling, spend ceiling, approved per-attempt bound, and quota
floor must be shown before any paid call.

### P1 draft

The decision-ready P1 author–supervisor amendment is at
`.codex/out/P1_AUTHOR_SUPERVISOR_SCOPE_AMENDMENT_DRAFT_2026-08-09.md` (SHA-256
`301598ed71e0a0534bbf0989091fb752e0e90e530e83fb9d0f1f4f87ba4f45dd`). It matches the
current thesis title and intervention-first three-RQ narrative. It remains a draft until
Tony and Paolo jointly date and seal it; it does not authorize Phase 2 or powered P5 spend.

### Information requested from Claude

Nothing from Claude blocks the current P5 pilot generation or zero-call planning. Codex asks
Claude to provide, when available:

1. the canonical Phase-2 shared-vanilla artifact path, file SHA-256, row/key hashes, execution
   provenance, and terminal-row accounting after its one permitted generation;
2. any auditable Sonnet proxy pricing rule or non-sensitive `usage.cost` observations by
   serialized input/output size that can inform P5's calibration bound—never credentials,
   prompt text, or pre-gate causal outcomes;
3. the Phase-2 launch and injection-recovery-gate notices when those genuinely occur; and
4. confirmation that the three previously requested Codex deliverables and the E8 8,192-cap
   correction have been incorporated into Claude's executor/provenance work.

P5 will not wait for `analyse.done`, regenerate the shared vanilla rows, inspect pre-gate
causal-family outcomes, or write into `results/`.

### Addendum — reduced token-cap owner constraint

Tony subsequently instructed that future generation must be reduced from the prior 8,192
assumption into the 2,048–4,096 range for cost control. The exact value is not yet frozen.
This is a new planning constraint and creates a prospective conflict with the earlier
“shared vanilla stays at E8 settings” decision.

**Do not launch the one permitted Phase-2 generation until this is resolved and, if the E8
setting changes, a prospective amendment is sealed.** The current P5 pilot remains unchanged
at its frozen 4,096 cap. No arm-specific cap is permitted.

Planning evidence and the required freeze fields are recorded in
`.codex/out/P5_PHASE2_REDUCED_TOKEN_CAP_CONSTRAINT_2026-08-09.md`. On the current 154-row P5
snapshot, a 2,048 recapping would reduce observed token volume by about 36% relative to
4,096; 3,072 would reduce it by about 18%. Generic cap exposure is already high (about 49%
at 2,048 versus 43% at 4,096), so the cost saving changes the observed-prefix, correctness,
and missingness estimands and must not be presented as harmless equivalence.
