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
