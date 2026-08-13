# P5 4,096-token analytic-prefix contract

**Status:** approved design implementation contract; no data processed and no spend authorized  
**Primary prefix:** first 4,096 original generated token IDs or earlier EOS  
**Tokenizer/decoder:** frozen base-R1 alias used by the Phase-2 generation contract

## Required source representation

Every successful shared or P5-owned generation row must retain the ordered generated token IDs losslessly, inline or in an immutable hash-bound sidecar. The sequence excludes the terminating EOS in accordance with the Phase-2 counting convention. `n_tokens` must equal the sequence length.

Decoded completion text alone is insufficient. Re-encoding decoded text can change token boundaries and cannot reconstruct an exact generated-token prefix. A row without lossless generated IDs is inadmissible for the primary P5 prefix estimand; it remains absent or provenance-unresolved rather than being approximately clipped by characters or words.

## Prefix construction

1. Verify the source row, source artefact/shard, generation configuration, tokenizer/template, and generated-ID hashes.
2. Let `k = min(n_tokens, 4096)` and take the exact first `k` stored generated IDs.
3. Decode that prefix once with the frozen base-R1 tokenizer alias and `skip_special_tokens=true`.
4. Persist the ordered prefix IDs or a lossless immutable sidecar, their canonical SHA-256, decoded prefix-text SHA-256, `k`, and the source-row SHA-256.
5. Mark `prefix_stop_reason=eos` only when the source terminated at EOS at or before 4,096. Mark `analytic_prefix_cap` whenever source generation continued beyond 4,096. Other source terminations at or before 4,096 retain `other` with their documented reason.
6. Deterministic sentence/clause source-unit construction and behavioural annotation operate only on the persisted decoded prefix.

## Validation

- token IDs are non-negative integers, not booleans;
- `n_tokens` equals the exact ID count;
- EOS token ID `151643` is absent from the stored pre-EOS sequence;
- a raw `length` stop has exactly the raw generation cap recorded in the source generation configuration;
- duplicate `(checkpoint_role, task_id)` keys must be byte-identical;
- a complete source row at or before 4,096 must decode byte-identically to its preserved raw completion text;
- missing IDs, source-hash mismatches, conflicting duplicates, or decode mismatches fail closed.

## Estimand boundary

This prefix is an analysis rule and does not amend or overwrite Phase 2's sealed 8,192-token raw artefact. Full-completion analyses are secondary sensitivities. Pilot variance transfers directly only to the 4,096-prefix estimand. Cap-hit and EOS support are reported by task and checkpoint; truncation is observed support, not a generation failure.
