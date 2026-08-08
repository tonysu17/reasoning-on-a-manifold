# P5–Phase-2 shared vanilla generation contract

**Date:** 8 August 2026  
**Status:** pre-generation compatibility contract; no model or API execution is authorised by this document.  
**Purpose:** ensure that Phase 2 produces the generic-reasoning vanilla outputs once and that P5 consumes those exact outputs as a read-only input artefact, rather than creating a second “vanilla” dataset.

## Decision

The Phase-2 E8-compatible generation settings satisfy P5's parser and scorer requirements. Phase 2 should generate its three vanilla arms once, on the immutable 100-task manifest, and expose them through one canonical artefact manifest. P5 will ingest those rows by artefact and row hash and **will never regenerate the overlapping Phase-2 vanilla rows**.

The one necessary clarification is the token cap. The E8-compatible Phase-2 cap is **8,192 new tokens**. P5's separate 4,096-token pilot, with a possible common increase to 6,144, applies only to P5-owned pilot and safety generation. It does not impose a shorter cap on the imported Phase-2 generic rows and is not a reason to create a second vanilla set.

## Scope and row cardinality

The canonical shared object covers the generic stratum only:

- 100 tasks from `results/prereg/phase2_task_manifest.json`, in its frozen order;
- one unsteered/vanilla generation for each of the three Phase-2 checkpoint roles: base R1, public STAR1, and DeepScaleR;
- therefore 300 planned `(checkpoint_role, task_id)` row keys, each represented by either one successful terminal record or one explicit terminal error after the identical-setting retry policy.

The owned safety and matched-control full-FT seed-42 checkpoints are not Phase-2 arms. If they enter P5's final generic comparison, P5 may generate only those additional checkpoint roles, under a separately frozen P5 run and spend authorisation. Those rows must use the same Phase-2 task bytes and input-format contract, but they are not part of—and must not overwrite or be described as—the canonical Phase-2 vanilla artefact.

## Exact input contract

For every task and every checkpoint role, use the **base R1 tokenizer and chat-template alias**, not each checkpoint's bundled tokenizer. The pinned base identity is:

- model/tokenizer snapshot: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B@ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`;
- `tokenizer.json` sha256: `88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773`;
- `tokenizer_config.json` sha256: `8ac8c85fb242563c2260baec0909debd69d718af6a0b3d90e6cab62b4d341cd5`;
- effective chat-template sha256: `56a1447ad31926fdc21fb07e56e5642bd9c850c4f52d8c8af7bbe5f079a84f5f`.

Format each prompt as a single user message with
`apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)`. The pinned template emits the assistant marker followed by `<think>\n`; the E8 fallback appends `<think>\n` if a compatible template does not. Tokenise that resulting string once with the base alias. No checkpoint-specific system prompt, chat template, BOS substitution, or other prompt prefix is permitted.

Before generation, record for each task:

- exact UTF-8 prompt-text sha256;
- ordered input-token IDs or an immutable lossless encoding of them;
- sha256 of the canonical integer input-ID sequence;
- input length;
- Phase-2 manifest file sha256 and its internal `ids_sha256`.

The input-ID hash for a task must be byte-for-byte identical across all three roles. A mismatch is a pre-generation refusal, not a tolerated model-specific exception. P5 will validate the imported hashes against the same base-tokenizer procedure after the Phase-2 manifest arrives.

## Exact decoding contract

Use one completion per row with the following common settings:

| Field | Required value |
|---|---|
| `max_new_tokens` | `8192` |
| semantic temperature | `0.0` |
| `do_sample` | `false` |
| number of samples | `1` |
| run/generation seed | `20260808` |
| top-p/top-k | inactive because sampling is disabled; do not enable either |
| repetition/length penalties | transformer defaults; record resolved values and do not add a custom penalty |
| EOS token | base-tokenizer `<｜end▁of▁sentence｜>`, token id `151643` |
| pad token during generation | EOS token id `151643` |
| custom stop strings | none |

The implementation may pass a placeholder numerical temperature such as `1.0` to `transformers.generate` only when `do_sample=false`, matching the E8 code path, but provenance must record both the semantic setting (`temperature=0.0`) and the exact resolved generation configuration. The seed is inert under greedy decoding but must still be recorded as `20260808`.

Stop only at the first EOS or the common 8,192-new-token cap. In particular, do not stop at `</think>`: the post-reasoning answer is part of the output needed for boxed-answer scoring. Decode the generated IDs before EOS with `skip_special_tokens=true`. Under the executed E8 batched convention, `n_tokens` is the number of generated IDs before the first EOS, excluding EOS itself. Record:

- `stop_reason="eos"` when EOS occurs;
- `stop_reason="length"` when 8,192 generated tokens are reached without EOS;
- `stop_reason="other"` only for a documented non-EOS/non-cap successful termination.

No parser-side clipping is permitted before the raw record is hashed. Sentence annotation, boxed-answer extraction, length, repetition, and backtracking-per-1,000 calculations all operate on the preserved decoded completion. A cap hit is a valid but explicitly truncated observation, not a generation failure.

## Canonical artefact and provenance schema

Phase 2 may store the rows in one file or immutable shards, but it must emit one top-level manifest that defines the canonical shared artefact. That manifest should provide:

- schema version and stable run ID;
- canonical artefact/shard paths and sha256 values;
- ordered row-key manifest and its sha256;
- Phase-2 task-manifest path, file sha256, internal `ids_sha256`, and prompt count;
- preregistration and amendment identifiers/hashes, including the decision-sealed A2 state;
- full 40-character code commit and dirty flag;
- generation configuration object and sha256;
- Python, PyTorch, Transformers, CUDA/driver, hardware, dtype, attention implementation, and batching configuration;
- start/finish times, authorisation record, planned/success/error counts, and complete attempt history;
- checkpoint IDs, immutable revisions, weight hashes, and config hashes;
- base-tokenizer, tokenizer-config, and chat-template hashes above.

Each terminal row must expose, directly or through an unambiguous indexed join:

- `schema_version`, `run_id`, `checkpoint_role`, checkpoint ID/revision/weight/config hashes;
- `task_id`, category, exact prompt text, prompt-text hash, task-manifest hash, and input-ID hash;
- generation-config hash, generation seed, attempt number/ID, timestamps, code commit/dirty state, environment hash, and hardware;
- for success: decoded completion text, generated-token count, stop reason, and preferably a hash/lossless encoding of generated token IDs;
- for error: error type and message, with no fabricated empty successful output;
- sha256 of the canonical serialisation of the row.

These fields are sufficient for P5's current generation-record validator. If Phase 2 uses a different native field layout, P5 may build a deterministic normalised view under `.codex/out/`, but that view must carry the source artefact sha256 and source-row sha256. It never replaces the Phase-2 artefact of record.

## Resume, retry, and failure rules

- Resume only under the identical prompt manifest, checkpoint identity, tokenizer/input IDs, and generation-config hash.
- Preserve every attempt; never silently replace an errored or partial attempt.
- Infrastructure retries do not change decoding settings, prompt bytes, batching semantics, or checkpoint identity.
- Never repair a missing row by switching to a shorter cap, sampling, another template, or checkpoint-specific tokenizer.
- Duplicate successful rows with the same logical key must be byte-identical. Conflicting duplicates make the artefact inadmissible until resolved in Phase 2.
- Missing and errored rows remain unresolved and are reported in P5's complete-pair and missingness counts; they are never scored as zero.

## P5 ingestion and no-regeneration rule

P5 treats the canonical Phase-2 vanilla set as an **external read-only input artefact**. Once handed its top-level path and sha256, P5 will:

1. verify the Phase-2 task-manifest hashes and exact prompt bytes;
2. verify the checkpoint, tokenizer/template, input-ID, generation-config, and row hashes;
3. import only the base R1, STAR1, and DeepScaleR vanilla terminal rows;
4. compute P5 observational endpoints or submit the preserved text to the frozen scorer/annotator;
5. write only derived normalised, scoring, and analysis products under `.codex/out/` until any later ledger-controlled promotion.

P5 will **not** call a model to recreate, fill, “clean up,” or validate any imported Phase-2 vanilla row. If the canonical artefact is absent, incompatible, corrupt, or lacks enough provenance to validate, the P5 final generic comparison remains blocked or is reported as provenance-unresolved. It does not launch a replacement vanilla run. Any authorised correction belongs to the Phase-2 execution/provenance process and must preserve its preregistration and amendment rules.

P5 also preserves the causal-family firewall: it may analyse the predeclared observational endpoints on the vanilla rows, but it must not inspect or consume Phase-2 intervention outcomes before the injection-recovery gate is evaluated.

## Compatibility conclusion

No P5 scorer or parser requirement requires a change to the sealed E8 decoding settings. The P5 tooling accepts full raw text, `n_tokens`, and `eos`/`length`/`other` stop reasons; it treats truncation as a guard and missing rows as unresolved. Therefore Phase 2 can retain the E8 8,192-token greedy protocol unchanged, provided it uses the common base-tokenizer alias and emits the hashes, terminal row states, and attempt lineage above.

Evidence checked read-only for this contract:

- `results/prereg/PHASE2_TRANSPORT_PREREG_2026-08-08.md` §§2–4 and A1;
- decision-sealed A2 handoff and `results/prereg/PHASE2_ADJUNCT_AMENDMENT_2026-08-08.md`;
- `results/eval/R1-1.5B__E1/provenance.json`;
- `configs/config.yaml` (`chains.max_new_tokens=8192`, `temperature=0.0`);
- `src/steered_inference.py`, `src/chain_gen.py`, and `src/model_adapters.py`;
- `.codex/out/P5_BEHAVIOURAL_EVALUATION_SPEC_2026-08-08.md`;
- `.codex/out/P5_INPUT_ID_GATE_2026-08-08.json`;
- `.codex/out/p5_preflight.py`.

E8's provenance record has `git_commit: null` and represents the cap as `max_new_tokens: null`, resolved through the configured default. This contract does not retroactively repair that execution provenance; it makes the Phase-2 setting explicit before the new generation run.
