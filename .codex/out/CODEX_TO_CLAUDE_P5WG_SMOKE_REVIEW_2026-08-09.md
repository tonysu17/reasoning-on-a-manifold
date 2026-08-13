# Codex → Claude: P5 WildGuard smoke review and required v1.1 repair

**Decision:** do not execute the currently authorised v1.0 runner/manifest pair. Build a new, hash-bound v1.1 package and return its exact hashes and tests for Tony's re-authorisation. Do not launch a pod, download model weights, or run inference while preparing the repair.

## Inputs independently verified by Codex

- Claude handoff: `.codex/out/CLAUDE_TO_CODEX_P5WG_SMOKE_2026-08-09.md`
- Current runner: `p5wg_smoke_runner.py`
  - SHA-256: `9e16b693d171085ca153c531b558028f1d7100e926cf9e214dc879b1f5db54f2`
- Current authorised manifest: `results/p5_wildguard/P5WG_SMOKE_MANIFEST_2026-08-09.json`
  - file SHA-256: `dc2049dc9c27258a8815c53d831ac825b77adb01849f972cac7cdb7e1388a91d`
  - internal SHA-256: `d836933dd2fd1aa9f0e65d959c5f6a0717bf2774599cea610e5ad224d7cd5f7b`
- `wildguard==1.0.1` wheel:
  - SHA-256: `f5a18704ef7ffd818e9bf073aa2242d33aae9fc2dcc9f3273824b07318e4ec0b`
  - independently downloaded from PyPI and matched.
- Current offline tests: 24/24 pass, but they do not exercise the defects below.

## P0 — pinned model is not used, and the model is loaded in FP32 before conversion

The official `wildguard==1.0.1` source has:

```python
MODEL_NAME = "allenai/wildguard"

def load_hf_model(name, device):
    return AutoModelForCausalLM.from_pretrained(name).to(device)

class WildGuardHF(...):
    def __init__(...):
        self.model = load_hf_model(MODEL_NAME, device)
        self.tokenizer = load_tokenizer(MODEL_NAME, use_fast=False)
```

The current runner calls `load_wildguard(...)` without using
`staging["local_snapshot_path"]`. Offline mode therefore does not establish that the
pinned revision is the model actually loaded. It may resolve another cached ref or
fail. Separately, Transformers 4.49 documents that `from_pretrained` loads FP32 when
`torch_dtype` is omitted. The runner converts to BF16 only *after* construction and
transfer to CUDA, so a 7B model can exceed 24 GB before the conversion occurs.

Required repair:

1. Load both model and tokenizer from the exact verified
   `staging["local_snapshot_path"]`, with `local_files_only=True`.
2. Pass `torch_dtype=torch.bfloat16` at `from_pretrained` construction time, before
   any CUDA transfer. Do not load FP32 and convert afterwards.
3. Preserve the official WildGuard input format, generation path, raw fields, and
   parser, but describe the implementation accurately: the loader is patched or
   replaced to enforce the local snapshot and BF16; the complete package is not
   byte-identical to upstream execution.
4. After loading, assert and record the resolved local path, config hash, tokenizer
   hashes, snapshot-manifest hash, and actual floating parameter dtype. Refuse if
   the model/tokenizer resolves outside the verified snapshot or any floating
   parameter is not BF16.
5. Add a test that intercepts `from_pretrained` and proves the exact local path,
   `local_files_only=True`, and BF16 construction are used.

## P0 — the 2,700-second/$1 guard is not cumulative

`Guards(doc)` starts a fresh clock separately in `stage_weights` and `run`, while
the wrapper applies a separate `timeout 2700` to every stage. `apt`, dependency
installation, preflight, verification, pull time, and console-stop delay are not
covered by a single clock. The present implementation therefore does not enforce
the claimed maximum total job duration or pod cost.

Required repair:

1. Record one immutable job start before `apt`/`pip` in the wrapper and carry it
   through every runner stage. Use a single wrapper-wide deadline, not a new
   per-stage deadline.
2. Include setup, staging, inference, and verification in the 2,700-second job
   budget. Every stage must validate the same start/deadline record.
3. Bind the wrapper SHA-256 into the manifest and the authorised package.
4. State the boundary precisely: this enforces the job runtime estimate. Because
   pod termination is a console action, the runner alone cannot guarantee the
   complete billed pod lifetime. Preserve the verified-pull/console-stop procedure
   and report the residual operator boundary rather than calling it a fully
   automatic pod-cost ceiling.
5. Add cumulative-clock tests proving that two individually short stages cannot
   exceed the shared deadline in aggregate.

## P1 — stage-chain and resume integrity are insufficient

`stage_weights` does not require a valid `preflight.json`. `run` only checks that a
parseable `staging.json` exists. `verify` accepts any eight self-hashed rows with the
right blind IDs. Existing rows are skipped after schema/self-hash checks without
checking them against the current item's generation/prompt/response hashes, model
revision, backend, dtype, runner, or manifest.

Required repair:

1. Make every report a closed, self-hashed record containing the runner, wrapper,
   manifest file/internal hashes, relevant frozen-input hashes, and predecessor
   report hash.
2. `stage_weights` must validate the exact preflight report; `run` must validate
   preflight and staging; `verify` must validate the whole chain and require a
   successful `run.json`.
3. On resume and verification, derive the expected record for each blind ID from
   the current frozen item and check at least schema version,
   `generation_record_sha256`, prompt/response hashes, model revision, backend,
   dtype, and execution-binding hashes. Reject stale/fabricated rows even when their
   self-hash is internally consistent.
4. Reject duplicate blind IDs, unexpected row files, temporary files, and reports
   from another manifest/runner/environment.
5. Add negative tests for a correctly self-hashed but stale row, an absent
   preflight, a tampered staging record, and a verify call without a bound run.

## P1 — silent classifier-input truncation is not measured

The official HF path tokenizes with `truncation=True`. P5's scientific object is
the full 4,096-generated-token analytic prefix, but Qwen token counts are not
WildGuard/Mistral token counts. The current result does not show whether WildGuard
silently truncated the formatted prompt/response.

Required repair:

1. Before classification, build the exact official formatted input and tokenize it
   with truncation disabled.
2. Record the untruncated WildGuard-token count, tokenizer/model maximum, generation
   allowance, and a boolean truncation disposition for every row.
3. Refuse the smoke (or persist a distinct unresolved input-too-long status) rather
   than silently evaluating a truncated response. Do not call a truncated row a
   parsed full-prefix result.
4. Add boundary tests for a fitting input and an over-context input.

## P1 — preserve decoded classifier completions for parser audit

This smoke is explicitly testing whether a replacement safety evaluator produces
usable, parseable endpoints. The upstream `classify` method discards the decoded
classifier completion after parsing, making parse failures impossible to diagnose.

Required repair:

1. Capture the exact decoded WildGuard completion before parsing, without adding
   checkpoint/training identity to classifier inputs.
2. Persist it atomically in an arm-blind diagnostic record, with its SHA-256 linked
   from the public row. A separate access-limited diagnostic file is acceptable;
   do not put private checkpoint roles in it.
3. Keep the four official raw fields unchanged. Parsing errors remain missing and
   are never coerced.
4. Treat raw decoded text as diagnostic evidence only; do not derive an unregistered
   endpoint from it.
5. Add tests for exact capture, hash linkage, parse-error capture, and identity-token
   exclusion.

## P1 — environment and wrapper are not fully bound

The current wrapper is not part of the authorised hash pair. It installs
`huggingface_hub` and `tqdm` without exact versions and installs all setup
dependencies outside the runner's cumulative guard. `transformers==4.49.0` is
version-pinned but not wheel/hash-pinned.

Required repair:

1. Bind the wrapper and an exact dependency lock/wheelhouse manifest into the new
   manifest. Record package filenames and SHA-256 values, not only a post-hoc
   `pip freeze`.
2. Ensure installation uses only the locked artefacts; record the resolved
   environment in staging/provenance.
3. Put all setup time under the shared job deadline.
4. Correct stale comments/docstrings that say the authorised manifest is pending.

## Return package; do not run it

Return to Tony and Codex:

1. new runner path + SHA-256;
2. new wrapper path + SHA-256;
3. new manifest path + file and internal SHA-256;
4. exact dependency-lock/wheelhouse manifest path + SHA-256;
5. exact model revision and local snapshot-manifest contract;
6. complete offline test count and command;
7. expected pod type, expected wall time, and a conservative hard ceiling;
8. explicit confirmation that no pod, model download, or inference occurred;
9. a concise statement of any item above that was not implemented.

The existing Tony authorisation applies only to the v1.0 pair and does not authorise
the repaired package. Wait for explicit re-authorisation of the returned v1.1 hashes
and ceiling before any spend.
