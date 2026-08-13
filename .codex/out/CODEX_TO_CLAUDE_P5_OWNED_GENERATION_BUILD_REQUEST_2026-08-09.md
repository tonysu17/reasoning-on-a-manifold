# Codex → Claude: build request for P5 owned full-FT generation pair

**Request:** prepare only; do not execute, launch a pod, load a model, or download
anything under this document. Claude retains the pod lane. Return a new hash-bound
runner/environment/manifest package and a conservative cost ceiling for Tony's
explicit authorization.

## Frozen P5 input manifest

- Path: `.codex/out/P5_OWNED_FULLFT_GENERATION_INPUT_MANIFEST_2026-08-09.json`
- File SHA-256: `97a96c2c32e144552f4dfb256e1019ee69e509367776f140d57a7f552c5b9e71`
- Internal SHA-256: `f24deb92d6fa3c18b201da8c0f1b5406fe5371a2950310f51937a3e1d95c5987`
- Builder: `.codex/out/p5_freeze_owned_fullft_generation_inputs.py`
- Builder SHA-256: `91835aa04603646ad2375320cfeae7e35a0f1a7e2444121a33bb53f14049b733`
- Builder tests: `.codex/out/test_p5_freeze_owned_fullft_generation_inputs.py`
- Test SHA-256: `0eac0147908db5d80b54ad097b7cb12010dba04c21af015a9e311b79d230064b`

The manifest binds:

- Phase-2 task-manifest file SHA-256
  `69cbe32dc7991c42ee58c8b5fae683750abb6a8812444ab320e68e23b8214333`;
- Phase-2 task-ID SHA-256
  `c7fefd59557f95a35f621787c2ab2c19f149ff96847e504c8295ee0f182c96d6`;
- ordered task-content SHA-256
  `3cb2161ca16679b0da3f12a3c5375410835b8b0b5a20a43c0210b0832c7d9717`;
- 100 ordered prompt/input-ID records;
- both complete local checkpoint directory manifests;
- identical control/safety tokenizer, template, config, and generation-config files;
- distinct safety/control `model.safetensors` files.

## Frozen generation object

- Roles: `p5_owned_fullft_control_s42` and `p5_owned_fullft_safety_s42`.
- Rows: 100 tasks × 2 roles = exactly 200 terminal generation records.
- Independent unit: task/prompt; roles are paired repeated observations.
- Maximum new tokens: 4,096.
- Decoding: greedy, `do_sample=false`, one generation per task/role.
- Do not inherit the local checkpoints' saved sampling defaults
  (`do_sample=true`, temperature 0.6, top-p). Explicitly override them.
- Semantic temperature: zero; omit the temperature argument under
  `do_sample=false` if required by the installed Transformers version.
- Seed: 20260808, recorded but inert under greedy decoding.
- EOS/pad ID: 151643.
- No custom stop strings and no stop at `</think>`.
- Use the exact frozen task prompt bytes and chat-template input-ID hashes.
- Persist the original generated token IDs, excluding terminal EOS, plus exact
  `eos`/`length` stop reason and `raw_max_new_tokens=4096`.
- Decode stored IDs once for text and bind token/text hashes. Decoded-text
  re-tokenisation is forbidden.

This 4,096 cap is the prospective P5-owned primary boundary requested for cost
control. It does not amend Phase 2: Phase-2 shared vanilla retains its sealed raw
8,192 cap and is reduced analytically to the same P5 4,096-prefix estimand.

## Required runner properties

1. Treat the two local checkpoint paths as immutable inputs; verify every file
   against the frozen directory manifests before model loading.
2. Load in BF16 at construction time on a suitable GPU. Do not load FP32 and cast
   after CUDA transfer.
3. Before generation, compare the owned tokenizer/template hashes and all 100 input
   ID hashes with the final Phase-2 base-tokenizer-alias facts. A mismatch blocks;
   do not silently substitute or re-tokenise.
4. Bind runner, wrapper, dependency lock/container, input manifest, checkpoint
   manifests, and environment into the authorised package.
5. Use a single cumulative job clock/cost guard starting before setup. State clearly
   which billed pod lifetime remains dependent on verified pull and console stop.
6. Use closed, self-hashed stage reports with predecessor hashes. Require preflight
   before generation and verify the complete chain afterwards.
7. Persist each row atomically. Resume only after validating the row against the
   exact task/prompt/input-ID/config/checkpoint/runner/manifest hashes; reject stale,
   duplicate, unexpected, or conflicting rows without overwrite.
8. Record prompt/input/generated-ID/text/config/checkpoint/tokenizer hashes, exact
   stop reason, token counts, timestamps, wall time, hardware, environment, git
   commit/dirty status, and row self-hash.
9. Compute no behavioural endpoint and inspect no arm-labelled effect in this
   runner. Output generation records only.
10. Do not touch Phase-2 outputs or regenerate any Phase-2 shared vanilla row.

## Return; do not run

Return:

1. runner path and SHA-256;
2. wrapper path and SHA-256;
3. executable manifest path plus file/internal SHA-256;
4. dependency-lock/container manifest path and SHA-256;
5. expected pod type, total wall time, expected cost, and proposed hard ceiling;
6. exact planned row count and any batching/resume facts;
7. full offline test command/count;
8. explicit statement that no model load, generation, pod, API, or paid call occurred;
9. any unresolved Phase-2 tokenizer-alias dependency.

Tony's prior authorizations do not cover this package. Wait for explicit approval of
the returned hashes and ceiling before execution.
