# Provenance completion — remote checkpoint metadata

**Date:** 2026-08-02  
**Scope:** tokenizer/config metadata only. No weights or adapters were downloaded.

Resolved **15 provenance rows** to **25 immutable checkpoint revisions**. The YAML now records a commit plus sha256_16 for each of `tokenizer.json`, `tokenizer_config.json`, and `config.json` when present, and an architecture summary derived from `config.json`.

## Metadata table

| checkpoint @ revision | commit | tokenizer sha16 | tokenizer-config sha16 | config sha16 | architecture |
|---|---|---|---|---|---|
| `Nickyang/FastCuRL-1.5B-Preview@main` | `44b07bdfea2b` | `e20ddafc659ba902` | `8c32f3b0afaf748b` | `36b484882a47c544` | 1536h/28L/131072ctx/bos151646 [qwen2/Qwen2ForCausalLM] |
| `Nickyang/FastCuRL-1.5B-V2@main` | `06f550128605` | `e20ddafc659ba902` | `8c32f3b0afaf748b` | `255f125f6c8af4d9` | 1536h/28L/131072ctx/bos151646 [qwen2/Qwen2ForCausalLM] |
| `Nickyang/FastCuRL-1.5B-V3@main` | `f765b72f534d` | `e20ddafc659ba902` | `8c32f3b0afaf748b` | `fc29c6312d1a89d9` | 1536h/28L/131072ctx/bos151646 [qwen2/Qwen2ForCausalLM] |
| `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B@v1` | `b89048893f95` | `e20ddafc659ba902` | `d9125f504eea1bb5` | `5b3c4e17b37a11c9` | 1536h/28L/131072ctx/bos151646 [qwen2/Qwen2ForCausalLM] |
| `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B@v2` | `a8e647c7a46f` | `e20ddafc659ba902` | `92a23085bfbd6862` | `f1b8eaf06009a58e` | 1536h/28L/131072ctx/bos151646 [qwen2/Qwen2ForCausalLM] |
| `knoveleng/Open-RS1@main` | `1f0d80f257de` | `a4256422650d141f` | `161b6e92d1d62443` | `1986788fc26f1ba8` | 1536h/28L/131072ctx/bos151643 [qwen2/Qwen2ForCausalLM] |
| `knoveleng/Open-RS2@main` | `5a080aa479de` | `a4256422650d141f` | `92a23085bfbd6862` | `1986788fc26f1ba8` | 1536h/28L/131072ctx/bos151643 [qwen2/Qwen2ForCausalLM] |
| `knoveleng/Open-RS3@main` | `b9ea78ba17ad` | `a4256422650d141f` | `92a23085bfbd6862` | `1986788fc26f1ba8` | 1536h/28L/131072ctx/bos151643 [qwen2/Qwen2ForCausalLM] |
| `RUC-AIBOX/STILL-3-1.5B-preview@main` | `88d330baa0f8` | `e20ddafc659ba902` | `7552ff5b82b05790` | `f510408a0b1bd78d` | 1536h/28L/131072ctx/bos151643 [qwen2/Qwen2ForCausalLM] |
| `hbx/JustRL-DeepSeek-1.5B@main` | `0637e4096c78` | `88145e3c3249adc2` | `8ac8c85fb242563c` | `f62729adb55c028e` | 1536h/28L/131072ctx/bos151643 [qwen2/Qwen2ForCausalLM] |
| `agentica-org/DeepCoder-1.5B-Preview@main` | `103033da54f1` | `e20ddafc659ba902` | `79e0359ba5ec6c33` | `b06206e84c910bb6` | 1536h/28L/131072ctx/bos151646 [qwen2/Qwen2ForCausalLM] |
| `nvidia/DLER-R1-1.5B-Research@main` | `d0523143ce16` | `e20ddafc659ba902` | `92a23085bfbd6862` | `d2c9d01a6785866e` | 1536h/28L/131072ctx/bos151646 [qwen2/Qwen2ForCausalLM] |
| `Zyphra/ZR1-1.5B@main` | `6108a26d0cd7` | `e20ddafc659ba902` | `79e0359ba5ec6c33` | `337cb800f14451b6` | 1536h/28L/131072ctx/bos151646 [qwen2/Qwen2ForCausalLM] |
| `l3lab/L1-Qwen-1.5B-Exact@main` | `b1fa57f192f0` | `e20ddafc659ba902` | `b869a935677e8f9d` | `82a91a93aaab1dcd` | 1536h/28L/131072ctx/bos151646 [qwen2/Qwen2ForCausalLM] |
| `l3lab/L1-Qwen-1.5B-Max@main` | `8d5eff2725e7` | `e20ddafc659ba902` | `b869a935677e8f9d` | `283188fa6d2c2471` | 1536h/28L/131072ctx/bos151646 [qwen2/Qwen2ForCausalLM] |
| `theshyustc/CoRT-Prompt-Hint-1.5B-RL@main` | `fb0bc7ce122a` | `e20ddafc659ba902` | `d5305a85802c9426` | `0f1713873c4ec526` | 1536h/28L/131072ctx/bos151646 [qwen2/Qwen2ForCausalLM] |
| `theshyustc/CoRT-Hint-Engineering-1.5B-RL@main` | `c764ddc8f553` | `e20ddafc659ba902` | `90c217294fc32c55` | `82d60d6f167d8e93` | 1536h/28L/131072ctx/bos151646 [qwen2/Qwen2ForCausalLM] |
| `huihui-ai/DeepSeek-R1-Distill-Qwen-1.5B-abliterated@main` | `a0f34fe37d34` | `e20ddafc659ba902` | `a8dade878894b7aa` | `be0ccfa8d3e5eeeb` | 1536h/28L/131072ctx/bos151643 [qwen2/Qwen2ForCausalLM] |
| `stepenZEN/DeepSeek-R1-Distill-Qwen-1.5B-Abliterated-dpo@main` | `facdb70bfe5b` | `a4256422650d141f` | `4018a43a2cf98792` | `422f9cdafeeaf232` | 1536h/28L/131072ctx/bos151643 [qwen2/Qwen2ForCausalLM] |
| `UCSC-VLAA/STAR1-R1-Distill-7B@main` | `34ed6c6b813b` | `e20ddafc659ba902` | `d9125f504eea1bb5` | `841664ed9d282c04` | 3584h/28L/131072ctx/bos151643 [qwen2/Qwen2ForCausalLM] |
| `UCSC-VLAA/STAR1-R1-Distill-8B@main` | `a17db837abbf` | `d91915040cfac999` | `d015c5337f32d7bb` | `61c111dfd62ca117` | 4096h/32L/131072ctx/bos128000 [llama/LlamaForCausalLM] |
| `UCSC-VLAA/STAR1-R1-Distill-14B@main` | `160bf2c5252f` | `e20ddafc659ba902` | `d9125f504eea1bb5` | `bf697f2df9c01fee` | 5120h/48L/131072ctx/bos151643 [qwen2/Qwen2ForCausalLM] |
| `UCSC-VLAA/STAR1-R1-Distill-32B@main` | `e0a3bb481c1d` | `e20ddafc659ba902` | `92a23085bfbd6862` | `8a9eed6e48724191` | 5120h/64L/131072ctx/bos151643 [qwen2/Qwen2ForCausalLM] |
| `openai/gpt-oss-20b@main` | `6cee5e81ee83` | `0614fe83cadab421` | `9279e942392b742d` | `3a2a26ded679375b` | 2880h/24L/131072ctx/bos? [gpt_oss/GptOssForCausalLM] |
| `openai/gpt-oss-safeguard-20b@main` | `8a11e17b25c9` | `0614fe83cadab421` | `9279e942392b742d` | `8fc6c94451a1a97b` | 2880h/24L/131072ctx/bos? [gpt_oss/GptOssForCausalLM] |

## CoRT deep-check

Reference R1 tokenizer hashes from the core provenance row: `tokenizer.json=88145e3c3249adc2`; STAR1/DeepScaleR alternate family hash `e20ddafc659ba902`.

### `theshyustc/CoRT-Prompt-Hint-1.5B-RL`

- tokenizer.json `e20ddafc659ba902`: **STAR1/DeepScaleR-exact**.
- chat template identical to R1 metadata: **False**.
  R1 template sha16 `56a1447ad31926fd`; candidate `54d400beedcd17f4`. Diff preview:

```diff
--- R1
+++ candidate
@@ -1 +1 @@
-{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = fal... [truncated]
+{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = fal... [truncated]
```
- selected config differences from R1:

  - `bos_token_id`: R1 `151643`; candidate `151646`
  - `sliding_window`: R1 `4096`; candidate `None`

### `theshyustc/CoRT-Hint-Engineering-1.5B-RL`

- tokenizer.json `e20ddafc659ba902`: **STAR1/DeepScaleR-exact**.
- chat template identical to R1 metadata: **False**.
  R1 template sha16 `56a1447ad31926fd`; candidate `54d400beedcd17f4`. Diff preview:

```diff
--- R1
+++ candidate
@@ -1 +1 @@
-{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = fal... [truncated]
+{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = fal... [truncated]
```
- selected config differences from R1:

  - `bos_token_id`: R1 `151643`; candidate `151646`
  - `sliding_window`: R1 `4096`; candidate `None`

## Weight-delta sanity-check specification — unrun

This section is a specification only; the metadata pass downloaded no weights.

1. Resolve R1 and each CoRT checkpoint to the immutable commits recorded above; verify identical tensor-key sets, shapes, and dtypes before subtraction.
2. Stream one safetensors shard at a time on CPU. For every floating tensor compute `||theta_CoRT-theta_R1||_F`, `||theta_R1||_F`, relative norm, maximum absolute delta, and nonzero fraction; hash the ordered tensor-key/shape manifest.
3. Aggregate squared norms globally and by layer/module (attention, MLP, embeddings, normalisation, LM head). Verify finite deltas and reject a purported descendant if keys/shapes differ outside a predeclared tied-weight exception.
4. Compare the two CoRT delta vectors by streaming dot product/cosine, and compare their norms with an identity reload (zero) and a same-lineage post-training reference. Do not infer lineage from a small norm alone.
5. Record source commits, per-file sha256, code commit/dirty state, accumulation dtype, and complete/partial status. Do not promote CoRT into direct-coordinate causal transport until this check and the byte-identical-input-id gate pass.

## Caveats

- A metadata hash establishes the exact small files inspected, not declared parentage or weight lineage.
- The combined YAML labels are planning aliases. The L1 row's unqualified checkpoint was resolved to the public `l3lab/L1-Qwen-1.5B-Exact`; this resolution remains explicit rather than silently changing the original row label.
- Missing metadata files remain `null`/`MISSING`; they are not synthesized from a parent checkpoint.
- The pre-existing file is named `.yaml` but is not currently accepted by PyYAML: some unquoted scalar values before `remote_verified` contain a colon. This pass was only authorised to append provenance fields, so it preserves that text and discloses the syntax debt rather than silently rewriting existing rows.
