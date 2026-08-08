# Checkpoint provenance — 2026-08-02 (Phase 0 item 4)

Admission rule (freeze §1.6): a checkpoint enters Phase 1 only when its row is complete.

**Tokenizer warning:** tokenizer.json is NOT byte-identical across the family (R1 88145e3c... vs STAR1/DeepScaleR e20ddafc...; config bos_token_id: R1 151643 vs DeepScaleR 151646). The C1 rule therefore applies to EVERY cross-checkpoint extraction: byte-identical input_ids via --tokenizer-alias 1.5b. 30_r1_compression matched-ids arms additionally carry gate_tokenizer.json.

| checkpoint | role | tokenizer.json sha16 | arch | verified |
|---|---|---|---|---|
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | base (thesis primary) | `88145e3c3249adc2` | 1536h/28L/131072ctx/bos151643 | local hash 2026-08-02 |
| UCSC-VLAA/STAR1-R1-Distill-1.5B | safety SFT descendant (Rung-0 pair) | `e20ddafc659ba902` | 1536h/28L/131072ctx/bos151643 | local hash 2026-08-02 |
| Qwen/Qwen2.5-Math-1.5B | pre-distillation parent (registry: baseline) | `c0382117ea329cdf` | 1536h/28L/4096ctx/bos151643 — 4096 ctx = the matched-ids grid mismatch | local hash 2026-08-02 |
| agentica-org/DeepScaleR-1.5B-Preview | RLVR descendant (best provenance: code+data+W&B public) | `e20ddafc659ba902` | 1536h/28L/131072ctx/bos151646 | local hash 2026-08-02 |

## Remote-verified (HF page 2026-08-02; hash before Phase-1 use)

- **Nickyang/FastCuRL-1.5B-Preview|V2|V3** ← R1-Distill-1.5B — staged curriculum RL (only public staged sequence)
- **nvidia/Nemotron-Research-Reasoning-Qwen-1.5B (v1,v2 revisions)** ← R1-Distill-1.5B — ProRL, 3000 steps, ref resets
- **knoveleng/Open-RS1|RS2|RS3** ← R1-Distill-1.5B — GRPO variants, $42 budget
- **RUC-AIBOX/STILL-3-1.5B-preview** ← R1-Distill-1.5B — RL (2503.04548)
- **hbx/JustRL-DeepSeek-1.5B** ← R1-Distill-1.5B — single-stage GRPO
- **agentica-org/DeepCoder-1.5B-Preview** ← R1-Distill-1.5B — RL for code
- **nvidia/DLER-R1-1.5B-Research** ← R1-Distill-1.5B — RL length-penalty
- **Zyphra/ZR1-1.5B** ← R1-Distill-1.5B — RL math+code
- **l3lab/L1-Qwen-1.5B, L1-Qwen-1.5B-Max** ← agentica-org/DeepScaleR-1.5B-Preview — LCPO length-control RL — 2 RL generations deep
- **theshyustc/CoRT-Prompt-Hint-1.5B-RL, CoRT-Hint-Engineering-1.5B-RL** ← R1-Distill-1.5B — tool-integrated RL (2506.09820); CONFIG-VERIFIED 2026-08-02; owed: tokenizer-file hash, chat-template diff, weight-delta sanity
- **huihui-ai/DeepSeek-R1-Distill-Qwen-1.5B-abliterated** ← R1-Distill-1.5B — refusal-direction ablation (weight surgery, training-free)
- **stepenZEN/DeepSeek-R1-Distill-Qwen-1.5B-Abliterated-dpo** ← huihui abliterated — DPO on top of surgery — 2-step chain
- **UCSC-VLAA/STAR1-R1-Distill-7B|8B|14B|32B** ← respective R1-Distill bases — safety SFT scale family
- **openai/gpt-oss-20b (registry: safety_gpt_oss)** ← OpenAI pretrain+deliberative alignment — safety-chapter model
- **openai/gpt-oss-safeguard-20b** ← openai/gpt-oss-20b — official safety-reasoning FT 2025-10-29 — pt21 pair (decision 7 open)

Machine-readable: `configs/analysis/checkpoint_provenance.yaml`
