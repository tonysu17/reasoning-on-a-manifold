# Literature positioning — 2026-07-13 (3-agent web-verified review)

Closest work to the three live experimental fronts. All IDs verified against verbatim
abstracts (two agents caught + corrected PDF-fetch confabulations; treat second-hand
summaries of 2605.09292 claiming steering/temperature usage as wrong).

## 1. Post-training method signatures (pt12 plane: SFT translate / DPO off-axis / GRPO pending)

| rank | work | verdict |
|---|---|---|
| 1 | **2606.23740** Weight-Space Geometry of Offline Reasoning Training (21 Jun 2026!) — same-data/different-objective on Qwen3-4B; SFT-family deltas colinear (cos≥0.97), **DPO near-orthogonal + higher rank**, GRPO ~67% orthogonal | **DIRECT OVERLAP — biggest threat to the method-dependent-direction claim.** Weight space only; our surviving deltas: residual-stream translation, KL dose axis, token-entropy sign, activation PR, reasoning-distill model |
| 2 | **2512.11838** D-STEER: DPO as steering-vector perturbation; upper-layer spectral entropy **collapse** | DIRECT OVERLAP + tension — must reconcile explicitly (their upper-layer spectral entropy ≠ our output-token entropy/PR) |
| 3 | **2511.08567** Path Not Taken: RLVR learns off-principals, spectrum-preserving | biggest threat to "behavioural-without-representational compression" — near-consensus in weight space; ours survives as activation-PR-on-matched-text + checkpoint ladder |
| 4 | **2505.11711** RL finetunes small subnetworks (cite this, NOT withdrawn 2507.17107) | complementary |
| 5 | **2605.16600** pretraining writes / alignment reads asymmetry | complementary |
| 6 | **2603.22446** RLVR sparse token-level shifts | borrowable (token-side companion) |
| 7 | **2605.11775** entropy polarity in RFT | borrowable mechanism vocabulary |
| 8 | **2506.11618** convergent EM direction (protocol-INVARIANT) | useful foil to our method-DEPENDENT finding |

**Borrow (cheap, on our checkpoints):** principal-angle + linear-mode-connectivity barrier
(2606.23740), subnetwork-recovery test (2505.11711).

## 2. R3 strategy-entropy (creativity strand)

| rank | work | verdict |
|---|---|---|
| 1 | **2605.09292** Beyond Accuracy: strategy diversity in LLM math (counts, prompting, frontier models) | DIRECT OVERLAP on measurement; no temperature ladder, no steering, no Shannon entropy, no token-dissociation |
| 2 | **2511.08305** SPREAD: Riemannian activation steering for diverse reasoning | **biggest threat to the "steerable" half** — unsupervised diversity-optimized vectors vs our behaviourally-identified pump |
| 3 | **2601.22010** STARS (Stiefel steering for diverse paths) | sibling of SPREAD |
| 4 | **2510.26122** Reasoning Path Divergence | **borrow as second, continuous strategy-distance metric** |
| 5 | **2506.09659** Intent Factored Generation ("token-level diversity ≠ exploration") | validates premise, mild pre-emption |
| 6-8 | 2510.01171 Verbalized Sampling; 2504.05228 NoveltyBench; 2506.23601 SemDiD | baselines / protocol standards |

**Surviving whitespace:** the conjunction — thermostat-vs-pump dissociation with token
4-gram diversity held fixed (~0.96) + per-task Shannon strategy-entropy + small
RLVR-distilled model + mechanistic (backtracking) direction.

## 3. gpt-oss H1 fingerprint

**H1 is UNCLAIMED as of 2026-07-13** — nobody has published a safety-REASONING subspace on
gpt-oss-20b internals. Runway: weeks-to-months, not open-ended.

| rank | work | verdict |
|---|---|---|
| 1 | **2505.14185** Ponkshe: safety subspaces not linearly distinct (ICLR'26) | the NULL H1 must beat (weight-subspace, non-reasoning models) |
| 2 | **2603.05773** Knowing without Acting: disentangled recognition/execution safety axes | partial scoop risk — one gpt-oss extension away; **borrow double-difference extraction** |
| 3 | **2509.23882** Probing GPT-OSS-20B (behavioural jailbreak taxonomy) | no internals threat; borrow failure taxonomy |
| 4 | **2604.20945** Breaking Bad: steering audits incl. gpt-oss | complementary; runway pressure signal |
| 5 | **2510.16968** MoE expert-signature distillation fingerprints | partial H2 anticipation (routing ≠ object shape); cite as precedent |
| 6 | **2605.05329** Annotator Policy Models (interpretation-free concepts) | **borrow for DSR annotator-circularity mitigation** |
| 7 | **2507.03167** Yamaguchi (already cited) | complementary |
| 8 | **2604.09665** Deliberative alignment deep, uncertainty remains | complementary |

Cautions to design in: **2605.16938** (gpt-oss `reasoning_effort` = token ceiling, not a
dial — treat as control variable; internal movement across effort would itself be novel);
**2605.26772** (CoT rebuilds refusal against static steering — causal checks must run
through generation).

**Net priority:** move on P0/P1 promptly (H1 window narrowing); cite 2606.23740 +
2512.11838 + 2511.08567 in any pt12 write-up and frame deltas explicitly; add the RPD
metric to R3's analysis as classifier-independent validation.
