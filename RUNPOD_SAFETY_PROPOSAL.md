# Proposal: RunPod runs for the safety-section results (2026-07-03)

> Written overnight while the $0 local pilot runs. **Nothing here is launched** —
> each run needs your go + a pod. Costs assume an RTX 4090 (~$0.35–0.70/hr,
> as used for E8). Timings extrapolate from measured rates: Mac MPS ≈ 3 min/chain
> teacher-forced (8k chains ~200 s); CUDA prefill for the same is ~1–2 s/chain,
> so a full-corpus extraction ≈ 45–90 min/model on a 4090.

## Context
- **Running tonight ($0, local):** Rung-0 pilot — STAR1-R1-Distill-1.5B vs
  R1-1.5B, same first-100 annotated chains → `results/safety_posttrain/spillover_star1_pilot100.json`.
- **Ready ($0 data built):** `data/safety_star1_sft.json` (the real STAR-1 1K
  recipe data) + `data/control_generic_sft.json` (size-matched non-safety
  control from R1's own chains, quantile length-matched p50 262 vs 245 words).
- Full-corpus R1-1.5B activations already exist locally (`data/activations/R1-1.5B`),
  so Rung-0 at full scale needs only ONE new extraction.

## Run A — Rung-0 at full scale (thesis-grade observational result)
STAR-1 full-corpus extraction (986 chains, all 28 layers) + `pt03` diff against
the existing R1-1.5B activations, plus the within-model split-half angle null
(the gating null the methodology critique requires).
- Compute: ~1–1.5 GPU-h. **Cost ≈ $1.**
- Output: per-behaviour principal angles, ΔID, direction drift, selectivity
  ranking at full power → first real numbers for the safety chapter's spillover arm.

## Run B — Rung-1 dose-response + control (the de-confounded, near-causal arm)
`pt02` LoRA on R1-1.5B: safety doses {100, 300, 1000} from the true STAR-1 data
+ the size-matched non-safety control (4 merged checkpoints; training itself
~10 min total on CUDA), then full-corpus extraction of each + `pt03` diffs.
- Compute: ~4–6 GPU-h. **Cost ≈ $3–5.**
- Output: the pre-registered spillover claim proper — geometric trajectory vs
  dose, with "any-SFT" subtracted by the control. This is what the safety
  chapter's spillover paragraph actually promises.

## Run C — optional replications (defer until A+B read)
- 7B pair (R1-Distill-7B → STAR1-7B): same protocol, ~2× cost. ≈ $5.
- gpt-oss-20b → gpt-oss-safeguard-20b: H1 fingerprint arm. Needs a bigger pod
  (A100/H100, MoE 20B) **and the safety-label annotation schema (API $)** —
  blocked on an annotation budget decision, not on GPU.

## API-side decision (separate from RunPod)
The proxy budget refreshed on 2026-07-03. The **non-builder re-annotation band**
for the steering chapter (Qwen3-235B re-scoring the E1 steered chains) is the
one purchase that upgrades an existing thesis result (backtracking: preliminary
→ PASS-gated). Trimmed version (headline cells only: backtracking arms + their
floors + vanilla, ~500 chains): **≈ $100–150** based on the E8 cost per chain.
Full 12-cell band ≈ $250–350. Needs your cap either way.

## Suggested order
1. Read the pilot report (this morning) — if the pipeline is clean, Run A the
   same day ($1).
2. Run B once A's split-half null is in place ($3–5).
3. Decide the non-builder band cap (the steering chapter benefits regardless of
   how A/B turn out).
