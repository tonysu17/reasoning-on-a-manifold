# Safety-spillover pilot: pre-registration hardening (2026-07-03, ~01:00)

> Written BEFORE any result was read, from a two-agent pass (adversarial
> methodology critique + STAR-1 recipe verification). The pilot's first launch
> was killed and re-launched because of finding C1 below. Analysis of
> `results/safety_posttrain/spillover_star1_pilot100.json` MUST honour the
> gates in §2 and the wording in §3.

## 0. The experiment
Rung-0 of `post_training_spillover_extension.md`: teacher-force the SAME first-100
annotated generic-reasoning chains through `R1-1.5B` and `UCSC-VLAA/STAR1-R1-Distill-1.5B`,
mean-pool per annotated span, diff per-behaviour residual-stream geometry
(`pt03_measure_spillover.py` → principal angles, Δd_eff, centroid/direction drift,
selectivity ranking).

## 1. CAUGHT AND FIXED BEFORE READING: the extra-BOS confound (C1)
STAR1's tokenizer post-processor prepends `<|begin_of_sentence|>` (151646) where the
base tokenizer does not; on the stored chains (whose text already carries a baked BOS)
this yields a **doubled BOS** and a +1 token shift for every STAR1 sequence — a global,
safety-unrelated activation shift that would have inflated every diff. Verified
first-hand (base `[151646, 151644, …]` vs STAR1 `[151646, 151646, 151644, …]`).
**Fix:** `04_extract_activations.py --tokenizer-alias 1.5b` forces byte-identical
`input_ids` through both checkpoints. First-attempt outputs deleted. Any future
cross-checkpoint extraction (7B pair, LoRA checkpoints, gpt-oss pair) must pass the
same input_ids-parity check before its diff is trusted. NOTE: `pt03` mode (b)
(internal extraction) still tokenizes per-model — use mode (a) with 04-extracted
dirs, or port the fix, before ever using mode (b).

## 2. Gates the analysis must apply (from the adversarial review, ranked)
1. **Input parity check first**: identical `row_index.json` row order + sequence-length
   parity between the two dirs, else stop.
2. **Angle floor (C2)**: report NO principal angle without a null floor.
   **AMENDED 2026-07-04, before any real data was read:** the originally preferred
   pooled label-permutation null FAILED synthetic known-answer validation — under a
   genuine subspace change the pooled distribution's top-k eigengap vanishes, the
   null inflates, and a true 45° single-axis rotation went undetected (observed
   12.3° vs pooled null 12.5°). The **within-model disjoint-subsample null** (angles
   between two disjoint size-m draws from the SAME model) is now the primary floor
   (`src/safety_posttrain/nulls.py::within_model_null`; 6 known-answer tests in
   `tests/test_spillover_nulls.py`); the pooled permutation is retained only as a
   secondary exchangeability diagnostic. Report observed matched-n angle, the
   within-model null mean/sd, **excess**, and the smoothed `p_within`. The paired
   displacement coherence uses a **sign-flip null** (the shuffled-pairing null was
   also found miscalibrated: it changes the norm denominator systematically).
3. **Matched-n (C3)**: per-behaviour n spans ~7× in the first-100 (deduction 2,355 …
   adding-knowledge 346); subsample all behaviours to common n before angles/d_eff;
   rank selectivity by excess-over-null at matched n, never raw angle.
4. **Surprisal control (C4)**: teacher-forcing shift is expected and uniform; correlate
   per-span drift with STAR1 per-span NLL — if the behaviour ranking tracks surprisal,
   the "selectivity" is distribution shift, not safety.
5. **Paired displacement (upgrade)**: with identical input_ids the rows pair exactly;
   per-row (post − base) displacement with a paired-vs-permuted null is more sensitive
   than two in-sample subspaces.
6. **Pre-committed layers (C9)**: primary read at the config peak layers {12, 16};
   full-depth profile may be SHOWN but no argmax-layer headline.
7. Disclose: first-100 = 97% mathematical-logic (C6); 55/100 chains truncated at cap
   (C7); duplicate span text 17–35% per behaviour (C8, report unique-span n);
   single annotator (de-circularise via the qwen3/nova annotation files on disk).

## 3. Claim wording (pilot, before the control arm exists)
Permitted, IF excess-over-null is positive: "a measurable difference in per-behaviour
residual geometry between R1-1.5B and its STAR-1 safety-SFT checkpoint, exceeding the
finite-sample floor at matched n; descriptive; 100 first-N, ~97%-math, single-seed,
single-annotator chains; **no non-safety control**."
FORBIDDEN until the Rung-1 matched control is diffed identically: any causal or
attributive phrasing ("safety training causes/reshapes/spills into"), "selective
safety-entanglement", PH1-as-established, the PH2 ranking as a result. A null reads
"no difference beyond noise on this pilot", never "safety PT is geometrically inert".

## 4. Recipe facts for the write-up (verified 2026-07-03)
- STAR1-R1-Distill-1.5B = **full-parameter SFT** (ZeRO-3, bf16, 5 epochs, batch 128,
  LR 1e-5 cosine + 5% warmup, AdamW 0.9/0.95, wd 1e-4, seq 8192, loss on CoT+answer
  only) on exactly the 1,000-row `UCSC-VLAA/STAR-1` set. Our pt02 LoRA arm is therefore
  an **approximation** of the recipe (declare LoRA-vs-full-FT explicitly).
- Reported 1.5B effects (paper v2 table): safety avg 35.8→90.3 (+54.5); reasoning avg
  44.2→43.0 (−1.2), sharpest AIME-2024 −6.7. This is the accuracy-side "safety tax"
  anchor our geometry story complements.
- Tokenizer/template byte-identical except the BOS post-processor issue above and
  cosmetic re-serialization; architecture identical (28L, 1536, qwen2).
- gpt-oss-safeguard-20b: exists, base = gpt-oss-20b, but it is **policy-conditioned
  classifier tuning, not refusal alignment** — frame the second pair honestly and do
  not present it as a STAR-1 analogue; recipe not reproducible (no published data/HPs).

## 5. Status
- Pilot v2 (shared tokenizer) launched 2026-07-03 01:00, ~10 h.
- $0 Rung-1 data ready: `data/safety_star1_sft.json` (true recipe data),
  `data/control_generic_sft.json` (quantile length-matched own-chains control).
- Bigger runs: see `RUNPOD_SAFETY_PROPOSAL.md` (Run A ≈ $1, Run B ≈ $3–5, non-builder
  band ≈ $100–150 trimmed) — all gated on Tony.
