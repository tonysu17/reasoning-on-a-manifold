# Session handoff — 2026-08-16 (J-space Phase-1 failure investigation → diagnostic prep)

**Context.** J-space Phase-1 (run `jspace-p1-20260810T224338Z-996077031021`, validated A5) stopped at the
external positive-control gate: only typo qualified (assoc 0/4 criteria, multihop 2/4 — L17 pair failed).
Registered consequence stands: *the fitted lens did not pass the prespecified validity gate.* Phase 2
prospective/unrun. This session diagnosed the failure (capability-ordered pattern: typo ≫ multihop-late ≫
association-absent), discovered the Neuronpedia hosted-lens ecosystem, and did engineering-only prep for a
sealed diagnostic protocol. **No scientific endpoint was touched tonight.**

## Established tonight (engineering, pre-seal admissible)

1. **Neuronpedia hosts 36 prefitted J-lenses** (`neuronpedia/jacobian-lens` on HF), incl. `qwen2.5-7b-it`,
   `qwen3-1.7b`, `gemma-3-1b`, `gpt-oss-20b`. Fit recipe is **identical to ours** (Salesforce/wikitext-103-raw-v1
   train, max_seq_len 128, BF16), early-stopped at 457–485 prompts. → If a hosted WikiText lens passes
   association on its own model, corpus recipe is exonerated; discrimination collapses to model/scale.
   Provenance + LFS sha256: `results/prereg/JSPACE_HOSTED_LENS_PROVENANCE_2026-08-16.json` (uncommitted).
2. **Local downloads** in `data/jlens_hosted/` (gitignored): qwen3-1.7b lens (226 MB), gemma-3-1b lens (66 MB),
   all 4 configs + convergence CSVs, `SHA256SUMS.txt`. 7B lens (694 MB) and gpt-oss lens (382 MB) NOT
   downloaded — pull directly on Spark/pod (LFS oids recorded).
3. **Lens format** confirmed: `torch.load` dict `{J: {layer: [d,d]}, n_prompts, source_layers, d_model}` —
   compatible with our scoring harness. Load with `weights_only=True`.
4. **Local env**: new `.venv-jlens/` (torch 2.13.0, MPS ✓, transformers 5.x). Mac = **M4, 16 GB** →
   1.5–2B models only; 7B work goes to DGX Spark. R1-1.5B cached locally, generates **11.9 tok/s** (bf16,
   greedy, MPS). D1 CoT arm ≈ ≤2.5 h local worst case, or ~30 min on Spark. Base model
   `Qwen/Qwen2.5-Math-1.5B` is also already cached (future base-vs-distill contrast).
5. **Blocker RESOLVED same evening**: `google/gemma-3-1b-pt` license accepted by Tony; HF login configured
   locally (`Tonysu172003`); weights pulled (2.04 GB, snapshot `fcf18a2a`); MPS load + lens dimension match
   verified (d_model 1152 ✓, source layers 0–24). D3 can run both cells (qwen3-1.7b ungated as before).

## TO-DO — 2026-08-17

**T0. Session ritual** — `git -C thesis status`; skim RESULTS_LEDGER.md / METHODOLOGY.md heads. (5 min)

**T1. Bookkeeping (owed regardless of path; ~45 min; then commit = Tony's call)**
   - RESULTS_LEDGER.md entry: stopped instrument-validation study; gate FAIL 1/3 (typo only);
     consequence verbatim; Phase 2 prospective/unrun; evidence status *current non-confirmatory*;
     hashes from validated A5 bundle; do-not-cite marker for any J-space number outside the appendix decision.
   - METHODOLOGY.md cross-ref: staged-gate design + L17-anchoring lesson (gate required suite validity at the
     steering site, which multihop's real-but-late signal could not satisfy).
   - Commit tonight's provenance JSON alongside.

**T2. Seal `results/prereg/JSPACE_R1_DIAGNOSTIC_PROTOCOL_2026-08-16.md` — DRAFTED 2026-08-16 evening,
   awaiting Tony's review/seal; execution blocked until seal commit exists.**
   Diagnostic-only, non-gating, cannot reclassify Phase 1; no thesis-claim license. Contents:
   - **D1 competence audit** (R1-1.5B, all eligible items from `jspace_r1_eval_eligibility_manifest.json`):
     two arms — free-CoT (greedy, max_new_tokens fixed, e.g. 384) vs forced-immediate (single forward /
     tiny budget). Scoring rule sealed BEFORE run (case-insensitive target-string match + tie rules; use the
     repo's own locator fields for multihop bridge probes — no freehand rephrasing). Competence thresholds
     declared in advance (e.g. suite "within competence" iff CoT accuracy ≥ 50%). Prediction ledger:
     externalization ⇒ multihop right-with-CoT/wrong-immediate; incapacity ⇒ wrong in both.
   - **D2 scorer positive control** (Spark): `Qwen/Qwen2.5-7B-Instruct` + hosted 7B lens through OUR
     pipeline (re-derive tokenizer eligibility for Qwen2.5-7B vocab). Pass = beats permutation p95 any-layer
     on ≥2 suites ⇒ scorer exonerated.
   - **D3 small-end ladder**: qwen3-1.7b (lens on disk; ungated) ± gemma-3-1b (license blocker).
     Question: does ANY ~1–2B model show association/multihop readout on the same suites?
   - **D4** prompt-specific vs averaged Jacobians (8 hash-selected multihop items, L17+L25). (~$1 or Spark)
   - **D5** FP32 vs BF16 unembedding, rank-25/26 margins (association gap too large for this — confirm minor).
   - **D6** provenance pinning — partially done tonight; finish inside protocol.
2b. Commit sealed protocol BEFORE any D-run starts (house rule).

**T3. Execute (order: D2/D3 on Spark while D1 runs local overnight if preferred)**
   - Verify Spark reachability first; pull 7B model + 7B lens there.
   - All runs ≈ $0 (local + Spark). Log per-run manifests under `results/jspace_r1_pilot/diagnostics/`.
   - **D1 COMPLETE 2026-08-16 23:16 local (52.7 min, MPS), run `d1-20260816T212205Z-ccdae99be0fd`:**
     multihop **within competence** (CoT 0.617) with **externalization signature TRUE**
     (immediate 0.099, CoT bridge-verbalization 0.716); association **outside competence**
     (CoT 0.061, immediate 0.000); typo descriptive 0.083 (meta-task caveat). Accounts (b) assoc
     + (c) multihop supported. Remaining tomorrow: D2 (Spark, gates D3 reading), D3, D4, D5.

**T4. Decision memo + fork (with Tony)** — failure-attribution table from D1–D5, then pick:
   - **A** rescue decomposition on 1.5B: CoT-domain lens (train-split chains only) + site-level
     linearization-fidelity gate at L17; claim language narrowed to "token-writeout cone". New prereg, <$10.
   - **B** scale/emergence story via full hosted ladder (± self-fit 7B R1 Phase-1, $5–20). Thesis paragraph,
     no decomposition (no causal handle above 1.5B).
   - **C** gpt-oss-20b DSR × hosted lens (safety chapter; note its lens idd 0.94 — check convergence CSV first).
   - **D** shelve: ledger + one thesis sentence (memo's safest default).

**T5. Thesis application decision (optional tomorrow)** — one-sentence main-text mention vs appendix
   subsection (drafted in `.codex/out/JSPACE_PHASE1_THESIS_INTEGRATION_MEMO_2026-08-11.md`); either requires
   evidence-snapshot freeze + claim-ledger entries first.

## Parking lot / placement
- **Ph2 arm-differential missingness sensitivity analysis is still REQUIRED** before citing any Ph2
  prevalence endpoint — bigger thesis blocker than anything J-space; don't let diagnostics displace it.
- Thesis reframe plan (2026-08-12, T0–T5, ~57 h) remains the authoritative schedule; J-space work is
  post-charter exploratory and should stay ≤ half a day unless a fork decision says otherwise.
- ~~Tony tonight: accept gemma license~~ DONE 2026-08-16 evening — D3 fully unblocked.

## Artefact index (tonight)
- `results/prereg/JSPACE_HOSTED_LENS_PROVENANCE_2026-08-16.json` (uncommitted)
- `data/jlens_hosted/` — 2 lens .pt + 4 configs + 4 convergence CSVs + SHA256SUMS.txt (gitignored)
- `.venv-jlens/` — torch 2.13.0 MPS env (ignored)
- Phase-1 evidence unchanged: validated A5 bundle + failure map (`.codex/out/jspace_phase1_posthoc_failure_map_2026-08-11/`)
