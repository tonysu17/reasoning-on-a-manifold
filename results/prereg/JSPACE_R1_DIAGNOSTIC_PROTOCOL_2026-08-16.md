# SEALED DIAGNOSTIC PROTOCOL — J-space Phase-1 failure attribution (D1–D6)

**Status:** DRAFT 2026-08-16, awaiting Tony's seal. Sealed upon Tony's written approval and the commit of
this file; **no diagnostic stage may start before the seal commit exists.** Any post-seal deviation
requires a dated amendment committed before the affected stage runs.

**Authority boundary.** Exploratory diagnostics only. Nothing below is confirmatory, nothing gates, and
nothing can reclassify Phase 1 (`jspace-p1-20260810T224338Z-996077031021`, validated
`jspace-p1v-20260810T231952Z-6ad3de6f3754`). The registered Phase-1 consequence stands verbatim: *the
fitted lens did not pass the prespecified validity gate.* Phase 2 remains prospective/unrun. No thesis
empirical claim is licensed by any result below; thesis use remains governed by the application gate in
`.codex/out/JSPACE_PHASE1_THESIS_INTEGRATION_MEMO_2026-08-11.md`. Outputs feed exactly one artefact: a
failure-attribution decision memo for the fork decision (rescue-on-1.5B / scale-story / gpt-oss spin-off /
shelve).

**Spend boundary.** Target $0: local M4 (MPS) and DGX Spark only. Any paid pod or API spend requires
Tony's separate sign-off. No LLM-judge scoring anywhere in this protocol — all endpoints are
deterministic string/rank computations (avoids introducing a new annotator-circularity surface).

## 1. Question

Phase 1 failed its external criterion-validity gate with a capability-ordered pattern (typo ≫
multihop-late ≫ association-absent). Which failure account does the evidence support:
**(a)** scorer/recipe defect; **(b)** behavioural incapacity of R1-1.5B on the eval items;
**(c)** capacity present only via externalized CoT (silent-pass deficit); **(d)** corpus-averaging loss;
**(e)** precision artefact; **(f)** generic ~1–2B-scale absence of verbalizable readout?
Each diagnostic targets one or two accounts. No diagnostic can rescue the lens.

## 2. Fixed inputs

| Role | Identity | Pin |
|---|---|---|
| R1 model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | rev `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`, BF16 |
| J-lens code | `anthropics/jacobian-lens` | commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` |
| Eval suites (native fields) | `data/evaluations/lens-eval-{association,typo,multihop}.json` at that commit | sha256 recorded in run manifest at fetch |
| Eligibility manifest | `results/prereg/jspace_r1_eval_eligibility_manifest.json` | sha256 `a193ce15ff18d1852a870703dba104763a01b06740c99ec8fd955e3a15927f7c` |
| Phase-1 validated report | watch-root `…-validated-a5/phase1_report.json` | sha256 `004145b129f289972e740ad23e20084eb0a9109ef0d89383645f3fc269fc35d9` |
| Phase-1 readout arrays | `external_readout_arrays.npz` (validated bundle) | sha256 `02ee47f173b034db744758ac915460d1bd7ed5c347bd4022ca97dd95760c6289` |
| Phase-1 merged lens | validated bundle lens artefact | hash per bundle `ARTIFACT_MANIFEST.json` (`4664395b…`) |
| Scoring code | `jspace_phase1_scoring.py` | sha256 `6f610e61dc13fb2adceb6d86151467cdf8c5cd4366522da39f9f3de1bc45d426` (byte-identical to Phase-1 fixed input) |
| Hosted-lens provenance | `results/prereg/JSPACE_HOSTED_LENS_PROVENANCE_2026-08-16.json` | sha256 `3f98318b0c8be9f4de33653a694b01a66fedc2c848aa323c40f407102b0e163c` |
| D2 model | `Qwen/Qwen2.5-7B-Instruct` | rev `a09a35458c702b33eeacc393d103063234e8bc28` |
| D2 lens | hosted `qwen2.5-7b-it` (fitted 485 prompts) | LFS sha256 oid per provenance JSON; verify on download |
| D3 model A | `Qwen/Qwen3-1.7B` | rev `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| D3 lens A | `data/jlens_hosted/qwen3-1.7b_jacobian_lens.pt` (466 prompts) | sha256 `6fcc79011bd921ffd87612255e2e99950a124fa519470ee44ebaf161c39be9d6` |
| D3 model B | `google/gemma-3-1b-pt` | rev `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` |
| D3 lens B | `data/jlens_hosted/gemma-3-1b_jacobian_lens.pt` (467 prompts) | sha256 `39f61074bfdb7c896b5027612a21e9648ca348d6516f0fbdbca6e5690c8b1ab1` |
| Repo state at seal | this repository | commit recorded in seal block below |

Item populations are the **eligible subsets** from the sealed eligibility manifest, keyed by item `name`:
association 98, multihop 81, typo 96. The repo eval files additionally contain suites
(`multilingual`, `order-ops`, `poetry`) that are **out of scope**.

## 3. D1 — behavioural competence audit (R1-1.5B) → accounts (b) vs (c)

**Rationale.** Phase-1 pass@25 is a joint test of representation existence and lens readability. D1
measures the model side alone, behaviourally, on the same items.

**Load and decoding.** BF16; greedy (`do_sample=False`); no repetition penalty; seed 20260816 recorded;
device MPS (local) or Spark CUDA, recorded. Arm-specific `max_new_tokens` below.

**Arms and prompts (wrappers frozen verbatim; `{prompt}` = repo `prompt` field, `{target}` = repo
`target`, `{label}` = eligible label / `intermediates` entry).**

- **MH-imm** (multihop, immediate): input = `{prompt}` raw, no chat template; 8 new tokens.
  Endpoint: `{target}`-match in the continuation. *Can it complete the composite silently?*
- **MH-cot** (multihop, CoT): chat template (`apply_chat_template`, `add_generation_prompt=True`), user
  message = `Complete this sentence with the correct final word or number: "{prompt}"`; 512 new tokens.
  Endpoints: (i) `{target}`-match anywhere in the full output including think text; (ii) bridge
  (`{label}`)-match anywhere — the **externalization signature**.
- **AS-imm** (association, immediate): input = `{prompt}` + `\nThe single word that best describes what is
  left unstated here is "` raw; 4 new tokens. Endpoint: `{label}`-match.
- **AS-cot** (association, CoT): chat template, user message = `Read this passage and name, in one word,
  the unstated concept it points to: "{prompt}"`; 512 new tokens. Endpoint: `{label}`-match anywhere.
- **TY-imm** (typo, descriptive only): input = `{prompt}` + `\nThe misspelled word in the sentence above,
  spelled correctly, is "` raw; 4 new tokens. Endpoint: `{label}`-match. No CoT arm.

**Scoring rule (deterministic, sealed).** Primary: case-insensitive regex `\b{string}\b` on the
whitespace-normalized generation (numeric targets as `\b{digits}\b`; multi-word strings matched whole).
Inflectional variants do **not** count (conservative; may undercount association competence — reported as
a stated bias). Secondary sensitivity count: case-insensitive substring without boundaries. Primary decides
all thresholds.

**Sealed thresholds (interpretive, non-gating).** Per suite on the CoT-arm primary endpoint:
*within competence* ≥ 0.50; *outside competence* ≤ 0.20; otherwise *indeterminate*.
**Externalization signature (multihop):** MH-cot target-rate ≥ 0.50 AND MH-imm target-rate ≤ 0.20; bridge
rate reported alongside. **Registered expectations:** association ≤ 0.20 in both arms (accounts b/f);
multihop shows the externalization signature (account c).

**Consequence map.** Suite *outside competence* ⇒ its Phase-1 miss cannot be attributed to the lens;
criterion validity is untestable on that suite for this model. Suite *within competence* with Phase-1 miss
⇒ genuine readout absence (silent-representation deficit or lens failure — D4 separates).

## 4. D2 — scorer positive control (Spark) → account (a)

Hosted `qwen2.5-7b-it` lens + `Qwen/Qwen2.5-7B-Instruct` (28 layers, sources 0–26 → target 27; verify
against the lens file's `source_layers` at load) through **our** Phase-1 scoring path
(`jspace_phase1_scoring.py`, unmodified) plus a thin model-generic adapter (tokenizer swap; committed
before the run). Eligibility re-derived under the Qwen2.5-7B-Instruct tokenizer by the Phase-1 contract:
scored form = one leading ASCII space + label, `add_special_tokens=False`, exactly one token, specials
excluded, head rows beyond the tokenizer vocabulary excluded. Raw prompts, readout at final prompt token,
BF16 model, lens applied in FP32.

**Endpoints:** per-suite any-layer union pass@25 and full layer profile; 1,000 within-suite label
permutations, seed 20260817. **No L17-analogue criterion — instrument-level only, by design.**

**Sealed pass rule:** scorer exonerated iff ≥ 2 of 3 suites have any-layer union > permutation p95 with
p ≤ 0.05. **Consequence:** pass ⇒ account (a) rejected; fail ⇒ scorer/recipe suspect, D3 interpretation
frozen and the memo says so; no code fix may be applied and re-scored without a dated amendment.

## 5. D3 — small-end scale ladder → account (f)

Cells: `Qwen3-1.7B` (raw prompts ⇒ no thinking mode) and `gemma-3-1b-pt` (lens sources 0–24 of 26
layers; verify at load), each with its hosted WikiText lens, scored exactly as D2 with per-model
eligibility re-derivation (gemma vocab 262,144 ⇒ eligible-N will differ; report per-cell eligible-N;
apply the Phase-1 minimum of 50 eligible items as a reporting flag, not a gate). Permutations n=1,000,
seed 20260817.

**Registered question:** does either ~1–2B model beat permutation p95 (any-layer union, p ≤ 0.05) on
association or multihop? **Interpretation (conditional on D2 pass):** neither cell ⇒ account (f)
strengthened — R1-1.5B's failure is the expected low anchor at this scale; either cell ⇒ (f) weakened —
an R1-specific deficit (math-base / distillation) gains weight. **Sealed caveat:** cross-tokenizer
eligibility differences and different fit corp sizes (466/467 vs our 100) limit exact comparability;
reported, not adjudicated.

## 6. D4 — prompt-specific vs corpus-averaged Jacobian (R1-1.5B) → account (d)

**Items:** the 8 eligible multihop items with lowest SHA-256(ASCII item name), ascending hex — fixed
before any computation. **Method:** per-item single-prompt lens via the pinned J-lens fit path
(`n_prompts=1`; BF16 model, FP32 accumulation, eager attention, released skip-16 rule; valid-position
count recorded — multihop prompts are short). **Endpoint:** bridge-label rank at the final prompt
position under (i) the single-prompt lens and (ii) the Phase-1 merged lens, at source layers 17 and 25,
Phase-1 rank domain (ranks > 25 censored to 26). Per-item table + medians.
**Registered directions:** averaging-destroys ⇒ single-prompt ranks materially better at L25 (and
possibly L17); never-existed ⇒ both poor everywhere. Descriptive only; no threshold.

## 7. D5 — precision margins (CPU) → account (e)

From the validated `external_readout_arrays.npz`, recompute all external readout ranks with FP32
unembedding and FP32 lens copy. **Endpoint:** per suite, count of (label, layer) hits crossing the 25/26
boundary in either direction, and any change to the any-layer union counts. **Registered expectation:**
association union changes by at most ±1 item (the observed gap is far too large for a boundary artefact).

## 8. D6 — provenance completion

Run manifests record: eval-file sha256 at fetch; downloaded D2 lens sha256 vs LFS oid; model snapshot
revisions actually resolved; environment fingerprints (python/torch/transformers, device, GPU/MPS);
repo commit at execution; wall time. Hosted-lens static provenance is already pinned
(`JSPACE_HOSTED_LENS_PROVENANCE_2026-08-16.json`, committed `e4bb882`).

## 9. Execution order and stop rules

1. **D2 first** (its failure freezes D3 interpretation). D1 may run concurrently (local overnight).
2. Then D3; D4/D5 in any order.
3. Crashes/OOM: diagnose, fix launch mechanics, rerun — engineering repairs are permitted and logged;
   **endpoint definitions, wrappers, thresholds and seeds may not change without a dated amendment.**
4. No result-contingent addition of suites, items, layers, or arms.
5. Everything lands in `results/jspace_r1_pilot/diagnostics/<Dk>/<run-uuid>/` with `manifest.json`,
   `report.json`, `REPORT.md`.

## 10. Decision memo and licensed wording

Final artefact: `results/jspace_r1_pilot/diagnostics/DECISION_MEMO.md` — the failure-attribution table
(accounts a–f × evidence) and a recommendation for the fork (A rescue-on-1.5B / B scale-story /
C gpt-oss spin-off / D shelve). The memo recommends; Tony decides.

**Licensed wording examples.** "Under the sealed rule, R1-1.5B's CoT-arm accuracy on association was
p̂ (n=98)"; "our scoring pipeline does/does not reproduce hosted-lens readout on Qwen2.5-7B-Instruct";
"neither 1–2B comparison model beat its permutation control on association".
**Forbidden:** any reclassification of Phase 1; "R1 has/lacks a global workspace"; cross-model
quantitative equivalence claims; any thesis-facing claim outside the integration-memo gate.

## 11. Seal block

- Drafted: 2026-08-16 (session), repo commit at draft `975a1029c788c220f0ceef2f373c3bfd0946efa4`.
- Tony's decision: ☑ **sealed as-is** (Tony, 2026-08-16, in-session: "sealed, commit it and start D1 overnight").
- Seal commit: the commit introducing this file IS the seal; execution may begin only after it exists.
