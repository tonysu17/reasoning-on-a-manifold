# gpt-oss H1 pilot — execution plan (PREP ONLY; no spend without Tony's explicit go)

**Drafted:** 2026-07-13 (HANDOFF_ANNOTATOR_SWAP.md Part 2). **Approved tier:** reliability pilot
only (~$30–60 API), pending final go. Design sources: thesis `safety.tex` H1 operationalisation
(capability control mandatory; H1 gates H2–H4) + `../safety_reasoning_extension.md` §14 (F1–F15,
esp. F1 CoT-span extraction, F3 capability control, F4 κ-gated labels, F6 effort confound) and
§14.3 run order (Gate B/C before GPU-heavy work; MPU = κ pilot + H1 + caveated fingerprint).

**Branch state (verified 2026-07-13):** `safety/gpt-oss-extraction` checked out at
`../rom-safety-worktree` (HEAD 05a75e3); **suite 205/205 PASS, 0 skips, 94 s** — no drift or
breakage. (Memory's "303 green" figure from 2026-06-13 does not match today's 205 collected;
likely counted a different snapshot — flagged, not a failure.)

## Pilot scope (the F4 Gate-B κ pilot, with the minimal GPU prelude it needs)

**P0 — chain generation (GPU, pod).** ~100 gpt-oss-20b chains: ~40 harmful (refusal-expected),
~40 matched benign (XSTest-style contrast pairs, F13), ~20 capability-control (hard non-safety
reasoning, F3's difficulty anchor). ONE effort level (medium) — the effort sweep is F6's later,
length-controlled job. Harmony analysis channel captured (the CoT object, F1); MXFP4 runs on
24 GB but 48 GB is the safe tier (A6000/A40). Prompts drawn from the branch's built prompt sets.
- Cost: A6000 ~$0.50/h × 2–4 h ≈ **$2–4** (+ the standard push/pull ops from this repo's playbook;
  truthful-sentinel + isolated-volume lessons apply).

**P1 — DSR 4-label annotation + κ gate (API via lab Bedrock proxy; the approved spend).**
~300 analysis-channel sentences stratified over refuse / safe-complete / comply chains (F4);
**3 LLM annotators** (Sonnet + Qwen3-235B + Nova-Pro — the R2.2 trio, proxy-available) + a
**≥100-sentence 2-human gold anchor** (who is human #2 — Tony decision). Report per-label κ AND
span-F1@IoU≥0.5 (both, per R2.1 precedent: κ 0.35–0.44 on the 6-label scheme).
- Cost: ~300 sentences × ~0.5–1k context tokens × 3 annotators ≈ 0.5–1.5M tokens ≈ **$10–35**
  proxy-dependent; budget envelope **$30–60** covers re-runs of mushy labels.

**Sealed gates (F4, fixed now):** per label — κ < 0.4 ⇒ that label's geometry is uninterpretable
(negative-only reporting); 0.4–0.6 ⇒ geometry must replicate across all three annotators'
labelings; ≥ 0.6 ⇒ citable. Expected: `decision` crisp; `spec_citation` moderate;
`adjudication`/`harm_recognition` mushy. **If `decision` itself fails κ ≥ 0.4, the DSR schema is
revised before any H1 spend** — that is the pilot's kill criterion and its entire point.

## What the pilot is NOT
No H1 geometry, no fingerprint, no forgery, no effort sweep, no Qwen ladder. Those wait on this
gate (and on Gate C's capability control, which runs CPU-side on existing 1.5B activations and
can proceed in parallel any time).

## Cost table per tier

| tier | what | compute | API | status |
|---|---|--:|--:|---|
| P0 | 100-chain gpt-oss-20b generation (1 effort level) | $2–4 pod | — | pending go |
| P1 | κ pilot: 300 sentences × 3 annotators + human anchor | — | $10–35 (envelope $30–60) | pending go (approved tier) |
| P2 | F1 CoT-span extraction + H1 under capability control | ~$5–15 pod (20B fwd passes) | — | NOT approved; gated on P1 |
| P3 | effort sweep (F6 controls), fingerprint, S3 forgery battery | pod + API | — | NOT approved; gated on H1 |

**Minimal publishable unit** (per §14.3): κ pilot + H1 yes/no under difficulty matching + caveated
output fingerprint — a null H1 is the honest MPU headline.

## Run order when the go lands
1. P0 generation (pod, this repo's ops playbook; ~half a day inc. verification).
2. P1 annotation via proxy + κ/span-F1 report → gate verdicts per label.
3. STOP and report to Tony — P2 is a separate decision with the gate table in hand.

## Consents recorded 2026-07-13 (Tony)
- **P1 spend:** HOLD FOR P0 REVIEW — P0 runs autonomously + self-verifies, then STOP and
  show Tony the chains + verification before ANY API spend. P1 does not auto-fire.
- **Human gold anchor:** TONY IS THE ANCHOR — prepare a ≥100-sentence gold-annotation file
  for Tony to label; P1's human-anchored κ waits on his labelling (LLM-only κ can preview).
- **Thesis integration (pt12/R3 verdicts):** AUTO-FOLD clean verdicts into thesis prose +
  rebuild + coherence check, show Tony the diff post-hoc (ledger already autonomous).
- Stimuli set: StrongREJECT (harmful) + XSTest (benign) → `data/gptoss_stimuli.json` (built,
  gitignored). P0 staged to chain onto Pod 3's post-RL grace window.
