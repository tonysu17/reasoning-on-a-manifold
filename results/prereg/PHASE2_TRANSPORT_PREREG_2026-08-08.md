# SEALED — Phase 2: causal transport of the grounded backtracking frame

**Status: SEALED 2026-08-08** on Tony's instruction ("seal the phase 2 prereg", this date).
Derived from `.codex/out/PHASE2_TRANSPORT_PREREG_DRAFT.md` (2026-08-02) with every
**[AWAITING]** item resolved below; the draft's question, claim boundary, controls, outcome
table, gates, and provenance payload are adopted verbatim unless restated here. Any deviation
after this date requires a dated AMENDMENT block appended to this file before the affected
stage runs.
**Authorises no spend by itself** — launch requires the §12 sign-offs.
**Red-team consequences incorporated** (`.codex/reviews/PHASE0_REDTEAM.md`): the Phase-0
first-cut manifold numbers (f\*=1.38 / n\*=366) are INVALID (wrong floor, unpooled) and appear
nowhere in this design; any sizing recomputation must use the authoritative pooling path
(`src/delta_floor.py`: `single_direction → energy_matched_random`,
`manifold_k5 → random_subspace_k5`) — the sizing inputs here are the T2 simulation
(`.codex/out/ph2_mde_sim.json`, seed 20260802) and the in-run injection-recovery gate.

## 1. Question and claim boundary — as draft §1

One grounded coordinate (base R1-1.5B backtracking DAS frame, L17, width 2). Verdict per
target checkpoint: **retained / rescaled / rotated / decoupled / disabled**, each bounded to
the tested model, layer, width, intervention family, battery, and achieved sensitivity. No
generalisation to other behaviours (E10.3), layers, or intervention classes.

## 2. Checkpoints — SEALED

| role | HF id | immutable snapshot (provenance table 2026-08-02) |
|---|---|---|
| base | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| target 1 (safety full-SFT) | UCSC-VLAA/STAR1-R1-Distill-1.5B | `f865d7ac5136370518986a5273f4d731d7a0f254` |
| target 2 (RLVR case study) | agentica-org/DeepScaleR-1.5B-Preview | `e3f524ce413a296b4d388e7560dd5c82c1c56725` |

No tool-use arm (Phase-4A gate + amendment required). Byte-identical input ids from the R1
tokenizer alias on every matched-input arm (C1; tokenizer non-identity is provenance-evidenced).
Layer hook, precision, and identity-reload controls as draft §2.

## 3. Units, splits, pairing — SEALED

- Independent unit = problem/task; pooling within problem before any resampling; cluster
  bootstrap by problem, B = 10,000, seed **20260808**, same resampled indices across base and
  target (paired design).
- **Evaluation battery: n = 100 problems**, drawn seeded (20260808) and stratified by task
  category from `data/tasks_final.json` EXCLUDING (a) the 986-chain annotation-corpus problems
  and (b) the E8 evaluation ids (`results/eval/R1-1.5B__E1/eval_task_ids.json`). The draw is
  executed once, hashed, and committed as `results/prereg/phase2_task_manifest.json` BEFORE any
  generation. Rationale for 100 over the simulator-sufficient 50: the paired sizing model is
  flagged optimistic (binomial sentence-noise assumption); f\*(100) ≈ 0.296 gives ~40% margin
  while staying one pod session.
- **Discovery set (re-fit / norm / whitening / gain estimation): the existing annotated-corpus
  spans teacher-forced through each target on byte-identical ids** (the spillover mechanism).
  Span labels transfer with the text; no new annotation is needed for re-fitting. Discovery and
  evaluation are disjoint by construction (annotation-corpus problems are excluded from the
  battery).
- Evaluation is also disjoint from the frame-builder data (the frame was built on
  annotation-corpus spans). E8-task reuse: none.

## 4. Frames, interventions, controls — draft §4 with these seals

- **Transport arms (three, predeclared):** raw coordinates; **scalar-norm-corrected** (per-model
  mean residual L2 ratio at L17, estimated on discovery spans only); **diagonally-whitened**
  (per-dimension std from discovery spans). No full-covariance whitening in this launch.
- **Gain correction** (for the *rescaled* verdict only): a single scalar on clamp strength,
  fitted on discovery spans only.
- **Target grounding-gate nulls: 20 sham frames (same builder/search budget) + 20
  random-orthogonal frames (same width/norm) per target.** Gate = transported frame exceeds the
  95th percentile of the pooled 40-frame target-null on BOTH coord-AUC and state-dependence.
  Base sealed absolute thresholds (0.65 / 2.0) are reported for continuity, not adjudication.
  Neighbouring-layer controls: L16 and L18, fixed now.
- **Re-fit:** E10 sealed recipe unchanged (layer L17 only; width ∈ {1, 2} per E10.2; original
  optimizer/steps/seed protocol; M5 — no target-specific tuning), pairs built from
  teacher-forced discovery spans. Alignment null = the random-orthogonal frame set.
- **Causal battery per model:** transported-raw(±), transported-norm(±), transported-whitened(±),
  re-fit(±), sham frame, random-orthogonal frame, count-matched floor, energy-matched floor,
  vanilla. Doses: the E10 P2 clamp protocol and the E8 sealed α (α\* = 1.0), one sample per
  task per arm, generation settings identical to the E8 sealed protocol (temperature, max
  tokens, seeds per its provenance record), run seed 20260808.
- **On-manifold floor: EXCLUDED** from this launch (not sealed in time; may enter only by
  amendment sealed before its generation).

## 5. Estimands — draft §5 with these seals

- **Primary:** paired change in held-out backtracking **sentence-fraction Δ_floor**
  (suppression-oriented, matched floor − active arm, pooled within task) of the transported-raw
  frame, base → target, vs sham and energy-matched controls. Attenuation
  f = 1 − Δ_target/Δ_base reported only with stable sign/denominator; raw effects + intervals
  always reported.
- **Secondary:** the corrected-dose arms, re-fit effect, alignment angle vs null, target gate
  statistics, per-1k estimand (variance-sensitivity only; cannot be promoted post hoc), damage
  and task-performance contrasts, norm/whitening gain diagnostics.
- **Annotator: Nova-Pro (non-builder) for all Phase-2 behavioural verdicts** (pt04c precedent);
  Sonnet permitted only as a diagnostic duplicate on ≤20% of rows. Correctness = boxed
  exact-match. M3/M5/M7 bind as house rules.

## 6. MDE + battery consequence — SEALED

- Sizing inputs: T2 simulation — paired f\*(100) ≈ 0.296 (sentence-fraction), 0.250 (per-1k);
  unpaired fallback n = 300.
- **In-run injection-recovery (mandatory, pre-outcome):** on the BASE model only, transported
  frame mixed with the energy-matched sham at attenuation **f ∈ {0.25, 0.5, 0.75}** × the full
  100-task battery (the f = 0 and f = 1 endpoints are the main battery's transported-raw and
  sham arms — no extra generation). Pass = ≥ 80% detection of f = 0.5 at one-sided α = .05
  under the executed annotation pipeline, evaluated BEFORE any target-model outcome is
  inspected.
- If the gate fails at n = 100: enlarge along 150 → 200 → 300 (each step = Tony spend
  approval). If cross-model pairing is unavailable: n = 300 fallback + rerun the gate. If no
  approved battery passes: **"retained" is removed from the vocabulary** — positive surviving
  effects report as "not disabled under the tested intervention (attenuation ≥ X% detectable)";
  unresolved nulls report as inconclusive, never "disabled".

## 7. Multiplicity — SEALED

Primary family = {STAR1, DeepScaleR} × primary attenuation test, Holm across the two cells.
Secondary families exactly as draft §7 (gates; re-fit recovery; damage; exploratory). No test
moves between families after p-values are seen.

## 8–10. Outcome table, gates, stop rules, provenance payload

Adopted verbatim from draft §8–§10, including: the five-outcome decision hierarchy
(decoupled precedes disabled whenever representational signal remains; inadequate sensitivity
⇒ downgraded vocabulary, never a five-way verdict), the six gates (provenance/alignment,
execution, damage, target-null, MDE, and the inherited twice-failed-stop rule), and the full
provenance payload. Damage-gate thresholds SEALED: active arm vs matched floors — chain-level
repetition (rep4 > 0.8) excess > 10 pp, truncation excess > 10 pp, or boxed-accuracy drop
> 15 pp vs vanilla ⇒ primary interpretation stops for that arm.

## 11. Cost envelope (for the §12 sign-off; not a seal)

≈ 12 cells × 100 tasks × 3 models + 300 injection-recovery generations ≈ **3,900–4,200
generations** ≈ one long pod session or two nights (~$20–40 on a 4090); annotation of the
battery by Nova-Pro ≈ **$100–250** (the dominant cost line, as flagged in Phase 0);
teacher-forced discovery extractions ≈ $5–10.

## 12. Launch preconditions (sealing ≠ launching)

1. **Phase-0 closure**: pod batch (s0–s3) landed and dispositioned — s3's full-sequence curve +
   SV family check appended to the Phase-0 freeze §4 (batch running at seal time, 2026-08-08).
2. Task manifest drawn, hashed, committed (§3).
3. Tony's explicit spend sign-off against §11.
4. Any post-seal design change = dated amendment above this line.

## AMENDMENT A1 (2026-08-08, sealed before the manifest stage ran)

**Discovery:** §3's draw rule is unsatisfiable as written. The annotation corpus covers ALL
1000 `data/tasks_final.json` ids (000–099 in each of the 10 categories), so exclusion (a)
empties the pool. E8 faced the same constraint and evaluated on beyond-pool ids (097–101 per
category, "last 5" of a since-truncated 102-per-category pool; note E8's 097–099 overlap the
corpus — a further reason not to inherit its set).

**Amended rule (intent preserved: disjoint, stratified, seeded, drawn once):** the evaluation
manifest = **100 FRESH tasks, 10 per category**, produced by the pool's own generator
(`src/task_gen.py::generate_tasks`, same category set, prompt schema, and difficulty mix,
Sonnet via the lab proxy), assigned ids from **_102 upward per category** — disjoint from the
corpus (000–099) and from E8 (097–101) by construction, with disjointness verified
programmatically against both exclusion lists. Generation parameters, seed 20260808, prompt
version, and the generator's model id are recorded in the manifest. Drawn once, hashed,
committed as `results/prereg/phase2_task_manifest.json` (tool: `ph2_manifest.py`, which
refuses to overwrite an existing manifest). Incremental cost ≈ $0.5 proxy — requires Tony's
go (proxy spend).

No other clause of this preregistration changes.

**A1 correction (2026-08-08, same day, before any draw — caught by the tool's guard test):**
the annotation-corpus ids reach **_116** in some categories (e.g., SPAT_116, PROB_115), so the
fixed "_102 upward" start would collide. Corrected rule: fresh ids start at **max excluded
suffix + 1, computed dynamically at draw time** (= _117 on current data) and recorded in the
manifest's `generator.id_start`. Disjointness remains programmatically verified against the
full exclusion set; nothing else changes.

## Seal record

Sealed 2026-08-08. Basis: unified plan Phase 2 + amendments A2/A3/A4/A6/A8/A10; codex draft
2026-08-02; red-team F1–F7 consequences; T2 simulation (seed 20260802). Sealed decisions made
by the session under Tony's "seal the phase 2 prereg" instruction; the §12 sign-offs remain
his. Files: this prereg; task manifest (pending, §3); simulation `.codex/out/ph2_mde_sim.json`.
