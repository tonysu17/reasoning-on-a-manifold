# Claude → Codex handoff (2026-08-09)

## Decisions recorded today (Tony, in chat)

1. **P1 joint scope seal APPROVED** ("we approve the P1 joint seal") — recorded
   in your draft's chat-record line, `RESULTS_LEDGER.md`, and session memory.
   Scope = approve as written (title, 3-RQ structure, Phase-2 + P5 prospective
   thesis evidence). **Resource checkboxes remain OPEN** — Phase-2 spend and the
   P5 powered/scoring ceilings still need Tony's separate signatures.
2. **Prereg AMENDMENT A4 sealed (annotation window):** Sonnet-only annotation
   cost re-costed against the corpus length distribution came to ≈$582
   full-chain (mean 5,052 tokens ⇒ ~19,300 chunked calls) — far above the
   Nova-priced §11 line. Tony directed cost down; A4 seals a **paragraph-aligned
   ~3,000-token annotation window** (uniform across arms/models; per-1k uses
   in-window denominators; damage/boxed/length/truncation stay full-chain;
   generation stays at the SEALED E8 8,192 cap, per your own correction).
   New annotation estimate **≈$275**. Window changeable only by dated
   correction before `stage_annotate` first runs.

## State you can rely on

- **Phase-2 executor is COMPLETE** (commits `f0f5998`..`0a47287`+ on
  `codex/phase0-support`): all 8 stages implemented; 49/49 pre-spend tests;
  repo suite green; MPS smoke on the pinned checkpoints passed (identity gate,
  clamp displacement, pair-state extraction). Your evidence-manifest field spec
  is implemented in `build_provenance` (compat keys byte-preserved;
  `rom-result-provenance-v1` companions; `amended = [A1,A2,A3,A4]`).
- **Shared-vanilla artifact location (your contract):** `stage_generate_battery`
  will emit `results/ph2/battery/base_vanilla_shared.json` +
  `base_vanilla_shared.sha256` — base-role vanilla rows on the sealed manifest,
  E8 settings (greedy, 8,192 cap, seed 20260808, base-tokenizer ids, no custom
  stop). Consume read-only; never regenerate overlapping rows.
- Your three delivered items (MDE pooling confirmation, shared-vanilla
  contract, field spec) are consumed; the E8-cap correction is reflected
  everywhere. Nothing further is owed to Phase-2 from your side.

## Asks / open items toward you

1. **Early stand-alone vanilla slice:** does your P5 powered-design freeze need
   the shared generic-vanilla artifact BEFORE the Phase-2 battery runs? If yes,
   say so — that triggers Tony's ~$3–5 early-slice decision; if no, P5 waits
   for the battery's export.
2. **Scorer v2.1 ceilings:** your request-count/cost ceiling ask goes to Tony
   directly (he now has it on his approval list). FYI: A4 windows Phase-2
   ANNOTATION only — your scorer is a separate measurement layer and its
   cost model is unaffected, but if you want a symmetric windowing policy for
   the generic stratum, that is your protocol decision to seal on your side.
3. **Provenance spot-check:** when the first Phase-2 artifacts exist, verify
   the emitted sidecars satisfy your claim-ledger consumer (shape implemented
   from your spec but not yet exercised against a real refresh pass).

## SECTION 2 (same day, after reading your 12:02 drops) — the cap fork

Your `P5_PHASE2_REDUCED_TOKEN_CAP_CONSTRAINT_2026-08-09.md` and my Amendment A4
operationalize the SAME owner cost instruction two different ways:

- **My A4 (sealed):** the ANNOTATION measurement window drops to ~3,000 tokens;
  **generation stays at the sealed E8 8,192**. Solves the named problem (the
  Sonnet annotation bill, $582→$275) with zero estimand damage — boxed
  correctness, truncation, damage gates all stay full-chain.
- **Your constraint:** future GENERATION frozen in 2,048–4,096 — which would
  require a prospective amendment to the sealed E8 generation settings, re-cut
  your shared-vanilla contract, and (your own numbers) put 42–49% of generic
  rows at the cap, destroying boxed-correctness availability on those rows.

**Status: an owner decision fork, surfaced to Tony with a recommendation
(keep generation sealed at 8,192; savings via A4; P5 powered may apply a
DECLARED post-hoc recap of the shared vanilla rows for its own same-cap
comparisons — your recapping table shows this is computable analytically, no
regeneration, satisfying your all-arms-same-cap rule as an analysis rule).
I honor your pause: the Phase-2 battery does NOT launch until Tony resolves
this fork, in addition to the spend sign-off.** If Tony instead freezes a
generation cap, Phase-2 needs a further dated amendment (A5) before
`generate_battery` — my A4 window then still applies on top.

Your addendum's four asks, answered:
1. Shared-vanilla artifact path/hashes: §"State you can rely on" above; full
   accounting emitted at generation time per your field spec.
2. Auditable Sonnet pricing: no `usage.cost` observations exist yet (manifest
   generation didn't persist them). `src/annotation._proxy_call` now logs
   `usage.cost` + `remaining_quota` per call — annotation runs will produce the
   per-request trail; the two-call calibration you have approved-in-principle
   remains the faster route to a bound.
3. Launch + injection-recovery-gate notices: will be posted to this channel
   when they genuinely occur.
4. Confirmed: your three deliverables + the E8 8,192 correction are
   incorporated (field spec implemented in `build_provenance`; pooling
   confirmation consumed by `stage_analyse`; contract honored by the battery's
   vanilla export).

Also consumed: `P5_V2_1_OWNER_DECISION_2026-08-09.md` (Tony approved your v2.1
protocol + both gates in principle; hash-bound ceilings still owed to Tony).

## SECTION 3 (same day, later) — CAP FORK RESOLVED: OPTION A (Tony in chat)

Tony chose Option A and approved the Phase-2 spend envelope. Closing your
`P5_PHASE2_REDUCED_TOKEN_CAP_CONSTRAINT_2026-08-09.md` freeze fields:

1. **Exact cap: 8,192 — unchanged, stays SEALED at the E8 setting. No
   amendment to generation settings; your pause on the one permitted Phase-2
   generation is lifted** (launch still waits on the pod + the §12 mechanics,
   not on the cap).
2. Generation budget at that cap: pod-side only, ~1–2 nights ≈ $10–20 (inside
   the approved $20–40 compute line). No API cost in generation.
3. Scoring/annotation-request budget after chunking: ≈9,100 Sonnet calls under
   the A4 3,000-token annotation window ≈ $230 central / $275 ceiling
   (verified rate card: a live probe billed exactly $3/M in + $15/M out —
   `usage.cost` present, `remaining_quota` NOT returned by the proxy anymore;
   your calibration should not rely on that field).
4. Cap-hit / terminal-correctness handling: unchanged full-chain rules
   (truncation an explicit endpoint; boxed correctness from full chains;
   missing terminal evidence unresolved).
5. Manifest/config: `phase2_task_manifest.json` ids_sha256
   `c7fefd59557f95a35f621787c2ab2c19f149ff96847e504c8295ee0f182c96d6`,
   E8-sealed generation settings per the prereg §4.

Consequence for P5: the shared-vanilla artifact will be generated ONCE at
8,192 per the standing contract. For your own same-cap comparisons, apply a
DECLARED analytic recap of the consumed rows (your recapping table already
computes this) — no regeneration, no arm-specific caps. Your powered-design
generation cap for P5's OWN arms remains yours to freeze with Tony.

## Operational notes

- P5 pilot: left LIVE and untouched (your workstream; run files intentionally
  uncommitted while churning; your pause record's SHA is superseded by the
  resumed run — reconcile when you checkpoint completion).
- Proxy etiquette unchanged: 29-s ceiling, ≤2 concurrent, shared with Phase-2
  annotation when it launches — coordinate scheduling to avoid contention
  during the 2–3-day annotation run.
