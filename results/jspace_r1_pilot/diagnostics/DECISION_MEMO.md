# J-space failure-attribution decision memo

**Date:** 2026-08-17. **Status:** exploratory diagnostics, non-gating; nothing here reclassifies
Phase 1 (registered outcome stands verbatim: *the fitted lens did not pass the prespecified validity
gate*; Phase 2 prospective/unrun). Sealed chain: protocol `752dee7`; amendments 1–3; runs
D1 `d1-20260816T212205Z-ccdae99be0fd` (local M4), D2 `d2-qwen2.5-7b-it-f1b6cdfdff03`,
D3 `d3-qwen3-1.7b-3bd7b4c16291` / `d3-gemma-3-1b-5bf6d1b42fd0`, D4 `d4-195b5eb7f2d2`,
D5 `d5-c480b291f5aa` (RunPod RTX 4090, pod 213.173.111.15; first attempt of D2/D3 failed on a volume
disk quota — stubs discarded, tracebacks preserved in `pod_logs/`; rerun clean).

## Failure-attribution table

| Account | Diagnostic | Verdict | Evidence |
|---|---|---|---|
| (a) scorer/recipe defect | D2 | **REJECTED** | Hosted 7B lens through our unmodified scorer qualifies **3/3 suites** (assoc 0.0612, typo 0.4583, multihop 0.6173; all p=0.001) |
| (b) association outside model competence | D1 | **CONFIRMED** | R1-1.5B association: 0.000 immediate, **0.061 with free CoT** (n=98) — no latent to read out |
| (c) multihop capacity only via externalized CoT | D1 (+D3) | **CONFIRMED** | CoT 0.617 vs silent 0.099; bridge verbalized in CoT on **0.716** of items; non-distilled 1–2B models show the silent readout R1 lacks |
| (d) corpus-averaging destroyed local signal | D4 | **REJECTED** | The **exact** position-local Jacobian is equally blind: L17 = 0/81 in top-25 for local, skip16, and merged alike (81/81 censored ties); at L25 merged (11/81) slightly *beats* local (6/81). Nothing existed for averaging to destroy (A3's pre-registered asymmetry) |
| (e) BF16/FP32 precision artefact | D5 | **REJECTED** | Association any-hit items 4→3 under FP32 readout (within registered ±1); typo boundary churn (110 gained/57 lost of 1414 hits) is margin wiggle with no directional effect |
| (f) generic ~1–2B-scale absence | D3 | **REJECTED for multihop; mostly rejected for association** | qwen3-1.7b qualifies on **all three** suites (multihop 0.5432; assoc 0.0306 p=0.018; typo 0.5938); gemma-3-1b qualifies 2/3 (multihop 0.3704; assoc exactly 0.0000) |

## All cells, any-layer union pass@25 (instrument-level)

| Model | Lens | assoc | typo | multihop | Qualifying |
|---|---|---:|---:|---:|---|
| R1-Distill-1.5B (Phase 1) | ours, 100 WikiText | 0.041ⁿˢ | 0.896 | 0.185* | 1/3 → gate FAIL |
| qwen3-1.7b | hosted, 466 WikiText | 0.031 | 0.594 | 0.543 | 3/3 |
| gemma-3-1b-pt | hosted, 467 WikiText | 0.000ⁿˢ | 0.542 | 0.370 | 2/3 |
| Qwen2.5-7B-Instruct | hosted, 485 WikiText | 0.061 | 0.458 | 0.617 | 3/3 |

ⁿˢ = fails permutation. *Phase-1 multihop passed all-layer but the registered gate also required L17,
where R1 = 0/81. Same corpus recipe across every row (Salesforce/wikitext, seq 128, BF16). Cross-model
caveats sealed in §5: different tokenizers/eligible-N; hosted fits are ~4.7× our prompt count (D2's
qualification exonerates the scorer under its own lens regardless).

## Synthesis — why R1-1.5B failed

The gate probed **silent semantic intermediates at L17**. R1-1.5B has none to probe: association is
outside its behavioural competence entirely (b), and multihop composition exists but only through the
generated token channel (c) — the signature its distillation recipe trains for. Scorer (a), precision
(e), and corpus-averaging (d) are all formally exonerated, and D3 shows same-scale non-distilled models
qualifying under identical corpus recipes — so the deficit is **model/recipe-specific, not
scale-generic**. The one genuinely scale-flavoured residue: association readout is absent at 1B
(gemma 0.000), marginal at 1.7B (0.031) and still weak at 7B (0.061).

## Fork implications

- **A — rescue decomposition on 1.5B.** D4 fired A3's pre-registered consequence: the exact local map
  ranks bridges no better than the average, which **removes the mechanistic motivation** for a
  locality/CoT-domain-matched lens. A narrowed Path A survives only on typo-validity grounds (token-
  identity readout at L17 is strong, 0.646, and backtracking markers are surface tokens): decompose
  δ_full in the token-writeout cone, claim language capped at "writeout cone", never "workspace".
  Legitimate, small (<$10), but now motivated by convenience of the instrument rather than mechanism.
- **B — scale story.** D3's actual finding is better than the planned emergence curve: a **recipe
  contrast** (non-distilled 1–2B read out silently; the R1 distill does not, and D1 shows it does the
  hop out loud instead). Connects directly to the thesis's post-training-origin theme. To harden beyond
  "descriptive": needs the base-model control (Qwen2.5-Math-1.5B lens — base is already cached
  locally) to separate distillation from math-base pretraining. ~one pod-hour + a new sealed sheet.
- **C — gpt-oss-20b DSR × hosted lens.** Untouched by these results; still the best value-per-dollar
  J-lens use for the safety chapter (own prereg; check its lens's high identity-distance 0.94 first).
- **D — shelve.** Ledger entry + the memo'd one-sentence thesis mention; the diagnostic story is
  complete and self-contained as an appendix note.

**Recommendation:** D2–D5 close the diagnostic question decisively; no further diagnostics are
warranted. For the thesis timeline (reframe plan T0–T5 is the authoritative budget), the defensible
default is **D now, with B's base-control as the one optional cheap follow-up** if the recipe-contrast
paragraph earns thesis space. A remains available but should be argued on its narrowed terms; the memo
recommends against C/A spending before the Ph2 missingness sensitivity analysis (still the larger
outstanding blocker) is done. Tony decides.

## Costs

Pod: one RTX 4090 community instance, ~1.6 h wall including the quota failure and rerun (≈ **$1.1**).
Local: D1 52.7 min on the M4 (free). Total diagnostic phase ≈ $1.1 + ~3 h wall.

## Artefact index

- `d1/…/` behavioural audit bundle (generations, sealed scoring, report)
- `d2/…/`, `d3/…/` hosted-lens readout bundles (per-suite npz top-25 arrays, manifests, reports)
- `d4/…/` two-arm Jacobian comparison (per-item ranks + n_valid_positions)
- `d5/…/` FP32 recompute arrays + boundary-crossing counts
- `pod_logs/` full pod stdout including the quota-failure tracebacks (first attempt) and rerun
