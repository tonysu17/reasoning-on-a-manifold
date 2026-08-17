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

## ADDENDUM 2026-08-17 evening — base-control cell executed (run `d3-qwen2.5-math-1.5b-base-2008de267117`)

Self-fit 100-prompt WikiText lens on **Qwen2.5-Math-1.5B** (R1-Distill-1.5B's own base), sealed sheet
`JSPACE_BASE_CONTROL_SHEET_2026-08-17.md`; 7.3 h fit + 4 min scoring, local M4, $0. Matched comparison:
identical scored vocabulary domain (151,665; 271 padding rows excluded) and identical eligible-item
counts (98/81/96) as the Phase-1 distill run; both lenses converged normally and near-identically
(mean_rel_change 0.026 @ n=50 distill vs 0.025 base, both inside the hosted-lens band).

| Endpoint | **Base** Qwen2.5-Math-1.5B | **Distill** R1-Distill-1.5B |
|---|---:|---:|
| multihop any-layer union | **0.4074** (p=.001, qualifies) | 0.1852 (p=.001) |
| multihop **at L17** | **0.0000** | **0.0000** |
| multihop peak layer | L23 (0.321) | L25 (0.136) |
| multihop logit-lens comparator | **0.4630** (> J-lens) | 0.1728 (< J-lens by .012) |
| association union | 0.0102 (p=.84, fails) | 0.0408 (p=.118, fails) |
| typo union / L17 | 0.8750 / 0.7396 | 0.8958 / 0.6458 |
| typo logit comparator | 0.6354 (J-lens +0.240) | 0.5521 (J-lens +0.344) |

**Registered matrix verdict (by the letter):** the base qualifies on multihop ⇒ the sheet's
"distillation-recipe-specific" branch is the licensed reading of the *union* endpoint.

**But the mechanism is NOT the branch's narrative, and this must be stated wherever the cell is cited.**
Three facts qualify it:

1. **Neither model carries the bridge at mid-stack.** Base L17 = 0.0000, identical to the distill.
   The base never had a mid-stack verbalizable workspace for these bridges either — so the Phase-1 L17
   failure **predates distillation** and cannot be attributed to it.
2. **What differs is late-layer bridge resolution, and only there.** The base's entire multihop signal
   lives at L20–L26 (peak L23), like the distill's L21–L26 (peak L25) — but 2.2× more of it.
3. **The base's late signal is not workspace-format.** Its logit-lens comparator (0.463) *exceeds* the
   J-lens (0.407): the bridge appears as proto-output, already rotated into the output basis, not as a
   cached pre-output intermediate. Contrast typo, where the J-lens beats the logit lens by +0.24 (base)
   and +0.34 (distill) — that is what a genuine cached intermediate looks like in this instrument.

**Corrected synthesis.** The earlier speculative "distillation relocated composition out of the silent
workspace into the token channel" is **not supported as stated** and should not be written that way:
there was no mid-stack workspace in the base to relocate from. The supported statement is narrower:
*at 1.5B, neither the math-base nor its CoT distill exposes bridge entities to Jacobian-lens readout at
mid-stack; both resolve bridges only in answer-adjacent layers and in proto-output form; and the
distillation step roughly halved how often that late resolution happens (0.407 → 0.185).* This is
consistent with D1's behavioural finding (the distill composes out loud: CoT 0.617 vs silent 0.099)
without requiring the workspace-relocation mechanism.

**Association:** base 0.0102 at chance (p=.84), distill 0.0408 at chance (p=.118) — both fail; the
association deficit is inherited from the math-narrow base, not created by distillation, as expected.

**Typo:** base 0.875 / distill 0.896, both with large J-lens advantages ⇒ the base lens is a working
instrument, so the multihop/association readings are not fit-quality artefacts. This is the cell's
internal positive control and it passes.

**Consequence for the fork.** The recipe-contrast paragraph (option B's motivation) survives but shrinks:
it is a claim about *late-layer bridge availability*, not about a workspace. The 7B pair would now
re-test a smaller and murkier effect (an amount-of-late-resolution difference whose readout is not even
workspace-format), so its value has **decreased**; recommendation against funding it stands, more firmly
than before. Option D (shelve + appendix) remains the recommendation, with the appendix stating the
corrected synthesis above rather than the relocation story.

## Costs

Pod: one RTX 4090 community instance, ~1.6 h wall including the quota failure and rerun (≈ **$1.1**).
Local: D1 52.7 min on the M4 (free). Total diagnostic phase ≈ $1.1 + ~3 h wall.

## Artefact index

- `d1/…/` behavioural audit bundle (generations, sealed scoring, report)
- `d2/…/`, `d3/…/` hosted-lens readout bundles (per-suite npz top-25 arrays, manifests, reports)
- `d4/…/` two-arm Jacobian comparison (per-item ranks + n_valid_positions)
- `d5/…/` FP32 recompute arrays + boundary-crossing counts
- `pod_logs/` full pod stdout including the quota-failure tracebacks (first attempt) and rerun
