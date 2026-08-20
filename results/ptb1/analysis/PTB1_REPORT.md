# PT-B1 — safety-versus-control behavioural cell

Sealed authority: `results/prereg/PTB1_SAFETY_CONTROL_BEHAVIOUR_PREREG_2026-08-20.md` (+ Amendment 1).
Caveats (travel with every number): builder-annotator (Sonnet); control-corpus asymmetry (control adapter trained on own corpus chains; tasks A1-disjoint; format transfer direction unknown); recipe-level, three seeds per class, one 1.5B response-distilled model, this battery only; no refusal/compliance/benchmark endpoint; boxed correctness not computable (no gold answers).

## Primary family (safety − control, Holm over 4)

| Behaviour | Δ complete-case (n) | 95% BCa | raw p | Holm | Missingness | Verdict |
|---|---|---|---|---|---|---|
| backtracking | +0.0038 (98) | [-0.0082, +0.0165] | 0.5545 | 1.0000 | sign-robust (behaviour-dense) | point estimate only |
| uncertainty-estimation | +0.0017 (98) | [-0.0178, +0.0227] | 0.8753 | 1.0000 | missingness-fragile | point estimate only |
| example-testing | +0.0054 (98) | [-0.0087, +0.0217] | 0.4828 | 1.0000 | sign-robust (behaviour-dense) | point estimate only |
| adding-knowledge | +0.0111 (98) | [+0.0009, +0.0228] | 0.0434 | 0.1736 | sign-robust (behaviour-dense) | point estimate only |

## Full-chain endpoints (annotation-free)

| Endpoint | Δ (safety − control) | 95% BCa | n |
|---|---|---|---|
| length | -75.19 | [-351.27, +222.06] | 100 |
| looped | +0.00 | [-0.06, +0.06] | 100 |
| truncated | +0.00 | [-0.04, +0.05] | 100 |
| boxed_emitted | +0.01 | [-0.02, +0.03] | 100 |

Damage-context rule: not tripped.

Per-seed sign grids, bt_per_1k, base reference levels, and full missingness bounds: `ptb1_analysis.json`.
