# R2 — entropy–value frontier (creativity–entropy rung 2)

Value axis: boxed completion rate (annotation-free; NOT correctness — CF-P). Prereg: R2_FRONTIER_PREREG.md

> **PREP FINDING 2026-07-12:** 4-gram diversity is SATURATED (~0.96, range 0.87–0.99, 0% of tasks <0.9) — surface token variation, not solution diversity, no dynamic range. Answer-level diversity has range but is SPARSE (only ~30–46% of the 50 general-reasoning eval tasks produce any \boxed answer). ⇒ the value-vs-diversity FRONTIER is not measurable on this task set; R2's clean result is the completion/length effect below. True diversity frontier deferred to R3's dedicated multi-solution task family.

## Knob A — PUMP (backtracking amplification α @ T=0.6)

| level | tasks | 4gram-div | ans-div (n≥2) | answered | boxed | collapse | mean_tok |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.0 | 50 | 0.962 | 0.722 (6) | 0.30 | 0.160 | 0.060 | 4905 |
| 0.5 | 50 | 0.961 | 0.894 (11) | 0.34 | 0.240 | 0.047 | 4521 |
| 1.0 | 50 | 0.963 | 0.821 (14) | 0.42 | 0.293 | 0.053 | 4249 |
| 1.5 | 50 | 0.962 | 0.905 (14) | 0.46 | 0.293 | 0.040 | 3867 |

## Knob T — THERMOSTAT (vanilla temperature)

| level | tasks | 4gram-div | ans-div (n≥2) | answered | boxed | collapse | mean_tok |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.6 | 50 | 0.962 | 0.722 (6) | 0.30 | 0.160 | 0.060 | 4905 |

## P-R2.1 — pump vs thermostat at matched diversity

_insufficient points (need ≥2 per knob with diversity)_


## P-R2.2 — value falls as diversity rises (both knobs)

- pump: {'spearman_rho': 0.738, 'p': 0.26213521262737816, 'n': 4}
- thermostat: {'status': 'too few points', 'n': 1}


> ⚠️ PRELIMINARY: thermostat frontier has <2 diversity points — the temperature sweep (T∈{0.3,0.9,1.2}) is not yet generated. Pump frontier stands on existing data; P-R2.1 awaits the sweep.

