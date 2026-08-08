# R3 FULL — strategy entropy on the multi-solution family

> ⚠️ **P-R2.1 CI SUPERSEDED 2026-07-19.** The "+0.016 CI95 [0.013, 0.019]" below bootstraps
> deterministic interpolants (see RESULTS_LEDGER downgrade). The correct task-level bootstrap
> (`33_r3_task_bootstrap.py` → `TASK_BOOTSTRAP.md`) gives CI95 [−0.007, +0.163] ⇒ the gap is
> **DIRECTIONAL ONLY**. Do not cite the interval below.

3584 chains / 64 tasks / 7 cells. Prereg: R3_PILOT_PREREG.md §'What the full R3 adds' + R2_FRONTIER_PREREG.md knob definitions (R2 folded into R3)

Value = TRUE correctness (computable golds; unparsed = incorrect). Strategy entropy = Shannon entropy (bits) of primary-strategy labels over a task's samples, unclassified excluded, None if <2 labelled; cell value = mean over tasks with defined entropy. Classifier: FROZEN pilot lexical classifier — RANGE-FINDER only (CF-T); judged labels are a declared follow-on.

## Substrate diagnostics at the anchor cell (pilot-gate analogues, informational)

- parse 0.898 | accuracy 0.883 | coverage 0.998 | multi-strategy tasks 0.641

## Cells

| cell | rows | acc | acc(uncol) | parse | strat-H (bits) | H-norm | multi-strat | ans-div | 4gram-div | collapse | cap-hit | mean tok |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pump_subtract_a0.5 | 512 | 0.891 | 0.891 | 0.91 | 0.509 (64) | 0.381 | 0.56 | 0.168 | 0.915 | 0.00 | 0.10 | 2739 |
| pump_subtract_a1 | 512 | 0.881 | 0.881 | 0.90 | 0.410 (64) | 0.307 | 0.48 | 0.181 | 0.910 | 0.00 | 0.11 | 2562 |
| pump_subtract_a1.5 | 512 | 0.854 | 0.854 | 0.88 | 0.447 (64) | 0.324 | 0.53 | 0.202 | 0.909 | 0.00 | 0.12 | 2520 |
| vanilla_T0.3 | 512 | 0.838 | 0.841 | 0.85 | 0.459 (64) | 0.345 | 0.50 | 0.172 | 0.888 | 0.00 | 0.16 | 3088 |
| vanilla_T0.6 | 512 | 0.883 | 0.883 | 0.90 | 0.541 (64) | 0.397 | 0.64 | 0.166 | 0.928 | 0.00 | 0.11 | 3055 |
| vanilla_T0.9 | 512 | 0.875 | 0.875 | 0.88 | 0.402 (64) | 0.289 | 0.52 | 0.158 | 0.955 | 0.00 | 0.14 | 3338 |
| vanilla_T1.2 | 512 | 0.561 | 0.561 | 0.59 | 0.397 (64) | 0.276 | 0.47 | 0.285 | 0.982 | 0.00 | 0.28 | 3665 |

## Value × strategy-entropy plane

- pump frontier: [(1.0, 0.41, 0.881), (1.5, 0.447, 0.854), (0.5, 0.509, 0.891), (0.0, 0.541, 0.883)]
- thermostat frontier: [(1.2, 0.397, 0.561), (0.9, 0.402, 0.875), (0.3, 0.459, 0.838), (0.6, 0.541, 0.883)]

### P-R2.1 (reframed) — pump vs thermostat at matched strategy entropy

Overlap H [0.41007835451473684, 0.5407662479249685]; mean accuracy gap (pump−thermo) **+0.016** CI95 [0.012892599736197686, 0.01891790808835551]; pump higher at 24/25 grid pts.

**P-R2.1 (reframed) SUPPORTED — pump retains more value at matched strategy entropy**


### P-R2.2 (reframed) — value falls as strategy entropy rises

- pump: {'spearman_rho': 0.6, 'p': 0.4, 'n': 4}
- thermostat: {'spearman_rho': 0.8, 'p': 0.2000000000000001, 'n': 4}


### P-R2.3 (reframed) — collapse asymmetry (secondary)

- collapse by pump level: {'vanilla_T0.6': 0.0, 'pump_subtract_a0.5': 0.0, 'pump_subtract_a1': 0.0, 'pump_subtract_a1.5': 0.0}
- collapse by thermo level: {'vanilla_T0.3': 0.004, 'vanilla_T0.6': 0.0, 'vanilla_T0.9': 0.0, 'vanilla_T1.2': 0.0}
- collapse-excluded matched gap: P-R2.1 (reframed) SUPPORTED — pump retains more value at matched strategy entropy


## Difficulty strata (CF-V)

| cell | stratum | n | parse | acc | strat-H | cap-hit |
|---|---|--:|--:|--:|--:|--:|
| pump_subtract_a0.5 | easy | 256 | 0.94 | 0.926 | 0.562 | 0.07 |
| pump_subtract_a0.5 | hard | 256 | 0.88 | 0.855 | 0.457 | 0.13 |
| pump_subtract_a1 | easy | 256 | 0.92 | 0.914 | 0.434 | 0.09 |
| pump_subtract_a1 | hard | 256 | 0.88 | 0.848 | 0.386 | 0.14 |
| pump_subtract_a1.5 | easy | 256 | 0.92 | 0.891 | 0.448 | 0.08 |
| pump_subtract_a1.5 | hard | 256 | 0.84 | 0.816 | 0.446 | 0.16 |
| vanilla_T0.3 | easy | 256 | 0.89 | 0.879 | 0.527 | 0.12 |
| vanilla_T0.3 | hard | 256 | 0.81 | 0.797 | 0.391 | 0.20 |
| vanilla_T0.6 | easy | 256 | 0.93 | 0.922 | 0.500 | 0.08 |
| vanilla_T0.6 | hard | 256 | 0.87 | 0.844 | 0.582 | 0.14 |
| vanilla_T0.9 | easy | 256 | 0.93 | 0.926 | 0.457 | 0.09 |
| vanilla_T0.9 | hard | 256 | 0.84 | 0.824 | 0.346 | 0.19 |
| vanilla_T1.2 | easy | 256 | 0.64 | 0.625 | 0.364 | 0.24 |
| vanilla_T1.2 | hard | 256 | 0.53 | 0.496 | 0.429 | 0.32 |

## Strategy × correctness (CF-U, pooled over cells)

| tpl | strategy | n | acc |
|---|---|--:|--:|
| T1 | casework | 9 | 0.0 |
| T1 | pattern | 187 | 0.658 |
| T1 | recursion | 252 | 0.325 |
| T2 | formula | 428 | 0.904 |
| T2 | telescoping | 20 | 0.35 |
| T3 | formula | 446 | 0.713 |
| T3 | recursion | 2 | 0.5 |
| T4 | elimination | 297 | 0.973 |
| T4 | matrix | 9 | 0.889 |
| T4 | substitution | 142 | 1.0 |
| T5 | formula | 448 | 0.855 |
| T6 | modular | 89 | 1.0 |
| T6 | pattern | 359 | 0.93 |
| T7 | casework | 104 | 0.875 |
| T7 | complement | 316 | 0.759 |
| T7 | unclassified | 28 | 0.857 |
| T8 | explicit_roots | 139 | 1.0 |
| T8 | identity | 46 | 0.978 |
| T8 | unclassified | 8 | 1.0 |
| T8 | vieta | 255 | 0.98 |

## Declared follow-ons NOT in this run

- LLM-judge strategy labels under the R2.2 multi-annotator κ protocol, compared against this lexical proxy (CF-T upgrade).
- R3(ii): excursion signature at within-chain strategy switches (R0 E-2 instruments, matched-position controls) on these chains.

