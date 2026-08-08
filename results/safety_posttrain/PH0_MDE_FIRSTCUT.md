# PH0 — transport-MDE first cut (sizing only, not inference)

Seed 20260802, B=10000. f* = smallest attenuation of the E8 delta_floor a two-battery
comparison could detect (alpha .05 one-sided, power .80, no pairing benefit assumed).

## backtracking | single_direction
- reported delta_floor +0.0538; estimand recovery: frac rel-err 0.0% (PASS)
- SE (paired/unpaired bootstrap): 0.0164 / 0.0198
- **f\* = 1.29** at n=49 → tasks needed for f\*=0.5: **328**

## backtracking | manifold_k5
- reported delta_floor +0.0698; estimand recovery: frac rel-err 28.4% (FAIL -> SE-ratio fallback only)
- SE (paired/unpaired bootstrap): 0.0178 / 0.0196
- **f\* = 1.38** at n=48 → tasks needed for f\*=0.5: **366**

Consequence rule (§1.4): Phase 2 prereg must size its battery so f* ≤ 0.5, or drop
the 'retained' verdict in favour of 'not disabled'.

## CORRECTION (2026-08-08 — red-team F1–F4, `.codex/reviews/PHASE0_REDTEAM.md`)

The **manifold_k5 rows above are INVALID**: this script paired manifold_k5 against
`energy_matched_random`, but the E8 contract (`src/delta_floor.py`) pairs it with
`random_subspace_k5`; replicated floor rows were overwritten rather than pooled (last-write-wins);
and the failed estimand gate still emitted a headline number contrary to the docstring.
Authoritative cell: Δ_floor = +0.06982, n = 49. The archived manifold f\* ≈ 1.38 / n ≈ 366 must
not be used anywhere. The **single_direction rows are unaffected** (correct floor; exact estimand
recovery), with one label fix (F4): its f\* = 1.29 embeds an extra-conservative arm/floor-unpairing
choice; the literal frozen-rule value is f\* = 1.071 (n\* = 225). Sizing authority is now the T2
simulation (`.codex/out/ph2_mde_sim.json`) plus the in-run injection-recovery gate sealed in
`results/prereg/PHASE2_TRANSPORT_PREREG_2026-08-08.md`.
