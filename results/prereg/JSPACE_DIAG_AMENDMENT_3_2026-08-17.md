# AMENDMENT 3 — D4 position-local Jacobian arm

**Date:** 2026-08-17, recorded **before D4 executes** (no D4 rank has been computed anywhere).
**Amends:** §6 of `JSPACE_R1_DIAGNOSTIC_PROTOCOL_2026-08-16.md` (seal `752dee7`), on top of
AMENDMENT 2. **Status: SEALED** by Tony in-session 2026-08-17 ("yes go ahead").

## Rationale

Amendment 2 fixed which items the released skip-16 estimator can process, but the deeper mismatch
remains: that estimator averages the Jacobian over a window calibrated for 128-token corpus rows, while
D4's question is about the map at a specific short prompt. Four of the eight A2 items rest on 1–3 valid
positions. The defect is the estimator window, not the stimuli; authoring longer prompts was considered
and rejected (post-hoc-composed stimuli would break the anchor to the registered items and introduce
construction freedom after the Phase-1 failures are known).

## Added arm (arm "local")

Using the pinned released path with its documented parameter — no code fork — compute, per item, the
**position-local Jacobian at the last maskable position**:

- population: **all 81 eligible multihop items** (no length filter needed; every item qualifies);
- call: `jacobian_for_prompt(model, prompt, source_layers=[17, 25], dim_batch=8, max_seq_len=128,
  skip_first=seq_len - 2)` → exactly one valid position, `seq_len - 2`;
- model load, endpoint, censoring, and comparator exactly as sealed §6: BF16, eager attention,
  bridge-label censored rank (>25 → 26) at source layers 17 and 25, against the Phase-1 merged lens
  on the same activations;
- the estimate is exact (deterministic computation of the local linear map), not a noisy estimate of a
  prompt-general object; per-item `seq_len` and `n_valid_positions` (=1) are recorded.

**Declared imperfection:** the released mask can never include the final position itself, so the local
Jacobian sits at `seq_len − 2`, one token before the Phase-1 readout position. This is stated wherever
arm-local results are reported.

**Registered directions (descriptive, no threshold):** averaging-destroys (account d) ⇒ local ranks
materially better than merged, especially at L25; never-existed ⇒ local ranks also poor. Asymmetry: the
local map is exact, so poor local ranks are strong evidence *against* account (d) — there was nothing
for averaging to destroy. Good local ranks with poor merged ranks would mechanistically motivate the
Path-A locality/domain-matched lens; their absence removes that motivation.

## Continuity

Arm "skip16" (AMENDMENT 2: the 8 length-eligible items under the released skip-16 rule) runs unchanged
alongside, for continuity with the sealed §6 design. Both arms report in one D4 run bundle. Estimated
added compute: ≤ 10 minutes on the RTX 4090 (81 single-position fits at 2 source layers on a 1.5B).
Non-gating, descriptive, cannot reclassify Phase 1 — unchanged.
