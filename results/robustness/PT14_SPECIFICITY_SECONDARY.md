# pt14 — behaviour-specificity null on a second annotator (ledger §F3)

**Result: the 2/4 behaviour-specificity verdict REPLICATES FULL 3-WAY (Sonnet +
Nova + Qwen3) — perfect 8/8 cell agreement.** The specificity test is no longer
single-annotator; it now matches R2.2's coverage of intrinsic dimension and the
curvature negative.

## Instrument
The exact primary null: chain-stratified within-chain label permutation on the
top-10 variance ratio (`src.nulls.full_null_hierarchy`, `top_k_variance_ratio`,
n_resamples=500, seed 42, tail='upper', CF-13 dedup), run identically for Sonnet
(`R1-1.5B`) and Nova (`R1-1.5B-novaspans`) and Qwen3 (`R1-1.5B-qwenspans`, extracted 2026-07-22) at L12/L16. A behaviour is "specific"
iff chain-stratified p < 0.05.

## chain-stratified p-values (specific = p < .05)
| behaviour | Sonnet L12/L16 | Nova L12/L16 | Qwen3 L12/L16 |
|---|---|---|---|
| backtracking | **.002 ✓ / .002 ✓** | **.002 ✓ / .002 ✓** | **.002 ✓ / .002 ✓** |
| uncertainty-estimation | **.002 ✓ / .002 ✓** | **.002 ✓ / .002 ✓** | **.002 ✓ / .002 ✓** |
| example-testing | .96 ✗ / .06 ✗ | 1.0 ✗ / 1.0 ✗ | 1.0 ✗ / 1.0 ✗ |
| adding-knowledge | 1.0 ✗ / 1.0 ✗ | 1.0 ✗ / 1.0 ✗ | 1.0 ✗ / 1.0 ✗ |

(.002 is the 500-resample permutation floor.)

## Reading
All three annotators, both layers: **backtracking and uncertainty-estimation**
concentrate their variance beyond the within-chain label-permutation null;
**example-testing and adding-knowledge** do not. The single-annotator caveat on
the behaviour-specificity 2/4 verdict is DISCHARGED — the specificity test now
replicates at the test level across all three annotators, matching the
intrinsic-dimension + curvature replication R2.2 already established. Notably this
holds despite the annotators disagreeing substantially on the labels themselves
(char-level kappa 0.35-0.44; up to 3x differences in per-label frequency): the
geometry is robust to that label noise.

The one near-miss (Sonnet example-testing L16 p=.06) sits just above threshold
and both Nova and Qwen3 read it firmly non-specific (p=1.0), so the call is
unchanged and the secondary annotators are more conservative.

## Scope
Full 3-way, complete. Qwen3 span activations were extracted 2026-07-22 on a 24 GB
pod (peak 6.7 GB; pod self-terminated after sync) and are local at
`data/activations/R1-1.5B-qwenspans/`. Nothing further owed on this item.

Data: `results/robustness/specificity_secondary_annotator.json`; code
`pt14_specificity_secondary.py`.
