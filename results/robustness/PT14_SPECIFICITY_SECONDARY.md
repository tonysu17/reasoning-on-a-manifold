# pt14 — behaviour-specificity null on a second annotator (ledger §F3)

**Result: the 2/4 behaviour-specificity verdict REPLICATES on a second annotator
(Nova) — perfect 8/8 cell agreement with Sonnet.** The specificity test is no
longer single-annotator.

## Instrument
The exact primary null: chain-stratified within-chain label permutation on the
top-10 variance ratio (`src.nulls.full_null_hierarchy`, `top_k_variance_ratio`,
n_resamples=500, seed 42, tail='upper', CF-13 dedup), run identically for Sonnet
(`R1-1.5B`) and Nova (`R1-1.5B-novaspans`) at L12/L16. A behaviour is "specific"
iff chain-stratified p < 0.05.

## chain-stratified p-values (specific = p < .05)
| behaviour | Sonnet L12 | Nova L12 | Sonnet L16 | Nova L16 |
|---|---|---|---|---|
| backtracking | **.002 ✓** | **.002 ✓** | **.002 ✓** | **.002 ✓** |
| uncertainty-estimation | **.002 ✓** | **.002 ✓** | **.002 ✓** | **.002 ✓** |
| example-testing | .96 ✗ | 1.0 ✗ | .06 ✗ | 1.0 ✗ |
| adding-knowledge | 1.0 ✗ | 1.0 ✗ | 1.0 ✗ | 1.0 ✗ |

(.002 is the 500-resample permutation floor.)

## Reading
Both annotators, both layers: **backtracking and uncertainty-estimation**
concentrate their variance beyond the within-chain label-permutation null;
**example-testing and adding-knowledge** do not. The single-annotator caveat on
the behaviour-specificity 2/4 verdict is discharged for the Nova arm — the
specificity test now replicates at the test level (Sonnet + Nova), matching the
intrinsic-dimension + curvature replication R2.2 already established.

The one near-miss (Sonnet example-testing L16 p=.06) sits just above threshold
and Nova reads it firmly non-specific (p=1.0), so the call is unchanged and if
anything Nova is more conservative.

## Scope / remaining
This is a **2-annotator** replication (Sonnet + Nova). The full 3-way needs the
Qwen3 span-pooled activations, which are not local (only the Qwen3 geometry-
robustness JSON was synced from the cluster, not its `qwenspans` matrices) — a
small re-extraction (one pod, the same 4 behaviours × 2 layers). Flagged, not run.

Data: `results/robustness/specificity_secondary_annotator.json`; code
`pt14_specificity_secondary.py`.
