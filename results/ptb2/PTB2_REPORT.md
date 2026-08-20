# PT-B2 — post-training and the behaviour-concentration map

Sealed authority: `results/prereg/PTB2_POSTTRAINING_CONCENTRATION_PREREG_2026-08-20.md`.

Estimand: fixed-top-ten variance concentration (RQ1 statistic); NOT correlation dimension, participation ratio, PCA variance-threshold dimension, subspace, or manifold.

**Layers.** L12/L16 are the only adapter-extracted depths and do NOT overlap RQ1's {11,14,17,20,27}; no PT-B2 cell is numerically comparable to an RQ1 cell.

**Environment rule.** primary safety-minus-control is WITHIN-session per seed; base-referenced values are cross-session and descriptive only (see PTB1_AMENDMENT_3).

## Primary: safety − control (paired within seed, Holm over 8)

| Cell | Δ | 95% BCa | boot p | Holm | seed-perm p | Verdict |
|---|---:|---|---:|---:|---:|---|
| L12|backtracking | -0.00016 | [-0.00023, -0.00011] | 0.0005 | 0.0040 | 0.25 | resolved nonzero (Holm, chain bootstrap governs) |
| L12|uncertainty-estimation | -0.00021 | [-0.00027, -0.00016] | 0.0005 | 0.0040 | 0.25 | resolved nonzero (Holm, chain bootstrap governs) |
| L12|example-testing | -0.00045 | [-0.00053, -0.00036] | 0.0005 | 0.0040 | 0.25 | resolved nonzero (Holm, chain bootstrap governs) |
| L12|adding-knowledge | -0.00019 | [-0.00025, -0.00013] | 0.0005 | 0.0040 | 0.25 | resolved nonzero (Holm, chain bootstrap governs) |
| L16|backtracking | -0.00002 | [-0.00009, +0.00006] | 0.5900 | 0.7080 | 0.75 | not resolved — point estimate only |
| L16|uncertainty-estimation | -0.00011 | [-0.00017, -0.00006] | 0.0010 | 0.0040 | 0.25 | resolved nonzero (Holm, chain bootstrap governs) |
| L16|example-testing | -0.00025 | [-0.00036, -0.00016] | 0.0005 | 0.0040 | 0.25 | resolved nonzero (Holm, chain bootstrap governs) |
| L16|adding-knowledge | +0.00003 | [-0.00005, +0.00010] | 0.3540 | 0.7080 | 0.50 | not resolved — point estimate only |

Seed-permutation p has a hard floor of 0.25 in a 3-vs-3 paired design; the chain bootstrap governs every claim.

## Secondary: concentration map (descriptive)

Cells passing the chain-stratified null at p<.05, by recipe class (safety/control counts are out of 3 seeds):

| Cell | base | safety | control |
|---|---|---:|---:|
| L12|backtracking | pass | 3/3 | 3/3 |
| L12|uncertainty-estimation | pass | 3/3 | 3/3 |
| L12|example-testing | fail | 0/3 | 0/3 |
| L12|adding-knowledge | fail | 0/3 | 0/3 |
| L16|backtracking | pass | 3/3 | 3/3 |
| L16|uncertainty-estimation | pass | 3/3 | 3/3 |
| L16|example-testing | fail | 0/3 | 0/3 |
| L16|adding-knowledge | fail | 0/3 | 0/3 |

descriptive map; no Holm claim attaches; smallest attainable p is 1/1001.

Point statistics, per-seed differences, specificity margins and full provenance: `ptb2_analysis.json`.
