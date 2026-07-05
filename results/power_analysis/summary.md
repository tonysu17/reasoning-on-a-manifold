# Curvature diagnostics — power analysis

Ambient dim d = 1536, intrinsic dim m = 10, noise sigma = 0.1, k = 30, reps = 50

## Minimum detectable curvature (AUC >= 0.95) per (N, diagnostic)

| N | diagnostic | min detectable curvature |
|---|------------|--------------------------|
| 100 | geodesic_euclidean | > max tested |
| 100 | local_vs_global | > max tested |
| 100 | tangent_variation | > max tested |
| 500 | geodesic_euclidean | > max tested |
| 500 | local_vs_global | 0.500 |
| 500 | tangent_variation | > max tested |
| 2000 | geodesic_euclidean | 0.500 |
| 2000 | local_vs_global | 0.500 |
| 2000 | tangent_variation | > max tested |

## Recommendation

Per-behaviour N expectations (R1-Distill, 1000 chains x ~27 sentences):
  - deduction:        ~10,000
  - initializing:     ~5,000
  - uncertainty:      ~4,500
  - backtracking:     ~2,500
  - adding-knowledge: ~1,500
  - example-testing:  ~1,500

Baseline model (Qwen-2.5-Math-1.5B-Instruct) per-behaviour N may drop to ~50-500 for rare behaviours.

Smallest N tested: 100. Largest: 2000.
Cross-reference the table above to determine which behaviours can support per-behaviour curvature analysis.