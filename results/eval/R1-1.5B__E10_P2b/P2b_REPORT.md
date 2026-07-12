# E10.1 P2 REDESIGN — matched-displacement collapse curves

Date: 2026-07-07 · collapse = 4-gram repetition > 0.8

| arm | n | collapse | displacement | cue/1k | boxed | tokens |
|---|---|---|---|---|---|---|
| das_clamp_b0.5 | 50 | 0.54 | 11.03 | 16.00 | 0.06 | 6946 |
| das_clamp_b1.0 | 50 | 0.44 | 20.61 | 254.83 | 0.00 | 8192 |
| das_proj_a0.5 | 50 | 0.50 | 4.02 | 12.39 | 0.06 | 6540 |
| das_proj_a1.0 | 50 | 0.56 | 7.32 | 18.43 | 0.02 | 7139 |
| das_proj_a2.0 | 50 | 0.92 | 11.91 | 34.26 | 0.00 | 8051 |
| das_proj_a3.0 | 50 | 0.96 | 16.53 | 37.90 | 0.00 | 8192 |
| dm_clamp_b0.5 | 50 | 0.56 | 13.13 | 12.17 | 0.04 | 6836 |
| dm_clamp_b1.0 | 50 | 0.86 | 24.28 | 19.55 | 0.00 | 7672 |
| dm_proj_a0.5 | 50 | 0.42 | 7.61 | 9.87 | 0.02 | 6708 |
| dm_proj_a1.0 | 50 | 0.62 | 16.40 | 12.20 | 0.02 | 7386 |
| dm_proj_a2.0 | 50 | 0.84 | 46.51 | 5.04 | 0.00 | 8192 |
| dm_proj_a3.0 | 50 | 1.00 | 79.56 | 1.51 | 0.00 | 8192 |
| vanilla | 50 | 0.32 | 0.00 | 9.79 | 0.06 | 5727 |

## Primary: clamp vs projective at MATCHED displacement (in-range now)
- **das_clamp_b0.5**: clamp collapse 0.54 vs projective 0.85 at displacement 11.03 (proj range 4.0–16.5, in-range=True); diff -0.31 CI95 [-0.44, -0.19] ⇒ **clamp GENTLER**
- **das_clamp_b1.0**: clamp collapse 0.44 vs projective 0.96 at displacement 20.61 (proj range 4.0–16.5, in-range=False); diff -0.52 CI95 [-0.66, -0.38] ⇒ **clamp GENTLER**
- **dm_clamp_b0.5**: clamp collapse 0.56 vs projective 0.55 at displacement 13.13 (proj range 7.6–79.6, in-range=True); diff +0.02 CI95 [-0.13, 0.16] ⇒ **type IRRELEVANT (equal)**
- **dm_clamp_b1.0**: clamp collapse 0.86 vs projective 0.68 at displacement 24.28 (proj range 7.6–79.6, in-range=True); diff +0.18 CI95 [0.07, 0.3] ⇒ **clamp WORSE**