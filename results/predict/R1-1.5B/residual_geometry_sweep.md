# Residual-geometry sweep (label-free)

| layer | disp. R² | resid/persist | resid/persist (gap1) | resid ID | delta ID | resid PR | dir-churn | norm-slope |
|---|---|---|---|---|---|---|---|---|
| 5 | +0.303 | 0.884 | 0.930 | 6.02 | 3.93 | 54.1 | 0.897 | -0.0294 |
| 11 | +0.305 | 0.880 | 0.921 | 5.08 | 3.77 | 43.2 | 0.900 | -0.0740 |
| 14 | +0.310 | 0.876 | 0.915 | 5.26 | 3.88 | 41.1 | 0.909 | -0.0453 |
| 17 | +0.310 | 0.876 | 0.909 | 4.85 | 3.74 | 41.1 | 0.907 | -0.0246 |
| 20 | +0.302 | 0.882 | 0.918 | 5.07 | 3.70 | 49.3 | 0.908 | -0.2084 |
| 23 | +0.293 | 0.889 | 0.930 | 5.17 | 3.72 | 57.9 | 0.907 | -0.6910 |
| 27 | +0.248 | 0.921 | 0.972 | 5.00 | 3.65 | 64.7 | 0.892 | -0.8460 |

R² > 0 ⇒ learned predictor beats persistence at that layer; resid ID < delta ID ⇒ the unpredictable part is lower-dimensional than the raw step.