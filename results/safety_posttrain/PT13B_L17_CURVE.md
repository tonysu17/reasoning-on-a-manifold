# PT13b — contraction curve on the R1-compression instrument (L17 windowed PR)

Date 2026-08-02 · prereg §1.1 of PHASE0_TRANSPORT_FREEZE · commit 5fd66637 (dirty)

**FAILED: Gate(s) B2 — substrate inadequate locally; mapping deferred to pod s3 (freeze §1.1 labelling rule) — no mapping reported.**

- B2 ratio 1.356; root cause: Xout_* is a CLASS-SELECTED NON-CONTIGUOUS token subsample (18_loop_geometry.py select_class_token_indices, tokens-per-class=192, out-of-loop tokens only), not a sequence excerpt; the windowed instrument is undefined on it. No local full-sequence source exists (main + _pod_archive checked). Mapping deferred to pod s3.

Substrate-invalid curve retained below for the audit record only — NOT valid for mapping.

| c | median dPR (invalid substrate) | mean var removed |
|---|---|---|
| 1.0 | +0.0000 | 0.0000 |
| 0.9 | -5.2652 | 0.1409 |
| 0.75 | -13.7446 | 0.3244 |
| 0.5 | -25.7339 | 0.5561 |
| 0.25 | -32.0104 | 0.6951 |
| 0.0 | -33.7850 | 0.7415 |
