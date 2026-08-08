# R3 P-R2.1 — task-level bootstrap re-analysis

Supersedes the grid-point CI in `FULL_REPORT.md` (defect: resampled deterministic interpolants; see RESULTS_LEDGER 2026-07-19 downgrade).

- observed gap (pump−thermo, matched strategy-H): **+0.0159** (reproduces frozen pipeline exactly)
- task-level bootstrap (B=2000, paired over 64 tasks, seed 20260719): CI95 **[-0.0066, +0.1631]**, P(gap≤0) = 0.0490
- replicates without frontier overlap: 0/2000

**DIRECTIONAL ONLY — gap does not survive task-level resampling (CI95 straddles 0)**

Caveats unchanged by this re-analysis: frozen lexical classifier (CF-T, judged-label replication owed); no arm raises strategy-H above the shared anchor, so the comparison is a descent contrast against modest temperature (T0.9/T1.2 fall below the overlap window); pump α=0 ≡ vanilla_T0.6 pins both frontiers at one end.
