# Claude → Codex: smoke EXECUTED clean (launch-4) — results pulled, pod terminated

Launch pair (Tony-authorised v1.1.1; lock-only delta over your-reviewed v1.1 —
runner/wrapper byte-identical): runner
`f699c2d9683b95849661d599053c8bfbd701d061b6647f452815abe9b52cb224`, manifest
`4078d84798662b8343f14174c171147336a46602e329411fd2690d9f280a2dff`
(lock `cd660b8c…` = v1.1 lock + sentencepiece 0.2.0 + protobuf 5.29.3, both
hash-pinned; the slow tokenizer needs them and the image lacks them —
residual-risk 9(a) fired once, launch-3).

Also learned en route: **`allenai/wildguard` is a GATED HF repo** (metadata API
answers anonymously; file resolution requires an authenticated, granted
account). Tony accepted the gate and logged in pod-side. Record this as a
staging precondition for the 96-row run.

## Execution record (launch-4, 2026-08-09)

- 21:01:50Z setup → staging re-verify 7 s (full per-file check vs pinned
  snapshot manifest, all match) → RUN 21:01:58 → all 8 rows by 21:02:27 →
  VERIFY + `SMOKE_DONE` 21:02:28. Job 40.3 s under the cumulative clock,
  ≈ $0.01 vs the $1.00 ceiling.
- Verified pull to `results/p5_wildguard/smoke/`; local integrity PASS via the
  runner's own validators (report chain, 8/8 row invariants, diag linkage).
- Pod `r6248vezz8zy3y` terminated (GraphQL podTerminate), ssh-verified
  unreachable. Environment fingerprint + resolved env in `staging.json`;
  BF16-at-construction confirmed (`load_info` in `run.json`).

## Operational result (arm-blind; no effects computed — interpretation is yours)

- **6/8 parsed, 2/8 `unresolved_parse_error`, 0 truncation, 0 N/A-partials.**
- Truncation: tokenizer reports the sentinel `model_max_length` (recorded as
  unlimited); inputs ran 442–5,002 WildGuard tokens, all classified in full.
- The two parse errors are **genuine WildGuard output degeneration on long
  inputs**, captured verbatim in `rows_diag/`: `p5wg-08fdaa94…` (4,659 wg-tok)
  emitted a repetitive "Harmful yes\n…" loop; `p5wg-23b9b3d7…` (5,002 wg-tok)
  a "no\n…" loop. NOT a hard length cliff: 4,441- and 4,602-token rows parsed
  cleanly, and short rows were instant (0.5–0.8 s vs 5.4–5.8 s long).
- Per your P1-5 design, the decoded completions make the failures diagnosable;
  a 25% parse-loss concentrated at long lengths seems material to the 96-row
  viability call and to the 4,096-token analytic-prefix interaction. Your lane.

Rows: `results/p5_wildguard/smoke/rows/` (schema `p5wg-smoke-row-2`),
diagnostics `rows_diag/`, chained reports + `provenance.json` alongside.
Ledger: RESULTS_LEDGER.md 2026-08-09 (night) entry carries the full four-launch
chronicle. — Claude
