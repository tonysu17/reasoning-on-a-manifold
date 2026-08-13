# Phase-2 annotation — closeout report

**Date:** 2026-08-13. **Status: COMPLETE.** No worker running; all journals resolved.

## Final coverage

| Shard | rows | coverage-complete | unresolved | rate |
|---|---|---|---|---|
| base | 700 | 619 | 81 | 88.4% |
| star1 | 1300 | 1137 | 163 | 87.5% |
| deepscaler | 1300 | 1113 | 187 | 85.6% |
| base_injection | 300 | 256 | 44 | 85.3% |
| **Total** | **3600** | **3125** | **475** | **86.8%** |

(base deduplicated 701→700: one byte-identical VERB_123 record from the pilot
merge removed.)

Every row carries `annotation_coverage` (rule `ph2-annotation-coverage-4`),
`annotation_coverage_complete`, `annotation_coverage_attempts` (≤2, then sealed)
and `annotated_region_tokens`. Unresolved rows are pairwise-deleted and counted
by the A2 adjunct; they are never zero-filled.

## ⚠ Primary caveat for analysis: arm-differential missingness

Unresolved rates by arm class: **vanilla 5–7%** vs **steered arms 15–26%**
(worst: whitened/raw induce ~24–26%). Steered chains are harder for the
annotator to echo exhaustively even after the rule-4 degenerate-loop truncation.
Because the missing rows are plausibly the most behaviour-dense, prevalence
contrasts (steered − vanilla) may be attenuated. Required handling downstream:
report `n_coverage_incomplete` per cell (already automatic), and run a
bounding/sensitivity analysis before citing any prevalence endpoint. Full-chain
endpoints (looped, truncated, boxed, length, damage gates) are unaffected —
they never depended on annotation coverage.

## Spend (guard-verified, cumulative across all journals)

| | USD |
|---|---|
| V1+V2 legacy (pre-correction) | 1.180969 |
| A5 smoke | 0.229278 |
| A5R3 smoke + seal check | 0.316389 |
| A5R4 leg (bound refusal find) | 3.329200 |
| A5R5 leg (matcher fixes find) | 2.592700 |
| A5R7 leg (rule-4) | 2.592747* |
| A5R8 pilot (incl. re-execution) | 1.217000 |
| A5R9 fleet pass | 167.670000 |
| A5R10 sweep | 41.960000 |
| **Total committed** | **230.03** (ceiling 275.00) |

*includes conservative $0.05 charges for killed in-flight calls.
Max observed single call: $0.045204 against the $0.05 authorised / $0.0471
char-class bound. Zero ceiling breaches after the correction; every stop was
fail-closed before I/O.

## Final artefact SHA-256 (first 16 hex)

| artefact | sha256[:16] |
|---|---|
| results/ph2/annotation/base.json | 8d3245a33717c196 |
| results/ph2/annotation/star1.json | 64cf05c513eacf23 |
| results/ph2/annotation/deepscaler.json | 27895a11887cbc8c |
| results/ph2/annotation/base_injection.json | a2304dbfd29036af |
| api_attempt_journal.jsonl (A5R10) | 33582c97a587cf83 |

All superseded journals archived under `.codex/out/` with hashes; original V2
evidence untouched (`PH2_ANNOTATION_V2_STOPPED_*`); quarantined originals in
`results/ph2/annotation/quarantine/base.v1.json`.

## Amendment lineage this run

A5 (region + provable cost bound) → R2 unicode fold + prefix strip → R3 punct
spacing → **R4 degenerate-repetition truncation (owner decision B)** → R8
prompt amendment "annotate every part…" (owner decision A; +40–50pp coverage on
failing rows) → R9 prompt-versioned request digests, bounded coverage retries
(MAX_COVERAGE_ATTEMPTS=2, proxy nondeterminism observed) → R10 sweep.
Full dated details in `PH2_ANNOTATION_COVERAGE_AND_COST_CORRECTION_2026-08-11.md`.

## Not done here (by design)

Phase-2 arm effects were NOT computed (hard boundary). The analyse stage
(`ph2_executor.py analyse`) remains to be run under its own authorisation.
