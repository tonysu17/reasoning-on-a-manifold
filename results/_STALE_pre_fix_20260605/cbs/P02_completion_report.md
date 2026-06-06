# P0.2 Completion Report — Anchor candidate emission

**Branch**: `cbs/p0-anchor-candidates`
**Date**: 2026-05-27
**Synthesis-plan reference**: §P0.2 / build-now step 4 (first halt point only)
**Status**: PARTIAL — candidates emitted, awaiting Tony's curation + tier-ranking re-run

## What was done

Emitted `results/cbs/anchor_candidates.csv` (60 rows). Stratification:

* 10 task categories × 2 behaviours × 3 sentences/cell = 60 sentences total.
* Behaviour balance: 30 adding-knowledge / 30 deduction (over-samples
  adding-knowledge relative to its ~15% corpus base rate, per Tony's pick at
  the P0.2 halt point).
* Per-category counts uniform at 6 (3 adding-knowledge + 3 deduction).

CSV columns: `task_id, sentence_idx, category, behaviour, task_domain_hint,
context_3, sentence, tier_estimate`.

## Code change

Added `balance_behaviours` parameter to
`src/cbs/annotation.py::build_anchor_candidates_csv` and a corresponding
`--behaviour-balance` flag to `08_annotate_cbs.py`. Backward-compatible:
default is the original category-only stratification.

Tests: `22 passed, 13 skipped` (no regressions).

## What's still outstanding

### (a) Sonnet tier-ranking re-run (Tony-side)

`tier_estimate` is empty for every row because `CLAUDE_PROXY_URL` /
`CLAUDE_PROXY_KEY` were not available in the runtime environment. To
re-emit with tier estimates (≈ $0.06 API):

```bash
source .env                          # sets CLAUDE_PROXY_URL / CLAUDE_PROXY_KEY
python 08_annotate_cbs.py --build-anchors --behaviour-balance \
    --anchors-per-category 6 --seed 0
```

The current candidate set is sorted by `tier_estimate` only; with empty
estimates the order falls back to the sampled order (per-cell random).
For curation Tony can sort by `(category, behaviour)` in a spreadsheet
either way.

### (b) Tony's manual curation (synthesis §P0.2)

Pick 15 anchors total: 5 per tier (1 retrieval / 2 recombination /
3 novel application). Write a locked anchor-block text file — typically a
short markdown with one example per anchor, formatted to drop into
`CBS_PROMPT_TEMPLATE.format(anchor_block=...)`. Pass the file via
`08_annotate_cbs.py --anchor-block-path <file>` for the pilot run.

### (c) Pilot run (synthesis §P0.2, second halt point)

This is a **run-phase** action, not a build-now action:

```bash
python 08_annotate_cbs.py --pilot --pilot-size 100 \
    --anchor-block-path <locked-anchor-block.md>
```

* Exits 0 on pass (κ ≥ 0.5 AND tier-3 rate ≥ 0.05).
* Exits 1 on fail and writes `results/cbs/R1-1.5B/FAILSTOP_M1.md` with the
  three options.
* Writes `results/cbs/R1-1.5B/pilot_for_human_review.csv` (50 rows for
  Tony's 50-sentence ground-truth labelling).

The pilot's ~$2 API cost lives in the run phase, not build-now.

## Deviations from synthesis

* Tony picked behaviour-balanced 30/30 over the default 6/category random
  mix because the corpus base rates (15% adding-knowledge vs 52%
  deduction) made the random mix tilt 5:1 toward deduction, which
  under-samples the rare behaviour anchors most need to be diverse on.

## Halt status

Build-now CONTINUES with **placeholder anchors** through M2–M6. Pilot run
is paused until Tony returns the locked anchor block. The placeholder is
unchanged from P0.3 and runs are tagged `pilot-only` until locked.

## Next milestone

M2 — geometry module + smoke run.
