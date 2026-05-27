# M1 Completion Report — CBS-tier + cross-domain annotator

**Branch**: `cbs/m1-annotation`
**Date**: 2026-05-27
**Synthesis-plan reference**: §M1.1–§M1.5
**Status**: COMPLETE (build phase only — pilot run gated on P0.2)

## What was implemented

### `src/cbs/annotation.py`

Full implementation per synthesis §M1.3:

* `CBS_PROMPT_TEMPLATE` — verbatim from §M1.2, with `{anchor_block}` /
  `{task_domain}` / `{task_prompt}` / `{prev_sentences}` / `{behaviour}` /
  `{sentence}` substitution points. `PLACEHOLDER_ANCHOR_BLOCK` is used until
  P0.2 anchor curation completes — pilot results emitted with the placeholder
  are tagged `pilot-only`.
* `TASK_DOMAIN_PROMPT` — short one-shot classifier into `TASK_DOMAINS`.
* `SonnetClient` Protocol + `ProxyClient` production client over
  `CLAUDE_PROXY_URL` / `CLAUDE_PROXY_KEY` (same transport as Phase 3).
* `_extract_json_object` — tolerates code-fence wrapping and pre/post-amble
  via a balanced-brace scanner.
* `annotate_task_domain(task, client) -> str` — returns one of TASK_DOMAINS;
  defaults to `"other"` on parse / model-output failure rather than blocking
  the pipeline on a single ambiguous task.
* `annotate_sentence_cbs(...) -> CBSResult` — full classification with
  seed→temperature mapping (seed 0 → T=0.0, seed ≥ 1 → T=0.3) for the
  dual-seed κ mechanism. Raises `ValueError` on malformed output / invalid
  tier so the caller can log + continue.
* `annotate_chains_cbs(...)` — sequential over chains (preserves the
  per-task task_domain cache), parallel within each chain via
  ThreadPoolExecutor with `max_workers` workers (default 8).
* `cohen_kappa_three_tier(a, b) -> float` — sklearn-backed, manual fallback.
* `build_anchor_candidates_csv(...)` — stratified per-category sampler with
  optional Sonnet pre-classification for ranking. Used by P0.2 to emit the
  CSV that Tony curates manually.

### `08_annotate_cbs.py`

CLI modes:

* default — full annotation, single seed → `--out` JSON.
* `--pilot` — stratified-100 dual-seed κ; emits
  `results/cbs/{model}/kappa_run1_run2.json` and
  `results/cbs/{model}/pilot_for_human_review.csv`. Hard-fail-stops on
  κ < 0.5 OR max(tier-3 rate) < 0.05 with a FAILSTOP report and three
  options per synthesis §P0.2.
* `--dual-seed-kappa` — full corpus dual-seed κ.
* `--build-anchors` — emit `results/cbs/anchor_candidates.csv` for Tony's
  P0.2 curation. Runs with a real client if `CLAUDE_PROXY_URL` is set
  (≈ $0.06 for 60 candidates); falls back to no-tier-estimate sampling
  otherwise.
* `--anchor-block-path` — substitute the locked anchor block after P0.2.

### Unit tests (`src/cbs/tests/test_annotation.py`)

17 tests, all mocked Sonnet client (`MockSonnetClient`):

* Module import smoke.
* `CBSResult` field validation (tier, knowledge_domain, confidence).
* CBS prompt + task-domain prompt format-string render.
* `annotate_task_domain`: happy path, invalid-domain fallback to `"other"`,
  malformed-JSON fallback to `"other"`.
* `annotate_sentence_cbs`: happy path with context-window assertion (only
  last 3 prev sentences embedded), dual-seed pass-through, malformed-JSON
  raises, invalid-tier raises.
* `annotate_chains_cbs`: adds fields only to targeted behaviours, uses
  task_domain cache (single domain call across two chains with same
  task_id), text-based per-sentence failure preserves other spans
  (deterministic under within-chain parallelism).
* `cohen_kappa_three_tier`: perfect agreement → 1, strong disagreement
  near 0, length-mismatch raises.

## Validation

```
$ python -m pytest src/cbs/tests/ -q
22 passed, 13 skipped
```

(The 22 active tests include P0.3's 7 plus 15 new M1 tests; the 13 skipped
remain placeholders for M2 / M3 / M4 / M5 / M6.)

CLI smoke:

```
$ python 08_annotate_cbs.py --help
usage: 08_annotate_cbs.py [-h] [--in IN_PATH] [--out OUT] [--seed SEED]
                          [--dual-seed-kappa] [--pilot] ...
```

## Deviations from synthesis

* Synthesis §M1.3 signature: `annotate_chains_cbs(... seed: int = 0,
  max_workers: int = 8)`. Implementation matches; parallelism added
  *within* each chain (not across chains) so the per-task task_domain cache
  remains hit-once.
* The `seed` parameter is mapped to a `temperature` value
  (0 → 0.0, ≥1 → 0.3) when the proxy back-end does not honour an
  OpenAI-style `seed` field. This is the dual-seed κ mechanism described
  in synthesis §P0.2 ("different temperature seeds").
* `_extract_json_object` added as a non-spec helper because Phase-3
  experience shows Sonnet sometimes returns code-fenced JSON despite the
  prompt. Kept tolerant; the only hard failure mode is "no JSON object
  found", which raises `ValueError` and is logged-and-skipped by
  `annotate_chains_cbs`.

## Known limitations

* `ProxyClient` requires `CLAUDE_PROXY_URL` / `CLAUDE_PROXY_KEY`; not used
  in the unit tests (all mocked).
* `08_annotate_cbs.py --pilot` is not exercised end-to-end at build time
  because the pilot run depends on (a) P0.4 truncation policy and (b) P0.2
  anchor curation — both gated on Tony.
* The runner does not yet add resume-from-checkpoint logic for very large
  full-corpus runs; if added later it should follow `src/annotation.py`'s
  pattern.

## Next milestone

P0.4 — write the truncation-policy decision template, halt, and ask Tony
to pick a policy. After Tony picks, P0.2 — emit anchor candidates, halt
for curation, then run the pilot with the locked anchor block.
