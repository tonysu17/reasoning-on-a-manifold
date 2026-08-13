# Phase-2 annotation last-minute preflight — 2026-08-10

## Disposition

**READY FOR HASH-BOUND ANNOTATION TOMORROW; NOT RUNNING TONIGHT.** The pod generation
artifacts are local, structurally complete, and byte-bound by the V2 annotation guard
manifest. Annotation is permitted only through `ph2_executor.py` with `--authorised`, the
exact guard-manifest path, and its exact SHA-256.

One guarded launch-response was attempted during preflight. It exposed that remaining quota
is nested at `metadata.remaining_quota.remaining_budget`, then stopped before a second call or
any annotation row. Its reported cost was USD 0.028434. The two-event V1 journal and empty
`[]` checkpoint were archived byte-for-byte under `.codex/out/`; the V2 limits subtract that
attempt and cost so the original aggregate ceilings remain unchanged.

The later analysis has one declared limitation: the frozen 100-task manifest contains prompts
but no expected-answer key. Therefore boxed exact-match accuracy is not identifiable from the
current artifacts. The code now refuses the former invalid fallback to boxed-answer presence,
records the damage endpoint as unresolved, and maps affected checkpoint verdicts to
`inconclusive`. This does not block behavioural annotation; it prevents overclaiming later.

## Artifact verification

| Source artifact | Rows | Initial calls | Windowed rows | SHA-256 |
|---|---:|---:|---:|---|
| `results/ph2/battery/base.json` | 700 | 2,039 | 484 | `fab5a0d1366a8158b77c2260147a8daa2928827d3752be784e452759d6983832` |
| `results/ph2/battery/star1.json` | 1,300 | 3,072 | 789 | `32b88bf0ea605b880502588bc8827cde444a3f0e8bff6127f6f8b11a2a7f7492` |
| `results/ph2/battery/deepscaler.json` | 1,300 | 4,077 | 1,025 | `a9112feed4a0f4a8e4b7cc426749aca524f6f365c55c11fe43a2cdd511484658` |
| `results/ph2/injection/base_injection.json` | 300 | 828 | 253 | `9b1474cd4b1f1b931cf2ca4f333223863e03583407cfd3b76ad315f8424d425e` |

Total: 3,600 logical rows and **10,016 deterministic initial API requests**. The task
manifest has 100 unique task ids. Deduplication keys are unique within the source artifacts;
no chain is blank. At final freeze, no annotation process, live journal, or result row exists.

The A4 measurement window is the paragraph-aligned first approximately 3,000 generated
tokens, uniformly applied. The full chain remains attached as `chain_full` for full-chain
length, repetition, truncation, and (where an answer key exists) correctness endpoints.

## Correctness repairs completed before first annotation

1. **CF-18 chunk merge:** Phase-2 now uses zero-overlap chunks and lossless concatenation,
   so genuine repeated sentences such as `Wait.` are not deleted at chunk seams.
2. **Unknown labels:** schema-invalid/unknown labels cause a bounded retry and ultimately an
   unresolved row; they are never silently coerced to `deduction`.
3. **Oversized paragraphs:** a single long paragraph is subdivided at a late sentence or
   whitespace boundary. The observed maximum chunk is 1,200 estimated input tokens.
4. **Partial annotations:** any explicit `annotation_complete=false` row is excluded from
   Phase-2 rates even when it contains some spans; it is never treated as a zero or complete
   observation.
5. **Proxy response parsing and accounting:** all text blocks are joined; absent text remains
   a retryable failure under the bounded allowance. Cost is read from `usage.cost`; remaining
   quota is read from `metadata.remaining_quota.remaining_budget`, with the old flat key only
   as a compatibility fallback.
6. **Damage accuracy:** boxed presence is no longer substituted for boxed exact-match.
   Missing answer keys produce an unresolved damage endpoint and an inconclusive verdict.

## Immutable operational guards

Guard manifest:
`.codex/out/PH2_ANNOTATION_GUARD_MANIFEST_V2_2026-08-10.json`

Manifest file SHA-256:
`0e97468a80a1de3d4ab9b5831dc7e874b21b8cb48d8948eebf31fddf9c6bc69f`

- Model: `eu.anthropic.claude-sonnet-4-5-20250929-v1:0`.
- Planned initial requests: 10,016 exactly; source/path/count drift refuses execution.
- Global retry capacity: 499 total V2 retries, separate from the initial-call pool.
- Per-chunk ceiling: three total attempts (initial plus at most two retries).
- V2 global ceiling: 10,515 attempts.
- V2 approved spend ceiling: USD 274.971566.
- Maximum reported or conservatively committed cost per call: USD 0.05.
- Remaining-quota floor: USD 5.00 after the first observable proxy response.
- Each attempt is reserved with `flock` + `fsync` before network I/O in an append-only
  journal. Interrupted and unaccounted calls are permanently charged at the per-call maximum.
- A successful response without finite numeric cost/quota telemetry stops the run. NaN and
  infinity are rejected in both manifests and journals.
- Prompt text, response text, endpoint, and credentials are never written to the budget
  journal.
- Annotation is sequential with per-row atomic checkpointing and resume by complete row.

Including the archived V1 diagnostic response, the aggregate ceilings remain **10,516 total
attempts and USD 275.00** exactly.

The guard manifest binds the four source artifacts and eight executing code/protocol
artifacts by SHA-256. The loader dry-run reproduced 10,016 calls, matched the complete source
set, and passed the proxy chunk-budget check (1,200 estimated input tokens; 1,620 estimated
worst output against a 2,300-token safety allowance). Wrong hash, absent credentials, absent
`--authorised`, source drift, code/protocol drift, retry exhaustion, spend exhaustion, quota
exhaustion, corrupt journal state, and missing telemetry all fail before a further call.

## Verification

- Focused Phase-2/annotation suite: **162 passed**.
- Full repository suite after final hardening: **954 passed, 9 warnings** in 124.46 seconds.
- The warnings are pre-existing sklearn zero-variance PCA, MPS `pin_memory`, and Python
  import deprecation warnings; none is in the annotation path.
- `python -m py_compile` passed for all touched execution modules.
- `git diff --check` passed.
- Exactly one guarded preflight response was incurred (USD 0.028434); it produced no retained
  annotation and is archived as an explicit diagnostic. No further API calls were made.

## Evidence-status boundary

Phase-2 behavioural results remain **builder-annotator scored** because Sonnet supplies both
the inherited behavioural frame and these verdict labels. This caveat must travel with every
behavioural-rate claim. The no-answer-key limitation must travel with every damage/verdict
claim; it is not evidence that accuracy was preserved or damaged.
