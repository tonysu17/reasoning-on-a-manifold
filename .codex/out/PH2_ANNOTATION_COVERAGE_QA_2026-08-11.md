# Phase-2 annotation — coverage failure QA and evidence disposition

**Date:** 2026-08-11
**Status:** DIAGNOSIS + IMPLEMENTATION COMPLETE — **NOT AUTHORISED TO RESUME**
**Spend in this pass:** $0.00 (zero API calls, zero pod/GPU activity)

---

## 1. Preserved evidence (immutable)

The halted run's artefacts are audit evidence. They were **not** overwritten,
cleaned, deleted, or reinterpreted in this pass. Hashes re-verified at the end
of the pass and byte-identical to the values at its start:

| Artefact | SHA-256 |
|---|---|
| `results/ph2/annotation/base.json` | `4273ff37ed62881c3aeb9aeed5ada525bcfe28796687400923626ea0887efb61` |
| `results/ph2/annotation/api_attempt_journal.jsonl` | `e41c7df0aa218e8faa22f315accf923b1fb3b3c487c04ee62b978adbbcbf9f38` |
| `results/ph2/battery/base.json` (source) | `fab5a0d1366a8158b77c2260147a8daa2928827d3752be784e452759d6983832` |

Checkpoint: 15 rows, 632 spans. Journal: 47 reservations, 45 outcomes, 2
conservatively charged dangling reservations. Reported V2 cost $1.052535.

`results/ph2/annotation/base.json.bak` (11:15, 143,727 B) is an earlier
automatic snapshot from `src.config.backup_existing` and is likewise untouched.

---

## 2. Failed-coverage disposition

Structural and provenance checks passed and are not in dispute. Semantic
coverage did not. Re-derived independently by
`src.annotation_coverage.validate_coverage` against the frozen bytes:

| Row | spans | `</think>` | coverage of reasoning region | defect |
|---|---|---|---|---|
| MATH_117 | 114 | no | 0.990 | — |
| MATH_118 | 16 | yes | 0.995 | — |
| **MATH_119** | 52 | no | 0.957 | 451-char gap: a **4th occurrence** of a looped sentence annotated only 3 times |
| **MATH_120** | 40 | yes | — | 4 spans drawn from post-`</think>` response text |
| **MATH_121** | 53 | no | 0.891 | 2 gaps; unique reasoning ("Alternatively, maybe a scenario where…") dropped |
| MATH_122 | 14 | yes | 0.996 | — |
| MATH_123 | 20 | yes | 0.994 | — |
| **MATH_124** | 60 | no | 0.876 | **1,404-char** internal omission (a full mod-8 symmetry derivation) |
| MATH_125 | 34 | yes | 0.994 | — |
| **MATH_126** | 43 | yes | 0.972 | 189-char gap **and** 10 post-`</think>` spans |
| SPAT_117 | 30 | no | 0.995 | — |
| SPAT_118 | 10 | yes | 0.990 | — |
| **SPAT_119** | 39 | yes | 0.997 | 2 post-`</think>` spans |
| **SPAT_120** | 51 | yes | 0.967 | unique omitted reasoning **and** 2 post-`</think>` spans |
| SPAT_121 | 56 | no | 0.994 | — |

* **Gap rows (5):** MATH_119, MATH_121, MATH_124, MATH_126, SPAT_120 — exactly
  the set named in the operator audit.
* **Post-`</think>` inconsistency:** annotated in 4 rows (MATH_120, MATH_126,
  SPAT_119, SPAT_120), omitted in 5 (MATH_118, 122, 123, 125, SPAT_118).
* **All 15 rows carry `annotation_complete: true`.**
* **Disposition: 7 rows coverage-incomplete, 8 rows coverage-complete** under
  the draft A5 region rule. No row's spans are fabricated or out of order.

This table is pinned as a regression test
(`tests/test_annotation_coverage.py::test_frozen_checkpoint_reproduces_the_audited_defects`),
which also asserts the checkpoint hash, so the evidence cannot drift silently.

---

## 3. Root cause

Two independent defects, both of the same shape: **a guarantee was asserted
from an expected value rather than derived from a bound.**

### 3.1 Completeness was a transport predicate, not a scientific one

`annotate_chain` returns `complete = not any_failed`, where a chunk "succeeded"
if `parse_annotation_response` yielded ≥1 span. Nothing compared the returned
spans against the source region. The inherited Venhoff prompt ends with
*"If there is a tail that has no annotation leave it out"*, which grants the
annotator explicit per-row discretion over the tail — and the annotator used it
inconsistently (4 rows in, 5 rows out).

This matters because the estimand is a ratio over returned spans:
`src.evaluation.behaviour_fraction` divides by `len(annotated_chain)`, and
`bt_per_1k` divides by an annotated-token count that included text the
annotator may or may not have labelled. A discretionary omission therefore
moves **both** the numerator and the denominator. MATH_119 is the sharpest
case: three of four occurrences of a looping sentence were kept and the fourth
dropped, in a chain whose scientific content *is* the loop. (MATH_117 carries
one sentence repeated 31 times — repeat counts are signal here, not noise.)

### 3.2 The $0.05 per-call ceiling was never enforceable

`AnnotationAttemptGuard.record_response` compares the **reported** cost to
`max_cost_per_attempt_usd` — i.e. after the call has been billed. The only
pre-call lever was `ANNOTATION_MAX_TOKENS = 4000`, chosen (2026-08-09) from an
*expected* ~1,620-token echo, with the explicit note that the cap "is NOT the
cost lever". At the frozen rate card the 4,000-token allowance alone prices at
**$0.060**, and the whole call at **$0.065112** worst case — so no combination
of declared parameters bounded a call at $0.05. The observed $0.055686 is
inside that unbounded envelope; the guard detected it only post-hoc.

---

## 4. What changed (all zero-spend, none of it executed)

See `PH2_ANNOTATION_COVERAGE_AND_COST_CORRECTION_2026-08-11.md` for the draft
dated amendment (**A5**), which is **NOT sealed** and requires Tony's approval.

* `src/annotation_coverage.py` (new) — deterministic annotation region +
  semantic coverage validator.
* `src/annotation_quarantine.py` (new) — versioned, approval-gated migration
  that preserves original rows.
* `src/annotation_budget.py` — frozen rate card; `max_output_tokens` /
  `max_prompt_chars` declared in the manifest; worst-case bound refused at load
  **and** before every call; prior committed spend carried across manifests.
* `src/annotation.py` — chunk plan resized so the bound holds; pre-I/O cost
  check; coverage verdict written on every row; resume keyed on coverage.
* `src/ph2_stages.py` — A5 region flag; region-scoped per-1k denominator;
  coverage-incomplete rows pairwise-deleted **and counted** in the A2 adjunct.
* `ph2_executor.py` — `ANNOTATION_MAX_TOKENS` 4000 → 2800; new guard
  parameters; coverage counts in status and provenance.

Guards retained unchanged: global spend ceiling, retry ceiling, per-scope
attempt ceiling (hard limit 3), quota floor, reservation-before-I/O journal,
fail-closed telemetry. All 17 pre-existing budget tests pass unmodified in
substance.

---

## 5. Proposed executable manifest

`.codex/out/PH2_ANNOTATION_GUARD_MANIFEST_A5_2026-08-11.json` — **proposed, not
authorised for use.** Dry-loaded through `AnnotationAttemptGuard.from_manifest`
with zero attempts reserved and zero calls made.

| | |
|---|---|
| Internal SHA-256 | `2d01388a7427f2231a998b9e75aab78ee02358f8cb8c56215c582d7edc723b1f` |
| File SHA-256 | `80bc7accbbf61b09aace67be945736f663d373ae1e8019a492f6c9d96189ab47` |

Supersedes `PH2_ANNOTATION_GUARD_MANIFEST_V2_2026-08-10.json`.

---

## 6. Open decision for the owner

The corrected plan's **conservative** cost projection exceeds the pre-approved
ceiling. At the pipeline's own 4-chars/token estimator the full 11,260-call
plan projects to **$189.73**; at the conservative 2.5-chars/token bound used
for the per-call guarantee it projects to **$303.57**, against
`MAX_PREAPPROVED_SPEND_USD = $275.00`. The guard will halt cleanly at the
ceiling rather than overrun, so the risk is an **incomplete run**, not an
overspend. This needs an explicit decision (raise the ceiling, or reduce
scope) and is called out in the approval package.
