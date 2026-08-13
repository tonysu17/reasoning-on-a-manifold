# DRAFT Amendment A5 — annotation region, coverage completeness, per-call cost bound

**Date:** 2026-08-11
**Status: SEALED — approved by Tony, 2026-08-11.**
Approval covers this amendment as written, the guard manifest with file SHA-256
`80bc7accbbf61b09aace67be945736f663d373ae1e8019a492f6c9d96189ab47`, quarantine
of the 7 coverage-incomplete rows, and reannotation of 3,592 rows / 11,239
chunks under a $275.00 ceiling with a $0.047112 worst-case per-call bound.
**Lineage:** A1, A2, A3, A4 sealed → **A5 sealed**
**Supersedes:** the 2026-08-09 `ANNOTATION_MAX_TOKENS = 4000` operational allowance only.
**Leaves untouched:** the A4 ~3,000-token annotation window; the SEALED 8,192
generation cap; full-chain retention (`chain_full`) for the full-chain damage,
boxed-correctness, length and truncation endpoints; the A3 builder-annotator
caveat; the arms, floors, seeds and decision hierarchy.

---

## A5.1 — The scientific annotation region

**Problem.** The inherited Venhoff Appendix-A prompt ends *"If there is a tail
that has no annotation leave it out."* That delegates a scope decision to the
annotator, per row. In the 2026-08-11 run the annotator labelled post-`</think>`
response text in 4 of 9 eligible rows and omitted it in 5. Because the
behavioural endpoints are ratios over returned spans, that discretion moves the
estimand.

**Rule (deterministic, applied by configuration, never per row).** For each
chain, within the unchanged A4 ~3,000-token paragraph-aligned window:

1. The annotation region **begins** at the start of the window.
2. The annotation region **ends** at the first `</think>`. The marker and all
   text after it — the model's user-facing response — are **outside** the
   region. Where the window contains no marker, the entire window is region.
3. A **trailing** `**Final Answer**` block inside the region is excluded as an
   answer restatement, **only** if it is at most 400 characters. A longer
   trailing block stays in scope and must be annotated.

Rule (3) fails safe by construction: an unusually long trailing block is
*retained* and surfaces as a coverage gap rather than being silently deleted,
which is the failure class this amendment exists to close. The two occurrences
in the frozen checkpoint are 187 and 144 characters.

**Justification for excluding the response.** The four target behaviours
(backtracking, uncertainty-estimation, example-testing, adding-knowledge) are
properties of the reasoning process; the post-`</think>` text is the answer,
not the deliberation. It is also the *majority* historical behaviour (5 of 9
rows omitted it entirely). Excluding it makes the region a strict function of
the chain, so every arm and model is scored on the same object.

**Coupled denominator change (required, not optional).** `bt_per_1k` must
divide by the tokens in the region actually annotated
(`annotated_region_tokens`), not the whole A4 window. Pairing a
region-restricted numerator with a window-wide denominator would deflate every
per-1k rate. Pre-A5 records without the field fall back to the A4 window and
then the full chain, so historical rows keep their historical denominator.

**Versioning.** The rule carries `COVERAGE_RULE_VERSION =
"ph2-annotation-coverage-1"`, persisted on every row. A future rule change is a
migration, never a reinterpretation.

---

## A5.2 — Completeness means coverage, not parse success

**Problem.** `annotation_complete` asserted only that every chunk returned ≥1
parsable span. Fifteen rows satisfied it while five dropped material reasoning.

**Rule.** A row is complete only when **both** hold:

* `annotation_complete` — every chunk's transport and parse succeeded; and
* `annotation_coverage_complete` — a verdict from the **current** rule version
  says the returned spans exhaust the region.

Coverage requires all of:

* whitespace normalised deterministically (runs → single space, ends stripped);
* every span matched **exactly** in the region, by a strictly forward scan;
* repeated occurrences mapped to **successive distinct positions** — genuine
  repeated reasoning is retained as separate observations and is **never**
  deduplicated;
* no span out of source order, none fabricated, none drawn from excluded
  material;
* no uncovered run of ≥ 25 characters anywhere in the region.

Only the exclusions defined in A5.1 (the `</think>` marker and response, and a
short trailing Final-Answer block) are permitted as unannotated material.

A schema-valid but coverage-incomplete chunk/row is **unresolved**, exactly as
a transport failure is. Downstream, the A2 adjunct pairwise-deletes such rows
per endpoint and reports `n_coverage_incomplete` alongside `n_unresolved`; the
`annotate` stage reports the same count in status and provenance. Rows written
before the validator existed carry no verdict and are therefore **not** trusted
as complete — they become eligible for reannotation.

---

## A5.3 — A provable per-call cost bound

**Problem.** The $0.05 per-call maximum was checked only against the cost the
proxy *reported*, i.e. after billing. The one pre-call lever,
`ANNOTATION_MAX_TOKENS = 4000`, was set from an expected echo length and was
documented as "NOT the cost lever". It never bounded a call:

| | worst-case input | worst-case output | worst-case call |
|---|---|---|---|
| Superseded (4,000 tokens) | $0.005112 | $0.060000 | **$0.065112** ✗ |
| **A5 (2,800 tokens)** | $0.005112 | $0.042000 | **$0.047112** ✓ |

**Frozen rate card** (`eu.anthropic.claude-sonnet-4-5-20250929-v1:0`):

* input **$3.00** / Mtok, output **$15.00** / Mtok;
* **`tokens_per_char_upper = 0.4`** (2.5 chars/token) — deliberately more
  conservative than the pipeline's own 4-chars/token chunking estimator, so
  the figure is an upper bound and not a central guess on maths-heavy text.

**Worst-case input derivation.** `chunk_chain` flushes *before* appending a
unit that would exceed the target, and `split_oversized` bounds every unit by
the target, so no chunk exceeds `CHUNK_TARGET_TOKENS`. Chains at or below
`CHUNK_THRESHOLD_TOKENS` bypass chunking, so that path bounds the prompt
instead. Hence
`max_prompt_chars = 4 × max(threshold, target + overlap) + prompt template +
continuation prefix` = **4,260 characters**. This is asserted against what
`chunk_chain` actually emits, over adversarial inputs, in
`test_the_chunk_plan_never_builds_a_prompt_over_the_declared_bound`.

**Parameter changes** (this is the whole of the operational change):

| Parameter | Was | A5 | Why |
|---|---|---|---|
| `CHUNK_THRESHOLD_TOKENS` | 1200 | **800** | smaller input buys output headroom under the bound |
| `CHUNK_TARGET_TOKENS` | 1000 | **800** | as above; also improves the 29-s margin |
| `CHUNK_OVERLAP_TOKENS` | 0 | **0** | unchanged (CF-18 — no overlap merge) |
| `ANNOTATION_MAX_TOKENS` | 4000 | **2800** | ≥ worst-case echo (1,080 est.) and ≤ $0.05 |

The 29-s API-Gateway margin **improves**: worst-case output estimate falls from
1,620 to 1,080 tokens against the 2,300-token budget.

**Enforcement (fail-closed, before network I/O).**

1. At manifest load: the declared rate card, `max_output_tokens` and
   `max_prompt_chars` must yield a worst case ≤ `max_cost_per_attempt_usd`, and
   that ceiling must itself be ≤ the hard project limit of $0.05. Otherwise the
   guard **refuses all API calls**.
2. Before every call: `assert_attempt_cost_bound` re-prices the **actual**
   prompt and the **actual** output allowance. A breach raises
   `AnnotationCostLimitError` with **no HTTP request issued and no attempt
   reservation consumed** — a refused call never happened.
3. `record_response` retains its post-hoc check as defence in depth.

**Retained unchanged:** global spend ceiling, retry ceiling, per-scope attempt
ceiling (hard limit 3), remaining-quota floor, reservation-before-I/O
append-only journal, fail-closed missing-telemetry behaviour.

**Spend continuity.** A new manifest necessarily starts a new journal (journal
events are bound to the manifest hash). `prior_committed_spend_usd` seeds the
committed total so a manifest replacement can never reset the cumulative
ceiling to zero. It is set to **$1.180969**, the full lineage made explicit:

| Component | USD |
|---|---|
| V1 attempt (telemetry-parser diagnostic, 0 rows committed) | 0.028434 |
| V2 reported cost (45 priced outcomes) | 1.052535 |
| V2 dangling reservations, 2 × $0.05 conservative charge | 0.100000 |
| **Total carried forward** | **1.180969** |

The V2 manifest had instead netted V1's $0.028434 out of its ceiling
(274.971566). A5 restores the ceiling to the true pre-approved **$275.00** and
carries every prior commitment explicitly, so the arithmetic is auditable in
one place rather than folded into a ceiling. Remaining headroom: **$273.819031**.

---

## A5.4 — Treatment of the existing 15 rows

The existing checkpoint is **not** mutated by this amendment. On an approved
resume, `src/annotation_quarantine.py` performs a versioned migration that:

* writes the 7 coverage-incomplete rows **verbatim**, with their verdicts and
  reasons, to `results/ph2/annotation/quarantine/base.v1.json`;
* writes a manifest recording the checkpoint hash before and after, the rule
  version, and the per-row reason for eligibility;
* leaves the 8 coverage-complete rows in place, so they are never re-billed.

It refuses unless explicitly approved **and** the checkpoint still hashes to
the value the plan was computed from.

---

## A5.5 — Deviation register

* Annotator remains Claude Sonnet 4.5 (A3); the builder-annotator caveat is
  unaffected and still travels with every behavioural verdict.
* The Venhoff Appendix-A prompt text is **unchanged**. A5 constrains the
  *scoring region*, not the prompt; the prompt's tail clause is now
  post-hoc enforced rather than trusted.
* Reported behavioural rates are computed over the reasoning region only. This
  must be stated wherever Phase-2 behavioural rates are reported, and is not
  comparable term-for-term with any earlier full-window rate.
