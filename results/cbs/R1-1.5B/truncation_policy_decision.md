# P0.4 — Chain truncation policy decision (R1-1.5B)

**Status**: TEMPLATE awaiting Tony's decision.
**Halt point**: synthesis §P0.4 / build-now step 3.
**Date drafted**: 2026-05-27
**Authoritative source**: `results/quality_reports/chains_R1-1.5B.md`

---

## The finding

The R1-1.5B chain corpus (`data/chains_R1-1.5B.json`, 1000 chains, generated
at `max_new_tokens=8192`) has a severe truncation problem:

| Quantity | Value |
|---|---|
| Chains at max_tokens | **502 (50.2%)** |
| Chains *without* closing `</think>` | **499 (49.9%)** |
| Chains hit-max **AND** no closing `</think>` (truncated mid-thinking) | 499 |
| Chains short **AND** closing `</think>` (clean finish) | 498 |

Per category:

| Category | % at max | clean half? |
|---|---|---|
| lateral_thinking | 95% | mostly truncated |
| spatial_reasoning | 71% | mostly truncated |
| pattern_recognition | 66% | mostly truncated |
| verbal_logic | 62% | majority truncated |
| probabilistic_thinking | 59% | majority truncated |
| mathematical_logic | 55% | majority truncated |
| creative_problem_solving | 44% | mixed |
| scientific_reasoning | 24% | mostly clean |
| causal_reasoning | 22% | mostly clean |
| systems_thinking | 4% | clean |

This is a confound for **every** downstream M2 / M3 / M4 test:

* M2 geometry: a sentence near a hard token cap may have unusual activations
  for reasons unrelated to CBS tier.
* M3 trajectory: arc-length distributions, curvature, and "trajectory end"
  features all change if chains are mid-thought truncated.
* M4 matched pairs: correct-vs-incorrect chains may be unbalanced for
  truncation, biasing the comparison.

---

## Three options (synthesis §P0.4)

### (a) Re-generate at `max_new_tokens=16384` for the truncated half

* **Cost**: ~15 cluster GPU-hours for ~500 chains × 16k tokens.
* **Wall-clock**: ~3 days (Phase 2 re-run + Phase 3 re-annotation).
* **Quality**: best — cleanest cohort, no confound.
* **Risk**: some chains may still truncate at 16k (lateral_thinking median was 8192 — the true distribution is unknown above that cap). The 16k cap may need to become 24k after one round, repeating the loop.
* **Blocking**: yes — postpones Phase 4 by ~3 days, then re-runs M1
  annotation (~$30 API).

### (b) Stratify all M2–M6 analyses by `truncated: bool`  **← RECOMMENDED**

* **Cost**: 0 GPU-hours, 0 API.
* **Wall-clock**: 0 days.
* **Quality**: clean-half is the primary analysis cohort (498 chains);
  truncated-half is reported as a sensitivity rerun. Effect sizes from the
  two cohorts are compared explicitly; divergence is itself a finding.
* **Risk**: power loss on tier-3 effects from the truncated half if those
  effects come predominantly from truncated chains (which is *a priori* odd).
* **Why recommend for the build phase**: does not gate any code construction.
  The truncation flag is a one-line column in the M3 summary parquet and
  every M2/M3/M4 runner already accepts a `truncated` filter via
  `--truncation-policy stratify`.

### (c) Filter to the clean half (`closing_think=True`, ~498 chains)

* **Cost**: 0.
* **Wall-clock**: 0.
* **Quality**: cleanest single-cohort framing.
* **Risk**: 50% sample-size hit. M3's per-sentence regression and M4's
  matched-pair Wilcoxon both lose power. M4's tier-3 base rate falls to
  ~$498 \times 0.05 / chain$ ≈ uninterpretably few tier-3 sentences if the
  per-chain rate is at the synthesis-§P0.2 floor.

---

## What Tony decides

Pick one of (a), (b), (c) and write the chosen letter and a 1-2 sentence
rationale under the line below. The decision feeds:

* M1 pilot sample size (P0.2): if (a), wait for re-generation before running
  the pilot; if (b) or (c), pilot runs immediately on the existing corpus
  with stratification / filter as appropriate.
* `--truncation-policy` flag default in `10_trajectory_build.py` and
  `11_trajectory_analysis.py`.
* Every M2 / M3 / M4 completion report's sample-size disclosure.

### Decision

```
Choice: (b) Stratify by truncated:bool
Rationale: Does not gate the build phase; the truncated cohort becomes a
           sensitivity rerun rather than a regenerate-and-block decision.
Date: 2026-05-27
Recorded-by: Claude Code, via AskUserQuestion at the P0.4 halt point.
```

Downstream defaults set accordingly:
* `10_trajectory_build.py --truncation-policy stratify`   (default)
* `11_trajectory_analysis.py` will report per-stratum effect sizes.
* M2 / M3 / M4 completion reports must include the per-stratum n.

After filling in, Claude Code resumed at P0.2 (anchor curation).

---

## Implementation notes for downstream milestones

The `truncated` per-chain flag is derivable from
`results/quality_reports/chains_R1-1.5B.md` cross-tabulation rule:

```python
truncated = (n_tokens >= max_new_tokens) and not chain.rstrip().endswith("</think>")
```

A small helper `src/cbs/cohort.py::is_truncated(chain)` can be added at M3
build-time so every runner downstream of P0.4 reads the same definition.

---

**Halt** — Claude Code will not proceed to P0.2 until this file contains
Tony's choice. Recommend (b) for the build phase.
