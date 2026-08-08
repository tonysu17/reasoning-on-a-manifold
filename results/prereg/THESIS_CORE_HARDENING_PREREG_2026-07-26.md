# Thesis core geometry hardening preregistration — 26 July 2026

## 1. Status, authority, and freeze

**Status: PROSPECTIVE / UNRUN.** This document freezes the Route A core
geometry hardening for RQ1 and RQ2 only. It is a methods commitment, not an
empirical result. No pilot, smoke test, benchmark, permutation, resampling,
estimator run, re-extraction, or other empirical analysis was executed while
preparing it.

**Freeze timestamp:** 2026-07-26, Europe/London. The statistical specification
is frozen when the author signs Section 18 and before any planned pilot output
is opened. The freeze is conditional on G1 and G2 in
`thesis/_planning/THESIS_REVISION_PLAN_2026-07-26.md`; signing this document
does not itself pass either gate.

**Working-tree provenance at drafting:** repository HEAD
`5fd66637f34981c6e79109c6f0ecbee04d71ff5b`; the working tree was dirty with
pre-existing tracked and untracked changes unrelated to this document.
Therefore the HEAD is a reference point, not a claim that this file or its
inputs came from a clean committed state. The execution manifest must record
its own commit, dirty flag, and complete input hashes.

This preregistration is subordinate to:

1. `thesis/_planning/THESIS_REVISION_CHARTER_2026-07-26.md`;
2. `thesis/_planning/THESIS_CLAIM_ARTIFACT_LEDGER_2026-07-26.md`; and
3. `thesis/_planning/THESIS_REVISION_PLAN_2026-07-26.md`.

The dated critique,
`thesis/_planning/THESIS_CRITIQUE_2026-07-26.md`, motivates the controls but is
not new evidence. If this document conflicts with the charter on scope or
claim status, the charter controls. If it conflicts with the ledger on an
existing artefact or provenance fact, the ledger controls. Unresolved
provenance remains unresolved; this document does not repair it by assertion.

**RQ1:** Do annotation-indexed reasoning activations show low estimated
intrinsic-dimensional occupancy beyond chain and finite-sample controls?

**RQ2:** Is the observed concentration behaviour-specific, and is there
evidence for nonlinear or curved structure?

The only licensed endpoint is estimated activation-cloud occupancy and the
bounded diagnostics named below. There is no hypothesis that an activation
cloud is a linear subspace, no hypothesis of global flatness, and no hypothesis
that post-training created the geometry.

## 2. Known-data disclosure and amendment status

This is a hardening preregistration after existing outputs were inspected, not
a claim of a blind original preregistration. The ledger records amended,
current non-confirmatory correlation-dimension estimates and a previously
executed one-sentence-per-chain sensitivity at old robustness layers
14/14/27/17 in
`results/robustness/R1-1.5B/geometry_robustness.json`. It also records a PCA
variance-concentration null, not a correlation-dimension null, at zero-based
layers 11, 14, 17, 20, and 27 in
`results/pca/R1-1.5B/null_pvalues_per_layer.json`; a three-model-annotator PCA
specificity result at layers 12 and 16 in
`results/robustness/specificity_secondary_annotator.json`; and an L27
mean/first/last PCA pooling sweep in
`results/pooling_sweep/R1-1.5B/pooling_sweep.json`. None is the prospective
direct correlation-dimension null defined here.

The current activation metadata says occurrence-aware extraction,
mean-pooling, one preceding token plus up to ten execution tokens, and
`clip_window_to_sentence_end=false` in
`data/activations/R1-1.5B/metadata.json` and
`data/activations/R1-1.5B/row_index.json`. The main corpus contains 37,851 raw
target rows across 993 chains; the ledger gives per-behaviour raw,
analysis-deduplicated, and chain counts. The model identifier is recorded, but
an immutable upstream checkpoint revision/weight hash is unresolved. Multiple
historical provenance records have `git_commit: null`; this uncertainty must
travel with any result.

The existing 80% point-subsampling intervals in `src/intrinsic_dim.py` are not
bootstrap confidence intervals. Any such interval used here is called a
**stability band**, never a CI, because sampling without replacement at
0.8 of the available units estimates estimator stability rather than a
standard n-out-of-n bootstrap sampling distribution.

## 3. Frozen scientific unit, observations, estimands, and target population

The independent scientific unit is the generated reasoning chain/task,
identified by `chain_id`/`task_id`. A labelled sentence-span activation is an
estimator observation nested within a chain, not an independent scientific
unit. Every output cell must report both unique-chain count and sentence-row
count before and after duplicate handling.

The primary estimand is:

> The correlation dimension of the activation span obtained from an equally
> weighted random chain containing the target model-annotated behaviour, under
> the frozen L27 canonical representation.

Operationally, each eligible chain contributes at most one target-labelled
activation, selected by the deterministic rule in Section 7. This gives equal
chain weight. The primary target population is the 1,000 generated chain
prefixes observed under the 8,192-token cap, not an unobserved population of
complete unconstrained solutions.

Secondary estimands are:

- the sentence-weighted correlation dimension after the same frozen
  duplicate rules;
- the equal-chain participation ratio (PR);
- representation-specific equal-chain correlation dimension and PR;
- complete-only, truncated-only, and category-matched equal-chain estimates;
- the local/global dimension-ratio diagnostic at the controlled layer; and
- existing PCA variance concentration, reported separately and never used as
  a substitute for a correlation-dimension null.

The direct within-chain label permutation and the one-sentence-per-chain
stability analysis are distinct. The permutation asks whether the target
labels select unusually low correlation dimension while preserving each
chain's label counts. The stability analysis asks how the point estimate moves
when chains and within-chain occurrences are sampled. Neither substitutes for
the other, and neither is to be described as the other.

## 4. Frozen model, labels, representation, and site

The sole primary representation model is
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (`R1-1.5B`; 28 transformer layers,
hidden width 1536). The unresolved immutable checkpoint revision/weight hash
will be reported as unresolved unless recovered before execution without
looking at new results.

The primary annotator is the repository's Sonnet arm, configured as
`eu.anthropic.claude-sonnet-4-5-20250929-v1:0` through the recorded proxy.
The four target labels, in their frozen order, are:

1. `backtracking`;
2. `uncertainty-estimation`;
3. `example-testing`; and
4. `adding-knowledge`.

The primary layer is **zero-based L27**, fixed for all four behaviours. L27 is
not claimed to be causal or uniquely optimal. It is frozen because the
declared Methods analysis used common L27 and because a single common layer
avoids outcome-informed per-behaviour PR-trough selection. The L27 matrix
exists for all four labels needed by the conditional within-chain
permutation. L16 is retained only as a secondary controlled robustness site
inside the three-model-annotation family and the separate curvature
diagnostic; it cannot rescue or redefine a primary L27 result.

The canonical representation is the residual activation mean over one token
preceding the annotated behaviour onset plus up to ten execution tokens. The
canonical window is **not clipped** at sentence end, matching
`clip_window_to_sentence_end=false`. Activations are converted to float64 for
distance and covariance calculations. They are not centred, z-scored,
unit-normalised, whitened, projected through PCA, or otherwise transformed
before correlation dimension. Euclidean distance is used.

The primary input matrices are exactly:

- `data/activations/R1-1.5B/backtracking_layer27.npy`;
- `data/activations/R1-1.5B/uncertainty-estimation_layer27.npy`;
- `data/activations/R1-1.5B/example-testing_layer27.npy`; and
- `data/activations/R1-1.5B/adding-knowledge_layer27.npy`.

They must be aligned through
`data/activations/R1-1.5B/row_index.json`; proxy chain identifiers are
prohibited. The annotated and truncation inputs are
`data/annotated_R1-1.5B.json` and `data/chains_R1-1.5B.json`.
If any primary L27 matrix is absent or its frozen Section 16 hash does not
match, the primary family is **UNRESOLVED — DO NOT EXECUTE**; no L16 or
behaviour-specific layer may substitute.

## 5. Frozen hypotheses

### H1 — RQ1: descriptive low operational correlation dimension

Under the frozen equal-chain, mean-pooled, unclipped Sonnet L27 representation,
each of the four behaviour clouds has a valid correlation-dimension point
estimate `0 < cdim <= 10.0`. The numeric bound is frozen before output as an
operational descriptive criterion: 10 is less than 1% of the ambient width
1536 and covers the ledger-disclosed earlier estimates without treating those
estimates as confirmatory evidence. H1 passes only if all four cells are valid
and meet the bound; one to three is mixed; zero is fail; any missing/invalid
cell is UNRUN. H1 is descriptive and does not use a null p-value.

### H2 — RQ2: four-cell behaviour specificity at L27

For each of the four primary Sonnet/L27/mean/unclipped cells, the observed
equal-chain cdim is lower than its within-chain label-permutation null.
Family-level support requires all four Holm-adjusted lower-tail p-values to be
at most 0.05 and unchanged in pass/fail status under the diagnostic seed.
One to three passing behaviours is mixed, zero is fail, and any missing or
invalid cell makes H2 UNRUN rather than shrinking the family.

The five-depth Sonnet and three-model-annotation families remain secondary
robustness analyses of H2, not hypotheses that can rescue H2. A behaviour is
depth-robust only if at least four of five layer-specific tests pass their
20-cell family Holm correction. It is annotator-robust only if L16 passes for
all three model annotations under the 24-cell family correction; L12 is a
second controlled site. Outcomes are reported by behaviour, layer, and model
annotation.

### H3 — RQ1/RQ2: chain and truncation robustness

For every primary behaviour, at least 475 of 500 chain-subsampling replicates
must be finite, the full-sample cdim and PR must lie inside their respective
2.5%–97.5% stability bands, and the finite-draw median for each statistic
differs from its full-sample statistic by at most 25% on the relative scale.
In addition,
complete-only and truncated-only cdim and PR must each differ by at most 25%
from the combined estimate, and the two cdim stability bands and two PR
stability bands must overlap. Where category-rate imbalance exceeds 10
percentage points, the mandatory matched-complete/matched-truncated diagnostic
must also satisfy Section 11. All four behaviours must pass both legs and any
required matching diagnostic for H3 pass; the exhaustive pass/mixed/fail/UNRUN
operator is frozen in Section 11.

### H4 — RQ1/RQ2: matched pooling/window sensitivity

On one common complete-case occurrence and chain set at L27, all six
mean/first/last by clipped/unclipped cells must produce finite equal-chain cdim
and PR. For each behaviour, every cell's cdim and PR must differ by no more
than 25% from that behaviour's mean/unclipped value computed on the same
common set, and all six lower-tail cdim tests must pass the single 24-cell Holm
family at alpha 0.05. H4 passes only if all 24 cells satisfy all conditions;
partial valid cells are mixed only when all 24 were run validly; any absent,
misaligned, non-finite, or under-resampled cell makes H4 UNRUN. No
"available-cell" or reduced-grid pass is permitted.

H1–H4 concern operational activation-cloud statistics. None asserts a linear
subspace, global flatness, or a post-training origin.

### Separate narrow curvature diagnostic — not H1–H4

Curvature is a prespecified, non-confirmatory L16 diagnostic only. It compares
the local/global dimension-ratio chain stability draws with paired same-size
random-row controls under Section 13. It cannot pass, fail, rescue, or redefine
H1–H4, and it is not a test of global flatness.

## 6. Analysis families and multiplicity

The **primary H2 family** contains four tests: four behaviours × Sonnet × L27
× canonical mean/unclipped representation. Holm's step-down family-wise error
correction is applied at alpha 0.05 across these four p-values. H1 has four
descriptive cells and no multiplicity-adjusted inferential test.

The **secondary five-depth family** contains 20 tests: four behaviours ×
Sonnet × zero-based layers {11, 14, 17, 20, 27}, canonical mean/unclipped
representation. Holm correction is applied across all 20 tests.

The **secondary three-annotator family** contains 24 tests: four behaviours ×
three model annotations {Sonnet, Qwen3-235B, Nova-Pro} × layers {12, 16},
canonical mean/unclipped representation. Holm correction is applied across all
24 tests. These are three model annotations, not statistically independent
human annotators. Extraction-vintage differences prohibit quantitative
cross-annotator effect-size comparisons.

The **H4 pooling/window family** contains 24 tests: four behaviours × three
pooling operators × two window rules, all at Sonnet/L27 on one common
complete-case set. Holm correction is applied once across all 24 lower-tail
p-values. Truncation, PR, sentence-weighted, stability-band, and curvature
analyses are otherwise scope/robustness analyses under their frozen thresholds.
None enters or can reverse the H2 four-cell family. No selective sub-family
may be reported as if it had been the primary family.

## 7. Duplicate handling and equal-chain construction

Inputs first pass `src/row_provenance.py` alignment checks. Each row's immutable
occurrence key is `(behaviour, chain_id, annotation_index, char_offset,
token_start)`. Exact repeated occurrence keys are collapsed before label
permutation by retaining the lexicographically first fully aligned record.
Exact activation-vector duplicates are then collapsed deterministically using
the lexicographically smallest occurrence key.

Any exact activation duplicate group spanning more than one chain or more than
one observed label is recorded separately. If retaining one member would
remove an independent chain or resolve a label conflict arbitrarily, the cell
fails validation and is not analysed until a dated amendment states the
non-outcome-based remedy. Duplicate fractions, group counts, cross-chain
groups, and cross-label groups must appear in every output.

Beginning with the 2026-07-26 23:58 BST pre-result full-run amendment, the
following additional deterministic rule applies after repeated-occurrence
collapse and before any estimator or permutation in every registered CPU
family. If an exact-vector group spans observed labels but all members have
the same `(chain_id, char_offset, token_start, n_positions)`, remove **every**
member of that group. No label is selected as a winner. Cross-chain groups and
cross-label groups spanning more than one such extraction-window key remain
hard failures. The rule was first applied to the amended B=25 resource-pilot
retry and is now extended prospectively to the full CPU analyses on the same
registered extraction vintages. Any new representation or annotator vintage
must pass its own pre-result provenance audit before this rule is applied.

For a target label, each eligible chain contributes one row. Selection uses one
canonical digest protocol in every family. Each string field is converted to
Unicode NFC; integer fields are base-10 JSON integers; missing required fields
are a validation failure. The payload is the JSON array:

`["thesis-core-hardening-v1",family_code,annotator_code,layer,behaviour_code,pooling_code,window_code,truncation_code,chain_id,annotation_index,char_offset,token_start,replicate_index,scope_key]`.

It is serialized exactly by Python
`json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)`
and UTF-8 encoded without a byte-order mark or trailing newline. Selection
uses the lexicographically lowest lowercase
`hashlib.sha256(serialized_bytes).hexdigest()`. `replicate_index=-1` denotes a
full-sample point statistic; stability replicates are zero-based 0–499.
`scope_key=""` except that category-matched truncation uses the NFC category.
Primary, depth, annotator, truncation, pooling/window, and curvature selections
all use their codes from the registry below; there is no pipe-delimited or
runtime-hash alternative.

This deterministic, activation-blind selection is applied identically to the
observed and every permuted label vector. It prevents long chains from receiving
more weight and avoids adding analyst discretion. A chain with no target row
after permutation does not contribute; within-chain permutation preserves
per-chain label counts, so its eligibility should not change. Any change is a
hard validation failure.

The sentence-weighted secondary estimate uses every remaining deduplicated row.
It is explicitly labelled sentence-weighted and cannot replace the primary
equal-chain estimate.

## 8. Correlation-dimension statistic

The frozen statistic is a thin named wrapper around
`src/intrinsic_dim.py::correlation_dimension_estimate` returning `.estimate`
with:

- `n_radii=20`;
- `random_state=42`;
- `n_bootstrap=0`;
- `subsample=2000`;
- float64 input; and
- no preprocessing beyond the duplicate/alignment rules above.

Because the primary construction supplies at most one point per chain and
there are 993 chains in the corpus, the estimator's internal 2,000-point
subsample should not activate. The implementation uses Euclidean pairwise
distances, radii from the 2nd to the 98th pairwise-distance percentiles, 20
log-spaced radii, and a linear slope over the middle 20%–80% of valid
log-radius/log-correlation-sum points. The wrapper must reject fewer than 100
points, non-finite values, non-positive slopes, degenerate distances, fewer
than three valid fit radii, or a non-finite fitted slope.

Participation ratio uses the same ordered equal-chain point matrix
`X` of shape `N × 1536`, converted to float64. PR alone is column-centred:
`Xc = X - mean(X, axis=0, dtype=float64)`. Compute
`s = svd(Xc, full_matrices=False, compute_uv=False)` and covariance
eigenvalues `lambda_i = s_i**2 / (N - 1)` for every returned singular value,
with no component cap or tolerance-based truncation. The frozen statistic is
`PR = (sum_i lambda_i)**2 / sum_i(lambda_i**2)`. `N<2`, any non-finite
quantity, or a zero denominator is invalid. This centring is part of the PR
definition only; cdim remains uncentred and untransformed. PR is descriptive
and is never called correlation dimension.

## 9. Monte Carlo conditional within-chain lower-tail permutation test

For each cell, pool the four aligned label matrices at the same model,
annotator, layer, pooling mode, and window. Within each chain independently,
permute the four target labels across eligible rows, preserving that chain's
target-label counts. No global or cross-chain shuffle is a primary null.

This is a Monte Carlo test, not enumeration of the exact permutation
distribution. The observed and randomized statistics use the deterministic
equal-chain construction in Section 7 and the point statistic in Section 8.
The alternative is unusually low dimension. Define the conditional assignment
set `V` as all within-chain label assignments that preserve every chain's label
counts and for which the prespecified Section 8 cdim is defined and valid. The
conditioning is necessary because the registered estimator has a
prespecified undefined region; it is not an outcome-dependent filter. The
observed assignment must itself be in `V`, or the cell and family are UNRUN.

Randomized assignments are proposed uniformly by independent within-chain
permutations. Rejection sampling is permitted **only** when the proposed
assignment's Section 8 estimator is invalid. Identity/no-op assignments,
including an assignment identical to the observed labels, are valid members
of `V`: compute and retain their statistic. Pre-analysis alignment, duplicate,
or label-count failures invalidate the cell/family; they are not
assignment-level rejection reasons.

With `B=2500` retained randomized assignments from `V`, the Phipson–Smyth
smoothed lower-tail Monte Carlo p-value is unambiguously:

`p = (1 + sum_{j=1}^B I(cdim_j <= observed_cdim)) / (B + 1)`.

The `+1` numerator and denominator include the valid observed assignment as
the reference draw. Any retained randomized identity assignment is also
counted among the `B` comparisons when its statistic is at most the observed
statistic; it is never discarded as a no-op.

The registered base seed is 20260726; the Monte Carlo diagnostic base seed is
20260727. Seed codes are one-based: family codes are primary-H2=1,
five-depth=2, three-annotator=3, truncation=4, pooling/window-H4=5, and
curvature=6; purpose codes are permutation=1, chain-stability=2,
truncation-match=3, curvature-pair=4; annotator codes are Sonnet=1,
Qwen3-235B=2, Nova-Pro=3; behaviour codes follow the frozen order 1–4;
pooling codes mean=1, first=2, last=3; window codes unclipped=1, clipped=2;
truncation codes combined=1, complete=2, truncated=3, category-matched=4.
Code 0 means "not applicable" for a coordinate not used by that family.
Codes are one-based while `cell_index` and replicate/attempt indices are
explicitly zero-based.

Every permutation attempt creates its RNG independently as
`default_rng(SeedSequence([base_seed, 1, family_code, cell_index, attempt_index]))`,
where `attempt_index` starts at 0. Thus a rejected attempt cannot shift later
streams. Chain stability uses
`default_rng(SeedSequence([20260726, 2, family_code, cell_index, replicate_index]))`.
No language-runtime hash function is used for seeds.

The complete zero-based `cell_index` registry is:

| Family | Zero-based indices | Exact mapping |
|---|---:|---|
| Primary H1/H2 L27 and its stability | 0–3 | `b`, where behaviour rank `b=0..3` follows Section 4 |
| Five-depth Sonnet | 4–23 | `4 + 5*b + d`, depth rank `d=0..4` maps to `{11,14,17,20,27}` |
| Three-annotator | 24–47 | `24 + ((a*2 + l)*4) + b`, annotator rank `a=0..2` maps Sonnet/Qwen3-235B/Nova-Pro and layer rank `l=0,1` maps L12/L16 |
| Truncation L27 | 48–63 | `48 + 4*t + b`, stratum rank `t=0..3` maps combined/complete/truncated/category-matched |
| Pooling/window H4 L27 | 64–87 | `64 + ((p*2 + w)*4) + b`, pooling rank `p=0..2` maps mean/first/last and window rank `w=0,1` maps unclipped/clipped |
| Secondary curvature L16 | 88–91 | `88 + b`; chain/control members share this index and use paired child-stream codes 0/1 |

Estimator kind (cdim, PR, or curvature ratio) is recorded separately and never
creates an implicit `cell_index`. Any cell outside this registry is
`UNRESOLVED — DO NOT EXECUTE` until a signed pre-result amendment adds it.

At least half of all eligible chains must contain more than one target label;
this frozen informativeness gate is evaluated before Monte Carlo sampling. A
cell below that threshold is UNRUN rather than a negative result. Every primary, five-depth,
three-annotator, and pooling/window H4 cell seeks exactly 2,500 valid resamples
and permits at most 2,750 attempted permutations. Every invalid attempt and
reason is retained. The only assignment-level rejection is a non-finite,
non-positive, degenerate, fewer-than-100-point, or otherwise Section 8-invalid
cdim. A proposed assignment that changes per-chain counts is an implementation
failure that stops the cell; it is not rejection sampling. If 2,500 retained
assignments are not obtained within 2,750 proposals, the cell and its full
multiplicity family are UNRUN; resamples are not silently dropped and the
family is not narrowed.

The registered primary family is repeated as a Monte Carlo stability diagnostic
with base seed 20260727 and the same B and rules. The 20260726 run supplies the
registered p-values. If any Holm pass/fail verdict changes under the diagnostic
seed, the affected result is classified mixed/Monte-Carlo-unstable and cannot
support H2 pass wording.

## 10. Chain uncertainty and one-sentence-per-chain stability

Point-estimate uncertainty is a duplicate-safe **chain stability band**, not a
bootstrap confidence interval. For each cell, run 500 deterministic
subsamples, each selecting 80% of eligible chains without replacement with
the Section 9 chain-stability coordinate. `floor(0.8*N)` chains are selected
from chain IDs sorted by their NFC UTF-8 byte sequence. Within every selected
chain, select one target occurrence using the exact Section 7 payload with the
zero-based replicate index. Recompute correlation dimension and PR end-to-end.
Quantiles are NumPy linear quantiles (`method="linear"`) at 0.025, 0.5, and
0.975 over finite draws only. Report all 500 statuses, finite-replicate count,
selected-chain count, median, endpoints, and failure reasons. Fewer than 475
valid draws makes the required cell UNRUN rather than producing a reduced band.

The band is a stability description under deletion of 20% of chains and
within-chain occurrence choice. It is not labelled a 95% CI, does not estimate
coverage, and does not replace the within-chain label-permutation specificity
test.

## 11. Truncation analysis

Truncation is derived from `data/chains_R1-1.5B.json` using the frozen rule in
`src/cbs/cohort.py`: `n_tokens >= 8192` and the stripped chain text does not end
with `</think>`. The analysis reports combined, complete-only, and
truncated-only estimates, with chain and row counts for every behaviour.

For each behaviour, the full-sample occurrence map is selected before assigning
truncation status, using the Section 7 digest with family code 4, Sonnet code
1, L27, the behaviour code, pooling/window/truncation codes all 0,
`replicate_index=-1`, and `scope_key=""`. It is then subset unchanged into
combined, complete, and truncated cells. For each stability replicate
`r=0..499`, first construct one new truncation-neutral occurrence map over all
eligible chains with the same coordinate except `replicate_index=r`; only
after that map is frozen may the replicate's combined, complete, and truncated
chain subsets be formed. The same chain's occurrence is therefore identical
wherever it appears within replicate `r`. No truncation code or
stratum-specific digest may change occurrence selection.

If the largest absolute difference in category share between the complete and
truncated groups exceeds 10 percentage points for an analysed behaviour, a
category-matched sensitivity is mandatory. Within each behaviour × category,
let `m=min(n_complete,n_truncated)`. Rank complete and truncated chain IDs
separately by the lowercase SHA-256 of the Section 7 payload with family code
4, Sonnet code 1, L27, the behaviour code, pooling/window codes 0,
truncation code 4, `chain_id`, `annotation_index=-1`, `char_offset=-1`,
`token_start=-1`, **`replicate_index=-1`**, and the NFC category as
`scope_key`. Retain the first `m` chains from each ranked list. No undefined
category-match rank enters the digest, and no category is pooled across a
missing counterpart.

Each behaviour's category-matched cell (indices 60–63) has an exact two-result
schema rather than one pooled result:
`matched_results={"matched_complete": R, "matched_truncated": R}`, where each
`R` requires `chain_ids: array[string]`, `n_chains: integer`,
`n_categories: integer`, `category_counts: object[string,integer]`,
`occurrence_list_sha256: string[64 hex]`, `cdim: number|null`,
`pr: number|null`, `cdim_stability: {n_valid,q025,median,q975}`,
`pr_stability: {n_valid,q025,median,q975}`, and
`status: enum["VALID","INVALID","UNRUN"]`. Both results must be valid or the
category-matched sensitivity is UNRUN. Behaviour frequency and normalised
within-chain position (`token_start / max(n_tokens,1)`) are reported
descriptively by truncation status.

Within the fixed matched-complete and matched-truncated chain lists, full-sample
occurrences use the neutral map at `replicate_index=-1`. For stability
replicate `r`, select 80% of each fixed matched chain list using its Section 9
chain-stability RNG, but obtain occurrences only from the single
replicate-wise neutral map for `r`.

Define `rel(a,b)=abs(a-b)/max(abs(a),abs(b),1e-12)`. The unadjusted truncation
rule passes for a behaviour only if complete and truncated cdim and PR each
differ from the combined statistic by at most 25% and their stability bands
overlap for both statistics. When category matching is required, the matched
rule passes only if `rel(matched_complete,matched_truncated)<=0.25` for both
cdim and PR and the matched stability bands overlap for both statistics.

Category matching has this frozen decision role:

- if category-rate imbalance is at most 10 percentage points, matching is not
  required and the behaviour is pass/fail according to the unadjusted rule;
- if imbalance exceeds 10 percentage points, both matched results and at least
  475 valid stability draws per statistic/side are mandatory; any missing or
  invalid required result makes the behaviour and H3 **UNRUN**;
- when required and both rules pass, the behaviour passes truncation
  robustness; when both rules fail, it fails; when one passes and the other
  fails, it is mixed due to category-composition dependence.

For each behaviour, first classify the **chain-stability leg** independently:
PASS requires both cdim and PR to have at least 475 valid draws, contain their
full-sample values within their frozen bands, and meet the 25% finite-draw
median rule; FAIL means all required outputs are valid but at least one of
those conditions fails; UNRUN means any required output is missing or invalid.
There is no MIXED value for this leg. Classify the
**truncation/category leg** as PASS, MIXED, FAIL, or UNRUN by the rules above.
The frozen per-behaviour cross-product is exhaustive:

| Chain-stability leg | Truncation/category leg | H3 behaviour status |
|---|---|---|
| PASS | PASS | PASS |
| PASS | MIXED | MIXED |
| PASS | FAIL | FAIL |
| PASS | UNRUN | UNRUN |
| FAIL | PASS | FAIL |
| FAIL | MIXED | FAIL |
| FAIL | FAIL | FAIL |
| FAIL | UNRUN | UNRUN |
| UNRUN | PASS | UNRUN |
| UNRUN | MIXED | UNRUN |
| UNRUN | FAIL | UNRUN |
| UNRUN | UNRUN | UNRUN |

Thus a valid-but-failing chain-stability leg can never be ignored, rescued, or
averaged with a truncation result, and any required UNRUN leg propagates to
that behaviour. The four-behaviour family operator is then:

1. if any behaviour is UNRUN, H3 is UNRUN;
2. otherwise, if all four behaviours are PASS, H3 is PASS;
3. otherwise, if all four behaviours are FAIL, H3 is FAIL; and
4. every other complete combination of PASS/MIXED/FAIL is H3 MIXED.

Material or discordant results force wording about observed prefixes,
truncation, and category composition; they are never averaged away.

## 12. Pooling and sentence-window sensitivity

At L27, the frozen representation grid is:

- pooling: mean, first, last;
- window: canonical unclipped and sentence-end clipped; and
- estimators: equal-chain correlation dimension and equal-chain PR, with
  sentence-weighted cdim reported separately.

Before any H4 statistic, align all six representations by the exact occurrence
key from Section 7 and construct their intersection. A clipped span with no
execution token uses the preceding token only when that token is present in
the aligned occurrence; otherwise that occurrence is absent from the
intersection and the attrition is reported. It is never replaced with an
unclipped activation.

Duplicate handling is joint. First collapse repeated occurrence keys once and
apply that retained key to all six matrices. Next, find exact-vector duplicate
groups separately in each representation; take the union of all occurrences
flagged as a loser in any representation and remove those occurrences from
all six. The deterministic winner is the lowest Section 7 digest computed
with the single representation-neutral coordinate: family code 5, Sonnet code
1, L27, the behaviour code, pooling/window/truncation codes all 0,
`replicate_index=-1`, and `scope_key="H4-common"`. Pooling/window cell codes
must never participate in duplicate-winner or occurrence selection. A
cross-chain or cross-label conflict in any representation invalidates the
entire H4 family. After this joint audit, for each behaviour and chain select
the lowest digest under that same neutral coordinate from the common
intersection once and use that exact ordered occurrence-key list in all six
cells. The SHA-256 of the canonical JSON list of selected occurrence keys is
recorded for every cell and must be identical across the six cells for a
behaviour.

Thus each behaviour has one common complete-case chain set and identical
digest-selected occurrences across all six cells. A chain lacking a surviving
complete-case occurrence is excluded from all six cells, with its reason
reported. If any one of the 24 behaviour × representation matrices, metadata,
row indices, occurrence-list digests, cdim/PR statistics, or permutation runs
is absent, misaligned, invalid, or non-finite, **H4 is UNRUN**. No result may
be based on "every available" cell and no reduced-grid pass is allowed.

Every H4 cdim null uses family code 5, `B=2500` valid within-chain
permutations, at most 2,750 attempts, the zero-based indices 64–87, and the
Section 9 conditional-validity, estimator-failure, alignment, and eligibility
rules. Identity/no-op assignments are retained. Holm correction
is applied once across all 24 p-values. The diagnostic base-seed rerun applies
to this complete family; a verdict change makes H4 mixed/Monte-Carlo-unstable,
not pass.

Current L27 mean/unclipped activations exist. The ledger records existing L27
mean/first/last spectral outputs, but they do not supply L27 correlation
dimension or sentence-end clipping. Missing L27 first/last and clipped
activations therefore require a selected-layer re-extraction before this grid
can be completed. Mean, first, and last must be extracted in one shared forward
pass; clipped and unclipped outputs must carry separate metadata and row
indices. If this exact six-cell extraction is infeasible, H4 remains UNRUN
pending amendment. No claim of pooling/window robustness is allowed from the
existing PCA sweep alone.

The exact planned H4 extraction root is
`data/activations/R1-1.5B-core-hardening-L27/`. Every path below is
**ABSENT / UNRUN — UNRESOLVED HASH — DO NOT EXECUTE** until the gated shared
forward pass produces it and a pre-statistic manifest records its SHA-256:

| Representation | Exact planned matrix paths | Exact planned alignment paths |
|---|---|---|
| mean/unclipped | `data/activations/R1-1.5B-core-hardening-L27/mean/unclipped/backtracking_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/mean/unclipped/uncertainty-estimation_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/mean/unclipped/example-testing_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/mean/unclipped/adding-knowledge_layer27.npy` | `data/activations/R1-1.5B-core-hardening-L27/mean/unclipped/metadata.json`; `data/activations/R1-1.5B-core-hardening-L27/mean/unclipped/row_index.json` |
| mean/clipped | `data/activations/R1-1.5B-core-hardening-L27/mean/clipped/backtracking_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/mean/clipped/uncertainty-estimation_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/mean/clipped/example-testing_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/mean/clipped/adding-knowledge_layer27.npy` | `data/activations/R1-1.5B-core-hardening-L27/mean/clipped/metadata.json`; `data/activations/R1-1.5B-core-hardening-L27/mean/clipped/row_index.json` |
| first/unclipped | `data/activations/R1-1.5B-core-hardening-L27/first/unclipped/backtracking_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/first/unclipped/uncertainty-estimation_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/first/unclipped/example-testing_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/first/unclipped/adding-knowledge_layer27.npy` | `data/activations/R1-1.5B-core-hardening-L27/first/unclipped/metadata.json`; `data/activations/R1-1.5B-core-hardening-L27/first/unclipped/row_index.json` |
| first/clipped | `data/activations/R1-1.5B-core-hardening-L27/first/clipped/backtracking_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/first/clipped/uncertainty-estimation_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/first/clipped/example-testing_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/first/clipped/adding-knowledge_layer27.npy` | `data/activations/R1-1.5B-core-hardening-L27/first/clipped/metadata.json`; `data/activations/R1-1.5B-core-hardening-L27/first/clipped/row_index.json` |
| last/unclipped | `data/activations/R1-1.5B-core-hardening-L27/last/unclipped/backtracking_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/last/unclipped/uncertainty-estimation_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/last/unclipped/example-testing_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/last/unclipped/adding-knowledge_layer27.npy` | `data/activations/R1-1.5B-core-hardening-L27/last/unclipped/metadata.json`; `data/activations/R1-1.5B-core-hardening-L27/last/unclipped/row_index.json` |
| last/clipped | `data/activations/R1-1.5B-core-hardening-L27/last/clipped/backtracking_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/last/clipped/uncertainty-estimation_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/last/clipped/example-testing_layer27.npy`; `data/activations/R1-1.5B-core-hardening-L27/last/clipped/adding-knowledge_layer27.npy` | `data/activations/R1-1.5B-core-hardening-L27/last/clipped/metadata.json`; `data/activations/R1-1.5B-core-hardening-L27/last/clipped/row_index.json` |

## 13. Curvature boundary

Only zero-based L16 is controlled for the separate secondary curvature
diagnostic. Its registered
instrument is
`src/curvature.py::local_vs_global_dim_ratio` with `k=10`,
`variance_threshold=0.90`, `n_anchors=150`, `random_state=42`, and
`n_bootstrap=0`, applied to float32 copies of separately constructed L16
equal-chain points. These are not the primary L27 or H4 points.
For each zero-based replicate `r=0..499`, obtain one parent
`SeedSequence([20260726,4,6,cell_index,r])` and call `.spawn(2)` exactly once.
Child 0 drives the 80%-without-replacement chain selection and Section 7
occurrence selection; child 1 draws the same number of rows without replacement
from the jointly deduplicated sentence-weighted cloud. The two statistics form
one pair `(ratio_chain,r, ratio_control,r)`. If either ratio is non-finite,
non-positive, has a zero/invalid denominator, or the diagnostic rejects its
neighbourhood calculation, discard the pair—not one member—and record the
reason. Fewer than 475 valid pairs makes the diagnostic UNRUN.

On valid pairs, define exactly
`D_chain = median_r(abs(ratio_chain,r - 1.0))` and
`D_control = median_r(abs(ratio_control,r - 1.0))`. Compute chain and control
2.5%, 50%, and 97.5% quantiles separately with NumPy
`quantile(..., method="linear")`. The bounded negative operator is
`(q_chain_0.025 <= 1.0 <= q_chain_0.975) AND (D_chain <= D_control)`.
If true, wording is "no beyond-chain curvature detected by this diagnostic at
L16"; if false, report a mixed/positive controlled-layer diagnostic. There is
no p-value, no family-wise test, and no global-flatness conclusion.

The geodesic/Euclidean ratio may be reported secondarily with `k=10`,
`n_pairs=350`, `random_state=42`, and `n_bootstrap=0`, but it cannot upgrade or
reverse the local/global decision. Synthetic calibration must be reported
honestly; the existing power record does not license "well-powered global
absence." No all-layer curvature run is planned because global-flatness
language is outside Route A.

## 14. Exact decision table and permitted wording

| Frozen outcome | Classification | Permitted thesis wording |
|---|---|---|
| All four valid primary cdim values are `0 < cdim <= 10.0` | H1 pass | "All four operational mean-pooled L27 activation clouds had correlation-dimension estimates at most 10." |
| One to three valid primary behaviours meet `0 < cdim <= 10.0`, with all four valid | H1 mixed | Name the behaviours; no all-four low-occupancy wording. |
| Zero of four valid primary behaviours meet the bound | H1 fail | "The preregistered descriptive low-cdim criterion was not met." |
| Any primary H1 cell is absent or invalid | H1 UNRUN | Report the failure and no H1 classification from a reduced set. |
| Four of four primary H2 Holm tests pass and diagnostic-seed verdicts agree | H2 pass | "At L27 under the frozen mean/unclipped equal-chain representation, all four model-annotation-indexed clouds had lower cdim than the conditional within-chain label-permutation null." |
| One to three of four valid H2 cells pass, or a diagnostic-seed verdict changes | H2 mixed | Name exact behaviours and Monte Carlo instability; no all-four specificity claim. |
| Zero of four valid H2 cells pass | H2 fail | "The direct cdim null did not support behaviour specificity"; existing PCA evidence remains a separate estimand. |
| Any H2 cell fails validation or 2,500/2,750 validity | H2 UNRUN | Retain every failed cell; do not narrow the four-test family. |
| All four primary behaviours pass the exhaustive Section 11 stability/truncation/category-matching operator | H3 pass | "The bounded L27 point estimates were stable to the registered chain deletion and truncation/category-composition checks." |
| Section 11 yields any complete combination other than four passes or four fails | H3 mixed | Name behaviour, statistic, unadjusted/matched verdict, and disagreement; narrow population wording. |
| Section 11 yields fail for all four behaviours | H3 fail | "The registered chain/truncation robustness criterion was not met." |
| Any required H3 or matched cell has fewer than 475/500 finite draws or is otherwise invalid | H3 UNRUN | No chain/truncation pass from remaining cells. |
| All 24 common-set H4 cells are valid; all cdim/PR deviations are at most 25%; all 24 Holm tests pass; diagnostic seed agrees | H4 pass | "The bounded L27 result persisted across the matched mean/first/last and clipped/unclipped grid." |
| All 24 H4 cells are valid but at least one and not all behaviours fail a deviation/null/seed rule | H4 mixed | Report all 24 cells and name representation-sensitive behaviours; no general pooling/window robustness claim. |
| All 24 H4 cells are valid and every behaviour fails at least one H4 rule | H4 fail | "The registered matched pooling/window robustness criterion was not met." |
| Any of the six representations or 24 cells is absent, misaligned, non-finite, under-resampled, or has a mismatched occurrence digest | H4 UNRUN | No incomplete-grid or "available-cell" pass; H4 remains prospective. |
| A behaviour passes at least four of five depth tests | Depth-robust for that behaviour | "The lower-tail cdim result recurs at four/five or five/five tested depths for [behaviour]." |
| A behaviour passes fewer than four of five valid depths | Depth-mixed/limited | Give exact passing layers; no across-depth claim. |
| Any of the 20 depth cells is unresolved/invalid | Depth family UNRUN | Do not report a reduced-depth robustness pass. |
| A behaviour passes L16 for all three model annotations after family correction | Annotator-robust for that behaviour | "The L16 result recurs across three model annotations," with agreement and extraction-vintage limits. |
| A behaviour fails any annotator at L16 | Annotator-mixed | Give exact annotator results; do not call the result annotation-robust. |
| Any of the 24 annotator cells is unresolved/invalid | Annotator family UNRUN | Do not report a reduced-annotator robustness pass. |
| Curvature bounded-negative operator is true with at least 475 valid pairs | Bounded curvature diagnostic negative | "No beyond-chain curvature was detected by this diagnostic at L16." |
| Curvature operator is false with at least 475 valid pairs | Curvature mixed/positive diagnostic | Report the exact controlled-layer departure; do not infer a global curved manifold. |
| Fewer than 475 curvature pairs or any required curvature input is invalid | Curvature UNRUN | No curvature conclusion. |
| H1, H2, H3, and H4 all pass | G3 strongest bounded outcome | "Low estimated intrinsic-dimensional, behaviour-specific activation geometry" only under the tested model, labels, L27, and matched representations. |
| H1 passes but H2 fails/mixed/UNRUN | G3 narrow outcome | "Low estimated dimension within annotation-indexed clouds"; specificity remains unsupported by the direct cdim contract. |
| Any core validation or resource gate fails | Stop/reframe | Report the gate failure; do not convert missing or invalid work into a null result. |

No table row licenses "six-dimensional linear subspace," "the representation
is flat," "all behaviours are specific everywhere," or "post-training created
the geometry."

## 15. Resource benchmark gate and stop rules

The first empirical action, after signatures, is the ledger-specified
backtracking/Sonnet/L27 pilot. Run B=25 and record cost before optionally
extending the benchmark to B=50. Its exact planned outputs are:

- `results/robustness/cdim_null_pilot/R1-1.5B/backtracking_L27_resource_only.json`; and
- `results/robustness/cdim_null_pilot/R1-1.5B/BENCHMARK.md`.

The pilot is a runtime/validity benchmark, not a thesis inferential result.
Its input cell is exactly canonical Sonnet/backtracking/L27 from Section 4.
The process may execute the statistic internally to determine validity, but
the resource-only writer must discard the returned scalar and must never
serialize, print, log, expose through an exception, or show the feasibility
reviewer any observed cdim, null draw, behaviour comparison, p-value, or
adjusted p-value. Permitted fields are limited to schema/status, input hashes,
B target, attempted/valid/invalid counts, enumerated invalid reasons,
wall-clock seconds, peak RSS bytes, scratch bytes, output bytes per attempt,
worker count, software/process/host facts, and projected resource totals.

For `A>0` attempted benchmark permutations and wall time `T` seconds, use:

- `seconds_per_attempt = T / A`;
- `projected_primary_seconds = 1.25 * seconds_per_attempt * 2750 * 4`;
- `projected_secondary_seconds = 1.25 * seconds_per_attempt * 2750 * 44`;
- `projected_H4_seconds = 1.25 * seconds_per_attempt * 2750 * 24`; and
- `projected_scratch_bytes(cell_count) = 1.25 * max(observed_scratch_bytes, output_bytes_per_attempt * 2750 * cell_count)`.

The 1.25 multiplier is a frozen safety factor. No parallel speedup may be
assumed to pass a gate; measured parallel performance may be recorded
separately. Feasibility review receives only the resource-only JSON validated
against the separate benchmark schema in Section 16.

The benchmark passes for the **primary family** only if: peak RSS is at most
48 GiB; at least 99% of attempted resamples are valid; no alignment or
unresolved cross-chain/cross-label duplicate failure occurs; projected wall time for the
four-cell primary B=2500 run on the available 16–32-core CPU resource is at
most 24 hours; and projected scratch/output use is at most 10 GiB. The
secondary CPU families pass only if their combined projected wall time is at
most 72 additional hours. H4 must separately project to at most 48 CPU-hours.
The GPU re-extraction gate passes only after a
free-space check confirms at least 100 GiB scratch, a one-chain extraction
smoke test validates all six representation metadata/row-index alignments, and
projected selected-layer extraction is at most 24 GPU-hours on a GPU with at
least 24 GiB VRAM.

If the primary gate fails, stop before full statistics. Optimisation that leaves
the statistic and estimand unchanged may be benchmarked and documented.
Narrowing B, layers, annotators, representations, or families is a material
amendment and must be signed before any full-result statistic is opened. If
the primary family is feasible but a secondary gate fails, run the primary
family and label the unavailable secondary work prospective/unrun. There is no
API spend. No paid resource is authorised by this document. The planned
maximum is 1–3 CPU node-days and, only after its gate, one selected-layer
24-GiB GPU-day with 50–100 GiB scratch.

Stop a cell immediately for row misalignment, proxy chain IDs, no mixed-label
chains, fewer than 100 equal-chain points, non-finite/degenerate estimator
output, unresolved duplicate conflicts defined in Section 7, or failure to reach 2,500
valid permutations within 2,750 attempts. Stop the family if any primary cell
stops. Never reinterpret an execution failure as evidence for the null.

## 16. Planned implementation, outputs, schemas, hashes, and tests

All paths in this subsection are plans, not claims that files exist.

**Planned implementation paths — ABSENT / UNRUN at freeze:**

- `configs/analysis/thesis_core_hardening_2026-07-26.yaml`;
- `scripts/thesis_core_hardening.py`;
- `schemas/thesis_core_hardening_result_cell_v1.schema.json`;
- `schemas/thesis_core_hardening_resource_benchmark_v1.schema.json`;
- `schemas/thesis_core_hardening_provenance_v1.schema.json`;
- `tests/test_thesis_core_hardening.py`.

All six were explicitly **ABSENT / UNRUN** at freeze. The planned
wrapper/runner symbols
are exact: `correlation_dimension_point`, `canonical_occurrence_digest`,
`build_common_complete_case_grid`, `run_within_chain_cdim_null`,
`run_chain_stability_band`, `run_truncation_sensitivity`,
`run_pooling_window_family`, `run_curvature_diagnostic`,
`run_resource_benchmark`, `write_resource_only_benchmark`, and `main` in
`scripts/thesis_core_hardening.py`. The 2026-07-26 23:58 BST pre-result
full-run amendment additionally authorises
`build_primary_family_document`, `write_primary_family_result`, and
`execute_registered_primary_from_disk`. No other differently named
implementation is authorised without a pre-result amendment.

**Planned full outputs — ABSENT / UNRUN at freeze:**

- `results/robustness/core_hardening/R1-1.5B/primary_L27_cdim_null.json`;
- `results/robustness/core_hardening/R1-1.5B/five_depth_cdim_null.json`;
- `results/robustness/core_hardening/R1-1.5B/three_annotator_cdim_null.json`;
- `results/robustness/core_hardening/R1-1.5B/chain_stability.json`;
- `results/robustness/core_hardening/R1-1.5B/truncation_sensitivity.json`;
- `results/robustness/core_hardening/R1-1.5B/pooling_window_sensitivity.json`;
- `results/robustness/core_hardening/R1-1.5B/curvature_L16.json`;
- `results/robustness/core_hardening/R1-1.5B/G3_DECISION.md`;
- `results/robustness/core_hardening/R1-1.5B/provenance.json`.

These outputs and the pilot outputs in Section 15 are explicitly
**ABSENT / UNRUN**.

The planned result-cell schema
`schemas/thesis_core_hardening_result_cell_v1.schema.json` has
`additionalProperties=false` and requires:

| Field | JSON type / enum |
|---|---|
| `schema_version` | string, enum `["thesis-core-hardening-result-cell-v1"]` |
| `status` | string, enum `["UNRUN","VALID","INVALID","STOPPED"]` |
| `family` | string, enum `["primary","five_depth","three_annotator","truncation","pooling_window","curvature"]` |
| `hypothesis` | string, enum `["H1","H2","H3","H4","CURVATURE_DIAGNOSTIC","SECONDARY"]` |
| `cell_index` | integer, minimum 0, maximum 91 |
| `codes` | object of integer `family`, `purpose`, `annotator`, `behaviour`, `pooling`, `window`, `truncation` codes |
| `model_id`, `annotator`, `label`, `pooling`, `window` | string |
| `layer_zero_based` | integer |
| `input_hashes`, `estimator_settings`, `preprocessing`, `duplicate_audit`, `provenance` | object |
| `occurrence_list_sha256` | string matching `^[0-9a-f]{64}$` |
| `row_counts`, `chain_counts`, `resamples` | object with non-negative integer counts |
| `observed_cdim`, `observed_pr`, `raw_p`, `holm_p` | number or null; must be null unless `status="VALID"` and relevant |
| `null_cdim` | array of numbers; exactly 2,500 items only for a valid inferential cell, otherwise empty |
| `stability_draws` | array of objects with integer replicate, enum status, number-or-null cdim/PR, and string-or-null failure |
| `matched_results` | object or null; for truncation indices 60–63 it is required and has exactly `matched_complete` and `matched_truncated`, each with the typed fields frozen in Section 11; null otherwise |
| `decision` | string, enum `["PASS","MIXED","FAIL","UNRUN","NOT_APPLICABLE"]` |
| `amendment_ids` | array of strings |

The planned resource schema
`schemas/thesis_core_hardening_resource_benchmark_v1.schema.json` also has
`additionalProperties=false`, forbids fields matching
`observed*`, `*cdim*`, `null*`, `*p_value*`, `raw_p`, or `holm_p`, and requires:

| Field | JSON type / enum |
|---|---|
| `schema_version` | string, enum `["thesis-core-hardening-resource-benchmark-v1"]` |
| `status` | string, enum `["UNRUN","COMPLETE","INVALID","STOPPED"]` |
| `input_hashes` | object of path-string to 64-character lowercase SHA-256 string |
| `b_target`, `attempted`, `valid`, `invalid`, `workers` | non-negative integer |
| `invalid_reasons` | object of string to non-negative integer |
| `wall_seconds`, `seconds_per_attempt`, `projected_primary_seconds`, `projected_secondary_seconds`, `projected_H4_seconds` | non-negative number |
| `peak_rss_bytes`, `scratch_bytes`, `output_bytes_per_attempt`, `projected_scratch_bytes` | non-negative integer |
| `software`, `process`, `host` | object |
| `gate` | string, enum `["PASS","FAIL","UNRUN"]` |

The planned provenance schema
`schemas/thesis_core_hardening_provenance_v1.schema.json` requires
`schema_version="thesis-core-hardening-provenance-v1"`, preregistration
path/hash, input path/hash manifest, exact code commit string or null,
`dirty` boolean, dirty-path array, Python/package versions, command argv,
UTC start/end strings, host/process objects, and amendment-ID array. It uses
`additionalProperties=false`.

Every result JSON must contain: schema version; explicit `status`;
preregistration path/hash; model ID and checkpoint revision/hash or unresolved
marker; annotator; label; zero-based layer; pooling/window; estimator settings;
preprocessing; raw/deduplicated row counts; independent-chain and mixed-chain
counts; duplicate audit; observed statistic; all null draws; attempted/valid
resamples and failure reasons; seed coordinates; raw and Holm-adjusted
p-values; stability draws/band where relevant; truncation/category summaries;
input SHA-256 hashes; code commit and dirty flag; Python/package versions;
host/process, wall time, peak RSS; and amendment identifiers. CSV tables must
be lossless views of JSON cells, not independent sources.

Planned output hashes are **ABSENT / UNRUN** and must not be filled until the
files are produced. At freeze, the following input SHA-256 hashes were recorded
without running an estimator:

| Existing input | SHA-256 |
|---|---|
| `src/intrinsic_dim.py` | `c3869d8b4462b7f06d004f3daee187e7e1a6d9d30a912cdb236bf4935464bfc5` |
| `src/nulls.py` | `22c91862fa7e7caf2f2615ab6340554655f8114f1bd5747ead5aba3985719759` |
| `src/row_provenance.py` | `9155ef8fd848a015c3637cf61dfa463eb9a0f8a187fbf2ac065506cf43e8a994` |
| `src/cbs/cohort.py` | `0d406320fd5d12eb034d92a3a1f6f4ba9d473a98c9b71c2bf55e8709e57954f2` |
| `data/activations/R1-1.5B/metadata.json` | `23220c350f165399e6a4b522af0fe3248b8912db2fc707f65ce044cfcaa4dfd1` |
| `data/activations/R1-1.5B/row_index.json` | `399b03af4fc77efbd7b742a2287538eb5ceb4f37ec18e69760f83cf8e2079734` |
| `data/annotated_R1-1.5B.json` | `9b24020ec3ce4ea53d1a47468a17090eed9d80062752728381e79dd38ccb78d5` |
| `data/chains_R1-1.5B.json` | `736201f490708349e9a1efb0321ed6e63fdbc6e93681e716130dbffe2741191c` |
| `data/activations/R1-1.5B/backtracking_layer27.npy` | `0432cade1316cdd6fc8f4cdde83016a663676c0d33746be716b3f38b5797e716` |
| `data/activations/R1-1.5B/uncertainty-estimation_layer27.npy` | `a557a11a680143af2fb2f45de748d2617cfb51bcba522757815c0659137fa1ab` |
| `data/activations/R1-1.5B/example-testing_layer27.npy` | `cf0a2c4acdad9595203d67a8e8b4897a0d887e278a9ad315e7635a7007e127b8` |
| `data/activations/R1-1.5B/adding-knowledge_layer27.npy` | `09d1df9d7d566002f85a5cd49fd25dd08247718da87c6a05513d7871129fce61` |

**Secondary five-depth Sonnet manifest (all existing; hashes frozen):**

| Existing input | SHA-256 |
|---|---|
| `data/activations/R1-1.5B/backtracking_layer11.npy` | `fc08efdac9919d33c7a52c9d2bf820e72eae09e6d9822df9ae048c4d682b9900` |
| `data/activations/R1-1.5B/backtracking_layer14.npy` | `64b015379a1736ac1f73adc1c80c8dcf1c2958d17e0de086403ffeb121c89b71` |
| `data/activations/R1-1.5B/backtracking_layer17.npy` | `37a0a82a0bca96c85e0d91862f30fcb0b0c9bcb7255aa4f2a510655f7cf2847d` |
| `data/activations/R1-1.5B/backtracking_layer20.npy` | `91dec4c61a0f70705a17a60becebb3c92bc6c1868bb02ce05a97b5aae3f90531` |
| `data/activations/R1-1.5B/backtracking_layer27.npy` | `0432cade1316cdd6fc8f4cdde83016a663676c0d33746be716b3f38b5797e716` |
| `data/activations/R1-1.5B/uncertainty-estimation_layer11.npy` | `488b1a1d7d05d8c64978323d86f9af628d15bd74d7cb9cdec24a0f7486d2e80e` |
| `data/activations/R1-1.5B/uncertainty-estimation_layer14.npy` | `857e03d4c8bfb64a3794fef4a454853caf68d23ffb7a92cf1631f782b90a5d45` |
| `data/activations/R1-1.5B/uncertainty-estimation_layer17.npy` | `0519637da893b60033f761241059061d2c7576a0d6c0c234fd2da6d379c12d5e` |
| `data/activations/R1-1.5B/uncertainty-estimation_layer20.npy` | `c6562a3b939b4e09e431553c49200b8e3c83d17713801923f7812ca59992cf9e` |
| `data/activations/R1-1.5B/uncertainty-estimation_layer27.npy` | `a557a11a680143af2fb2f45de748d2617cfb51bcba522757815c0659137fa1ab` |
| `data/activations/R1-1.5B/example-testing_layer11.npy` | `587d8e51a6fc3eb0d45723dfd116593f64c39df2f0bac947c530913a0b0c3d0d` |
| `data/activations/R1-1.5B/example-testing_layer14.npy` | `51ab67cde7d0c7ad90f0bdf909c34ced1eb934e1676bbedffab8ad07f87401ca` |
| `data/activations/R1-1.5B/example-testing_layer17.npy` | `1dad5968083a1f4ba1bbaff3c7decc61e59bec355b19313f4c219c99ef8a4c9e` |
| `data/activations/R1-1.5B/example-testing_layer20.npy` | `7bd6680906a7e224f71721edcba70d2d6e45ed8ebb8780be54db4a787a79de7a` |
| `data/activations/R1-1.5B/example-testing_layer27.npy` | `cf0a2c4acdad9595203d67a8e8b4897a0d887e278a9ad315e7635a7007e127b8` |
| `data/activations/R1-1.5B/adding-knowledge_layer11.npy` | `9d9ac5b3aab24045204ce8cad14ae543a8cb450128b59e9329b03390296282f9` |
| `data/activations/R1-1.5B/adding-knowledge_layer14.npy` | `a14e601a3fe45a42de50f6f076a11e6d5ff940b5bddecd6a1711b85dda9bc44f` |
| `data/activations/R1-1.5B/adding-knowledge_layer17.npy` | `f4a6dbe4a9ad95c7f584b744fe5ea5aacd4d12b703f8c19cf3d1d1821b31c6cd` |
| `data/activations/R1-1.5B/adding-knowledge_layer20.npy` | `d8def9bc0a1a62968c7bcefba06af8e3e1361b20191675d311d030ed4c15a98c` |
| `data/activations/R1-1.5B/adding-knowledge_layer27.npy` | `09d1df9d7d566002f85a5cd49fd25dd08247718da87c6a05513d7871129fce61` |
| `data/activations/R1-1.5B/metadata.json` | `23220c350f165399e6a4b522af0fe3248b8912db2fc707f65ce044cfcaa4dfd1` |
| `data/activations/R1-1.5B/row_index.json` | `399b03af4fc77efbd7b742a2287538eb5ceb4f37ec18e69760f83cf8e2079734` |
| `data/annotated_R1-1.5B.json` | `9b24020ec3ce4ea53d1a47468a17090eed9d80062752728381e79dd38ccb78d5` |

**Secondary three-model-annotation manifest (all existing; hashes frozen):**

| Arm/input | SHA-256 |
|---|---|
| canonical `data/activations/R1-1.5B/backtracking_layer12.npy` | `95581893832bd7b7ab17f46778732dc1110c3feb1de5682794f7d426bdf2df4a` |
| canonical `data/activations/R1-1.5B/uncertainty-estimation_layer12.npy` | `14c65ea8a12e4619da2abc2b066cd6e3d22eba4c26a95e64410f8a0ca8df12e8` |
| canonical `data/activations/R1-1.5B/example-testing_layer12.npy` | `32b3264ffd798c2f1802996176a57c984512adb2be66e7aa495779da8e665953` |
| canonical `data/activations/R1-1.5B/adding-knowledge_layer12.npy` | `eef357e98ac35e0ee457daaad5958166a21d19bbe3d25fd71300e11412b6e3f3` |
| canonical `data/activations/R1-1.5B/backtracking_layer16.npy` | `bdd363c7e8315d2e46c0d9ae073de830c22cf8ac4a567532e98413a5a27f308b` |
| canonical `data/activations/R1-1.5B/uncertainty-estimation_layer16.npy` | `b744acae600df41e542c789d73773b1dc521b7a7df55b0b5e5f0baffbd4635f7` |
| canonical `data/activations/R1-1.5B/example-testing_layer16.npy` | `8a46bfed968e93347e028d3e7e388fb967b8fe048d188ba70f23a8d493304485` |
| canonical `data/activations/R1-1.5B/adding-knowledge_layer16.npy` | `037d6407fd6a1846974c844fce5ce4ff539da15435051d3bb322db089294f33a` |
| canonical `data/activations/R1-1.5B/metadata.json` | `23220c350f165399e6a4b522af0fe3248b8912db2fc707f65ce044cfcaa4dfd1` |
| canonical `data/activations/R1-1.5B/row_index.json` | `399b03af4fc77efbd7b742a2287538eb5ceb4f37ec18e69760f83cf8e2079734` |
| canonical `data/annotated_R1-1.5B.json` | `9b24020ec3ce4ea53d1a47468a17090eed9d80062752728381e79dd38ccb78d5` |
| qwenspans `data/activations/R1-1.5B-qwenspans/backtracking_layer12.npy` | `b1781b5025bc3eff03b77f92dc5c3dca2564e3318d9b70f97fdf3616f941aeac` |
| qwenspans `data/activations/R1-1.5B-qwenspans/uncertainty-estimation_layer12.npy` | `a08cfcdbd671565581dd0f4f41c67bcc7f7d463a4ce9bac485156263f90f68a5` |
| qwenspans `data/activations/R1-1.5B-qwenspans/example-testing_layer12.npy` | `c0238166e244157dfec20eac58ce23a4ba2c788ac4e1942aecc15c9e373902fb` |
| qwenspans `data/activations/R1-1.5B-qwenspans/adding-knowledge_layer12.npy` | `6ffd0b41c77316e5e321b6be4286f90cdedde556c6c625e943b63af5442fc3b9` |
| qwenspans `data/activations/R1-1.5B-qwenspans/backtracking_layer16.npy` | `5dc9880b127e84f1c0fccc06aad0a5abff58676cb914fd7052c0f7fcad872865` |
| qwenspans `data/activations/R1-1.5B-qwenspans/uncertainty-estimation_layer16.npy` | `9b88955c57d04cdd38dfd432f6f2629ebb6b89644db6a48567dfeb7611f6f884` |
| qwenspans `data/activations/R1-1.5B-qwenspans/example-testing_layer16.npy` | `594a69bb2c818a344290f4874905081b65c2e4b8f7349b06dde2d627558d5804` |
| qwenspans `data/activations/R1-1.5B-qwenspans/adding-knowledge_layer16.npy` | `0da0af0bd5b91742603f62769d10ba461154cd137b9f99aecbf51cbfd161027e` |
| qwenspans `data/activations/R1-1.5B-qwenspans/metadata.json` | `cde8a906b7d60716cfed0fffdfdb7f38a75a170b559518814ddab8ceb10e524a` |
| qwenspans `data/activations/R1-1.5B-qwenspans/row_index.json` | `10f4bc33655cf0d67b4dcaeca35c88c815d8b01e0d612dd357ec4b83237c84d6` |
| qwenspans `data/annotated_R1-1.5B__qwen3-235b.json` | `fd0ae79316b1ac84fe0c5770af074af8ae11fbd891568d0522af2c06f6f5ad93` |
| novaspans `data/activations/R1-1.5B-novaspans/backtracking_layer12.npy` | `e601c3debaf188b8f4af30bd14b00ea96eae364f81063e50255116f760126ae7` |
| novaspans `data/activations/R1-1.5B-novaspans/uncertainty-estimation_layer12.npy` | `408335b1f5de96bdefe8c0efe7f0dc3d01879fdee2e8f4bd6ccd57e576d06a55` |
| novaspans `data/activations/R1-1.5B-novaspans/example-testing_layer12.npy` | `68a035fc18daa2b329092809c01697f4a707e7e6a505325c3da17ed9f2354997` |
| novaspans `data/activations/R1-1.5B-novaspans/adding-knowledge_layer12.npy` | `2cea223d6fa9e27f61ee5ec13f592897509fde366db84cc3ce58cd72540babb7` |
| novaspans `data/activations/R1-1.5B-novaspans/backtracking_layer16.npy` | `34565fcf9547395b8da07e65de31a910e09fb60e2854cda70e959b3a55b50176` |
| novaspans `data/activations/R1-1.5B-novaspans/uncertainty-estimation_layer16.npy` | `28223859eb0634ae2e599a30d539067d57b5f475e6c2c21947c4eb88c267dea4` |
| novaspans `data/activations/R1-1.5B-novaspans/example-testing_layer16.npy` | `f08c2ef97d330aa7d13d15f923dbfaae5701a43aaaf3267fc08c245ff72b79bf` |
| novaspans `data/activations/R1-1.5B-novaspans/adding-knowledge_layer16.npy` | `2f1f3872b9613d444a3a03c8cec5332161cf17a9322a58b2d014c329b213a0c6` |
| novaspans `data/activations/R1-1.5B-novaspans/metadata.json` | `5e9b77faa9b7d10e245982638388572a7a0fc7fa6d1cafc44a7581803462bfc7` |
| novaspans `data/activations/R1-1.5B-novaspans/row_index.json` | `cad6548ea2aaa79b1ca1e1f74aa1a3c841feb3f6b8929fbd9e5a75b423b0227e` |
| novaspans `data/annotated_R1-1.5B__nova-pro.json` | `1ebddde7d3f92fd1b70ee9bb569a5e41ce827116e1adb96c275024ca8c781729` |

If any path/hash pairing is ambiguous to the runner, it is **UNRESOLVED — DO
NOT EXECUTE** rather than guessed. The canonical, qwenspans, and novaspans
vintages must never share an annotation or row index by substitution.

Before execution, hashes must be recomputed and compared. A mismatch pauses
execution and requires provenance review; it is not silently accepted.
Missing clipped/first/last L27 activation, metadata, row-index, and annotation
hashes remain **ABSENT / UNRUN** until the gated six-cell extraction. Any
unresolved secondary input makes its whole family
**UNRESOLVED — DO NOT EXECUTE**.

The planned test suite is **ABSENT / UNRUN**. Before the pilot it must test:
known finite/degenerate cdim inputs; wrapper equivalence to
`correlation_dimension_estimate(..., n_bootstrap=0)`; lower-tail p-value and
Holm known answers; deterministic seed coordinates and digest row selection;
within-chain count preservation; identity/no-op inclusion; row alignment; occurrence
and exact-vector duplicate cases, including cross-chain/label hard failures;
equal one-chain weighting; invalid-resample accounting; truncation boundary
cases; schema validation; and a tiny synthetic permutation recovery check.
Existing `tests/test_intrinsic_dim.py` is not sufficient by itself.

## 17. Amendments, results firewall, and reporting

An amendment record must be appended below, never overwritten. Each entry must
state date/time, author, affected sections, old and new text/values, reason,
whether any benchmark or result was accessible, files/hashes accessed, and
author/supervisor signatures. A computational optimisation that changes only
speed still enters the execution log; any change to estimand, statistic,
preprocessing, unit, label, layer, family, tail, B, multiplicity, threshold,
annotator, pooling/window, truncation, duplicate, seed, or failure rule is a
material amendment.

Before the preregistration and statistical-review signatures, no pilot may be
run. After signing, the runner may inspect benchmark runtime, memory,
valid-resample rate, failure messages, and schema conformance, but not the
observed cdim, null draws, p-value, or behaviour comparison when deciding
whether to optimise or narrow. Benchmark JSON must therefore support a
firewalled resource-only view. Any person who sees statistical pilot contents
before a resource amendment must disclose that access.

Full outputs are written atomically to a sealed results directory. The
implementation log and hashes are generated before result summaries. H1 is
adjudicated first, then the secondary families, then scope diagnostics. All
failed, mixed, and null outcomes are retained. No family is rerun with altered
settings because its result is inconvenient. The thesis must lead with the
amended/prospective status and the exact bounded decision-table wording.

The primary output is one strict family envelope at the already frozen
`primary_L27_cdim_null.json` path. It contains exactly four schema-valid H1
descriptive cells, four schema-valid registered-seed H2 cells, four
schema-valid diagnostic-seed H2 cells, the frozen H1/H2 adjudications, the
input and schema hashes, shared execution provenance, and amendment IDs. Every
cell is validated against
`thesis_core_hardening_result_cell_v1.schema.json` plus the invariant checker
before the envelope is atomically sealed. An atomic progress artefact may be
written during execution for interruption recovery, but is removed only after
the sealed result validates; it is never substituted for the final output.

**Amendment register:**

- 2026-07-26, author/design draft; Sections 1, 5–7, 9–10, and 12–18.
  Pre-result formal-spec repair: separated H1 descriptive occupancy from H2
  specificity, reassigned H3/H4, froze the complete-case grid, code/index/digest
  registries, truncation-neutral and representation-neutral selection,
  two-result category matching, exact centred-SVD PR, curvature operator,
  resource-only benchmark, H4 extraction paths, schemas, and secondary hashes;
  fixed the primary at methods-declared common L27; and froze the Monte Carlo
  conditional valid-assignment test with identity assignments retained.
  No benchmark, estimator output, null draw, p-value, or
  result was accessible or generated during the change. Files accessed were
  the approved planning artefacts and existing code/data paths whose hashes
  are recorded above. Author signature: ____________________; supervisor
  signature: ____________________. Until both are signed, this entry and the
  preregistration remain unauthorised for execution.

- 2026-07-26 23:16 BST, author-only resource-pilot amendment; Sections 15,
  17, and 18. Old rule: the blank author, statistical-reviewer, amendment, and
  supervisor sign-offs blocked every pilot. New rule: the author/design owner
  may authorise exactly one CPU-only, no-API, resource-only B=25 pilot for the
  canonical Sonnet/backtracking/L27 cell without the otherwise-required
  statistical-reviewer and supervisor sign-offs. This exception replaces
  those sign-offs for this B=25 resource pilot only. It does not authorise
  B=50, any full or confirmatory analysis, GPU use, API use, data
  regeneration, inspection or serialization of statistical values, or git
  operations. The reason is to measure runtime, memory, validity rate, and
  resource feasibility behind the frozen firewall before seeking authority
  for any further work. No empirical benchmark or result was accessible when
  this amendment was recorded. The registered primary input hashes in
  Section 16 were recomputed and matched before the amendment; the implemented
  runner, tests, and resource schema had SHA-256
  `6f45d9d79d6416a8c20d6d98ffaf44e908476168466fa2ac20aef6634295b26a`,
  `760f158d59f8fe2b0957b71b6f0bacf966808c26c64227ebaa32afa2820874d1`,
  and `b404fa8b86a8598402950772c0ba64a5f3f33631bf9e4c11ef3a1f3113e4f154`,
  respectively. Author/design owner electronic signature: Tony Su, explicit
  consent in the Codex task on 2026-07-26 before execution.

- 2026-07-26 23:20 BST, fail-closed reporting correction after the authorised
  pilot preflight stopped on the registered cross-label exact-vector duplicate
  rule, before any permutation attempt. Affected Sections 15–17 and the
  implementation only. Old behaviour: the CLI raised after the hard stop and
  wrote neither planned resource artefact. New behaviour: that same already
  reached stop condition can be serialized as a zero-attempt `STOPPED`
  resource document with `duplicate_hard_failure`, primary gate `FAIL`, and
  every other gate `UNRUN`; the exception text, duplicate counts, vectors, and
  all statistical values remain excluded. This is a reporting-only correction,
  not authority to change the duplicate rule, inspect the conflict, or rerun
  the empirical pilot. No observed estimate, randomized statistic, comparison,
  or p-value was computed or accessible. Runner and test SHA-256 changed from
  `6f45d9d79d6416a8c20d6d98ffaf44e908476168466fa2ac20aef6634295b26a`
  and
  `760f158d59f8fe2b0957b71b6f0bacf966808c26c64227ebaa32afa2820874d1`
  to
  `972e4a613859dac65e4c0949f70357348456da3c76b41539a69d269b84e2901c`
  and
  `a8169a276d73536ee47d9d2e2fd6b6caea75b20fedb148ba4624d41577ec3bce`,
  respectively. Author/design owner electronic signature: Tony Su, under the
  resource-only reporting authority above.

- 2026-07-26 23:22 BST, execution close-out. The one-time B=25 authorisation
  was consumed by the registered duplicate-audit hard stop. Zero permutation
  attempts ran; zero valid or invalid randomized attempts were produced; no
  estimator output or other statistical value was computed. The canonical
  resource JSON and Markdown hashes are
  `290cd99650202df0763bbfb8ef15c0665107a6daf0f4c5284728ecca83977518`
  and
  `5d9cf192bb0e358b56e12039443da07aba7edd73896d56316381b920e21da382`.
  The empirical B=25 gate is now relocked, B=50 remains unauthorised, and this
  entry does not authorise a retry or a duplicate-rule remedy.

- 2026-07-26 23:45 BST, pre-result duplicate-provenance and resource-pilot
  retry amendment; Sections 7, 15, and 17. Old rule: every cross-label exact
  activation-vector group was an unconditional hard failure. New rule for one
  CPU-only, resource-only B=25 canonical Sonnet/backtracking/L27 retry: after
  occurrence-key collapse, remove every member of a cross-label exact-vector
  group only when all members have the same
  `(chain_id, char_offset, token_start, n_positions)`; never select a label
  winner. Cross-chain groups and cross-label groups spanning distinct
  extraction-window keys remain hard failures. This is a preprocessing-rule
  amendment limited to the resource-pilot retry. It does not authorise B=50,
  full or confirmatory analysis, GPU or API use, data regeneration, thesis
  result-claim editing, or git operations.

  The reason is a completed, read-only provenance audit performed after the
  original hard stop and before this amendment. Across 37,851 registered L27
  rows it found 313 exact-vector duplicate groups: 257 within-label groups and
  56 cross-label groups. All 56 cross-label groups were within one chain and
  one extraction-window key; none was cross-chain, zero-vector,
  constant-vector, or a distinct-window cross-label match. The registered
  annotation records in every cross-label group resolved to the recorded
  chain offset; their texts were prefix-related, consistent with overlapping
  or repeated annotation aliases at a common activation window. The symmetric
  rule removes 136 conflicted rows plus the 391 ordinary within-label
  duplicate losers, leaving 37,324 rows, all 993 chains, all 705 backtracking
  chains, 10,094 backtracking rows, and 956/993 mixed-label chains. Thus it
  makes no outcome-based choice, removes no independent chain, preserves the
  target-chain eligibility set, exceeds the 100-point minimum, and preserves
  the mixed-label informativeness gate.

  No estimator, permutation, randomized attempt, pilot retry, observed
  statistic, null draw, comparison, or p-value was run or accessible during
  the audit or amendment. The audit JSON and Markdown SHA-256 values are
  `560f6174ea91664008629614745e49302dd85c24a53e25c6bb727ab29337ffc6`
  and
  `a8afcf5422014633c4aeb2c030800487c85e032485e4b1471871e8c6b4c55f21`.
  The amended runner, runner tests, auditor, and auditor tests passed 90 tests
  and have SHA-256
  `c405f437149872cd003486e33c0c5fff6ee21fbcae47fb612c92680905e3c5fa`,
  `dbf1a721e1d030696c5632b9577b482a3c01afaeb90e81c90020abd48eef426e`,
  `c418996eed7926e63ce19e050d382afb05f26b90bd2949ae7817b42883f5541a`,
  and
  `87cf9d8326af92ee981c7b010faee729a5916db6d0d28798ca8cd66937c788ac`,
  respectively. The original STOPPED JSON and Markdown remain preserved under
  pre-amendment names with their registered hashes. Author/design owner
  electronic signature: Tony Su, under the explicit instruction in the Codex
  task to proceed autonomously if the provenance result was positive.

- 2026-07-26 23:48 BST, amended resource-pilot retry execution close-out.
  Exactly 25 CPU-only randomized resource attempts ran after the amended
  duplicate preflight; all 25 were estimator-valid and none was invalid.
  Runtime was 3.430844 seconds in total and 0.137234 seconds per attempt; peak
  RSS was 2,780,102,656 bytes. With the frozen 1.25 safety factor, projected
  wall times were 1,886.964200 seconds for the primary family,
  20,756.606200 seconds for the secondary CPU families, and 11,321.785200
  seconds for CPU H4. Projected scratch use was zero. The primary, secondary,
  and CPU-H4 resource gates passed; GPU-dependent H4 remains `UNRUN` because
  no GPU prerequisites were supplied or authorised.

  The firewall retained no observed estimate, randomized estimate, null draw,
  comparison, or p-value. The canonical resource JSON and Markdown SHA-256
  values are
  `9e0dbb7d19e0c36923bc2a1fb5b9c5c14414f04a674faa6f1276129deecb1fe4`
  and
  `924e206930ff83b1cf20609a7e0478df41501a256e17f4727db32f81767e9891`.
  The original zero-attempt STOPPED artifacts remain preserved as
  `backtracking_L27_resource_only_PRE_AMENDMENT_STOPPED.json` and
  `BENCHMARK_PRE_AMENDMENT_STOPPED.md`, with their original registered hashes.
  The retry authorisation is consumed and relocked. B=50, full or
  confirmatory analysis, GPU/API use, data regeneration, thesis result-claim
  editing, and git operations remain unauthorised.

- 2026-07-26 23:58 BST, pre-result full-run authority, duplicate-rule scope,
  execution-order, and output-envelope amendment; Sections 1, 7, 15–18. The
  author explicitly authorised all frozen analysis runs in the Codex task and
  instructed autonomous continuation for the next execution steps. The author
  had previously reported in the same task that the supervisor approved the
  preregistration; no supervisor identity, date, or independently verifiable
  signature was supplied, so this register records the report without
  fabricating a signature. This amendment treats the author's explicit
  execution authority as sufficient for the no-new-spend CPU runs while the
  formal signature blanks remain visible.

  The same-window all-members-removal rule in Section 7 is extended
  prospectively from the resource retry to the registered CPU families. This
  extension is outcome-independent: before this entry, only resource
  feasibility fields were accessible; no observed cdim, randomized cdim, null
  draw, comparison, raw p-value, or adjusted p-value had been retained or
  inspected. Each extraction/annotation vintage must independently show no
  cross-chain or distinct-window cross-label conflict before the amended rule
  can be used.

  Execution order is frozen as follows: (1) hash/alignment preflight; (2) the
  complete four-cell primary Sonnet/L27 family at registered seed 20260726;
  (3) the complete four-cell diagnostic-seed repetition at 20260727; (4)
  atomic schema/invariant validation and sealing of the primary output; (5)
  the registered, currently available CPU secondary families in their frozen
  order; and (6) scope diagnostics whose complete inputs pass their own
  gates. No primary cell or seed may be omitted after another cell is seen,
  and no family may be rerun with altered settings. GPU-dependent H4
  re-extraction remains conditional on its free-space, smoke-alignment,
  projected-time, and VRAM gates. There is still no API spend and no git
  operation. Thesis result-claim edits are deferred until the sealed families
  are independently checked.

  The primary file uses the strict family envelope described above and the
  additional runner/writer symbols named in Section 16. Their implementation
  and test hashes must be appended in a pre-result implementation-lock entry
  before the primary execution gate is activated. Author/design owner
  electronic signature: Tony Su, explicit consent in the Codex task on
  2026-07-26 before any full-result statistic was executed.

- 2026-07-27 00:04 BST, pre-result primary implementation lock; Sections 16
  and 17. The complete primary executor, strict family-envelope builder,
  schema/invariant cell validator, atomic no-overwrite writer, and atomic
  interruption-progress writer were implemented under the 23:58 amendment.
  The executor fixes B=2,500, attempt cap 2,750, registered seed 20260726,
  diagnostic seed 20260727, the four frozen labels/cell indices, and the
  amended same-window all-members-removal rule. It cannot omit a cell, change
  a seed, substitute a destination, overwrite a sealed output, or run while
  the one-time primary gate is false/consumed.

  Ninety-four focused synthetic/unit/schema tests passed. The runner, test
  file, and frozen result-cell schema SHA-256 values are
  `15b49cd29fb6f6f1903a1c438693b8c87d9181fa23bb0bf70287827216751e22`,
  `9b70418cb8fb528aae90c23c1ad3e66c65738c582ab14dc71f0691656eb71d5e`,
  and
  `28ec119e2877560912c70a346b71c351cdbe1df90b249a9d10afde5ca983c21b`,
  respectively. The planned final and progress outputs were absent when these
  hashes were recorded. No full primary estimator, permutation, observed
  cdim, null draw, comparison, raw p-value, or adjusted p-value had been
  executed or accessed. This lock authorises activation of the one-time CPU
  primary gate only; it does not alter any scientific setting.

- 2026-07-27 01:02 BST, sealed primary execution close-out. The one-time CPU
  primary gate was activated only after the 00:04 implementation lock. The
  registered four-cell Sonnet/L27 family ran first with base seed 20260726,
  followed by the complete four-cell diagnostic-seed repetition with base
  seed 20260727. The process exited successfully after 3,395.152717 seconds
  using one worker and recorded peak RSS of 3,325,739,008 bytes. The progress
  checkpoint was removed only after the complete family document passed its
  strict schema and invariant checks and was atomically written. The final
  result is
  `results/robustness/core_hardening/R1-1.5B/primary_L27_cdim_null.json`,
  SHA-256
  `99f9aec51af8447bdb576d46513da51db7418ad13354148d53226819f17abbd1`.
  Its embedded pre-result preregistration SHA-256 is
  `5ae729adf67bd9d9c047342024a700aedac20d80a0e095cf5ba26959fee43065`;
  this close-out necessarily changes the live document hash but not the
  sealed result's execution-time provenance.

  The duplicate/alignment audit retained 37,324 of 37,851 input rows, found
  zero cross-chain groups, and symmetrically removed all members of the 56
  same-window cross-label groups under the registered amendment. The four
  equal-chain samples contained 705, 909, 635, and 880 chains in frozen
  behaviour order. Their observed cdim estimates were 7.207893, 7.364468,
  7.040283, and 8.251715. All four cells were valid and at most 10, so H1 is
  **PASS** under its descriptive decision rule.

  Each of the four registered H2 cells obtained exactly 2,500 valid null
  draws in 2,500 attempts, with zero invalid draws. Registered lower-tail raw
  p-values were 0.842463, 1.000000, 0.894842, and 1.000000; all four
  Holm-adjusted p-values were 1.000000. Thus zero behaviours passed and H2 is
  **FAIL**. The diagnostic-seed raw p-values were 0.825270, 0.998800,
  0.894442, and 1.000000, again with all adjusted p-values 1.000000 and all
  four verdicts unchanged. Across the registered and diagnostic repetitions,
  all 20,000 requested null draws were valid and no cell was
  Monte-Carlo-unstable. This result supports only the frozen operational
  low-dimensional description; it provides no confirmatory evidence that
  the observed behaviour clouds have unusually low cdim under the registered
  within-chain label-permutation null.

  Immediately after successful finalisation, the primary authorisation was
  consumed and relocked, real-matrix loading was disabled, and
  synthetic-validation-only mode was restored. No API, GPU, data
  regeneration, or git operation occurred. Secondary CPU families remain
  prospective until their independent input, duplicate, implementation, and
  atomic-output preflights are complete.

- 2026-07-27 01:11 BST, pre-result CPU-secondary implementation and input
  lock; Sections 6, 7, 9, 14, 16, and 17. The registered secondary loader now
  resolves every path/hash pair directly from the frozen manifest, rejects
  any absent or ambiguous pairing, recomputes every digest, preserves each
  extraction vintage's own metadata, row index, and annotation file, and
  constructs exactly the registered cell-index grids. All 56 manifest
  entries (53 unique files) matched their frozen SHA-256 values. The real
  loader produced exactly 20 five-depth cells at indices 4–23 from five
  distinct layer pools and exactly 24 three-model-annotation cells at indices
  24–47 from six distinct annotator/layer pools.

  Duplicate/alignment preflights were run without calling an estimator or
  generating a randomized assignment. At each of Sonnet layers
  `{11,14,17,20,27}`, the amended rule retained 37,324/37,851 rows, found
  zero cross-chain and zero unresolved cross-label groups, retained all 993
  chains, and left 956 mixed-label chains. At both Qwen3-235B layers
  `{12,16}`, it retained 34,328/35,184 rows, found zero cross-chain and zero
  unresolved cross-label groups, retained 985 chains, and left 899
  mixed-label chains. At both Nova-Pro layers `{12,16}`, it retained
  34,210/35,028 rows, found zero cross-chain and zero unresolved cross-label
  groups, retained 956 chains, and left 837 mixed-label chains. Every arm
  therefore passed the frozen 50% informativeness gate. The smallest
  post-rule behaviour sample was 387 chains, above the 100-point estimator
  minimum.

  The strict secondary executor fixes B=2,500, attempt cap 2,750, registered
  seed 20260726, the complete frozen grids, the amended duplicate rule, and
  the five-depth-before-three-annotation execution order. It writes a
  schema-validated cell after each registered cell only to an atomic progress
  checkpoint; it exposes no sealed family output until every cell completes,
  Holm adjustment is applied across all 20 or all 24 cells, the family
  decision is adjudicated, and the strict family envelope validates. It
  refuses partial grids, altered coordinates, substituted destinations,
  existing progress artifacts, and overwrite of sealed results. Secondary
  families use only the registered seed as specified; no unregistered
  diagnostic-seed secondary family is added.

  Ninety-six focused synthetic/unit/schema tests passed. The runner, test
  file, and frozen result-cell schema SHA-256 values are
  `a524af4d6ab55b931df948494771bb6da83e8e2660f9de1fb22f71af9e775ab0`,
  `d347cbc8e729db74231b67338efefb384d97b416faea6fce6dbf53034d7cea68`,
  and
  `28ec119e2877560912c70a346b71c351cdbe1df90b249a9d10afde5ca983c21b`,
  respectively. Both planned final outputs and both progress outputs were
  absent when these hashes were recorded. No secondary observed cdim, null
  draw, comparison, raw p-value, or adjusted p-value had been executed or
  accessed. This lock permits activation of the previously authorised
  one-time CPU-secondary gates in frozen order; it changes no scientific
  setting and authorises no API, GPU, data-regeneration, or git action.

- 2026-07-27 03:34 BST, sealed five-depth secondary execution close-out. The
  complete 20-cell Sonnet family ran in the frozen cell-index order 4–23 with
  registered base seed 20260726. It exited successfully after 8,528.348477
  seconds using one worker and recorded peak RSS of 3,249,094,656 bytes. All
  20 cells obtained exactly 2,500 valid draws in 2,500 attempts, for
  50,000/50,000 valid draws and zero invalid attempts. The progress checkpoint
  was removed only after 20-cell Holm adjustment, family adjudication, strict
  cell/envelope validation, and atomic finalisation. The sealed result is
  `results/robustness/core_hardening/R1-1.5B/five_depth_cdim_null.json`,
  SHA-256
  `7122d85cb9b8d3f75f18ba84da3b33bb1c3b473302a4bc61710fd079caf8f369`.
  Its embedded pre-result preregistration SHA-256 is
  `161872eec588589dc9fe7d7825e435fc277b7ad6e4d425ccb3d3217111b457df`.

  In behaviour order, the observed cdim values across zero-based layers
  `{11,14,17,20,27}` were:

  - backtracking: `{6.655157, 6.302002, 6.470758, 7.561742, 7.276046}`;
  - uncertainty estimation:
    `{6.842255, 6.850615, 7.083604, 7.601434, 7.045847}`;
  - example testing:
    `{7.287494, 7.380942, 7.672520, 8.665347, 6.738013}`; and
  - adding knowledge:
    `{8.583414, 8.389510, 8.571687, 9.360851, 7.982956}`.

  The corresponding registered lower-tail raw p-values were:

  - backtracking:
    `{0.919232, 0.009196, 0.103958, 0.882447, 0.952419}`;
  - uncertainty estimation:
    `{0.790484, 0.839264, 0.888844, 0.693323, 0.978409}`;
  - example testing:
    `{0.999200, 0.998401, 0.999600, 1.000000, 0.605758}`; and
  - adding knowledge:
    `{1.000000, 1.000000, 1.000000, 1.000000, 1.000000}`.

  No cell passed the single 20-test Holm correction. The smallest adjusted
  p-value was 0.183926 for backtracking at L14; every other adjusted p-value
  was 1.000000. Consequently every behaviour is **depth-mixed/limited** with
  zero of five corrected layer tests passing, and the complete five-depth
  family decision is **FAIL**. This family does not rescue or redefine the
  failed primary H2 result. Immediately after finalisation, the five-depth
  one-time gate was marked consumed. The separately frozen
  three-model-annotation gate remains eligible; no API, GPU, data
  regeneration, or git operation occurred.

- 2026-07-27 06:13 BST, sealed three-model-annotation secondary execution
  close-out. The complete 24-cell family ran in frozen cell-index order 24–47
  with registered base seed 20260726 and without substituting data between
  the Sonnet, Qwen3-235B, and Nova-Pro extraction vintages. It exited
  successfully after 9,434.364229 seconds using one worker and recorded peak
  RSS of 3,393,273,856 bytes. All 24 cells obtained exactly 2,500 valid draws
  in 2,500 attempts, for 60,000/60,000 valid draws and zero invalid attempts.
  The progress checkpoint was removed only after the single 24-cell Holm
  adjustment, family adjudication, strict cell/envelope validation, and
  atomic finalisation. The sealed result is
  `results/robustness/core_hardening/R1-1.5B/three_annotator_cdim_null.json`,
  SHA-256
  `502e00f2862a0fc74f50f8997ce5fab3419ab803572bc21d18740159d442c4ed`.
  Its embedded pre-result preregistration SHA-256 is
  `ad985829e1e2084033cd499bc62c1a40bf96abb7fd5978b6bd4d2387a5a810d9`.
  Recomputed post-run hashes matched all 35 sealed immutable input/code/schema
  paths; only the execution-gate config was intentionally changed after
  sealing to consume and relock the run.

  In frozen behaviour order, the observed cdim values and raw lower-tail
  p-values were:

  - Sonnet L12 cdim `{6.378017, 6.971651, 7.355870, 8.855763}`, raw p
    `{0.211515, 0.998401, 0.978808, 1.000000}`;
  - Sonnet L16 cdim `{6.516491, 7.072737, 7.556666, 8.724277}`, raw p
    `{0.087565, 0.941224, 1.000000, 1.000000}`;
  - Qwen3-235B L12 cdim
    `{6.580709, 6.894760, 7.528953, 8.167999}`, raw p
    `{0.285486, 0.601759, 0.999200, 1.000000}`;
  - Qwen3-235B L16 cdim
    `{6.483215, 6.835079, 7.651611, 8.444653}`, raw p
    `{0.057177, 0.102759, 0.999600, 1.000000}`;
  - Nova-Pro L12 cdim `{6.402738, 7.055895, 7.665020, 8.374933}`, raw p
    `{0.998401, 1.000000, 1.000000, 1.000000}`; and
  - Nova-Pro L16 cdim `{6.439567, 7.099829, 7.799468, 8.534609}`, raw p
    `{0.851659, 0.999200, 0.996801, 1.000000}`.

  Every Holm-adjusted p-value was 1.000000. Consequently no L16 behaviour
  passed for all three model annotations; all four behaviours are
  **annotator-mixed**, and the complete three-model-annotation family decision
  is **FAIL**. This family does not rescue or redefine the failed primary H2
  result. Immediately after finalisation, the three-annotation one-time gate
  was marked consumed, the common CPU-secondary authorisation was disabled,
  real-matrix loading was disabled, and synthetic-validation-only mode was
  restored. No API, GPU, data regeneration, or git operation occurred.

- 2026-07-27 06:28 BST, pre-result CPU scope-diagnostic implementation lock
  and schema-alignment amendment; Sections 10–13 and 16–18. This entry was
  recorded after the registered CPU secondary families were sealed but before
  any H3 or L16 curvature estimator was run. The author had explicitly
  authorised all frozen no-new-spend runs and autonomous continuation in the
  Codex task. The complete Sonnet L27 H3 and Sonnet L16 curvature inputs were
  present; their registered path/hash pairs were rechecked. The H4 extraction
  root and all required first/last/clipped representation sets remained
  absent, so H4 is classified **UNRUN** without GPU use or data regeneration.

  The implementation adds exact one-use CPU gates, strict hash-gated L16
  loading, complete four-behaviour H3 and curvature executors, interruption
  checkpoints, no-overwrite atomic writers, and strict family envelopes. It
  passes the already amended same-extraction-window all-members-removal policy
  explicitly into H3 and curvature duplicate audits; cross-chain and
  distinct-window cross-label conflicts remain hard failures. H3 computes the
  chain-stability and truncation/category results once and writes the planned
  `chain_stability.json` and `truncation_sensitivity.json` envelopes from the
  same exhaustive results, preventing divergent duplicate runs.

  One pre-result schema inconsistency was repaired without changing an
  estimand or decision rule. Old schema behaviour required each valid
  category-matched cell 60–63 to expose one pooled top-level cdim/PR and 500
  top-level draws, contradicting Section 11's exact two-result
  `matched_complete`/`matched_truncated` contract. New schema behaviour
  requires top-level cdim/PR null, zero top-level draws/resamples, and the two
  separately typed valid matched results; their independently computed
  500-draw stability summaries remain inside those two results. The
  curvature cell schema's generic `cdim` and `pr` stability-draw fields carry
  `ratio_chain` and `ratio_control`, respectively, and the family envelope
  records that mapping explicitly; neither is reported as correlation
  dimension or participation ratio.

  Synthetic timing only, using random matrices at the empirical row/width
  scale, measured approximately 0.013–0.036 seconds per PR computation,
  0.035–0.163 seconds per correlation-dimension computation, and 1.588
  seconds per paired-diagnostic ratio call at 560 rows. No empirical
  statistic, comparison, diagnostic draw, or decision was computed or
  accessible during this timing. Ninety-eight focused tests passed. The
  runner, tests, repaired cell schema, and frozen curvature instrument hashes
  are:

  | Existing input / implementation | SHA-256 |
  |---|---|
  | `scripts/thesis_core_hardening.py` | `753a2a2b75959d27f1c64446b6bc307f1f3a7ebd1cc5055d0148a1263dabc1c2` |
  | `tests/test_thesis_core_hardening.py` | `703a3b0842f424722cfecc5f1d57eb8dc90c1a9b1551b8c2b4c41895fc0511b0` |
  | `schemas/thesis_core_hardening_result_cell_v1.schema.json` | `7342997425a708e102ee0f71246f72784736c6770f0e66f16869659fe3d9442b` |
  | `src/curvature.py` | `741218c7544cfc4c955d2b38187882926f552c0c21839891b5aaf507bee9c2a5` |

  The three planned scope outputs were absent when these hashes were
  recorded. This lock activates only the CPU H3 and L16 curvature
  diagnostics in that order. It does not authorise H4 extraction, GPU/API
  use, data regeneration, or git operations.

- 2026-07-27 06:44 BST, sealed H3 execution close-out. The exhaustive
  four-behaviour Sonnet/L27 chain-deletion, truncation, and required
  category-matching diagnostic ran first among the scope diagnostics and
  exited successfully after 791.403887 seconds with one worker and peak RSS
  3,477,094,400 bytes. Every chain, stratum, and matched-side stability
  component obtained 500/500 valid draws; no reduced behaviour, stratum, or
  matched analysis was used. The two outputs were constructed from the same
  in-memory behaviour results, independently validated, atomically sealed,
  and then had their shared progress checkpoint removed:

  - `results/robustness/core_hardening/R1-1.5B/chain_stability.json`,
    SHA-256
    `5d067729be2c55d05ab9ec4848ea9076d8008e22bf4328d51fe437d3d1ceb48f`;
  - `results/robustness/core_hardening/R1-1.5B/truncation_sensitivity.json`,
    SHA-256
    `e0ee2090a2ee4a3a02bb90d62a1c1908d439bc2c7186bdd4297127c14b7fd929`.

  In behaviour order, the full-sample chain-stability cdim/PR pairs were
  `{7.207893/39.127055, 7.364468/37.299701, 7.040283/28.660771,
  8.251715/50.244862}`. The chain-stability leg passed for backtracking,
  uncertainty estimation, and example testing. Adding knowledge failed
  because its full cdim 8.251715 lay just above the registered stability band
  `[7.534143, 8.220373]`; its PR condition remained valid. The chain-only
  family description is therefore mixed, three passes and one failure.

  Every behaviour exceeded the 10-percentage-point category-share trigger:
  `{0.153667, 0.183863, 0.134385, 0.161222}`, so all four exact
  category-matched comparisons were mandatory. The matched complete/truncated
  chain counts were `{197/197, 254/254, 199/199, 249/249}`. All four
  unadjusted truncation rules failed and all four matched rules failed.
  Correlation-dimension differences were within 25% throughout, but the PR
  stability bands did not overlap in any unadjusted or matched comparison.
  Additionally, backtracking's complete-vs-combined PR difference was
  30.31%, and matched uncertainty/example-testing PR differences were 26.14%
  and 29.23%. Consequently each behaviour's truncation/category leg is FAIL;
  the exhaustive cross-product makes all four H3 behaviour statuses **FAIL**
  and the complete H3 family **FAIL**. The registered wording is: “The
  registered chain/truncation robustness criterion was not met.”

  All embedded immutable input, runner, and schema hashes were rechecked with
  zero mismatches; the execution-gate config is intentionally mutable after
  sealing and is excluded from that post-run equality check. The H3 one-use
  gate is now consumed. Curvature remains authorised next in frozen order;
  H4 remains UNRUN. No GPU, API, data regeneration, or git operation occurred.

- 2026-07-27 06:55 BST, sealed L16 curvature diagnostic close-out. The
  complete four-behaviour paired chain/control diagnostic ran after H3 in
  frozen cell order 88–91 and exited successfully after 568.449385 seconds
  with one sequential worker and peak RSS 3,361,488,896 bytes. Each behaviour
  obtained exactly 500/500 finite paired draws and zero invalid pairs. The
  progress checkpoint was removed only after the four cells, field-semantics
  mapping, summaries, and family envelope validated and the planned output
  was atomically sealed:
  `results/robustness/core_hardening/R1-1.5B/curvature_L16.json`, SHA-256
  `9697654295e8f6a533aa2f523a264d2fdd45ddab2dd13be4adf84cf23a21262f`.

  The registered per-behaviour results were:

  - backtracking: chain ratio band `[1.014051, 1.052660]`, median
    `1.033867`, `D_chain=0.033867`, `D_control=0.011101`; bounded-negative
    operator **false**;
  - uncertainty estimation: chain ratio band `[0.998209, 1.033904]`, median
    `1.015315`, `D_chain=0.015315`, `D_control=0.019635`;
    bounded-negative operator **true**;
  - example testing: chain ratio band `[0.983528, 1.018344]`, median
    `1.000000`, `D_chain=0.006092`, `D_control=0.009337`;
    bounded-negative operator **true**; and
  - adding knowledge: chain ratio band `[0.946469, 0.983649]`, median
    `0.964852`, `D_chain=0.035148`, `D_control=0.016620`;
    bounded-negative operator **false**.

  Thus the permitted “no beyond-chain curvature detected by this diagnostic
  at L16” wording applies only to uncertainty estimation and example testing.
  Backtracking is a controlled-layer nonconforming/mixed result because its
  band lies above one and its departure exceeds the control. Adding knowledge
  is a positive controlled-layer result because its band lies below one and
  its departure exceeds the control. The complete four-behaviour curvature
  diagnostic is **MIXED**. It supplies no p-value, no all-layer result, and no
  global-flatness conclusion.

  All embedded immutable input, runner, instrument, and schema hashes were
  rechecked with zero mismatches; the execution-gate config is intentionally
  mutable after sealing and excluded from that post-run equality check. The
  curvature one-use gate is now consumed, all CPU scope execution is
  relocked, and real-matrix loading is disabled. H4 remains **UNRUN** because
  the six aligned representation inputs do not exist and its GPU extraction
  gates were never activated. No GPU, API, data regeneration, or git
  operation occurred.

  After relocking, the only test-file change from the pre-result lock was the
  gate-state assertion from authorised/unconsumed to false/consumed for both
  completed diagnostics. The final focused suite passed 98/98 tests; the
  post-close-out test SHA-256 is
  `62b1f98987e551035969c6ace3b56e28d05c0a02acb640bda8621f9be79606a9`.
  The locked runner and schema hashes remained unchanged.

## 18. Sign-offs and completeness checklist

### Required sign-offs before any pilot

- Author/design owner: ____________________  Date/time: ____________________
- Statistical reviewer: ___________________  Date/time: ____________________
- Supervisor disposition:
  - [ ] Approve as written and authorise the no-paid-resource pilot.
  - [ ] Approve the primary CPU pilot only; hold secondary/GPU work.
  - [ ] Request a pre-result amendment; do not execute.
  - [ ] Decline execution; retain as prospective/unrun.
- Supervisor: _____________________________  Date/time: ____________________

### Author-only resource-pilot exception

- [x] Author/design owner: Tony Su
- [x] Electronic consent recorded in the Codex task before execution.
- [x] Date/time: 2026-07-26 23:16 BST.
- [x] Scope: CPU-only, resource-only B=25 canonical
  Sonnet/backtracking/L27 pilot.
- [x] Statistical values remain firewalled; all broader execution remains
  blocked by the blank sign-offs above.

### Author-only amended resource-pilot retry

- [x] Author/design owner: Tony Su
- [x] Conditional electronic consent recorded in the Codex task before the
  provenance audit and retry.
- [x] Date/time: 2026-07-26 23:45 BST.
- [x] Scope: one CPU-only, resource-only B=25 canonical
  Sonnet/backtracking/L27 retry under the same-window all-members-removal rule.
- [x] Retry completed and gate relocked at 2026-07-26 23:48 BST.
- [x] Statistical values remain firewalled; all broader execution remains
  blocked by the blank sign-offs above.

### Author full-run execution authority

- [x] Author/design owner: Tony Su
- [x] Explicit full-run consent recorded in the Codex task before execution.
- [x] Date/time: 2026-07-26 23:58 BST.
- [x] Scope: frozen no-new-spend CPU primary, complete CPU secondary families
  whose registered inputs validate, and conditionally gated scope
  diagnostics.
- [x] Supervisor approval was reported by the author in the Codex task;
  independent signature details were not supplied and are not invented here.
- [x] No API or git operation; GPU/re-extraction remains conditional on every
  preregistered resource and alignment gate.

### Examiner-audit completeness checklist

- [x] Dated authority, status, freeze, RQ1/RQ2 scope, and known-data disclosure.
- [x] Original prospective/unrun status and every later execution close-out
  are retained; no confirmatory-status upgrade.
- [x] Chain scientific unit, row observation, equal-chain primary estimand, and
  sentence-weighted secondary estimand.
- [x] Exact model, model annotator, labels, canonical pooling/window,
  preprocessing, duplicate rule, and fixed common zero-based primary L27.
- [x] H1–H4 correctly separated and explicit exclusion of linear-subspace, global-flatness, and
  post-training-origin hypotheses.
- [x] Monte Carlo conditional lower-tail within-chain permutation statistic through
  `src/intrinsic_dim.py`, B, seeds, smoothing, failure handling, and Holm
  families.
- [x] Primary four-behaviour L27 family, secondary five-depth Sonnet family,
  and secondary three-annotator L12/L16 family.
- [x] Specificity permutation separated from one-sentence-per-chain stability.
- [x] Duplicate-safe 80%-chain stability band explicitly not called a
  bootstrap CI.
- [x] Truncation, category matching, target-prefix population, and materiality
  rule.
- [ ] All six L27 mean/first/last × clipped/unclipped matrices, metadata, and
  row indices exist and validate on one common occurrence digest. Currently
  ABSENT / UNRUN; H4 cannot execute or pass.
- [x] One-layer curvature boundary and no global-flatness claim.
- [x] Exact decision table with pass, mixed, fail, and stop wording.
- [x] The resource-only benchmark writer/schema is implemented and verified
  not to expose cdim/null/p-values; the amended B=25 retry is complete.
- [x] Planned scripts, JSON schemas, resource outputs, and tests are
  implemented. The primary and both registered CPU-secondary outputs are
  complete and sealed; H3 and L16 curvature scope diagnostics are complete
  and sealed; H4 remains prospective/UNRUN because its inputs are absent.
- [x] Existing input hashes recorded; unresolved checkpoint/commit provenance
  remains unresolved.
- [x] Amendment register, resource-only firewall, and supervisor choices.
- [ ] Author, statistical reviewer, amendment, and supervisor signatures are
  complete. The blank independent-signature fields remain visible; the
  separately recorded author full-run authority governed the completed
  primary, CPU-secondary, H3, and L16 curvature executions.

**PRIMARY, REGISTERED CPU SECONDARIES, H3, AND L16 CURVATURE EXECUTED AND
SEALED; H4 UNRUN BECAUSE ITS SIX ALIGNED REPRESENTATION INPUTS ARE ABSENT.**
