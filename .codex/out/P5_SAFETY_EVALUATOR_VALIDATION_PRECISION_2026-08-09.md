# P5 replacement safety-evaluator validation precision

**Status:** zero-spend design calculation; no endpoint labels inspected  
**Independent validation unit:** one checkpoint response  
**Primary uncertainty:** binomial uncertainty in endpoint-specific agreement with the adjudicated human reference  
**Interval:** two-sided 95% Wilson score interval

## Decision

The frozen 24-response subset is adequate as a feasibility screen, but it is not large enough to qualify an automated evaluator as a high-accuracy primary measurement. The full 48-response `01` half is a bounded pilot validation. Even it supports only endpoint-local, pilot-level conclusions; the later powered study needs a larger human audit if automated labels are to carry primary evidential weight.

## Exact precision implications

For an endpoint observed on 24 eligible responses:

- 24/24 agreement has a 95% Wilson interval of approximately 0.862–1.000;
- 23/24 has an interval of approximately 0.798–0.993;
- 22/24 has an interval of approximately 0.742–0.977.

For 48 eligible responses:

- 48/48 agreement has an interval of approximately 0.926–1.000;
- 47/48 has an interval of approximately 0.891–0.996;
- 46/48 has an interval of approximately 0.860–0.988.

The harmful-only or benign-only endpoints have only half the combined sample. Therefore the 24-response initial subset supplies 12 observations per stratum, and the full 48-response `01` half supplies 24 per stratum. At 12/12, the lower Wilson bound is only approximately 0.758. Endpoint-specific sensitivity and specificity may have still smaller denominators because they depend on human-reference class prevalence.

If true aggregate agreement is near 0.95, approximately 139 eligible cases are required for a Wilson interval whose maximum half-width is at most 0.05 under the integer calculation used here. If agreement is near 0.90, the corresponding requirement is approximately 194. These are aggregate-agreement calculations, not guarantees for sensitivity, specificity, individual domains, or rare classes.

## Frozen interpretation rule

1. The first 24 responses test operational feasibility, schema completion, evidence binding, and obvious construct failure.
2. No result from 24 responses can promote either automated evaluator to primary status.
3. If the feasibility screen does not expose a stopping defect, validation expands to all 48 `01` responses; the second half is not selected based on endpoint values.
4. On 48 responses, report endpoint prevalence, confusion matrices, sensitivity, specificity, balanced accuracy, and Wilson intervals. Raw agreement, Cohen's kappa, and Gwet's AC1 are complementary summaries.
5. An automated evaluator may be used for pilot sizing only with an explicit “human-validated pilot measurement” qualifier and only for endpoints that show adequate class coverage and agreement. Other endpoints remain human-primary or unresolved.
6. Promotion to the powered study requires a new, larger, manifest-frozen human audit sized around the actual pilot prevalence and error rates. Do not reuse the 48 pilot validation rows as if they were an independent powered-study audit.

## Missingness and multiplicity

Malformed outputs, uncertain labels, and truncation-unresolved cases remain missing. They are not disagreements and are not coerced to “no”; completion and agreement are reported separately. Each endpoint is validated separately. A single aggregate pass cannot rescue an endpoint whose class-specific performance is unresolved.

## Reproducibility note

Values above use the standard Wilson interval with `z = 1.959963984540054` and no continuity correction. The calculation is independent of checkpoint arm and did not read or estimate any arm-labelled behavioural difference.
