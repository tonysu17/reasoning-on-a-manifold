# PT-B1 Amendment 3 — identity-gate baseline correction

**Date:** 2026-08-20, committed before any PT-B1 arm passes its identity gate
and before any arm's generation is retained. **Post-hoc and disclosed**: this
correction was found while diagnosing a G1 failure. It is a
**measurement-validity fix**, not a relaxation — the G1/G2 bands (0.98 / 0.95 /
0.50), the recipe, the arms, the endpoints, and the ceilings are all unchanged.

**Amends:** the identity-gate baseline in the pre-registration §2 only.

## The defect

§2 specifies "the pooled mean displacement vector (arm minus base)" without
stating *which* extraction supplies the base term. The implementation used the
**stored July base activations** as the subtrahend while the arm term came from
a **fresh extraction on the execution pod**. The resulting vector is therefore

    (arm_new − base_July) = adapter effect + environment drift

and the adapter effect being measured is roughly 0.5% of activation norm — far
too small to survive contamination of that kind.

## Quantitative evidence

The base checkpoint was extracted on the execution pod with **no adapter of any
kind**, on the identical 5,570-row verification subset, and compared with the
stored July base extraction:

| Layer | ‖stored adapter displacement‖ | ‖pure environment drift‖ | ratio | cos(drift, adapter) |
|---|---:|---:|---:|---:|
| 12 | 0.2229 | **1.3404** | **6.01×** | −0.171 |
| 16 | 0.2829 | **1.8356** | **6.49×** | −0.151 |

The first G1 evaluation on `safety1000-s42` reported new-direction norms of
1.3145 (L12) and 1.8051 (L16) — statistically indistinguishable from the pure
drift magnitudes above. The gate was measuring environment drift, in which the
adapter's genuine contribution was a ~15% component. The reported cosines
(−0.006, +0.002) were therefore uninformative about adapter identity, and the
resulting STOP was a false negative.

## The correction

The base term must come from the **same environment as the arm term**, so that
environment drift cancels in the difference:

    direction(arm) = mean(arm_new − base_new)

The pod extracts the base checkpoint on the verification subset **once per
session**, caches it, and every arm's G1/G2 comparison uses that cached
same-environment baseline. The stored July arm and base extractions continue
to supply the reference direction, which is itself a within-July difference and
so is equally uncontaminated. Both sides of the cosine are now
within-environment differences, which is the comparison §2 intended.

## Standing methodological consequence

At this effect size, **activation displacements are not comparable across
extraction environments**: a library/hardware change moved these activations
about six times further than the entire post-training intervention does. Two
consequences, both already consistent with existing practice:

1. Every activation-level contrast in this project must be within-environment.
   The executed RQ3 comparisons satisfy this (base and adapter arms were
   extracted in one July session, with row-identity parity checks recorded),
   so no existing thesis result is affected by this amendment.
2. It independently reinforces Amendment 2's downgrade of PT-B1
   base-referenced comparisons, and extends the caution from generated text to
   activations.

No other section of the pre-registration is modified.
