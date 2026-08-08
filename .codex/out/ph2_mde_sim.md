# Phase-2 transport MDE simulation

**Status:** sizing model only; not inferential evidence and not a Phase-2 result.

Seed 20260802; 5,000 Monte Carlo trials per cell; one-sided alpha=0.05; target power=0.80.

## Calibration acceptance

The exact Phase-0 `single_direction` assumptions reproduce **f*=1.293** at n=49 (target 1.293; relative error 0.00%; PASS against the ±10% check).
This calibration is retained as a legacy reference, not used as the paired design's variance model.

## MDE curves

### sentence_fraction

Empirical E8 Delta_floor: +0.053789 over 49 paired tasks. Estimated shared-task fraction of observed delta variance: 86.5%.

| design | f*(50) | f*(100) | f*(150) | f*(200) | f*(300) | f*(400) | first tested n with f* <= 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| unpaired_across_models | 0.987 | 0.690 | 0.565 | 0.502 | 0.421 | 0.369 | 300 |
| task_paired_across_models | 0.435 | 0.296 | 0.250 | 0.207 | 0.177 | 0.158 | 50 |

### per_1k_tokens

Empirical E8 Delta_floor: +0.925493 over 49 paired tasks. Estimated shared-task fraction of observed delta variance: 91.0%.

| design | f*(50) | f*(100) | f*(150) | f*(200) | f*(300) | f*(400) | first tested n with f* <= 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| unpaired_across_models | 0.999 | 0.692 | 0.583 | 0.500 | 0.432 | 0.365 | 300 |
| task_paired_across_models | 0.369 | 0.250 | 0.193 | 0.177 | 0.156 | 0.135 | 50 |

## README — what this simulation does and does not establish

- The independent unit is the task. The E8 arm and its energy-matched floor are pooled within task before any resampling. The primary source cell is `backtracking|single_direction` at sealed alpha=1.0.
- `unpaired_across_models` samples different empirical task profiles for the base and target batteries. `task_paired_across_models` samples the same task profile for both, while drawing independent arm noise. This keeps cross-model task pairing separate from within-model arm/floor pairing.
- The hierarchical variance split is model-based. Observed per-task Delta_floor variance is decomposed into a binomial sentence-label noise floor and a residual shared task component. Sentence labels within a chain are not guaranteed independent; the binomial component may therefore understate arm noise and make the paired design optimistic.
- The target battery is simulated by multiplying both the mean effect and its task-specific component by `(1 - attenuation)`. Independent model-specific arm noise is not attenuated. Other transport failures can have different variance structures.
- Power uses the 95th percentile of a separately simulated null distribution of the mean base-minus-target effect. `f*` is linearly interpolated only between the sealed 0.1 attenuation grid points after applying a monotone power envelope to Monte Carlo jitter.
- Counts above the 49 observed E8 tasks are bootstrap extrapolations from the same empirical task distribution. They do not demonstrate that a newly collected 300-task battery has the same difficulty mix, effect variance, annotation quality, or support.
- The per-1k estimand counts target-labelled sentences per 1,000 generated tokens. It is a variance-reduced candidate only if Phase 2 seals it before outcome inspection; it is not the E8 headline estimand.
- The simulator sizes attenuation of the existing grounded single-direction effect. It does not rescue the archived manifold first cut, whose floor was misassigned. Target-model sham/random-orthogonal gates, matched count/energy floors, multiplicity, and injection-recovery still belong in the sealed Phase-2 protocol.
