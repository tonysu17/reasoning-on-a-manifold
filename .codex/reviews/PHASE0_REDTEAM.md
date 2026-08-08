# Phase-0 red-team review

**Date:** 2026-08-02  
**Scope:** read-only audit of `pt13b_l17_curve.py`, `ph0_mde_firstcut.py`,
`results/prereg/PHASE0_TRANSPORT_FREEZE_2026-08-02.md`, and their executing/reference
paths. No reviewed file or result artefact was edited or regenerated.  
**Repository state:** reviewed at `5fd66637f34981c6e79109c6f0ecbee04d71ff5b`; the
reviewed Phase-0 scripts, freeze, and result artefacts were untracked in a dirty worktree.
The numbers below are reproducible from the current files, but execution provenance remains
the provenance recorded by the artefacts (`git_dirty: true`), not a clean committed run.

## Verdict

`pt13b_l17_curve.py` correctly reproduces the R1-compression aggregation, applies the sealed
SVD tail-shrink operation, and withholds its mapping after B2 fails. Its local substrate is
nevertheless structurally invalid, and B2's numeric test is not sufficient to enforce that
fact on a future rerun.

`ph0_mde_firstcut.py` is valid for the current `single_direction` sentence-fraction data only
after its extra conservatism is stated. Its `manifold_k5` result is not an E8
`manifold_k5` MDE: the script uses the wrong floor, does not pool replicated subspace records,
and emits a numeric fallback after its own estimand gate fails. The authoritative E8 value is
`+0.06982169433792763` on 49 tasks against `random_subspace_k5`; the reported `+0.049992...`
on 48 tasks is a different arm/floor contrast.

## Findings

### F1 — CONFIRMED (critical): `manifold_k5` is paired to the wrong floor

- `ph0_mde_firstcut.py:27` sets one global floor, `energy_matched_random`, for both arms.
- The authoritative contract is `src/delta_floor.py:16-19,67-80`:
  `single_direction -> energy_matched_random`, but
  `manifold_k5 -> random_subspace_k5`.
- `delta_floor_report.json` records the same pairings. The authoritative manifold cell is
  `n=49`, `delta_floor=+0.06982169433792763`.
- The Phase-0 recomputation instead intersects `manifold_k5` with
  `energy_matched_random`, giving `n=48`, `+0.04999247691412346`. This fully resolves the
  handoff puzzle; neither a per-1k estimand nor alpha handling caused it.

Consequence: the archived manifold `f*=1.38` and `n=366` do not size the sealed manifold
contrast and must not be promoted into the Phase-2 preregistration.

### F2 — CONFIRMED (high): replicated subspace floors are overwritten, not pooled

`ph0_mde_firstcut._rates` assigns `out[base_task_id] = ...`, so the last row wins. E8 requires
all `#s` and `#rs` records to be pooled within task before resampling
(`src/delta_floor.py:100-130`). The current backtracking `random_subspace_k5` data contain 98
annotated rows over 50 base tasks. Merely correcting the floor name while retaining `_rates`
would yield `+0.06234156510076139`, still not the authoritative `+0.06982169433792763`.

Consequence: the simulation implementation should call the authoritative per-task extraction
or reproduce its indexing and pooling exactly; it should not reuse `_rates` as written.

### F3 — CONFIRMED (high): a failed estimand gate still produces a headline MDE

The docstring says a failed recovery reports “only an SE-ratio fallback.” The code instead
sets `delta_for_f` to the mismatched recomputed delta and calculates a normal headline
(`ph0_mde_firstcut.py:109-120`); the Markdown then prints it identically to a passing cell.
No SE ratio to an authoritative estimand is calculated.

With the correct floor and authoritative pooling, the manifold result under the script's
extra-conservative variance assumptions is approximately `f*=0.983`, `n*=190`, not
`f*=1.38`, `n*=366`. Under the literal preregistered paired-within-battery SE it is
approximately `f*=0.776`, `n*=119`. These are audit recomputations, not promoted results;
the planned simulation remains the sizing authority.

### F4 — CONFIRMED (medium): the code adds an undeclared second loss of pairing

Freeze section 1.4 says to bootstrap the paired E8 task-level `Delta_floor` to obtain the
single-battery SE, then multiply by `sqrt(2)` for an unpaired comparison between two model
batteries. The code additionally breaks arm/floor task pairing inside each battery, takes the
maximum of paired and arm/floor-unpaired SEs, and then applies `sqrt(2)`
(`ph0_mde_firstcut.py:103-110`). This is conservative, but it is not the exact frozen
assumption and “no cross-model task pairing” does not by itself justify discarding the
within-model arm/floor pairing.

For `single_direction`, the literal frozen calculation gives `f*=1.071` and `n*=225`; the
additional conservative choice gives the archived `f*=1.293` and `n*=328`. The handoff's
“about 330” is reproducible under that extra assumption, but should be labelled accordingly.

### F5 — CONFIRMED (medium): estimand “recovery” is a circular checksum, not validation

The script chooses between sentence fraction and per-1k by whichever is closest to the
reported delta (`ph0_mde_firstcut.py:86-97`). That comparison cannot independently validate
the report. The authoritative analysis removes the ambiguity directly:
`src/delta_floor.py:56,100-130` calls `behaviour_fraction`, and `src/evaluation.py:28-33`
defines the estimand as target-labelled sentences divided by annotated sentences.

The exact `single_direction` match is therefore a useful implementation checksum once the
source contract is known, but not independent evidence that the estimand was discovered
correctly.

### F6 — CONFIRMED (medium): Gate B2 is numeric but the invalidity is structural

The stored `Xout_*` rows are class-selected, non-contiguous token samples created by
`select_class_token_indices` (`18_loop_geometry.py:215-228` and
`src/loop_geometry.py:265-300`). A windowed sequential-state statistic is undefined on those
rows regardless of whether their median PR happens to lie within 20% of the full-sequence
median. The current B2 ratio is 1.356 and fails, so this execution safely withholds mapping.
However, `pt13b_l17_curve.py:139-159` would permit mapping if the same structurally invalid
data happened to pass the numeric tolerance.

Consequence: pod s3 full-sequence states remain required. Any future local version should
make contiguity/provenance a hard gate, with the distributional check secondary.

### F7 — CONFIRMED (low): aggregate variance-removed statistic was not frozen

The taskwise formula at `pt13b_l17_curve.py:129` is correct, but the curve maps a median dPR
to a **mean** taskwise variance-removed fraction (`:134-136`). Freeze section 1.1 specifies
the median aggregation for dPR but does not specify the aggregation for variance removed.
This is not the cause of the current failure, and no mapping is reported, but the full-sequence
analysis should seal whether the target is mean taskwise variance removal, median taskwise
variance removal, or a taskwise inverse followed by aggregation before interpreting a
percentage.

## Checks that were refuted as defects

### R1 — REFUTED: aggregation-mirror error

Gate A mirrors `30_r1_compression._rep_summaries` and `stage_analyse` for PR: per-task
`nanmean` over the window grid, paired checkpoint-minus-R1 deltas, then median over tasks.
It reproduces DeepScaleR `-0.016708630210242603` and STAR1
`-0.2169802000394334` exactly for 200 tasks.

### R2 — REFUTED: SVD tail-shrink or taskwise variance formula error

The implementation centers the task matrix, preserves the first `k` singular values, scales
the tail singular values by `c`, reconstructs, and re-runs the sealed windowed PR instrument.
The removed centered variance is exactly
`1 - sum(s_scaled^2) / sum(s^2)`, consistent with `pt13_contraction_mde.py:55-70`.
The identity round trip has worst absolute dPR `3.55e-13`, comfortably inside Gate C.

### R3 — REFUTED: wrong one-sided critical values or an intrinsically wrong `sqrt(2)`

`z_.95 = 1.6449` and `z_.80 = 0.8416` are appropriate for alpha=.05 one-sided and
80% power under a normal approximation. `sqrt(2)` is appropriate when comparing two
independent, equal-variance battery estimates. The defect in F4 is which single-battery SE
the code feeds into that formula, not the formula itself.

### R4 — REFUTED: alpha or per-1k explanation for the manifold discrepancy

Both authoritative cells use sealed `alpha=1.0`, and the E8 headline estimand is the sentence
fraction. Correct floor selection plus authoritative pooling reproduces both report values to
machine precision.

## Runnable read-only reproductions

Run from the analysis repository root. These snippets only read files and print results.

### 1. Reproduce the authoritative pairings and deltas

```sh
python - <<'PY'
import json
from src.delta_floor import floor_for_arm, per_task_fraction

p = 'results/eval/R1-1.5B__E1/'
ann = json.load(open(p + 'annotated_steered.json'))
steered = json.load(open(p + 'steering_results.json'))
for arm in ('single_direction', 'manifold_k5'):
    floor = floor_for_arm(arm)
    a = per_task_fraction(steered, ann, 'backtracking', arm, 1.0)
    f = per_task_fraction(steered, ann, 'backtracking', floor, 1.0)
    tasks = sorted(set(a) & set(f))
    delta = sum(f[t] - a[t] for t in tasks) / len(tasks)
    print(arm, '->', floor, 'n=', len(tasks), 'delta=', repr(delta))
PY
```

Expected:

```text
single_direction -> energy_matched_random n= 49 delta= 0.05378923565449617
manifold_k5 -> random_subspace_k5 n= 49 delta= 0.06982169433792763
```

### 2. Show the wrong-floor result and the last-write-wins pooling error

```sh
python - <<'PY'
import importlib.util, json

spec = importlib.util.spec_from_file_location('mde', 'ph0_mde_firstcut.py')
mde = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mde)
rows = json.load(open('results/eval/R1-1.5B__E1/annotated_steered.json'))

arm = mde._rates(rows, 'manifold_k5', 'backtracking')
for floor_name in ('energy_matched_random', 'random_subspace_k5'):
    floor = mde._rates(rows, floor_name, 'backtracking')
    tasks = sorted(set(arm) & set(floor))
    delta = sum(floor[t]['frac'] - arm[t]['frac'] for t in tasks) / len(tasks)
    print(floor_name, 'n=', len(tasks), 'delta=', repr(delta))
PY
```

Expected: the wrong floor gives 48 / `0.04999247691412346`; the correct floor with
last-write-wins instead of E8 pooling gives 49 / `0.06234156510076139`.

### 3. Reproduce the paired-versus-extra-conservative MDE calculations

```sh
python - <<'PY'
import json, numpy as np
from src.delta_floor import floor_for_arm, per_task_fraction

p = 'results/eval/R1-1.5B__E1/'
ann = json.load(open(p + 'annotated_steered.json'))
steered = json.load(open(p + 'steering_results.json'))
rng = np.random.default_rng(20260802)
B, Z = 10_000, 1.6449 + 0.8416
for arm in ('single_direction', 'manifold_k5'):
    floor = floor_for_arm(arm)
    aa = per_task_fraction(steered, ann, 'backtracking', arm, 1.0)
    ff = per_task_fraction(steered, ann, 'backtracking', floor, 1.0)
    tasks = sorted(set(aa) & set(ff))
    a = np.array([aa[t] for t in tasks])
    f = np.array([ff[t] for t in tasks])
    n, delta = len(tasks), (f - a).mean()
    i = rng.integers(0, n, size=(B, n))
    j = rng.integers(0, n, size=(B, n))
    se_paired = (f[i] - a[i]).mean(1).std(ddof=1)
    se_unpaired_arms = (f[i].mean(1) - a[j].mean(1)).std(ddof=1)
    for label, se in [('frozen-paired-SE', se_paired),
                      ('extra-conservative', max(se_paired, se_unpaired_arms))]:
        fstar = Z * np.sqrt(2) * se / abs(delta)
        nstar = int(np.ceil(n * (fstar / 0.5) ** 2))
        print(arm, label, 'f*=', round(fstar, 6), 'n*=', nstar)
PY
```

### 4. Confirm the `Xout_*` source and current gate outcomes

```sh
rg -n 'select_class_token_indices|Xout_' 18_loop_geometry.py src/loop_geometry.py
python - <<'PY'
import json
r = json.load(open('results/safety_posttrain/pt13b_l17_curve.json'))
print(json.dumps(r['gates'], indent=2))
assert r['mapping'] is None
PY
```

## Phase-2 handoff

The simulation should treat the current `single_direction` value as a reproducibility target
only under its explicitly named extra-conservative assumptions. For primary simulation inputs,
derive per-task outcomes via the authoritative E8 pooling path; keep arm/floor pairing,
cross-model task pairing, and shared-task effects as separate design switches. Do not use the
archived manifold first-cut MDE.
