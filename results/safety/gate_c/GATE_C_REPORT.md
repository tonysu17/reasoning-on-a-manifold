# Gate C report

**Verdict: PASS**

Gate C = *F3 capability control passing on existing 1.5B activations + F2 engine de-confounding merged*.

## What ran

The F3 capability control (`src/safety/capability.py`) was validated on real
`R1-1.5B` Phase-4 pooled residual activations. R1-Distill-1.5B has no safety
contrast, so the control is exercised on positive/negative controls built from
real rows — the real-activation analogue of `tests/test_safety_capability.py`.
Two of them are the gate:

1. **Difficulty-confounded contrast (must FAIL).** The "safety" labels ARE the
   difficulty labels: harmful = backtracking *hard* rows, harmless = *moderate*
   rows, with the capability axis fit on the same hard-vs-moderate difficulty
   contrast. Both pre-registered failure signatures must fire.
2. **Difficulty-orthogonal contrast (must PASS).** A genuine non-difficulty
   separation the control must not over-kill: backtracking-vs-uncertainty
   behaviour identity, difficulty-matched via `difficulty_matched_indices`, with
   the capability axis from held-out `example-testing` difficulty rows.

Two diagnostics contextualise the gate (not gating):

- **Cross-behaviour difficulty confound** — same confound, capability axis from a
  *different* behaviour (the gpt-oss-like transfer case).
- **Independent-estimate behaviour confound** — a sharp, reproducible nuisance
  axis (behaviour identity) estimated from *disjoint chains*, so the failure is
  not algebraically forced the way a same-source confound is.

Deterministic (seed 42); no wall-clock in outputs.

## Difficulty operationalisation

`data/tasks_final.json authored difficulty grade`, scalar
{'moderate': 0.0, 'hard': 1.0}. Model-independent
(authoring-time rated grade — the *rated difficulty* docstring candidate), fixed
before any safety activations. Base-model solve-rate is model-dependent and no
reference-answer length is stored in the task metadata, so this is the admissible
proxy present.

## Result at the primary layer (L16, pre-registered fraction 0.6)

- **Difficulty-confounded (must FAIL):** passed=False  retention=0.000 (min 0.5)  |cos|=1.000 (max 0.5)  d_full=0.395  d_ctrl=0.000
  - failure signature 1 (separation collapsed after partialling): **True**
  - failure signature 2 (refusal axis collinear with capability): **True**
  - correct FAIL: **True**
- **Difficulty-orthogonal (must PASS):** passed=True  retention=1.000 (min 0.5)  |cos|=0.030 (max 0.5)  d_full=1.301  d_ctrl=1.301
  - correct PASS: **True**
- **Diagnostic — cross-behaviour difficulty confound:** passed=True  retention=1.013 (min 0.5)  |cos|=0.446 (max 0.5)  d_full=0.395  d_ctrl=0.401
- **Diagnostic — independent-estimate behaviour confound:** passed=False  retention=0.557 (min 0.5)  |cos|=0.946 (max 0.5)  d_full=1.266  d_ctrl=0.706
- **Axis diffuseness:** difficulty cross-half cos =
  0.5424, difficulty cross-behaviour cos =
  0.4455, behaviour cross-half cos =
  0.9463.
- **F2 engine liveness** (grouped held-out d, chain-grouped): mean d =
  1.270, AUROC = 0.813, folds used
  5 — the de-confounded engine runs on real activations.

Thresholds (pre-registered in `capability.py`): retention_min =
0.5, alignment_max = 0.5.

## Layer sweep (robustness)

| layer | conf.passed | conf.retention | conf.\|cos\| | orth.passed | orth.retention | orth.\|cos\| |
|---|---|---|---|---|---|---|
| 11 | False | 0.000 | 1.000 | True | 1.006 | 0.063 |
| 13 | False | 0.000 | 1.000 | True | 1.006 | 0.054 |
| 16 | False | 0.000 | 1.000 | True | 1.000 | 0.030 |
| 19 | False | 0.000 | 1.000 | True | 0.994 | 0.052 |
| 22 | False | 0.000 | 1.000 | True | 0.998 | 0.037 |

The confounded contrast must show `passed=False`; the orthogonal contrast
`passed=True`, across the sweep.

## F2 verdict

**Merged: True.** src/safety/fingerprint.py (commit
02fad06) — de-confounded fingerprint engine (grouped held-out Cohen's d, permutation null, bootstrap CI, fractional-depth layer selection) is an ancestor of HEAD.

## What the instrument's two legs actually do on real activations

- **Collinearity (signature 2)** is the robust detector: |cos| is ~1.0 for a
  confound whose axis matches the capability axis and ~0.03 for the
  difficulty-orthogonal contrast — clean separation.
- **Partialling/retention (signature 1)** collapses to 0 for a same-source
  confound (removing the exact mean-difference direction, then re-taking
  diff-of-means, is zero by construction). For an *independent-estimate* confound
  it is conservative on real multi-dimensional signals: the
  behaviour-identity confound retains 0.56 at |cos|
  0.95, still FAILing via collinearity.
  Both legs must pass for `passed=True`, so the control fails whenever either
  fires — the intended asymmetry.

## Caveats

- **R1 difficulty is a diffuse axis.** The authored hard/moderate grade does not
  form a sharp, reproducible capability direction (cross-half cos
  0.5424, cross-behaviour cos
  0.4455), unlike behaviour identity
  (cross-half cos 0.9463). A difficulty confound
  is therefore only stageable as a *same-source* contrast; an
  independent-estimate difficulty axis is (correctly) not flagged, because a
  non-reproducible axis is not a genuine confound. On gpt-oss the capability arm
  and harmful/benign difficulty come from different prompt sets, so this
  reproducibility question must be checked there before the partialling leg is
  trusted.
- Validation runs at R1-1.5B width d=1536 with N in the thousands per side
  (N > d): the diff-of-means separation is well-powered here. The gpt-oss H1 read
  will be N≈300 harmful/benign at d=2880 (N << d), the winner's-curse regime the
  F2 held-out / permutation-null engine exists to handle — this gate validates
  the control's *decision logic*, not that regime's power.
- Site/layer differ from what H1 will read on gpt-oss: these are Phase-4
  per-sentence pooled residual rows from R1 reasoning chains, not DSR-labelled
  gpt-oss analysis-channel spans; the primary layer is chosen by fractional depth
  on 28 R1 layers, not gpt-oss's 24.
- Difficulty here is a binary authored grade (hard/moderate); the gpt-oss P0
  capability arm carries an *integer* rated grade (4/5), and the P0 harmful/benign
  chains carry `difficulty=null` — harmful<->benign difficulty matching there
  needs grades populated on those chains (or read from the P2 shard metadata)
  before the control can match strata.
