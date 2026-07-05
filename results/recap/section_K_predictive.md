## K. Predictive-Geometry Extension (Scripts 14–17, current branch)

*Branch: `predictive-geometry-of-reasoning`. Library: `src/predict/`. Runners: `14_label_correctness.py`, `15_predict_gate.py`, `16_residual_geometry_sweep.py`, `17_rung2_compare.py`. Results: `results/predict/R1-1.5B/` (flawed) and `.../corrected/` (citable).*

---

### K.0 What this stage is, in one breath

This is the only stage in the repo that turns the thesis's *static* "reasoning is a geometric process" claim into a *dynamical* one. Instead of describing where reasoning-step embeddings sit, it trains a **next-step predictor** over the ordered sequence of chain-of-thought step embeddings `x_1 … x_T`, and studies the **residual** `r_t = x_{t+1} − f(x_≤t)` — the part of the next reasoning step the predictor could not anticipate. The bet (from `PREDICTIVE_GEOMETRY.md` §3):

> "given the embedding of reasoning steps `x_1 … x_t`, predict `x_{t+1}`. Study the **residual** … the part of the next step the predictor could *not* anticipate. The bet is that where the predictor breaks (sharp residuals) localizes the *branch points / backtracks* of reasoning, and that the structure of those breaks separates correct from incorrect chains."

This is the JEPA (Joint-Embedding Predictive Architecture) idea — predict the future in a learned latent space, discard unpredictable detail — applied to reasoning trajectories. The framing motivation is that ch07 of the thesis found per-behaviour *curvature* confounded by within-chain autocorrelation and explicitly "relocated it as an open question, to the trajectory"; this extension is the apparatus the thesis promised but never built.

### K.1 The staged "rung" ladder (why it is gated)

The method is an **escalation ladder** (`PREDICTIVE_GEOMETRY.md` §6), each rung gated on beating the previous rung *and* two nulls:

- **Rung 0 — predictor-free.** Raw-trajectory Frenet curvature on difficulty-matched chains (the SSP-style smoothness baseline). "Tells us if *any* trajectory geometry separates correctness before we build anything."
- **Rung 1 — linear, no anti-collapse.** A ridge predictor of the **displacement** `x_{t+1}−x_t`. Crucially the residual carries **no anti-collapse regulariser**, so reading geometry off it is not circular. *(Built and tested.)*
- **Rung 2 — small JEPA.** A compact learned MLP predictor with a SIGReg or Barlow-Twins anti-collapse term, pursued only to *beat* Rung 1.
- **Rung 3 — apex.** Goal-conditioned latent rollout + **causal steering** along the predicted next-step direction via `src/steered_inference.SteeredModel`. **Not built.**

The gating discipline is hard-coded in `15_predict_gate.py::_gate_layer`:

```python
beats = (np.isfinite(r1)
         and r1 > max(a["rung0_curvature"]["auc_oof"],
                      a["persistence_step"]["auc_oof"],
                      a["length_gap_trunc"]["auc_oof"])
         and lp.p_value < 0.05 and ss.p_value < 0.05)
res_blocks["status"] = "rung1_beats_baselines_and_nulls" if beats else "no_rung1_advantage"
```

A Rung-1 result is declared interesting only if its residual-AUC beats curvature, persistence, *and* the length/gap/truncation control, *and* both nulls reject. This is the same confound register as the rest of the thesis.

### K.2 The predictor itself (Rung 1)

The core object lives in `src/predict/predictor.py::oof_residuals`. It is a ridge regression on the **displacement** under chain-grouped cross-validation. The "delta" target is the load-bearing design choice — it makes the predictor a residual/skip connection so the residual measures only the *unpredictable* part of the step and the learned map can only *improve* on persistence (`f(x_t)=x_t`):

```python
# src/predict/predictor.py :: oof_residuals
# Fit on the displacement (delta) or the absolute next embedding.
fit_target = (Xn - Xh) if config.target == "delta" else Xn
pred = np.full_like(Xn, np.nan)  # predicted NEXT embedding either way
gkf = GroupKFold(n_splits=n_splits)
for tr, te in gkf.split(Xh, groups=groups):
    if config.standardize:
        scaler = StandardScaler().fit(Xh[tr])
        Xtr, Xte = scaler.transform(Xh[tr]), scaler.transform(Xh[te])
    else:
        Xtr, Xte = Xh[tr], Xh[te]
    model = Ridge(alpha=config.alpha)
    model.fit(Xtr, fit_target[tr])
    yhat = model.predict(Xte)
    pred[te] = (Xh[te] + yhat) if config.target == "delta" else yhat
residuals = Xn - pred
```

`GroupKFold(groups=chain_id)` enforces **CF-2** (the chain confound): a chain never appears in train and test of the same fold, so the effective N is the chain count, not the step count. The scaler is fit on train folds only.

The residual is then aggregated to **per-chain features** (`chain_residual_features`), five numbers per chain:

```python
# src/predict/predictor.py :: RESIDUAL_FEATURE_NAMES
"resid_mean",      # mean residual norm over the chain's pairs
"resid_std",       # spread of residual norm (bursty vs uniform error)
"resid_max",       # largest single-step prediction error (sharpest branch)
"resid_slope",     # trend of residual norm across the trajectory
"resid_dir_churn", # mean (1 - cos) between consecutive residual vectors
```

`resid_max`/`resid_dir_churn` are the candidate *branch-point / backtrack* localizers (H3); `resid_mean` is the overall-predictability magnitude (H1). Pairs are sorted by `pos` within each chain before computing the temporal features (slope, churn), so the order-dependent features are meaningful.

### K.3 The gate metric (chain-grouped AUC) and the two nulls

The functional test ("does residual geometry predict chain correctness?") is a **chain-grouped logistic probe** giving a pooled out-of-fold ROC-AUC (`src/predict/evaluation.py::grouped_auc`). Each chain is one row and its own group; rows with NaN labels are dropped; it returns NaN gracefully when underpowered:

```python
# src/predict/evaluation.py :: grouped_auc
mask = ~np.isnan(labels)
X, y, g = features[mask], labels[mask].astype(int), groups[mask]
...
oof = np.full(y.shape[0], np.nan)
gkf = GroupKFold(n_splits=k)
for tr, te in gkf.split(X, y, groups=g):
    if np.unique(y[tr]).size < 2:
        continue
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(Xtr, y[tr])
    oof[te] = clf.predict_proba(Xte)[:, 1]
...
out["auc_oof"] = float(roc_auc_score(y[valid], oof[valid]))
```

Two nulls (`src/predict/nulls_predict.py`), both reusing the project's Phipson–Smyth smoothed permutation p-value (`src/nulls._smoothed_p`), discipline every positive claim:

- **`label_permutation_null`** — permutes correctness labels (optionally **within difficulty strata**, so difficulty composition is held fixed) and recomputes the grouped AUC. Tests whether the residual features carry correctness info beyond chance/strata composition.
- **`step_shuffle_null`** — permutes the **order** of reasoning steps within each chain (`step_shuffle_within_chain`) and recomputes the *entire* residual-AUC statistic. Tests whether the signal needs the genuine temporal order, or survives on the static cloud of steps. This is the sharpest test: it directly asks whether the "predictive geometry" framing is doing any work over a static-cloud probe.

The unit tests (`tests/test_predict.py`) are careful here: `test_step_shuffle_null_kills_order_dependent_signal` plants a separation that lives *only* in temporal predictability and confirms the shuffle erases it; `test_step_shuffle_null_ignores_static_magnitude_signal` plants a residual-*magnitude* separation and confirms the step-shuffle null does **not** call it significant. This documents exactly what the null tests — and foreshadows the real result.

### K.4 Rung 2 (the JEPA) and the anti-collapse circularity guard

`src/predict/jepa.py::oof_residuals_jepa` is a deliberate **drop-in** for the ridge `oof_residuals` — same `pairs` input, same `ResidualResult` output — so the whole downstream stack (features, AUC, nulls) works unchanged. The model is a tiny one-hidden-layer GELU MLP. Two anti-collapse terms are offered (`barlow`, `sigreg`); the critical design note is that anti-collapse lives in the *loss*, never in the geometry read-out:

```python
# src/predict/jepa.py (module docstring)
# CRITICAL (circularity): we never read absolute intrinsic-dim / isotropy off the
# JEPA *latent* — that would be inflated by the very regulariser above. Geometry is
# read only off the RESIDUAL (x_{t+1} - pred), and only relative/functional claims
# (JEPA-vs-ridge, success-vs-failure) are made.
```

The Barlow term drives the prediction/target cross-correlation toward identity (decorrelating predicted coordinates so outputs cannot collapse onto a low-rank set); the SIGReg term is a LeJEPA-style sketched isotropic-Gaussian penalty over random unit projections (Cramér–Wold). The JEPA path is fully deterministic (`torch.use_deterministic_algorithms(True)`, explicit per-fold `Generator`), and `test_ar1_recovery_determinism` asserts bitwise-identical residuals across runs.

### K.5 The data + the labels (Script 14, the only API spend)

The trajectories are built by `src/predict/trajectory_dataset.py` from the **existing** Phase-4 mean-pooled activations for the 4 behaviours (backtracking, uncertainty-estimation, example-testing, adding-knowledge), all 28 layers. A chain's trajectory is therefore a **sparse 4-behaviour subsequence** of its real reasoning; each pair carries its `gaps = orig_indices[t+1] − orig_indices[t]`, with a `max_gap=1` adjacency control for **CF-6** (sparsity).

`14_label_correctness.py` is the *only* step that spends API budget. The tasks are open-ended proofs/lateral-thinking with no exact-match grader, so correctness comes from an **LLM judge** (Claude Sonnet on the AWS Bedrock proxy — `src/predict/labels.py`). The judge prompt grades the final answer only:

```python
# src/predict/labels.py :: _JUDGE_PROMPT (excerpt)
Judge ONLY whether the model's FINAL answer / conclusion is correct and adequately
justified for the task. Ignore style, length, and the path taken. Use "uncertain"
if the reasoning was cut off before a final answer, or if the task has no
objectively correct answer and the proposed solution is not clearly valid.

Respond with ONLY a JSON object and nothing else:
{{"verdict": "correct|incorrect|uncertain", "confidence": "high|medium|low", "rationale": "<=25 words"}}
```

Three confound-aware choices: (a) `correct=None` ("uncertain") is **excluded downstream, never coerced** — an unparseable reply or API failure never fakes a label (`parse_verdict`, `judge_chain`); (b) `confidence` is recorded for a confidence floor; (c) `truncated` is carried so the correctness↔truncation confound (**CF-8**) can be controlled. Selection (`select_balanced_pilot`) balances across (category, difficulty) strata and prefers complete chains. Judging is checkpointed + resume-safe (`generate_correctness_labels`).

### K.6 What was actually run — and the inflation that was caught

A **200-chain pilot** was judged → **183 usable** (84 correct / 99 incorrect; 11 uncertain excluded). This is the *entire* empirical result of the stage so far. There were two runs, and the first was **wrong**:

- The **flawed** gate (`results/predict/R1-1.5B/gate_pilot.md`, now banner-marked **SUPERSEDED — DO NOT CITE**) trained the predictor on the **labelled subset only** (~183 chains). Its residuals encoded chain idiosyncrasies that correlate with correctness, inflating AUC to ≈0.61 (e.g. L27 0.618, label-perm p=0.006) and pushing the residual/persistence ratio **above 1** (1.54–1.59, i.e. "worse than persistence").
- The **corrected** run (`.../corrected/`) trains the label-agnostic predictor on **all 986 chains** (chain-grouped OOF) and evaluates correctness AUC only on the labelled subset. This is now wired into `15`/`17` as a comment-documented invariant ("Restricting predictor training to the labelled subset underpowers it badly"). The honest numbers (`corrected/SUMMARY.md`):

| layer | ridge AUC (lp p) | JEPA AUC (lp p) | best baseline | step-shuffle null |
|---|---|---|---|---|
| 14 | 0.542 (0.128) | **0.582 (0.026)** | 0.549 | n.s. (p=0.72) |
| 17 | 0.486 (0.421) | 0.541 (0.146) | 0.506 | n.s. (p=0.77) |
| 27 | **0.578 (0.034)** | **0.591 (0.024)** | 0.543 | — |

Note this **reverses** the "early read" still printed in `PREDICTIVE_GEOMETRY.md` §8 ("linear cross-chain map is *worse than persistence*, residual/step ≈ 1.4–1.5 ⇒ Rung 1 may be a null"). That early read was the *flawed* train-on-subset artifact. The corrected label-free sweep (`16`, `residual_geometry_sweep.md`) shows the opposite: the predictor **beats persistence at every layer** (displacement R² ≈ +0.25 to +0.31, residual/persistence ≈ 0.88), and the unpredictable residual is *higher*-dimensional than the raw displacement (resid ID ≈ 5–6 vs delta ID ≈ 3.7). So Rung 1 is *not* a trivial null; it is a small, real signal.

### K.7 What the corrected pilot actually shows (the honest read)

From `corrected/SUMMARY.md`:

- **The predictor works** and generalises across chains (R²≈0.3, resid/persist≈0.88).
- **The correctness signal is real but small** — AUC ≈ 0.58–0.59 at L27 (both predictors) and L14 (JEPA), beating all baselines and surviving the label-permutation null. **L17 is a dead spot** (AUC ≤ 0.54, no null rejects).
- **Order does not matter.** The step-shuffle null *never* rejects (p≈0.72–0.77). So the correctness signal is in residual **magnitude** (overall chain predictability), **not** trajectory order/branch geometry. This **supports H1** and **kills H3** for this data — exactly the static-magnitude case the unit test warned the step-shuffle null is blind to.
- **Rung-2 > Rung-1, but barely**, and it does not scale: a 60-epoch convergence check (`corrected_ep60/`) shows the JEPA *worsens* (L14 ratio 1.139→1.573, overfitting the displacement; L27 AUC 0.591→0.580). The conclusion: the signal is "largely **linear-accessible**; extra nonlinear capacity buys little."

### K.8 Critique — is the signal real, and how strong is the claim?

1. **It is a weak signal on a noisy label.** AUC 0.58–0.59 with n=183 chains (84/99 split) is barely above chance, and `corrected/SUMMARY.md` itself flags the estimate as "noisy" and recommends scaling to ~1000 labels. With only 5 residual features and one informative axis (`resid_mean`), the standard error on AUC at this n is roughly ±0.04 — the L27 0.578 (p=0.034) is one bad fold away from null. The whole stage rests on a single ~200-chain pilot.

2. **The label is circular with the very behaviour annotations the geometry is built on.** Correctness comes from a Claude judge; the behaviour spans the trajectory is built from also came from an LLM annotator (κ=0.35–0.44 elsewhere in the thesis). The judge sees the *full text* but the predictor sees only the *4-behaviour mean-pooled subsequence* — so they are not identically circular, but both inherit annotator/judge idiosyncrasy. There is no human-graded correctness check and no second-judge agreement reported for these 183 labels.

3. **The headline H3 (branch-point geometry) is dead, and that is the interesting part.** The step-shuffle null never rejects: the signal is pure residual *magnitude*, i.e. "how predictable is this chain overall," which is exactly what PHi (arXiv:2503.13431) already reports at the per-token level. The novel wedge claimed in §5 — that residual *geometry over time* localizes branch/backtrack points — **finds no support in the data**. What survives is the least novel sub-claim.

4. **Mean-pooling + sparse 4-behaviour subsequence is a serious confound (CF-6).** The "trajectory" is not the reasoning trajectory; it is the subsequence of 4 annotated behaviours, mean-pooled over each span. A "step-to-step displacement" between, say, a backtracking span and an adding-knowledge span 8 sentences later (gap 8) is not a local reasoning increment. The `max_gap=1` control exists but the headline numbers use all gaps. `PREDICTIVE_GEOMETRY.md` §7 itself concedes mean-pooled local data is "suggestive, not confirmatory" and that a last-token / un-pooled GPU re-extraction is the confirmatory arm — which has **not** been run.

5. **An inflation bug was caught only after the first writeup.** The flawed train-on-subset gate produced "significant" AUCs ≈0.61 with p<0.01 that were entirely an artifact. This is well-handled now (banner + corrected dir + code comments), but it is a reminder that the residual-feature pipeline is delicate and that "the null rejected" is not self-certifying here.

6. **Layer profile is non-monotone and partly post-hoc.** Signal at L14 and L27 but a "dead spot" at L17, with no mechanistic story for why. With three layers and noise of ±0.04, picking L14/L27 as "the layers that work" risks being a multiple-comparisons artifact.

### K.9 Status: run vs unrun, citable vs quarantined

- **Built + unit-tested (green):** all of `src/predict/` (dataset, predictor, JEPA, evaluation, nulls, labels); `tests/test_predict.py` and `tests/test_predict_jepa.py`. Plumbing validated via `--smoke` synthetic-label paths in 15/17.
- **Run (citable):** the corrected 183-chain pilot — `results/predict/R1-1.5B/corrected/` (gate, Rung-2 compare, ep60 convergence check, SUMMARY). Cite **these**.
- **Quarantined (DO NOT CITE):** `results/predict/R1-1.5B/gate_pilot.md` (the inflated train-on-subset gate) — retained only for before/after.
- **Unrun:** the full ~1000-label run (5× the pilot spend, "user decision"); the un-pooled / last-token GPU re-extraction (the CF-6 confirmatory arm); Rung 0 difficulty-matched curvature as a standalone; and **all of Rung 3** (latent rollout + causal steering).

### K.10 Connection to the steering decision the researcher is about to make

This is the key linkage, and it is **mostly a negative**:

- **Rung 3 of this ladder *is* a steering experiment** — "causal steering along the predicted next-step direction via `src/steered_inference.SteeredModel`" (H4, the apex). So in principle this stage could hand the Phase-7 steering experiment a *predicted-direction* steering vector to compare against the behaviour-difference vectors. But Rung 3 is unbuilt, and H4 is explicitly framed as "confirmation, not novelty."

- **On layer choice:** the corrected pilot's only directional hint is that the correctness signal concentrates at **later layers (L27)** and L14, with **L17 a dead spot**. That is consistent with — but much weaker than — the steering memo's plan to build mid-layers (≈11/16/19) plus **L27**. If anything, the predictive-geometry pilot is a faint independent vote for **L27 carrying late, correctness-relevant structure**. It does *not* adjudicate the mid-layer (Venhoff-recipe) choice.

- **On methodology:** the predictor's residual signal is **magnitude, not direction, and not order-dependent**. That means it offers *no* validated "predicted next-step direction" to steer along — the one thing Rung 3 would need. The step-shuffle null killing H3 specifically undercuts the idea that there is a clean geometric "branch direction" to push on.

**Bottom line for the steering go/no-go:** this stage does not gate the steering spend and should not be on its critical path. It contributes a weak, late-layer correctness signal (AUC≈0.58, magnitude-only) that modestly corroborates including **L27** in the steering layer set, but provides no usable steering direction and no methodological dependency. Treat it as a parallel, descriptive Movement-1 result — not a prerequisite — and do **not** let its unrun Rung-3 ambitions expand the steering budget.
