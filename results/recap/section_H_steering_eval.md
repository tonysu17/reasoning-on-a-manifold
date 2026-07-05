## H. Steering Evaluation Harness — Phase 7 (BUILT, UNRUN)

This section documents the Phase-7 *causal* stage: the harness that takes the steering vectors built in Phase 6 (`06_build_steering.py` / `build_steering_arms.py`) and asks whether subtracting a behaviour's direction from the residual stream during generation actually *reduces that behaviour in the model's output*. Phase 7 is the experiment that converts the standing geometric correlations (low-dimensional per-behaviour subspaces) into a causal claim. It is fully implemented, exhaustively unit- and integration-tested, and a `$0` real-GPU smoke run has passed — but **the real run that would produce a behaviour-fraction number has not been executed.** Nothing here is citable yet.

Files covered: `07_evaluate_steering.py`, `src/steered_inference.py`, `src/evaluation.py`, `src/layer_sweep.py`, `tests/test_phase7_integration.py`, `tests/test_steered_inference_arms.py`, `tests/test_steered_inference_engine.py`, `build_steering_arms.py`, `run_overnight_bakeoff.sh`, `run_trim_bakeoff.sh`, `GPU_GUIDE.md`.

---

### H.1 What the stage does, end to end

`07_evaluate_steering.py` is a thin orchestrator. It (1) loads the category-stratified held-out eval split — the *same* tasks the vector builders excluded from vector construction, so Phase 7 is a genuine out-of-sample test; (2) loads the steering vectors; (3) loads the model; (4) calls `run_steering_experiment` to generate steered chains for every (behaviour, arm, α, task); then (5) either writes annotation-free "damage" metrics (`--skip-annotation`) or re-annotates the steered chains with the annotator LLM and aggregates a **behaviour-fraction** outcome.

The two-pass workflow is deliberate and is the spine of the cost discipline. The docstring states it plainly:

```python
# 07_evaluate_steering.py (module docstring)
# Workflow: run with --skip-annotation first (generation-first — inspect
# degenerate_rate/repetition before paying), then re-run without it to annotate
# (it resumes). --annotator-model picks a NON-builder annotator to de-circularise.
```

So generation (free, GPU-only) is fully decoupled from annotation (paid, API). You generate everything, look at the cheap damage metrics, and only *then* pay to annotate — and the annotation pass *resumes from the saved generations*, so the GPU work is never repeated.

The eval split is loaded from the canonical corpus and stratified, with an explicit caveat baked into the comment:

```python
# 07_evaluate_steering.py, main()
tasks = load_tasks(Path("data/tasks_final.json"))
# ... With default-holdout vectors, Phase 7 is a true
# out-of-sample test; with --no-holdout vectors it is an on-corpus causal
# effect (check the vectors' metadata provenance "holdout" field).
# Residual caveat either way: the steering LAYER choice was informed by
# full-corpus analyses (and Huang's published layer 27).
from src.task_gen import stratified_eval_split
test_tasks, split_rule = stratified_eval_split(tasks, args.n_test)
```

That last comment is the single most important honesty flag in the runner and recurs in the critique below: even with held-out *tasks*, the steering *layer* was chosen with full-corpus information, so the layer choice is not itself held out.

### H.2 The injection mechanism — Huang Eq. 3, projective, every position

The causal core is `SteeredModel._hook_fn` in `src/steered_inference.py`. A forward hook on one decoder block computes the projection of the hidden state onto the unit steering vector `r` and subtracts (or adds) a scaled multiple of that projection back. This is the *projective* form (Huang et al. Eq. 3): it removes the component of `h` along `r`, not a fixed vector, so the magnitude of the edit adapts to how much of `r` is already present.

```python
# src/steered_inference.py, SteeredModel._hook_fn
is_tuple = isinstance(output, tuple)
hidden = output[0] if is_tuple else output
h = hidden.float()             # (batch, seq, hidden)
r = self._r                    # (hidden,)
proj = torch.einsum("bsd,d->bs", h, r).unsqueeze(-1)   # (batch, seq, 1)
self._abs_proj_sum += float(proj.abs().sum().item())
self._abs_proj_count += int(proj.numel())
if self.mode == "measure":
    return output
delta = (self.alpha * self.energy_scale) * proj * r.view(1, 1, -1)
if self.mode == "subtract":
    h = h - delta
else:
    h = h + delta
h = h.to(hidden.dtype)
return ((h,) + output[1:]) if is_tuple else h
```

Several design choices are defensible and load-bearing:

- **Projective, not additive.** `delta ∝ (rᵀh)·r`. With unit-norm `r`, `α` is the entire scale knob, and `α·1·(rᵀh)·r` at `α=1` exactly ablates the `r`-component. This matches Huang and means α=0 (subtract mode) is the identity — which is *why* the runner can reuse one shared vanilla baseline for the α=0 column (`steered_alphas = [a for a in alpha_values if a > 0]`).
- **Every position, prompt prefill included.** The module docstring: *"The hook applies to every position, prompt prefill included, matching Huang."* This is a faithful replication choice, not a tuning choice.
- **fp32 projection, restore to layer dtype.** `h.float()` then `h.to(hidden.dtype)` keeps the einsum in full precision even under fp16/bf16 weights; `self._r` is stored float32 (`test_hook_preserves_low_precision_dtype` asserts this).
- **Tuple-vs-bare-tensor defence.** transformers <5 returns a tuple `(hidden, ...)`; 5.x can return a bare tensor. Indexing `output[0]` on a bare tensor would silently grab batch element 0 — a real, silent integration bug. `test_hook_bare_tensor_not_indexed_as_batch` guards it directly.
- **`energy_scale`.** A second multiplicative gain, =1.0 for all behaviour and norm-matched arms, used *only* by the energy-matched-random control (see H.4).
- **`mode="measure"`.** The same hook, run as a forward-only probe, records mean |rᵀh| on the *unperturbed* stream and passes hidden states through untouched — this is the common reference used to calibrate the energy-matched arm so behaviour and random arms are measured against identical activations.

The hook is installed only inside `generate()` and removed in a `finally` block, so a CUDA OOM mid-generation cannot leave a poisoned hook on the model (`test_generate_removes_hook_even_on_error`).

### H.3 The arms: 14 arms + lean-6 curated

`_build_arms` constructs every arm for one behaviour, purely (no model). The arm set is the product of a long adversarial-review process; each arm exists to kill a specific alternative explanation. The runner docstring lists them:

```python
# 07_evaluate_steering.py (module docstring)
#   vanilla                  — unsteered baseline (one shared generation per task)
#   single_direction         — Venhoff-style difference-of-means vector
#   manifold_k{1,3,5,10}/auto — HEADLINE k-sweep (every built k, not just auto)
#   random_subspace_k{k}     — equal-k random-subspace control (R replicates)
#   random_direction         — norm-matched random (sanity floor)
#   energy_matched_random    — random rescaled to equal injected energy (real floor)
#   orthogonal_complement    — off-subspace (I-P_k)r component alone
```

Counting concrete method labels at default settings: `vanilla`, `single_direction`, five manifold arms (`manifold_k1/k3/k5/k10/auto`), five `random_subspace_k{1,3,5,10,auto}` (each ×3 replicates), `random_direction`, `energy_matched_random`, `orthogonal_complement` → **14 method labels** (the `test_phase7_integration.py` `expected_methods` set lists exactly these, minus the bare `vanilla`, as 14 steered/method entries). The "lean-6 curated" set is the trimmed configuration used for the contended cluster: `single_direction` + `manifold_auto` only, which is what `run_trim_bakeoff.sh` runs (`--no-random-control --no-random-subspace --no-energy-matched --no-orthogonal-complement`, k trimmed to auto).

The scientific content of each control:

- **`single_direction` vs the manifold k-sweep** is the headline comparison. The k-sweep exists because the adversarial review pointed out that at `k=auto` the manifold vector is nearly identical to the single direction (`cos 0.95–0.97`), so "the manifold helps" was untestable. Running every k (1/3/5/10/auto) makes the granularity an empirical question.
- **`random_subspace_k{k}`** is the cleverest control: it projects the *same* single direction onto a *random* k-dim subspace (QR of a seeded Gaussian), renormalised *identically*. This isolates "the behaviour's *own* PCA subspace matters" from "any k-dim projection + renorm helps":

```python
# src/steered_inference.py, random_subspace_projection
G = rng.standard_normal((d, k))
Q, _ = np.linalg.qr(G)              # (d, k), orthonormal columns
coords = Q.T @ r                    # (k,)
r_proj = Q @ coords                 # (d,) — projection back into ambient
norm = float(np.linalg.norm(r_proj))
if norm < 1e-10:
    v = r / (np.linalg.norm(r) or 1.0)
    return v.astype(np.asarray(reference).dtype, copy=False)
return (r_proj / norm).astype(np.asarray(reference).dtype, copy=False)
```

- **`random_direction`** is explicitly demoted from "the baseline" to a *sanity floor*. The code is candid that a norm-matched random vector injects ~19× *less* energy than a behaviour-aligned one (`|rᵀh| ≈ 4.9 vs 94.6`), because a random direction has a smaller projection onto `h`. So beating it proves almost nothing.
- **`energy_matched_random`** is the *real* generic-perturbation floor: a random direction rescaled so its delivered |rᵀh| equals the behaviour arm's, calibrated against the model at run time:

```python
# src/steered_inference.py, energy_matched_scale
# A random unit vector has smaller |v·h| than a behaviour-aligned one, so the
# gain is > 1 (it rescales the random direction UP). ...
return float(behaviour_mean_abs_proj / random_mean_abs_proj)
```

The calibration uses the first ≤5 eval tasks in `mode="measure"` and is *skipped entirely on a fully-resumed run* so resume stays cheap.

- **`orthogonal_complement`** steers the off-subspace component `(I−P_k)r` alone, to test whether the part the manifold projection *discards* is pure collateral (the mechanism behind any manifold advantage).

Alongside the arms, `_build_arms` emits **geometry bounds** per (behaviour, k): `cos(single, manifold_k)` and `retained_energy = ‖P_k r‖/‖r‖`. These are not optional flavour — they *bound how large any manifold effect can be*. If `retained_energy ≈ 1`, the discarded complement is tiny and manifold≈single by construction, so a null result there is expected, not interesting. This is exactly the kind of pre-registered ceiling that protects against over-reading a small effect.

### H.4 The outcome metric: behaviour-fraction

The headline outcome is in `src/evaluation.py`:

```python
# src/evaluation.py
def behaviour_fraction(annotated_chain: list[dict], target: str) -> float:
    """Fraction of sentences in *annotated_chain* classified as *target*."""
    if not annotated_chain:
        return 0.0
    n_target = sum(1 for a in annotated_chain if a["label"] == target)
    return n_target / len(annotated_chain)
```

i.e. re-annotate the steered chain sentence-by-sentence with the annotator LLM, and report the fraction of sentences labelled with the target behaviour. Subtracting the behaviour's direction should *lower* this fraction relative to vanilla; adding should raise it. `aggregate_results` rolls this into a nested `{behaviour: {method: {alpha: cell}}}` summary.

The aggregation has three pieces of hard-won correctness that directly defend a thesis claim:

1. **Degenerate annotations are not scored 0.0.** A missing re-annotation is counted in `n_missing`, an empty list in `n_empty` — *not* scored as "behaviour absent". The comment names the exact bias this avoids:

```python
# src/evaluation.py, aggregate_results docstring
# * A MISSING re-annotation (no record for the key) is SKIPPED and counted
#   in "n_missing" — previously it silently scored 0.0, deflating whichever
#   arm produced more annotation failures (which correlates with how
#   destructive that arm's steering is — exactly the comparison under study).
```

This is the crux: destructive steering produces garbage, garbage fails to annotate, and scoring failures as 0.0 would *manufacture* a behaviour-suppression effect out of pure collateral damage. The fix makes the comparison honest.

2. **Cheap damage metrics computed from text, no annotation spend.** `degenerate_rate` (chains < 32 tokens), `repetition_rate` (1 − distinct/total 4-grams), and `mean_n_tokens` are computed straight from generation. These are the controls that distinguish *suppression* from *damage*: an arm can "reduce the behaviour" simply by wrecking generation, and these surface it before a dollar is spent.

3. **Off-target leakage.** For a cell steering behaviour *b*, `leakage_by_behaviour` reports each *other* behaviour's mean fraction in the same chains — does suppressing backtracking drag uncertainty-estimation down too? This reuses the free per-sentence labels the annotator already produces (no extra spend).

There is also an `aggregate_accuracy` path (paired `accuracy_drop_vs_vanilla` over the intersection of task ids) for the "did steering break task-solving" check, though correctness labels are supplied externally and that pipe is not wired into the runner yet.

### H.5 Crash-safety: checkpoint every chain, atomic write, resume

The run can be hours long on a flaky tunnel, so resilience is built in. `run_steering_experiment` checkpoints after every (behaviour, α, method) sweep, dedups on a 5-tuple key, and writes atomically:

```python
# src/steered_inference.py
done = {
    (r["behaviour"], r["method"], r["alpha"], r["task_id"],
     r.get("subspace_replicate"))
    for r in results
}
# ... per record:
key = (beh, method_name, alpha, eff_task_id, rep)
if key in done:
    continue
```

```python
# src/steered_inference.py, _save_json — atomic
tmp = path.with_suffix(".tmp")
with open(tmp, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
tmp.rename(path)
```

The resume logic also actively *cleans* the checkpoint: legacy per-behaviour vanilla records (a pre-hoist schema) are dropped on load because they would "fake a vanilla-vs-α curve in aggregation" (`test_resume_drops_legacy_per_behaviour_vanilla`). The multi-sample machinery folds replicate and sample indices into the `task_id` (`#rs{rep}`, `#s{j}`) via `effective_task_id`, so N temperature draws are distinct points that resume independently and pool into one cell downstream — and `n_samples>1` with `temperature≤0` is *rejected* because N greedy draws are byte-identical and would "manufacture N data points from one — a silent variance fraud" (`test_greedy_multisample_is_rejected`). The bake-off shell scripts wrap the runner in an 8-attempt retry loop precisely because resume makes a transient crash lose ≤1 sweep.

### H.6 What is RUN vs UNRUN

**RUN — and only this:**
- The full unit + integration test suite. `test_steered_inference_arms.py` (pure arm construction), `test_steered_inference_engine.py` (hook math, dtype, lifecycle, resume, multi-sample), and `test_phase7_integration.py` (a *stubbed* end-to-end dry run through `runner.main()` with a real `nn.Module` whose `model.layers[L]` fires the actual forward hook). The integration test proves the wiring — vectors → arms → generate → energy calibration → checkpoint → aggregate → `generation_metrics.json`, and the mocked-annotator branch → `eval_summary.json` — runs end to end with no GPU and no API spend.
- A `$0` real-GPU smoke test passed: `--smoke --skip-annotation` (3 tasks, α∈{0,1}, generation only). `test_smoke_is_cheap_three_tasks_two_alphas` enforces that `--smoke` truncates to exactly 3 tasks and 2 alphas and writes *no* `eval_summary.json` (confirming the API branch is untouched, hence `$0`).
- The generation-only **layer bake-off** scripts exist (`run_overnight_bakeoff.sh`, `run_trim_bakeoff.sh`) and are designed to run L27 vs L16 at `$0` (no annotation), inspecting only `generation_metrics.json` damage metrics.

**UNRUN — the actual result:**
- **No behaviour-fraction has ever been computed.** The only output that establishes whether steering causally suppresses a behaviour is `eval_summary.json` (or the annotated aggregation), and that requires the annotation pass, which requires proxy credentials. Per the runner, when `CLAUDE_PROXY_URL`/`CLAUDE_PROXY_KEY` are unset it logs a warning and *skips re-annotation entirely*. The blocker is real-world: the proxy creds are not set anywhere the agent can reach.
- The full 14-arm, 50-task, α-grid run has not been executed at scale; the bake-off configs are sub-runs (10–15 tasks, 1–3 alphas, lean arms) for *layer selection only*, not the headline experiment.

So the correct status line is: **Phase 7 is built, QA'd, and smoke-verified, but it has produced zero citable causal results.** Any thesis sentence claiming "subtracting the behaviour direction reduces the behaviour" is, as of now, unsupported by this stage.

### H.7 How this connects to the steering decision the researcher is about to make

This is the operative question. Two decisions are pending: **which layer** to steer at, and **which methodology** (single vs manifold-k, which controls).

- **Layer.** `src/layer_sweep.py` is the de-confounded layer pre-filter that *feeds* this decision. It exists because the earlier attribution-patching approach (`07c`) came back confounded — a read-out-proximity artefact of a linear estimator that ramps monotonically to L26–27 with no interior peak. `layer_sweep.py` instead measures the *actual non-linear forward-intervention effect*: per-layer diff-of-means direction, output read-out (KL primary, onset-logprob secondary), subtract a norm-matched random null, α-normalise per layer by median ‖h‖, and bootstrap the argmax into a *shortlist*, framed explicitly as "a pre-filter for Phase 7 to confirm, NOT a single load-bearing argmax." The hook in `layer_sweep.py` (`_register_steer_hook`) is the *same* projective form as `SteeredModel._hook_fn`, so the pre-filter and the real run intervene identically — a deliberate consistency. The bake-off scripts then let Phase 7 *itself* arbitrate L16 (≈Venhoff mid-layer) vs L27 (Huang's published choice) on real generation damage. So the layer decision is meant to be a *triangulation*: layer-sweep shortlist + bake-off damage metrics + (eventually) behaviour-fraction.
- **Methodology.** The arm design *is* the methodology decision rendered as an experiment. The researcher does not have to pick "single vs manifold" a priori; the k-sweep + random-subspace + orthogonal-complement + energy-matched controls are built to *answer* it empirically, bounded by the per-(beh,k) `retained_energy` ceiling. The pooling (MEAN), the `8192` token cap (to avoid confounding α with truncation), and the energy-matched floor are all settled.

The one thing this stage *cannot* settle without spend is the outcome itself. The layer bake-off runs at `$0` but selects a layer on *damage/coherence proxies*, not on behaviour-fraction — and the runner's own comment warns "the layer pick needs annotation anyway." So the honest sequence before committing real budget is: (1) run `layer_sweep` for the shortlist, (2) run the `$0` bake-off for damage, (3) accept that the *final* layer + methodology confirmation needs the paid annotation pass.

### H.8 Critique — confounds, fragilities, untested assumptions

1. **Annotation-in-the-loop circularity is the headline threat.** The behaviour vectors were *built* from spans the annotator (Sonnet 4.5) labelled, and the outcome metric *re-annotates* with the same family of annotator. Builder-scores-its-own-output is a circularity the code itself flags ("RESULTS_LEDGER §C"). The mitigation — `--annotator-model` to use a non-builder annotator (e.g. Qwen3-235B) — is implemented and tested (`test_annotate_path_wiring_with_mocked_annotator` asserts the knob threads through) but the doc notes the non-builder annotator id is "not yet located," so the de-circularised run cannot actually be launched. Until it is, a positive behaviour-fraction result is partly confounded with annotator self-consistency.

2. **Suppression vs damage is mitigated but not eliminated.** The damage metrics (`degenerate_rate`, `repetition_rate`) and the no-score-0.0 rule are exactly the right guardrails, and the energy-matched-random floor is the right comparator. But `repetition_rate` and a 32-token floor are blunt: a chain can be coherent, on-length, and *subtly* derailed (e.g. solving a different problem) in a way that lowers the target-behaviour fraction without being "degenerate." The `aggregate_accuracy` path (does steering break task-solving?) is the proper control for this, and it is **not wired into the runner** — correctness labels are an external input with no producer in Phase 7. That is a real gap: the strongest "is this suppression or sabotage?" check is built but unplumbed.

3. **The layer is not held out, and the bake-off selects on a proxy.** The runner's own comment concedes the steering layer was informed by full-corpus analyses and Huang's L27. The `$0` bake-off picks L16 vs L27 on damage metrics, not behaviour-fraction, so the layer that "looks healthiest" need not be the layer where the behaviour-specific causal effect is largest. `layer_sweep` is the principled fix, but its output is a *shortlist* with an explicitly unstable bootstrap argmax — and `layer_sweep` itself is also unrun on the real model. There is a risk of selecting a layer on coherence and then reporting an effect there as if the layer were pre-registered.

4. **`retained_energy ≈ 1` quietly predicts a manifold null.** The geometry bounds are honest, but they also mean the headline single-vs-manifold comparison may be *structurally* unable to show a manifold advantage at k=auto (cos 0.95–0.97). If the interesting k turns out to be small (k=1/3) where retained energy is low, the manifold arm is mostly the *random-subspace* arm's twin in spirit — the result then hinges entirely on the random-subspace control being clean, which depends on R=3 replicates being enough. Three replicates is thin for a control whose whole job is to establish a null distribution.

5. **Cost/scale levers are sharp and under-explored.** Cost scales as roughly `arms × tasks × samples × |α>0|`. At 14 arms (random-subspace ×3), 50 tasks, 7 positive alphas, 5 samples, that is `~14×50×7×5 ≈ 24,500` generations *plus* an annotation call per generation — multi-week and well past a `$260` budget. The levers are explicit in the argparse help (`--n-samples`, arm-disable flags, `--max-eval-tasks`), and the lean-6 / 50-task greedy config is the intended trim, but the harness does not *enforce* a budget; an operator can trivially launch the full grid. There is no `--max-cost` guard, and the annotation pass has no token-cap argument surfaced in the runner.

6. **RunPod vs cluster is an operational, not scientific, fragility — but it bites.** Memory notes the cluster is 94% contended, disk-full (~15 GB), with a flaky cloudflared tunnel (~2–3 min/gen vs ~1 min dedicated). The bake-off scripts are hard-coded to `/home/tony/reasoning-on-manifold` and a cluster venv, so the "RUN ON RUNPOD" guidance and these scripts are not yet reconciled — the scripts encode the *contended-cluster* trim, not a RunPod config. `GPU_GUIDE.md` is also stale relative to the current pipeline: it references `src/utils/model_utils.py`, `src/data/generate_chains.py`, `scripts/run_phase1.py` and a 1000-token cap — none of which match the current `src/` layout or the 8192-token corpus cap. A researcher following `GPU_GUIDE.md` literally would set up the wrong environment.

7. **Multi-sample variance is honest but the default is single greedy.** The N-vs-N sampling design is correct and well-tested, but the *default* run is `n_samples=1, temperature=0` — a single greedy point per cell, no within-cell variance, so error bars on the behaviour-fraction come only from the 50-task spread. For a causal claim that single-vs-manifold differ, that is the difference between a publishable interval and a point estimate; the budget pressure pushes toward the weaker default.

**Bottom line.** The harness is unusually disciplined: the controls are the right controls, the no-score-0.0 fix is exactly the bias that would otherwise fake the headline effect, and the generation/annotation split protects the budget. The exposure is that (a) nothing causal has been measured yet, (b) the de-circularised annotator is implemented but cannot be launched, (c) the accuracy-preservation control is built but unplumbed, and (d) layer selection currently rests on proxies and full-corpus information rather than a held-out, behaviour-fraction-confirmed choice.
