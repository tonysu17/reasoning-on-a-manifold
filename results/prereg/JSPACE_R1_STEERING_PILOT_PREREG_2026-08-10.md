# SEALED DESIGN — R1 J-lens validation and steering-component pilot

**Status:** design sealed 2026-08-10 on Tony's instruction to proceed.  
**Execution status:** prospective / unrun.  
**Authorization boundary:** this document authorizes non-spend preparation and
offline validation only. It does **not** authorize a GPU rental, paid API use,
model inference/fitting, thesis-scope change, or thesis claim edit. Those each
require Tony's explicit sign-off after the Phase-0 cost report exists.

This is an additive pilot. It does not overwrite or reclassify any existing
steering result. Any deviation after sealing requires a dated amendment before
the affected stage runs.

## 1. Question and claim boundary

### Primary question

For the existing R1-1.5B backtracking single-direction intervention at its
executed layer L17, does the component of the **effective suppression
perturbation** recovered by a validated Jacobian-lens dictionary reproduce more
of the task-held-out behavioural effect than the complementary residual
component, after each is compared with its own energy-matched random floor?

The effective full perturbation is

\[
\delta_{\mathrm{full}}=-\alpha v_{\mathrm{BT}}, \qquad \alpha=1,
\]

because the retained execution used subtract/suppression steering. The conic
J-space decomposition is applied to \(\delta_{\mathrm{full}}\), not silently to
\(+v_{\mathrm{BT}}\); a nonnegative cone is not invariant to a sign flip.

### Claims this pilot cannot license

The pilot cannot establish that:

- J-space is a low-dimensional subspace, manifold, or explanation of the
  thesis's correlation-dimension, participation-ratio, PCA-threshold, or
  fixed-top-ten estimands;
- R1-1.5B has or lacks a global workspace generally;
- a failed lens or failed component intervention indicates surface imitation;
- J-space causes reasoning generally, or the result transfers to other models,
  behaviours, layers, doses, annotators, or intervention operators;
- RL or distillation created the measured component.

A negative result is local non-detection under this instrument and protocol. A
positive result is bounded to this model, checkpoint, L17 direction, task set,
dose, decomposition, and endpoint.

## 2. Fixed inputs

| Role | Path / identity | Fixed fact |
|---|---|---|
| Model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`; BF16 analysis load |
| J-lens code | `anthropics/jacobian-lens` | commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` |
| Primary vector | `results/steering_vectors/R1-1.5B__E1_pooled/backtracking_single.npy` | L17; SHA-256 `10b12bdd2e90afafee4198e13156d30e585cde41f29993dca2ca91d7c221b2ca` |
| Vector metadata | `results/steering_vectors/R1-1.5B__E1_pooled/metadata.json` | SHA-256 `0f756b9cbe1ec8f470625000784bdae1a07338987e01e2fcb1f7268e0eddd886` |
| Evaluation tasks | `results/eval/R1-1.5B__E1/eval_task_ids.json` | same 50-task held-out split; SHA-256 `c8f2d921e4f8506c68b488442ed51f05e6a14dbac638c66fc41569d7bbd37bcd` |
| Task source | `data/tasks_final.json` | canonical source; SHA-256 `a4ba180d5722e016c95945867ca6fbcf7abb48640f4b8d0c9e6111c572051b1d` |

The original vector has unresolved execution provenance (`git_commit: null`)
and the original causal result is amended and provisional. That uncertainty is
not repaired by reusing the vector; it travels with this pilot. The full-vector
replication gate below prevents a component comparison from being interpreted
when the source intervention itself is not reproduced.

## 3. Phase 0 — five-prompt compute and compatibility benchmark

The exact prompts and source hashes are frozen in
`results/prereg/JSPACE_R1_BENCHMARK_MANIFEST_2026-08-10.json`. They are
calibration inputs only and never enter a scientific endpoint.

### Fixed estimator settings

- source layers: block outputs 0--26;
- target layer: block output 27;
- maximum sequence length: 128 tokens;
- skip first 16 positions, as in the released reference code;
- default `dim_batch=8`;
- model load: BF16; Jacobian accumulation: released FP32 CPU representation;
- attention backend: eager; compilation disabled; deterministic algorithms
  requested with seed `20260810`;
- raw task-prompt text, without a chat template;
- one fresh process per attempted `dim_batch` setting.

### Execution sequence

1. Run `jspace_pilot_preflight.py`; any hash, shape, layer, revision, or prompt
   mismatch stops the benchmark.
2. On `MATH_000`, attempt `dim_batch` in the order 8, 4, 2, 1, stopping at the
   first setting that completes. An OOM process is terminated; its partially
   written output is never resumed or treated as a lens.
3. Run all five prompts at the first completing setting.
4. Record, per prompt: token count, valid-position count, backward-call count,
   wall time, peak allocated and reserved GPU memory, peak CPU RSS, exception,
   and per-layer matrix shape/dtype/finite status.
5. On `MATH_000`, capture L17 once through the existing steering-hook path and
   once through the J-lens `ActivationRecorder`. The tensors must agree at the
   same token positions to `rtol=1e-5, atol=1e-6`; this pins block-output
   indexing before the shared label "L17" is used scientifically.
6. Record whole-run hardware, CUDA, PyTorch, Transformers, tokenizer and model
   fingerprints. Save no generated continuation.

### Phase-0 gate and consequence

All five prompts must complete; all 27 matrices must be finite and shaped
`[1536,1536]`; the observed peak must leave at least 2 GiB free on the target
device; and repeated fitting of `MATH_000` at the selected setting must agree
to relative Frobenius tolerance `1e-6`. The report extrapolates 100-prompt cost
from measured wall time and records uncertainty; it does not infer runtime from
peak FLOPs.

If any gate fails, all fitting and causal stages stop. The only licensed wording
is an engineering incompatibility or resource-gate failure. A smaller layer set,
different precision, shorter context, or modified estimator requires an
amendment before another benchmark.

## 4. Phase 1 — lens corpus and fitting

Phase 1 remains unrun until a corpus manifest is created and committed before
any model fitting.

### Corpus rule

- Dataset: `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`, train
  split, revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Encode each raw non-blank row using the pinned R1 tokenizer without a chat
  template. Eligible rows contain at least 128 tokens.
- Seed a NumPy PCG64 permutation of eligible source-row indices with
  `20260810`; take the first 120 rows.
- The first 50 selected rows form fit A, the next 50 fit B, and the final 20
  are held-out readout-stability rows. Each prompt is stored verbatim with its
  source index, UTF-8 SHA-256 and exact token IDs in
  `results/prereg/jspace_r1_fit_manifest.json`.
- The manifest generator refuses to overwrite an existing manifest. Its file
  hash is appended in a dated pre-execution amendment; no fit may begin before
  that amendment.

Fit one all-source-layer lens on A and one on B. The 100-prompt lens is the
released `n_prompts`-weighted merge of A and B, not a third refit. No
math-domain lens enters the primary pilot. A later domain-sensitivity lens
requires an amendment and cannot replace the generic lens post hoc.

## 5. Phase 1 validity gates

### Numerical and merge identity

- all matrices finite and shaped `[1536,1536]`;
- source-layer set exactly 0--26;
- merged lens equals `(J_A + J_B)/2` to relative Frobenius tolerance `1e-6`;
- save/load round trip agrees to `1e-3` in FP16 storage and `1e-6` in an FP32
  diagnostic copy.

### Held-out readout stability

On the 20 held-out WikiText rows, compare fit A with fit B at every valid
position. The registered summary is the task/row-level median Jaccard overlap
of the top-25 token sets, reported by layer. A vocabulary-permuted-lens control
uses seed `20260811`.

The gate requires the A--B median to exceed the 95th percentile of the
permuted control in a contiguous band of at least four source layers that
includes L17. This is an instrument-stability requirement, not evidence of a
global workspace.

### External positive-control readout

Use the pinned J-lens repository's prompt-only evaluation files at the same
commit for `lens-eval-association`, `lens-eval-typo`, and
`lens-eval-multihop`. The estimand is pass@25 as defined in the repository's
evaluation README. Targets that are not single R1 tokenizer tokens are marked
ineligible before ranks are inspected.

For each evaluation, compare the merged J-lens against 1,000 within-evaluation
target-label permutations (seed `20260812`). The gate requires the observed
J-lens pass@25 to exceed the permutation p95 in at least two of the three
evaluations. Fit A and fit B must agree within 0.10 absolute pass@25 on at least
two of three. Logit-lens results are reported as a comparator but are not a
pass/fail threshold.

Failure of either the held-out stability gate or external positive-control
gate stops the decomposition experiment. It licenses only "the fitted lens did
not pass the prespecified validity gate."

## 6. Registered J-space decomposition

The reference repository fits/applies the Jacobian lens but does not release
the paper's gradient-pursuit implementation. This pilot therefore registers a
deterministic **nonnegative orthogonal matching-pursuit operationalization** and
does not describe it as an exact reproduction of Anthropic's optimizer.

At L17, form the activation-space token dictionary

\[
d_t = \operatorname{unit}\!\left(J_{17}^{\mathsf T}G w_t\right),
\]

where \(w_t\) is row \(t\) of the pinned model's unembedding and \(G\) is the
diagonal gain of its final RMSNorm. The RMS denominator is a shared positive
scalar at a position and therefore does not alter token ranks; the learned
elementwise gain cannot be omitted. Exclude tokenizer special tokens and
zero/non-finite directions; retain all other vocabulary entries. Verify the
orientation by comparing ranks from `W_U G J_17 h` with the released
`lens.apply` logits on the 20 held-out rows. With this pinned model (no logit
softcap), top-25 token sets must be identical at every checked position; a
failed orientation check stops the stage.

For primary sparsity `k=16`, initialize residual `r=delta_full` and empty
support. At each step:

1. add the unused dictionary entry with maximum strictly positive `d_t @ r`;
2. refit all selected coefficients by nonnegative least squares against
   `delta_full`;
3. update the residual;
4. stop at 16 entries or when no positive entry remains.

Define

\[
\delta_J=D_S c, \qquad
\delta_R=\delta_{\mathrm{full}}-\delta_J.
\]

Registered offline sensitivities use `k in {8,25}` but cannot replace k=16.
The other three E1-pooled single directions are decomposed descriptively only;
they receive no causal endpoint in this pilot.

### Decomposition gates

- exact reconstruction `||delta_full-delta_J-delta_R|| / ||delta_full|| < 1e-6`;
- coefficients finite and nonnegative to tolerance `1e-8`;
- selected-support condition number below `1e4`;
- neither component norm below 5% of the full-vector norm (avoids rescaling a
  nearly absent component by more than 20x);
- fit-A and fit-B component directions each have cosine at least 0.80 with the
  merged-lens component, and their explained squared-norm fractions differ by
  at most 0.05;
- synthetic sparse-cone recovery tests pass exactly for orthogonal dictionaries
  and meet registered tolerance on correlated dictionaries before the real
  vector is decomposed.

Any failure stops the causal component comparison. Support token labels,
coefficients, component norms, reconstruction error and stability statistics
are always reported and never interpreted as occupancy or intrinsic dimension.

## 7. Causal battery

### Independent unit and split

The independent unit is the task. Use exactly the existing 50 held-out task
IDs, with no replacement, exclusion or task-specific tuning. Missing or failed
generations remain unresolved rather than zero.

### Primary behaviour and endpoint

Backtracking is the only primary behaviour because it is the only existing
single-direction cell with retained, bounded intervention evidence. Let
`b_i(a)` be the annotated backtracking sentence fraction for task `i` and arm
`a`. For active arm `a` with its own energy-matched floor `f(a)`, define

\[
S_i(a)=b_i(\mathrm{vanilla})-b_i(a),\qquad
\Delta_i(a)=S_i(a)-S_i(f(a))=b_i(f(a))-b_i(a).
\]

Positive \(\Delta\) means more suppression than the matched generic
perturbation. The primary estimand is the paired task-level mean

\[
Q=\mathbb E_i[\Delta_i(J)-\Delta_i(R)].
\]

Report a paired task bootstrap interval with 10,000 resamples, seed `20260813`.
The primary test is two-sided at alpha .05. There is one primary cell and no
multiplicity adjustment. Component/full ratios are secondary and reported only
when the full-effect denominator has a stable nonzero sign.

### Arms and dose

- vanilla;
- original full suppression perturbation at alpha 1;
- J component, rescaled to `||delta_full||`;
- residual component, rescaled to `||delta_full||`;
- one independently seeded energy-matched random floor for each of full, J and
  residual components, calibrated by the existing injected-energy mechanism;
- one support-shuffled J control: the primary nonnegative coefficients assigned
  to seeded random eligible token directions (`20260814`), NNLS-refit forbidden,
  then rescaled to the same norm.

Generation settings, prompt formatting, hook site, greedy decoding, 8,192-token
cap and alpha otherwise match the E8 execution. Every new output goes to a new
`results/jspace_r1_pilot/` tree; no existing E8 artifact is overwritten or used
as a substitute for a missing new arm.

### Full-vector replication gate

Before component outcomes are inspected, the newly generated full-vector arm
must:

1. reproduce the original zero-dose/vanilla generation exactly under the pinned
   execution environment;
2. produce a positive floor-adjusted backtracking suppression estimate;
3. have a 95% paired-bootstrap interval excluding zero in the registered
   positive direction; and
4. pass the standing damage gates: no greater than 10 percentage-point excess
   in repetition/collapse or truncation and no greater than 15 percentage-point
   boxed-accuracy drop versus vanilla.

If this gate fails, component effects are reported descriptively without a
carrying/mediation claim. The source intervention is not called "refuted"; the
pilot reports a failure to reproduce it under the new execution.

### Interpretation table

| Observed bounded result | Licensed interpretation |
|---|---|
| Full gate passes; `Q>0` with interval excluding zero; J clears its floor | J component carries more of the tested intervention effect than the residual component under equal-norm perturbations. |
| Full gate passes; `Q<0` with interval excluding zero; residual clears its floor | Residual component carries more of the tested intervention effect under equal-norm perturbations. |
| Full gate passes; interval for `Q` includes zero | The pilot does not resolve a difference between components at achieved sensitivity. |
| Neither component clears its own floor | Component rescaling did not reproduce the full effect under the tested decomposition. |
| Both components clear floors | The full effect is not uniquely localized by this decomposition; report both effects and their non-additivity. |
| Lens, decomposition, damage, or full-replication gate fails | Stop the corresponding claim; report the gate failure locally. |

No outcome licenses the word "mediation" unless a later, separately registered
clamp prevents the complementary component from being re-derived downstream.

## 8. Annotation, provenance and result status

- Annotation uses the same executed within-annotator pipeline as the source E8
  result unless an amendment fixes a non-builder alternative before generation.
  The within-annotator circularity qualifier remains load-bearing.
- Save raw full chains before annotation. Generation-first damage inspection
  occurs before any paid annotation.
- Required provenance: git commit and dirty paths; prereg and manifest hashes;
  model/repository/dataset revisions; tokenizer/config/weight hashes; exact
  task and vector hashes; hardware/software/dtype; named seeds; stage command;
  per-stage start/end status; exclusions and exceptions.
- `amended` is a protocol marker, not an evidence status.
- Before thesis use: execution must complete, results must be independently
  validated, the claim ledger must receive a new row, and the thesis evidence
  snapshot must be refreshed. Until then every result is prospective/unrun.

## 9. Stop rules and authorization checkpoints

1. **Now authorized:** create this preregistration, the benchmark manifest, and
   offline preflight/tests.
2. **Tony approval required:** any model load or five-prompt hardware benchmark,
   even if the weights are already cached.
3. **Tony spend approval required:** GPU rental or paid external API use, after
   the benchmark reports a measured 100-prompt projection.
4. **Separate launch approval required:** the 100-prompt fit and causal battery.
5. **Separate thesis-scope decision required:** any authoritative Methods,
   Results, abstract, contribution or conclusion change.

At every boundary, no approval is inferred from the existence of this sealed
design.

## Seal record

Sealed 2026-08-10 before any J-lens fit, J-space decomposition, component
generation, or component annotation. Basis: Tony's instruction "proceed" after
the literature/methods review and the explicit no-spend preparation boundary.
Primary external references:
[Anthropic J-space paper](https://transformer-circuits.pub/2026/workspace/index.html)
and [reference implementation](https://github.com/anthropics/jacobian-lens).
