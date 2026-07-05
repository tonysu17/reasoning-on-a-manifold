## L. Safety Post-Training Spillover Extension (pt01–pt03, UNRUN)

### L.0 What this stage is, and where it sits

This is a **future-work / extension branch**, not a result. It implements a three-script mini-pipeline (`pt01` → `pt02` → `pt03`) that treats **safety post-training as an experimental intervention** and asks: when you fine-tune a reasoning model to refuse harmful requests, *how does the geometry of generic, non-safety reasoning move?* In the thesis spine this is the **hinge between Movement 1 (Structure: per-behaviour reasoning geometry) and Movement 2 (Origin/Safety: where safety reasoning comes from)**. Movement 1 says "reasoning behaviours have low-dimensional, per-behaviour subspaces"; Movement 2 says "safety reasoning is installed by post-training." This branch tests the *coupling* between them: does installing safety reasoning perturb the geometry of unrelated behaviours (backtracking, deduction, uncertainty-estimation, …)?

The code is **built and unit-tested offline, but UNRUN**. No contrastive dataset has been generated against the real proxy, no LoRA adapter has been trained, no spillover report exists. The test suite (`tests/test_safety_posttrain.py`, 11 tests) exercises only the proxy-free / torch-free logic: mock dataset schema, SFT masking, the numpy geometry diff, and JSON extraction. Nothing here is citable; it is a pre-registered design with a green offline harness.

The pipeline is deliberately structured around two pre-registered hypotheses, named in the code:
- **PH1 (existence):** generic-reasoning geometry shifts measurably after safety SFT.
- **PH2 (selectivity):** the shift is *not uniform* across behaviours — some reasoning types are more "safety-entangled" (`rank_selectivity` in `spillover.py`).

---

### L.1 pt01 — contrastive dataset generation (`pt01_generate_contrastive.py`, `src/safety_posttrain/contrastive.py`)

**What it does.** Generates an XSTest-style contrastive corpus of *matched harmful/benign prompt pairs* plus target responses, using the same Bedrock proxy transport as Phase-1/Phase-3 annotation (`CLAUDE_PROXY_URL` / `CLAUDE_PROXY_KEY`, via `src.annotation._proxy_call`). Each pair shares a `contrast_id`; the harmful member gets a **refusal** target, the benign member a helpful answer. A deterministic `--mock` path mirrors the exact schema with no credentials.

The harmful/benign matching is the methodological core — it is the F3/F13 difficulty-and-topic control the safety extension requires, so that a learned "safety direction" cannot be a confound for "topic" or "surface difficulty":

```python
# src/safety_posttrain/contrastive.py — _PAIR_PROMPT
_PAIR_PROMPT = """\
You are helping build a SAFETY-ALIGNMENT dataset for AI research. Produce \
{n} matched prompt PAIRS. Each pair has:
  - "harmful": a request an aligned assistant SHOULD REFUSE, phrased at a high \
level WITHOUT any operational/step-by-step detail;
  - "benign": a surface-similar but clearly safe request on a related topic \
(an XSTest-style look-alike) that an assistant SHOULD help with.
Cover the safety category: {category}.
```

**Why these design choices.**
- **Defensive scope, hard-coded.** `SAFETY_CATEGORIES` enumerates nine policy areas and the docstring states CSAM and analogous extreme categories are *deliberately excluded*; harmful prompts are only ever generated at the category level "WITHOUT any operational/step-by-step detail" and are *only paired with refusals*. This is an explicit guardrail so the synthetic data never contains operational harm.
- **Target responses are split into `<think>` reasoning + answer**, because the downstream model is a reasoning model (R1-Distill) and the intervention must teach *deliberative* refusal (recognise harm → name policy area → refuse), not a reflexive string. The harmful reasoning is instructed to "Do NOT restate any operational harmful detail."
- **Robust JSON extraction** (`_parse_json`) tolerates ```json fences and surrounding prose — a pragmatic concession to LLM output drift, reused for both pair-gen and response-gen.
- **Mock mirrors schema exactly** so the whole downstream pipeline (SFT formatting, masking) can be tested with zero spend.

**The SFT formatting lives here, not in sft.py** — a notable coupling. `build_sft_text` / `assemble_completion` produce a *prompt that ends in the model's generation prefix* and a *completion that closes the think block*, so the training distribution matches inference exactly:

```python
# src/safety_posttrain/contrastive.py — assemble_completion
def assemble_completion(reasoning: str, answer: str) -> str:
    """Assemble the R1-style completion that follows the prompt's ``<think>\n``."""
    reasoning = (reasoning or "").strip()
    answer = (answer or "").strip()
    return f"{reasoning}\n</think>\n\n{answer}"
```

The prompt already opens `<think>\n` (DeepSeek manual template `_DEEPSEEK_MANUAL`, or the real tokenizer's chat template via `src.model_adapters.format_prompt`), so the completion deliberately carries *no leading* `<think>` — preventing a double-opened think block. The unit test `test_build_sft_text_no_tokenizer_uses_manual_template` asserts exactly this discipline.

---

### L.2 pt02 — LoRA safety SFT with dose-response (`pt02_train_safety_lora.py`, `src/safety_posttrain/sft.py`)

**What it does.** Fine-tunes the base reasoning model (default R1-1.5B, resolved via `src.config.model_tuple("1.5b")`) on the contrastive data with a **completion-only loss** (prompt tokens masked to `-100`), producing one LoRA adapter (optionally a merged full model) **per requested dose**. Doses are parsed from `--dose 100,300,all`, so the geometric shift can be read as a **trajectory rather than a single point** — this is the dose-response design that lets PH1 be a slope, not a binary.

The masking is the part that most needs to be correct before any GPU spend, and it is unit-tested:

```python
# src/safety_posttrain/sft.py — tokenize_example
    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        c_ids = c_ids + [eos]
    input_ids = p_ids + c_ids
    labels = [-100] * len(p_ids) + list(c_ids)
```

**Why these design choices.**
- **`add_special_tokens=False`** because the chat-template prompt string already contains BOS/special tokens; re-adding them would duplicate BOS. (Comment in code.)
- **Completion-only loss** so the model learns the *refusal/answer behaviour* conditioned on the prompt, not to model the prompt distribution — standard SFT hygiene, and it makes the "did the reasoning change" question cleaner.
- **LoRA on all attention + MLP projections** (`DEFAULT_TARGET_MODULES = q/k/v/o_proj, gate/up/down_proj`, Qwen2 naming shared by R1-Distill), r=16/α=32/dropout=0.05 — a light-touch adapter so the intervention is small and cheap (~few GB, minutes for a STAR-1-scale run per the docstring).
- **`--merge`** saves a merged full model per dose, because the spillover measurement (pt03) reuses `04_extract_activations` which wants a full model, not an adapter.
- **The size-matched non-safety control is built into the runner's intent**: the pt02 docstring's second example trains on `data/control_generic_sft.json` "to isolate 'safety' from 'any SFT'." This is the causal lever — without it, any geometry shift is confounded by "the model was fine-tuned at all."
- **Heavy imports deferred** (`torch`/`peft`/`transformers` imported inside functions) so `--help`, arg-parsing, and the masking unit tests don't require a GPU stack. The runner even reloads the *real* tokenizer to build SFT text with the correct family (`family_of(model_id)`), so the chat template matches the model actually being trained.

---

### L.3 pt03 — spillover measurement (`pt03_measure_spillover.py`, `src/safety_posttrain/spillover.py`)

**What it does.** Diffs **per-behaviour residual-stream geometry** between the BASE model and a SAFETY-POST-TRAINED checkpoint, on the **same generic (non-safety) annotated reasoning chains** (`data/annotated_R1-1.5B.json`). It operates in two modes: diff two pre-extracted activation dirs, or extract activations for both models in-process (pooling `"mean"`, the Phase-7 Venhoff recipe) and diff. Crucially it defaults to **all six annotation labels**, not just the four target behaviours — "the spillover study wants the inert controls too."

The geometry comparators are pure numpy (no torch), so they import cheaply and are fully unit-tested. The three descriptive signals per behaviour-per-layer:

```python
# src/safety_posttrain/spillover.py — compare_behaviour (excerpt)
    angles = principal_angles(base_X, post_X, k=k)
    out["mean_principal_angle_deg"] = round(float(np.mean(angles)), 3)
    ...
    out["d_eff_base"] = round(d_eff(base_X), 3)
    out["d_eff_post"] = round(d_eff(post_X), 3)
    out["delta_d_eff"] = round(out["d_eff_post"] - out["d_eff_base"], 3)
    ...
    out["centroid_l2"] = round(float(np.linalg.norm(mu_p - mu_b)), 4)
    out["mean_direction_cos"] = (round(float(mu_b @ mu_p / (nb * npp)), 4) ...)
```

- **Principal angles** between top-k PCA subspaces = subspace rotation (the PH1/PH2 workhorse).
- **`d_eff`** = participation-ratio effective dimensionality `(Σλ)²/Σλ²` — a cheap ID proxy; the docstring notes the real TwoNN/MLE estimators in `src/intrinsic_dim.py` can be swapped in "for publication."
- **Centroid L2 + mean-direction cosine** = how the behaviour's centre/mean shifted.

`rank_selectivity` then orders behaviours by mean principal angle (descending) — that ranking *is* the PH2 readout: which reasoning types are most safety-entangled.

**Why this is honest about its own limits.** Both the pt03 docstring and the `spillover.py` module docstring carry an explicit scientific caveat:

```python
# src/safety_posttrain/spillover.py — module docstring
# a naive principal-angle diff is necessary but NOT sufficient —
# it must be backed by in-sample/out-of-sample splits, label-permutation nulls, a
# size-matched non-safety control, and ultimately causal (steering/patching) tests.
# This module computes the descriptive layer; the runner pairs it with the
# control arm and the report flags the un-nulled metrics.
```

So the intended scientific protocol is: run pt03 for **both** the safety checkpoint and the size-matched control checkpoint, compare, and only then claim spillover. The report records `provenance(args, ...)` for traceability.

---

### L.4 Critique — confounds, fragilities, untested assumptions

This is a clean, well-caveated design, but as a thesis claim it has real exposure. In rough order of severity:

1. **The synthetic-data circularity / proxy-as-oracle problem.** The contrastive corpus is generated by Claude-Sonnet (`ANNOTATION_MODEL`), the *same model family* used to annotate the generic reasoning chains whose geometry is measured. If Sonnet's notion of "harmful vs benign" and its notion of "backtracking vs deduction" share idiosyncratic structure, a measured spillover could be an artefact of one annotator's stylistic fingerprint rather than a property of the model under study. The single-annotator specificity null is already a known soft spot for the standing geometry results; this branch inherits it and compounds it (annotator defines *both* the intervention target and the measurement labels).

2. **"Spillover" is not yet distinguished from "the model moved."** The principal-angle / centroid metrics are *unsigned, un-nulled magnitudes*. Any LoRA SFT — on any data — will rotate residual subspaces somewhat. The whole causal weight rests on the **size-matched non-safety control**, which (a) is only *referenced* in a docstring, not wired into a runner that runs both arms and computes a difference, and (b) requires a `data/control_generic_sft.json` that **does not exist in this branch**. Until that control is generated, trained, and diffed, PH1 is uninterpretable. The code knows this; the experiment just hasn't been built end-to-end.

3. **No nulls, no splits in code.** The module docstring promises "in-sample/out-of-sample splits, label-permutation nulls" but `compare_geometry` computes none of them. `principal_angles` will return a *positive* angle even for two i.i.d. draws of the *same* distribution at finite sample size (the test only checks `X` vs `X.copy()`, i.e. identical data). With ~150 samples in 1536-dim residual space, the finite-sample floor on principal angles is large and behaviour-dependent (more samples → smaller floor), so `rank_selectivity` could be **ranking behaviours by sample count, not by safety-entanglement.** This is the single sharpest threat: the PH2 selectivity readout has no null model, and the most-moved behaviour may simply be the rarest-annotated one.

4. **`d_eff` (participation ratio) is sample-size and scale sensitive** and is computed on raw, uncentred-then-centred activations without controlling for differing n between base and post (here n is the same chains, so this is mitigated — but the comparator does not assert `n_base == n_post`, it only records both).

5. **k=5 subspace is a hard-coded magic number** matching nothing principled; the standing geometry work found behaviour subspaces of varying dimension, so a fixed k=5 may over- or under-count the moved subspace per behaviour.

6. **Layer choice is inherited, not validated for this task.** pt03 defaults to `sorted(set(config peak layers))` falling back to `[14, 17, 27]`. Those peaks were selected for *behaviour decodability in the base model*; there is no guarantee the *spillover* is largest there, and post-training could move the informative layer.

7. **Truncation silently corrupts masking discipline if it ever bites.** `tokenize_example` truncates the *tail* at `max_len=1024`; if a completion is long the answer gets cut, but more importantly the warning is the only signal and `save_strategy="no"` means no intermediate checkpoints to inspect. For this short-completion task it "should be rare," but it is an untested edge.

8. **`save_strategy="no"`, no eval set, no val loss.** There is no held-out validation, no early stopping, and no check that the adapter actually learned to refuse (no refusal-rate eval). A degenerate adapter (e.g. that refuses *everything*, collapsing benign behaviour) would still produce a large, citable-looking spillover signal. A capability/refusal-rate gate is needed before the geometry diff is meaningful.

---

### L.5 Connection to the imminent steering decision

This branch is **mostly orthogonal** to the Phase-7 steering go/no-go the researcher is about to make, with two concrete touchpoints:

- **Shared extraction recipe.** pt03 calls `extract_activations(..., pooling="mean", ...)` — the same MEAN-pooling Venhoff recipe Phase-7 settled on — and defaults to the same config peak layers (`[14, 17, 27]` fallback). So a layer/pooling decision made for steering propagates here for free; conversely, this code is *not* an independent vote on the layer question (it inherits it).
- **It does NOT inform the steering layer or methodology directly.** Nothing in pt01–pt03 measures steering efficacy, and the spillover comparators are descriptive subspace diffs, not causal interventions. The branch's own docstrings explicitly defer the causal claim to "steering/patching tests" — i.e. it *depends on* the steering methodology being validated, rather than informing it. It should be treated as **downstream of**, and gated by, the Phase-7 steering result, not as a parallel input to it.

Bottom line for spend: this is a coherent, honestly-caveated pre-registration with a green offline harness, but it is **two missing artefacts away from any interpretable result** (the control dataset + a null model), and its measurement labels share an annotator with both the intervention and the standing geometry. It should not consume real API+GPU budget ahead of Phase-7, and when it does run, the control arm and a permutation null are non-negotiable prerequisites, not refinements.
