## B. Data Generation — Tasks, Chains, Pilot Gate, Baselines

This section documents the **input-manufacturing** stage of the pipeline: how the
task corpus is produced (Phase 1), how reasoning chains-of-thought are sampled
from the model under study (Phase 2), the pilot gate intended to de-risk the
expensive scale-up (`00_pilot_gate.py`, `validate_pilot_lengths.py`), the
non-reasoning **baseline/control** corpus (Phase 2b) that exists specifically to
neutralise the chain confound **CF-2**, and the QC / model-identity verifiers
(`check_chain_quality.py`, `verify_base_model.py`). Everything downstream —
annotation, activation extraction, the per-behaviour subspaces, and the steering
vectors the researcher is about to spend real money building — inherits whatever
biases are baked in here. The headline critique, established with the actual
on-disk artefacts, is that **the corpus is 50.2% truncated at the token ceiling,
the pilot gate that should have caught this either never ran or was overridden,
and the CF-2 baseline (Phase 2b) has not been generated at full scale at all.**

---

### B.1 Phase 1 — Task generation (`01_generate_tasks.py`, `src/task_gen.py`)

**What it does.** It asks Claude (`anthropic.claude-sonnet-4-5-20250929-v1:0`,
via the lab proxy, NOT the R1 model) to write a balanced corpus of **1000
reasoning tasks = 100 tasks × 10 hand-defined categories**. The categories are a
fixed taxonomy of *reasoning domains* (note: domains, not the Venhoff *behaviour*
labels that get annotated later):

```python
# src/task_gen.py — CATEGORIES (top of module)
CATEGORIES: dict[str, str] = {
    "mathematical_logic":   "Problems requiring formal logic, proofs, and mathematical reasoning",
    "spatial_reasoning":    "Tasks involving spatial relationships, geometry, and visualisation",
    "verbal_logic":         "Syllogisms, verbal analogies, and language-based reasoning",
    "pattern_recognition":  "Identifying and continuing abstract sequences or patterns",
    "lateral_thinking":     "Problems requiring creative, non-linear approaches",
    "causal_reasoning":     "Cause-and-effect relationships and causal inference",
    "probabilistic_thinking":"Uncertainty, probability, and statistical reasoning",
    "systems_thinking":     "Complex systems, interdependencies, and emergent behaviour",
    "creative_problem_solving":"Open-ended problems requiring novel approaches",
    "scientific_reasoning": "Hypothesis formation, experimental design, evidence evaluation",
}
```

**Why these choices.**
- **Generator ≠ subject model.** Tasks come from a strong frontier model so the
  prompts are diverse, self-contained, and genuinely multi-step (the system/user
  prompt demands "at least 3–5 reasoning steps", "self-contained", and
  explicitly "Do NOT include the answer, solution hints, or worked examples").
  This keeps the *stimulus* distribution decoupled from the *subject* (R1-1.5B),
  so the chains are real model behaviour rather than echoes of a leaked solution.
- **De-duplication via in-context blocklist.** Within a category, each batch is
  generated with the prior prompts (first 120 chars) injected as a "do not
  repeat" context, so the model does not keep re-emitting the same textbook
  classics:

```python
# src/task_gen.py — generate_tasks() inner loop
context = [t["prompt"][:120] for t in cat_tasks] if cat_tasks else None
batch = _call_api(cat_name, cat_desc, prefix, start, n_this,
                  proxy_url, proxy_key,
                  context_summaries=context)
cat_tasks.extend(batch)
time.sleep(0.5)
```

- **Sampling temperature 0.8** in `_proxy_call` (diversity is wanted here, the
  opposite of the chain stage), with a 3-attempt retry + exponential backoff and
  a `json.loads` parse guard that strips ```` ``` ```` fences.

**The eval-split single-source-of-truth.** `stratified_eval_split()` is the one
function both the steering-vector builder and `07_evaluate_steering` must call so
the Phase-7 hold-out is identical on both sides. Its own docstring flags the
landmine that motivated it:

```python
# src/task_gen.py — stratified_eval_split()
# tasks_final.json is perfectly category-blocked, so the naive
# `tasks[-n_test:]` rule selected 50 tasks of a single category.
per_cat = max(1, n_test // len(by_cat))
test_tasks = [t for cat in sorted(by_cat) for t in by_cat[cat][-per_cat:]]
return test_tasks, f"last {per_cat} per category"
```

This matters directly for the steering decision: if the vector builder and the
evaluator ever drift on this rule, the "hold-out" silently leaks and any steering
effect is contaminated.

**Critique.**
- The 10 task categories are an *a priori* convenience taxonomy, not validated
  against anything; the actual scientific unit later is the Venhoff behaviour
  label, which is orthogonal. Category balance (100 each) does **not** guarantee
  behaviour balance — and indeed `adding-knowledge` is the behaviour that keeps
  failing specificity downstream.
- The corpus on disk is `data/tasks_final.json` (1000 tasks), produced after a
  dedup/cleanup pass (`data/tasks.json → tasks_deduped → tasks_500balanced →
  tasks_final`). The 500-balanced and deduped intermediates show the corpus was
  reworked; the pilot scripts still point at the *stale* `data/tasks.json` (see
  B.3), a provenance seam.

---

### B.2 Phase 2 — Chain generation (`02_generate_chains.py`, `src/chain_gen.py`)

**Subject model.** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (alias `1.5b`,
short name `R1-1.5B`; 28 layers, hidden 1536). The prompt is delegated to
`src/model_adapters` so the same harness drives DeepSeek `<think>` CoT, gpt-oss
harmony, and non-thinking bases — but the DeepSeek default is the path used for
the real corpus.

**Sampling = greedy (temperature 0).** This is the load-bearing methodological
choice and it follows Venhoff et al.:

```python
# src/chain_gen.py — generate_chain()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
```

With `temperature=0.0`, `do_sample=False` → deterministic greedy decoding. The
docstring is explicit that **the seed is a no-op under greedy** ("Greedy decoding
(T=0) ignores the RNG state entirely"). The multi-seed plumbing (`--seeds`,
`dedup_keys=("task_id","seed")`, `*_multiseed.json`) only bites when someone
re-runs with `temperature>0` for the §M4.5 robustness check.

**Generation budget = 8192 tokens** (`--max-tokens 8192`, overriding the
library default of 2048). Records are written with a schema that stores
`prompt`, `chain`, and `full_text = prompt + chain` so Phase 4 can reconstruct
*exact* token positions for activation extraction. Generation is checkpointed
(atomic tmp+rename) every 10 chains and is resume-safe; a batched variant
(`generate_chains_batched`, left-padded for decoder-only correctness, per-task
fallback on OOM) exists for throughput on shared GPUs.

**Why greedy.** Determinism makes the chain a *fixed* object: the same task
always yields the same chain, so activation extraction, annotation, and steering
all operate on one canonical artefact, and the "geometry" is not an artefact of
sampling noise. It also matches the comparison literature so behaviour fractions
are comparable to Venhoff Fig. 2.

**Critique — the truncation problem (the single most damaging finding here).**
The actual on-disk corpus `data/chains_R1-1.5B.json` (1000 chains) has:

- **mean = 5052 tokens, median = 8192, P95 = 8192, max = 8192**
- **502 / 1000 chains (50.2%) sit exactly at the 8192 ceiling**
- **only 501 / 1000 chains contain a closing `</think>`**

So **half the corpus is truncated mid-reasoning.** This is not a cosmetic data
issue — it is a structural confound for *this very project*:
1. Truncated chains have systematically *missing* late-CoT behaviours.
   Backtracking and uncertainty-estimation tend to cluster late in a chain; if
   half the chains are cut off, the behaviour fractions and the late-token
   activations are biased toward whatever fits in 8192 tokens.
2. It directly couples **behaviour** to **length/position** — which *is* CF-2
   (below). A "backtracking" subspace learned partly from where-in-the-chain a
   truncation lands is suspect.
3. `check_chain_quality.py` is built precisely to surface this — its truncation
   cross-tab separates "hit max AND no closing think (truncated mid-thinking)"
   from "short AND has closing think (clean finish)" — but a report existing is
   not the same as the corpus being clean.

---

### B.3 The pilot gate — designed, then bypassed (`00_pilot_gate.py`, `validate_pilot_lengths.py`)

**Intent.** Run a stratified 20-chain pilot (2 tasks × 10 categories) through
Phases 2–3 and check **5 gates** before authorising the full (expensive) run:

```text
# 00_pilot_gate.py docstring — Checks (all must pass before full Phase 2)
1. All 20 chains end with </think>                (no truncation)
2. Mean chain length ≤ 2,500 tokens               (cost/time bound)
3. Annotation parses into [(label, text), ...]    (format correct)
4. All 6 Venhoff labels appear ≥ 1× across 20 chains
5. Sentence fractions roughly match Venhoff Fig. 2 (±10 pp tolerance)
```

The expected Venhoff fractions are hard-coded (`deduction 0.52`,
`adding-knowledge 0.15`, `uncertainty 0.09`, `initializing 0.07`,
`example-testing 0.06`, `backtracking 0.04`) and used as the target distribution.

**It never ran as written, and it is now hard-disabled.** The very first line of
`main()` is a hard exit:

```python
# 00_pilot_gate.py — main()
sys.exit(
    "00_pilot_gate.py is HISTORICAL and cannot run: its annotation "
    "subcommands import batch-API functions that never existed in "
    "src/annotation.py, and it reads the stale data/tasks.json. The pilot "
    "ran via `03_annotate_chains.py --pilot`; use check_chain_quality.py "
    "and verify_annotation_completeness.py for the corpus-level checks. "
)
```

So Checks 1–5 were **never executed programmatically.** The docstring concedes
the corpus-level equivalents migrated to `check_chain_quality.py` and
`verify_annotation_completeness.py`. This is honest bookkeeping, but it means the
gate that was supposed to *prevent scale-up on a bad config did not gate
anything.*

**The length-only gate that DID run — and what it should have said.**
`validate_pilot_lengths.py` is a standalone check with a sharp decision rule:

```python
# validate_pilot_lengths.py — decision rule
if p95 < 6500 and n_ceiling <= 2:
    print(f"PASS: P95={p95:.0f} < 6500 and {n_ceiling}/20 hit ceiling")
    ...
else:
    print(f"STOP: P95={p95:.0f}, {n_ceiling}/20 hit ceiling — report before proceeding")
    sys.exit(1)
```

Running its logic against the **actual** pilot file `data/chains_pilot.json`:

- pilot **P95 = 8192**, **11 / 20 chains hit the 8192 ceiling**, mean ≈ 5082.

That is a **massive STOP** (rule fires on either `P95 ≥ 6500` *or* `≥3/20 at
ceiling`; here it is 8192 and 11/20). The pilot already showed that >half of
chains would truncate at 8192 — and the full run was generated anyway, producing
exactly the 50.2% truncation observed in B.2. **The pilot correctly predicted the
problem; the STOP was not honoured.** (Two further fragilities: the pilot file
lacks a `difficulty` field, so the script's `c['difficulty']` print would raise
`KeyError` before reaching the decision line; and `00_pilot_gate.py`'s Check 2
target of ≤2500 tokens is wildly inconsistent with the 8192 budget actually used
— the gate's own thresholds were never reconciled with the run config.)

**Critique.** The original pilot Check 1 ("all 20 chains end with `</think>`")
would have *failed outright* on the real pilot (only a fraction terminate
cleanly). The remediation — moving checks into corpus-level QC scripts — converts
a *blocking* gate into a *descriptive* report, which is exactly the wrong
direction for de-risking spend. For the steering experiment the lesson is
concrete: **before spending on Phase 7, re-run the chain corpus with a higher
ceiling (or budget-adaptive stopping) so the activations feeding the steering
vectors are not drawn from a 50%-truncated, length-confounded population.**

---

### B.4 Phase 2b — Baseline / control chains, and the CF-2 confound (`02b_generate_baseline_chains.py`)

**Why this script exists — CF-2 stated.** The central claim of the thesis is that
specific *reasoning behaviours* (backtracking, uncertainty-estimation,
example-testing, knowledge-augmentation) each occupy their own low-dimensional
subspace. **CF-2 is the objection that any apparent "behaviour geometry" could
instead be a geometry of chain *length* or token *position*** — long elaborate
chains differ from short ones in activation space for reasons that have nothing
to do with the behaviour label. Phase 2b builds the control that is supposed to
break that alternative explanation: the *same model family without the reasoning
post-training*, on the *same tasks*, with an *identical output schema*, so that
"how much behaviour is *added* by R1 distillation" can be measured rather than
assumed.

**The baseline model.** `Qwen/Qwen2.5-Math-1.5B` — chosen because
`verify_base_model.py` *empirically* established it is the true base of
R1-Distill-Qwen-1.5B (embed cosine **0.9936** and lm_head **0.9726**, vs only
**0.84 / 0.85** for the Instruct variant; aggregate delta 0.098 vs 0.192). The
verdict file on disk reads "**Likely:** `qwen-math` is the base; margin over
runner-up is 0.0937." This is the correct control: same weights modulo the
distillation delta, so a base-vs-distilled contrast isolates what RL/distillation
added.

**The construction — no chat template, no `<think>`, raw Q/A scaffold:**

```python
# 02b_generate_baseline_chains.py — format_baseline_prompt()
def format_baseline_prompt(instruction: str) -> str:
    """Q/A scaffold for non-reasoning base models.
    Matches the typical math chain-of-thought corpus format that
    Qwen2.5-Math was pretrained on. NO chat template, NO <think>.
    """
    return f"Question: {instruction}\n\nAnswer:"
```

**Schema parity is deliberate.** The records are byte-for-byte the same shape as
Phase 2 (`task_id, category, instruction, prompt, chain, full_text, n_tokens`)
"so downstream phases (annotation, activation extraction, PCA, steering) work
unchanged via `--model-short QwenMath-1.5B`." Generation is greedy
(`do_sample=(temperature>0)`), checkpointed, resume-safe, and snapshots the
existing file before touching it (a guard 02b "previously didn't" have).

**The token-budget defence (and why it is a CF-2 fix in itself).** The
`--max-new-tokens` default was raised to 8192 with an explicit rationale that is
itself a CF-2 argument:

```python
# 02b_generate_baseline_chains.py — main() arg help
"Generation budget (default: 8192 — MUST match 02_generate_chains.py's cap:
 base Qwen-Math models repetition-loop, and truncating the baseline 4x sooner
 than R1 biases every length/completeness-sensitive base-vs-distilled
 comparison. Baselines are typically short, so the higher cap rarely costs
 anything.)"
```

i.e. if the baseline were truncated at a *different* length than R1, the
base-vs-distilled contrast would be polluted by exactly the length artefact CF-2
warns about — so the caps are forced equal.

**Critique — does 02b actually neutralise CF-2? Partially, and it is UNRUN.**

1. **It is not generated at full scale.** On disk there is only
   `data/chains_QwenMath-1.5B_smoke.json` (**20 tasks**); there is **no**
   `data/chains_QwenMath-1.5B.json`. The CF-2 control has been *designed and
   smoke-tested but never produced for the 1000-task corpus*, so no
   base-vs-distilled behaviour-fraction or geometry comparison has actually been
   computed from it. This is the sharpest gap: **the confound's antidote does not
   yet exist as data.**
2. **A control on a different stimulus format is itself confounded.** The
   baseline uses a `"Question:…\n\nAnswer:"` scaffold while R1 uses the `<think>`
   chat template. So base-vs-distilled differs in (a) post-training **and** (b)
   prompt format simultaneously. Any geometry difference cannot be cleanly
   attributed to the reasoning post-training alone — the prompt is part of the
   treatment. The authors' defence is that the scaffold matches Qwen-Math's
   pretraining corpus (so the base is being used *in distribution*), which is
   reasonable but does not fully de-confound.
3. **CF-2-within-R1 is untouched by 02b.** 02b addresses the *base-vs-distilled*
   axis. It does **nothing** for the within-R1 worry that a behaviour subspace is
   really a length/position subspace — and given B.2's 50.2% truncation, that
   within-corpus length confound is live and large. Neutralising CF-2 properly
   needs *length-/position-matched* negatives within the R1 corpus, not just a
   non-thinking sibling model.
4. **The baseline may emit almost no annotatable behaviour.** Qwen-Math base is
   expected to produce short, non-deliberative answers; if it has near-zero
   backtracking/uncertainty, the "control" is a floor near zero, which makes the
   "behaviour is added by distillation" claim easy but tells you little about
   whether the *geometry within R1* is behaviour-specific vs length-specific.

---

### B.5 QC and identity verification (`check_chain_quality.py`, `verify_base_model.py`)

**`check_chain_quality.py`** is a pure-function corpus auditor (no I/O in
`quality_check`) that emits a Markdown + JSON report. It checks structural
integrity (missing fields, dup `task_id`s, and crucially
`full_text != prompt + chain` — the exact invariant Phase 4 relies on for token
alignment), per-category token distributions, a **truncation cross-tab**, prompt
integrity (counts prompts containing `<think>`), and content anomalies
(non-ASCII, CJK/kana language-drift, and regex-detected repetition loops). It is
the de-facto replacement for pilot Checks 1–2. Its existence is good practice;
its limitation is that it is *descriptive* — it reports 50% truncation, it does
not block on it.

**`verify_base_model.py`** settles the base-model identity empirically rather
than trusting DeepSeek's (silent) report, using three weight-space probes:
embed/lm_head cosine, and per-layer Frobenius fractional deltas on `q_proj` /
`gate_proj`:

```python
# verify_base_model.py — fractional_delta()
def fractional_delta(A, B) -> float:
    """Frobenius(A - B) / Frobenius(B). How much A diverges from B, relative
    to B's scale."""
    if A.shape != B.shape:
        return float("nan")
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))
```

This is the right way to ground the CF-2 control: the baseline's scientific
validity depends entirely on it really being R1's base, and the script gives a
defensible **0.9936 embed cosine** to back the `configs/config.yaml` claim. (Mild
caveat: the verdict is graded "Likely," not "Confident," because the winner's
aggregate delta 0.098 is just under the 0.10 "likely" threshold and the margin
over the Instruct variant, while clear, is modest in absolute terms.)

---

### B.6 Connection to the steering decision (layer + methodology)

This stage does **not** itself pick a steering layer — `configs/config.yaml`
carries `steering_layer: 27` for R1-1.5B (a Huang-et-al. late-layer choice) and
the baseline mirrors it at 27 "for direct comparability." But Phase B gates the
steering experiment in three concrete ways:

1. **The activations that become steering vectors are drawn from this corpus.**
   If 50.2% of chains are truncated at 8192, the residual-stream rows feeding the
   behaviour subspaces are biased toward early/mid-chain content and against the
   late-chain behaviours (backtracking, uncertainty) the steering experiment most
   wants to manipulate. **Recommendation before spend: regenerate (or
   length-filter/extend) so the steered behaviours are well-represented and not
   length-confounded.**
2. **CF-2 is still open as data.** The base-vs-distilled control (Phase 2b) is
   designed but only smoke-run. A steering result claiming "we moved a
   *behaviour*" is exposed to the reviewer's CF-2 objection — "you moved chain
   length/position" — until the matched-cap baseline corpus exists and a
   length-matched within-R1 control is added. The matched 8192 cap in 02b is the
   right instinct; it just has to actually be run.
3. **The eval hold-out is defined here.** `stratified_eval_split()` is the single
   source of truth the steering builder must exclude and the evaluator must
   score on. Any divergence silently leaks the hold-out and inflates the steering
   effect — verify both call sites use it before committing GPU/API budget.

**Bottom line for the researcher:** the generation machinery is well-engineered
(deterministic, resume-safe, schema-consistent, identity-verified), but two
load-bearing facts undercut citable claims as the corpus stands today — **(i) the
R1 corpus is half-truncated and the pilot's own STOP rule was overridden, and
(ii) the CF-2 control exists only as a 20-task smoke.** Both should be closed
before money is spent on steering.
