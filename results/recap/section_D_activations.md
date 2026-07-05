## D. Activation Extraction & Pooling

This stage (Phase 4) is the bridge between *labelled text* and *geometry*. Phases 1–3 produce a corpus of reasoning chains in which each sentence carries a behaviour label (`backtracking`, `uncertainty-estimation`, `example-testing`, `adding-knowledge`, plus the non-target labels). Phase 4 re-runs every annotated chain through the model with forward hooks, captures the residual stream at every layer, locates the token positions belonging to each labelled sentence, and pools those positions into one vector per (behaviour, layer, instance). The output is the set of `{behaviour}_layer{n}.npy` matrices on which **every downstream geometry claim and the entire steering programme rest.** Get this stage wrong and nothing above it is salvageable — which is exactly the history this code carries (CF-13…CF-16).

The relevant files:

- `04_extract_activations.py` — the runner (CLI, model load, skip-if-done guard).
- `src/activation_extraction.py` — the method (span→positions, pooling, sharded accumulation, integrity flag, sidecar, sweep).
- `src/hooks.py` — `ActivationCache`, the forward-hook residual-stream capture.
- `src/model_adapters.py` — decoder-layer location + harmony/DeepSeek family handling.
- `src/text_offsets.py` — the canonical occurrence-aware sentence→char-offset locator (CF-13 fix).
- `src/row_provenance.py` — row-id sidecar loading + duplicate hygiene (CF-13/CF-14 fixes).
- `pooling_sweep.py` / `tests/test_pooling.py` — the mean-vs-last sensitivity machinery.

---

### D.1 What is captured: residual stream, post-block, every layer, CPU-offloaded

Activations are captured by registering a PyTorch **forward hook on each decoder block** and grabbing that block's output — the post-block residual stream, *not* the MLP sub-module output, *not* attention, *not* a LayerNorm'd read. This matters: "residual stream" here means the hidden state flowing between transformer blocks (`hidden_states` after block *i*), which is the object steering vectors are added to. The hook deliberately handles both the tuple- and bare-tensor return conventions of HF blocks across versions, and immediately detaches + moves to CPU as float32:

```python
# src/hooks.py — ActivationCache._make_hook
def _hook(module, input, output):
    # HF transformer blocks may return either a tuple
    # (hidden_states, present_kv, ...) or just the hidden_states tensor
    # depending on the version. Handle both — we want the full 3D
    # (batch, seq_len, hidden_dim) tensor in the cache.
    h = output[0] if isinstance(output, tuple) else output
    self._cache[idx] = h.detach().cpu().float()
return _hook
```

```python
# src/hooks.py — ActivationCache._register
def _register(self) -> None:
    self._remove()
    for idx in self.layers:
        h = self._all_layers[idx].register_forward_hook(self._make_hook(idx))
        self._hooks.append(h)
```

Design rationale, defended as the authors would:

- **Post-block residual stream** is the canonical interpretability read for behaviour directions and the object that an activation-addition steering intervention perturbs. The module docstring justifies it as "the same convention as Huang et al. and Venhoff et al." — i.e. the recipe the steering literature this thesis competes with uses. Choosing the residual stream (not MLP-out) means the captured directions are *in the same space* a steering hook would write to, so a direction found here can be added back later without a basis change.
- **`detach().cpu().float()` per layer** is a memory decision: long R1 chains × 28 layers × hidden-dim in fp16 on GPU would OOM. Off-loading to CPU float32 immediately keeps GPU memory flat across chains. The cost is host RAM and a fp16→fp32 widening, accepted because the downstream PCA/geometry wants float32 anyway.
- **Layer coverage is ALL layers by default.** In the runner `layers=args.layers` defaults to `None`, and the library expands `None` to the full stack:

```python
# src/activation_extraction.py — extract_activations
if layers is None:
    layers = list(range(len(model.model.layers)))
```

For R1-Distill-1.5B that is **layers 0–27 (28 layers), hidden_dim 1536**. The runner's storage/runtime note ("~3–4 hours for 1000 chains × 28 layers", "~500 MB per model") confirms the intended full-stack extraction. `config.yaml extraction.layers: null` keeps this default.

The decoder-layer list is found via a fast path with a loud fallback. `ActivationCache.__init__` tries `model.model.layers` directly (DeepSeek/Qwen/Llama/gpt-oss all expose it) and only on `AttributeError` calls `locate_decoder_layers`, which walks a candidate list and **raises rather than silently mis-hooking** an unknown architecture:

```python
# src/model_adapters.py — locate_decoder_layers
candidates = (
    lambda m: m.model.layers,        # Qwen2 / Llama / GptOss
    lambda m: m.model.model.layers,  # some wrapped CausalLMs
    lambda m: m.transformer.h,       # GPT-2 / NeoX-style
    ...
)
```

This is the right defensiveness: a hook silently attached to the wrong module would produce plausible-looking but meaningless geometry, the worst kind of failure.

---

### D.2 From a labelled sentence to token positions: the Venhoff 1+10 window

Each annotated sentence is a *character span* in the chain text; the model sees *tokens*. The extractor tokenises the full `prompt + chain` text once per chain **with offset mapping**, then maps each sentence's char offset to a token window of **one preceding token + the first N=10 execution tokens** (`n_preceding=1`, `n_execution=10`, set in `config.yaml`). This is the Venhoff et al. recipe and the docstring states it explicitly ("one preceding token + first N execution tokens, following Venhoff et al.").

```python
# src/activation_extraction.py — _sentence_to_token_positions
onset = None
for tok_idx, (tok_start, tok_end) in enumerate(offsets):
    if tok_end > sentence_start_char:
        onset = tok_idx
        break
if onset is None:
    return []
...
for offset in range(n_preceding, 0, -1):
    p = onset - offset
    if p >= 0:
        positions.append(p)
for offset in range(min(n_execution, seq_len - onset)):
    p = onset + offset
    ...
    positions.append(p)
```

Why this window and not the whole sentence: the "first 10 tokens after the onset marker" captures the *behaviour signal at its surface*—the model committing to (e.g.) a backtrack—while keeping the per-instance vector count fixed-ish and avoiding diluting the signal across an arbitrarily long sentence. The single preceding token gives a small amount of pre-onset context (the residual the model decodes the onset *from*).

The crucial subtlety the code flags itself: **15.8% of target sentences are shorter than the 1+10 window**, so the unclipped window bleeds into the *next* sentence's tokens — a measured onset/surface-lexis confound. This is the `clip_window_to_sentence_end` knob (CF-flavoured robustness arm). It is **default OFF** (`config.yaml clip_window_to_sentence_end: false`) explicitly "to stay comparable with existing extractions" — i.e. the canonical matrices contain this bleed. The clip, when on, stops execution tokens at the sentence end char:

```python
# src/activation_extraction.py — _sentence_to_token_positions
if (sentence_end_char is not None and offset > 0
        and offsets[p][0] >= sentence_end_char):
    break  # token starts past the sentence end — next sentence's text
```

This is a genuine, acknowledged confound in the canonical data: a non-trivial fraction of "behaviour" vectors are partly made of the *following* sentence. The team chose comparability over correctness here; for the thesis it means any per-behaviour direction is partly contaminated by adjacency.

---

### D.3 Pooling: MEAN is canonical, but the code argues for LAST

Each (behaviour, instance) is a `(n_positions, hidden)` slice. `_pool` collapses it to `(hidden,)`. Four modes exist; **mean** is the configured default (Venhoff recipe), but the docstring is unusually candid that **last** is arguably more principled:

```python
# src/activation_extraction.py — _pool
#   mean  — average over positions (order-invariant; smears the within-span
#           trajectory and can cancel opposing directions).
#   last  — last execution token: in a causal transformer this is the only
#           position that has attended over the whole span (most
#           context-complete), and it is the residual the model decodes from.
if mode == "mean":
    return acts.mean(dim=0)
if mode == "last":
    return acts[-1]
if mode == "first":
    return acts[0]
if mode == "max":
    return acts.max(dim=0).values
```

The tension is real and pinned by tests: `test_mean_loses_order_information` asserts `mean` is permutation-invariant while `last` is not. So:

- **Why mean was chosen:** it is the Venhoff recipe (comparability with the prior literature the thesis benchmarks against), it averages out per-token noise, and it gives a stable centroid for a behaviour cloud. For *measuring* the geometry of a behaviour cluster, a smeared centroid is defensible.
- **Why the code keeps warning about it:** mean is order-invariant and "can cancel opposing directions." In a causal transformer only the **last** token has attended over the whole span and is the residual the model actually decodes from — which is precisely the position a steering intervention is most analogous to. So for the *steering* question (D.6), `last` has a stronger first-principles claim than `mean`.

To resolve this without re-running the GPU, the `pooling_sweep` machinery pools **multiple modes from one shared forward pass** (`streams = [pooling] + extra_modes`), writing extra modes to `pool_<mode>/` and a `pooling_sweep.json` comparing, at the steering layer, `cos(single_direction[mean], single_direction[last])` per behaviour and `d_eff_70` per mode:

```python
# src/activation_extraction.py — extract_activations (inner accumulation)
for layer_idx in layers:
    sl = cache[layer_idx][0, positions, :]   # (n_positions, hidden)
    for m in streams:   # primary + sweep extras, same forward pass
        acc[m][cat][layer_idx].append(_pool(sl, m).numpy().astype(np.float32))
```

`config.yaml` has this **ARMED**: `pooling_sweep: [mean, last]`, so every (re)extraction now also produces the `last` activations and the mean-vs-last cosine report at ~2× disk and ≈0 extra GPU. If `cos≈1`, the steering direction is pooling-robust; if `cos≪1`, the geometry is a pooling artefact and `last` should be preferred. This is the single most important sensitivity check feeding the steering decision, and it is wired to run automatically.

---

### D.4 Occurrence-aware span extraction — the CF-13 keystone fix

The original locator used `str.find`, which maps **every verbatim repeat of a sentence to its FIRST occurrence**. R1 chains loop and repeat (especially truncated ones and canned backtracking phrases), so repeated annotations all bound to the *same* token span, producing **35–56% byte-identical activation rows**. Zero-distance neighbours destroy every kNN-based intrinsic-dimension/curvature estimator (TwoNN, Levina–Bickel, geodesic graphs), and — worse — duplicates concentrate *within* a behaviour label, so the real per-behaviour matrix is duplicate-rich while label-permuted nulls are duplicate-poor, biasing the permutation test anti-conservatively. The 2026-06-08 tier1 layer-27 nulls (the keystone significance test) were **superseded** by this.

The fix (`src/text_offsets.locate_annotation_offsets`) runs a **forward cursor over the chain's ordered annotations**, searching each sentence from just past the previous match so the i-th repeat binds to the i-th occurrence:

```python
# src/text_offsets.py — locate_annotation_offsets
cursor = 0
for sent in sentences:
    idx = find_sentence_offset(chain_text, sent, start=cursor)
    offsets.append(idx)
    if idx is not None:
        # +1 (not +len) so overlapping/nested annotation boundaries can
        # still match while guaranteeing strict forward progress for
        # identical repeats. Never move the cursor backwards on an
        # out-of-order fallback match.
        cursor = max(cursor, idx + 1)
```

Two design details worth flagging: it advances by `+1` (not `+len`) so nested/overlapping annotation spans still match, and it falls back to a from-start search for out-of-order annotations so nothing that matched before fails. The extractor calls this over the **full** annotation list (all labels, not just the four targets) so the cursor advances identically whether or not a sentence is a target — that determinism is what lets analysis-side loaders reconstruct rows:

```python
# src/activation_extraction.py — extract_activations
# Locate ALL annotations up front (occurrence-aware): the cursor must
# advance over every annotation — including non-target labels — so
# repeats bind to successive occurrences deterministically (CF-13).
sent_offsets = locate_annotation_offsets(
    chain_text, [a.get("text", "") for a in annotations]
)
```

Empirically the docs report this drops the duplicate-row fraction from **51.9% → 1.2%**. A residual ~1.2% (over-annotation collisions + nested spans) remains, which is why `dedup_rows` in `src/row_provenance.py` is still applied downstream. Importantly, the stats review found the load-bearing Levina–Bickel "real < null" compression signal **survives dedup** (backtracking 9.12 vs null 10.65, ~18 SD) — duplicates understated absolute dims ~3× but did not manufacture the compression direction; TwoNN's nonsense values *were* the duplicate artefact. So the headline geometry claim survived this fix; the keystone null still owes a clean re-run.

---

### D.5 Provenance stamping, integrity flags, and crash-safety (CF-14/CF-15/CF-16 plumbing)

Activation matrices carry no row ids. Previously every analysis script reconstructed per-row chain-ids by *replaying* the Phase-4 iteration and **silently fell back to a proxy chain-id on a count mismatch** — under which the chain-stratified permutation null degenerates to a no-op (null ≡ real, p≈1.0) while looking legitimate (CF-14). The data-integrity review found the skip-blind replay mis-assigned 93.5% of uncertainty-estimation rows. Phase 4 now writes a `row_index.json` sidecar — one record per accepted row, in exact row order, identical across pooling streams:

```python
# src/activation_extraction.py — extract_activations (per accepted row)
row_index[cat].append({
    "chain_id": chain_id,
    "annotation_index": ann_idx,
    "char_offset": sent_offset,
    "token_start": positions[0],
    "n_positions": len(positions),
})
```

`src/row_provenance.chain_ids_for` loads this sidecar first and only replays for legacy extractions; `require_aligned` turns any chain-id/row mismatch into a **hard error** instead of a silent proxy. The matching rule is versioned (`SENTENCE_MATCHING_VERSION = "occurrence_aware_v1"`) and recorded in both `metadata.json` and the sidecar so analyses can refuse mixed-rule data.

Two more robustness mechanisms:

- **Sharded, crash-safe accumulation.** Accumulators flush to per-(stream, behaviour, layer) shard files every `flush_every` chains, then concatenate one (behaviour, layer) array at a time at the end. The docstring records why: an OOM-during-write killed the 2026-06-12 full-corpus run after 7/230 files and earlier truncated a behaviour to 1/28 layers. Peak memory at the final write is now bounded to one array, not the whole corpus.
- **An explicit `complete` integrity flag**, written LAST, true only when every behaviour with instances has all its layers on disk; `verify_extraction_complete` lets downstream refuse a partial set rather than silently building geometry on a truncated extraction. The runner's skip-if-done guard also re-runs if any target behaviour has zero extractions.

---

### D.6 Connection to the steering decision (layer choice + methodology)

This stage **constrains but does not finalise** the two open steering decisions (per the steering-status memory: pooling=MEAN resolved via Venhoff recipe; layer NOT finalised).

**Layer choice.** Because extraction defaults to **all 28 layers**, the captured data does *not* constrain which layer is steered — every candidate (mid-stack 11/16/19, and L27) is already in the canonical matrices, so Phase 7 can pick the steering layer post-hoc without re-extraction. The one place a layer is hard-coded is the *sweep report*, which compares pooling at `steer_layer = 27 if 27 in layers else max(layers)`:

```python
# src/activation_extraction.py — _write_pooling_sweep_report
steer_layer = 27 if 27 in layers else max(layers)
```

So the **pooling sensitivity check is evaluated at L27 by default** even though the steering layer is unsettled — a mild mismatch: if Phase 7 ultimately steers at a mid-stack layer (the Venhoff-recipe region), the mean-vs-last cosine the team will cite was computed at L27, not at the steered layer. Re-pointing the sweep's `steer_layer` to the chosen layer (or sweeping it) would close that gap.

**Methodology (mean vs last).** The steering vector is built from these pooled activations, so the pooling choice *is* part of the steering methodology. The canonical direction is mean-pooled (comparability), but the code's own argument — only `last` is context-complete and is the residual the model decodes/steers from — means the steering experiment should at minimum report the mean-vs-last cosine from `pooling_sweep.json` and, if it is low, prefer `last`. The sweep is armed in config precisely so this number exists before any GPU+API spend on steering.

---

### D.7 Status: RUN vs UNRUN, citability

- The **extraction code** is fixed and tested (`tests/test_pooling.py` pins all four pooling semantics + the sweep report; CF-13/14/15/16 fixes landed 2026-06-12).
- The **canonical matrices** in `data/activations/` predate or were regenerated around the Gate-0 work; per CONFOUNDS, the geometry is now **citable** (Gate-0 DONE: subspace survives, curvature is a clean negative/chain-artefact). The duplicate fix did *not* overturn the compression direction.
- The **pooling sweep with `last`** is *armed* but the canonical `pooling_sweep.json` (mean-vs-last cosine per behaviour at the steering layer) is the artefact the steering decision most needs and should be confirmed present/regenerated before committing spend.
- **Phase 7 steering itself is UNRUN.**

---

### D.8 Critique — confounds, fragilities, untested assumptions

1. **Window bleed into the next sentence (15.8%).** Default `clip_window_to_sentence_end: false` means a measured ~16% of target vectors are partly built from the *following* sentence's onset tokens. For short behaviour sentences this is exactly the behaviours most likely to be canned/short (backtracking, uncertainty). A per-behaviour direction contaminated by adjacency is a real threat to behaviour-specificity claims, and the clipped arm is a robustness check that (as far as the canonical data goes) is not the default. **Run the clipped arm and show the directions are stable.**

2. **Mean pooling cancels opposing directions — and the team knows it.** The headline geometry and steering directions are mean-pooled. The code itself argues `last` is more principled for a causal model. If `pooling_sweep.json` shows `cos(mean,last)` is low for any target behaviour, that behaviour's direction is a pooling artefact and any steering claim on it is fragile. This is the single sharpest internal critique and it is testable cheaply (the sweep is armed).

3. **Steering-layer / sweep-layer mismatch.** The pooling sweep is hard-pinned to L27, but the steering layer is unsettled and the leading candidates are mid-stack (Venhoff region). The pooling-robustness evidence the steering write-up will cite may therefore be at the wrong layer. Sweep the actual candidate layer.

4. **The keystone null still owes a clean re-run.** CF-13/14/16 were code-fixed but the load-bearing chain-stratified significance test on the *deduplicated, sidecar-provenanced* matrices is marked "re-run owed" in CONFOUNDS. The compression signal survives informally (~18 SD), but the formal p-value with B≥2239 and Holm–Bonferroni on the reported layers is the citable number and should be regenerated, not inherited from the 2026-06-08 superseded run.

5. **Null-pool scope (CF-16, standing design caveat).** Activations are extracted only for the four target behaviours; `deduction`/`initializing` (≈51% of sentences) are excluded. The permutation null therefore draws only from the four target labels — a narrower null than the full reasoning distribution. Extracting the non-target behaviours' activations would close this; until then "behaviour-specific" is relative to a 4-way contrast.

6. **fp16→fp32 and post-block read are assumptions, not validated.** The hook reads block *output*; for some architectures the "residual stream a steering hook writes to" is the block *input* (pre-LN) or a specific residual add point. For R1-Distill (standard Qwen2 decoder) block-output ≈ residual stream, so this is almost certainly fine — but the steering hook in Phase 7 must add at the *same* point this captured, or the direction is in a subtly different basis. Worth an explicit assertion that the steering write-point and the capture-point coincide.

7. **Replay fallback is still reachable for any legacy/foreign extraction.** `chain_ids_for` falls back to replay if no sidecar; `require_aligned` will hard-fail rather than proxy, which is correct, but it means analyses pointed at an old activation dir will *error* (good) — the operational risk is someone re-pointing at a pre-2026-06-12 directory and not re-extracting. Ensure the steering run consumes only sidecar-bearing matrices.
