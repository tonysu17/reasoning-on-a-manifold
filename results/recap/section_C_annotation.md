## C. Annotation — Behaviour Labelling & Multi-Annotator Robustness

This stage takes the raw reasoning chains generated in Phase 2 and decorates every
sentence with a *reasoning-behaviour* label. Those labels are the load-bearing
input to everything downstream: the activation extractor (Phase 4) slices the
residual stream at the token spans these labels mark, the PCA / geometry phases
build per-behaviour subspaces from those slices, and the steering vectors you are
about to spend real money on are difference-of-means over exactly these labelled
populations. If the annotation is noisy or biased, every geometric claim inherits
that noise. This section documents what the code actually does, defends the design
choices, and then attacks the reliability of the labels with the real agreement
numbers now on disk.

---

### C.1 What the stage produces

The pipeline entry point is `03_annotate_chains.py`; the engine is
`src/annotation.py`. For each chain it sends the full thinking text to an LLM
annotator and asks it to re-emit the text wrapped in delimiter tags, one tag per
behaviour span. The parsed output per chain is a list of
`{"label": str, "text": str}` dicts stored under `"annotations"`, plus a boolean
`"annotation_complete"`.

The taxonomy is the six-label Venhoff et al. (arXiv:2506.18167) scheme, copied
verbatim from that paper's Appendix A. From `src/annotation.py`:

```python
# src/annotation.py — module constants
VALID_LABELS = frozenset({
    "initializing",
    "deduction",
    "adding-knowledge",
    "example-testing",
    "uncertainty-estimation",
    "backtracking",
})

# The 4 behaviours we care about (distinct to thinking models).
TARGET_BEHAVIOURS = [
    "backtracking",
    "uncertainty-estimation",
    "example-testing",
    "adding-knowledge",
]
```

Two of the six labels (`deduction`, `initializing`) are generic step types that
also appear in non-thinking models; the four `TARGET_BEHAVIOURS` are the
"distinct to thinking models" behaviours the whole thesis is about. The
`TARGET_BEHAVIOURS` list is explicitly the *single source of truth* — its raw
hyphenated strings are interpolated directly into the downstream filenames
(`{behaviour}_manifold_k3.npy` etc.), so any rename here silently orphans the
steering-vector files.

The prompt is reproduced verbatim, single user message, **no system prompt**, to
match the paper's protocol exactly:

```python
# src/annotation.py — _PROMPT_TEMPLATE (verbatim Venhoff Appendix A)
_PROMPT_TEMPLATE = """\
Please split the following reasoning chain of an LLM into \
annotated parts using labels and the following format ["label\
"]...["end-section"]. A sentence should be split into multiple \
parts if it incorporates multiple behaviours indicated by the \
labels.
...
0. initializing -> The model is rephrasing the given task and \
states initial thoughts.
...
5. backtracking -> The model decides to change its approach.
...
Answer only with the annotated text. Only use the labels outlined \
above. If there is a tail that has no annotation leave it out.\
"""
```

**Why verbatim:** the design goal is replication of Venhoff's behaviour
fractions, which gives an external sanity check that the labelling is "in family"
with prior work. `03_annotate_chains.py` operationalises this — after a pilot run
it compares observed sentence fractions against `VENHOFF_FRACTIONS` (deduction
0.52, adding-knowledge 0.15, …) and PASSES the pilot only if the four target
fractions land within ±50% of the paper's Figure 2 values. This is a deliberately
loose gate (small-N pilot), tightened to ±10 percentage points for the full run.

---

### C.2 Which model annotates — AWS Bedrock proxy, not OpenAI

The single most important deviation from the source paper is the annotator
identity, and the code flags it loudly:

```python
# 03_annotate_chains.py — module docstring
# Note: Venhoff used GPT-4o; GPT-4o-2024-11-20 is unavailable on the AWS proxy
# used in this project. Claude Sonnet 4.5 is used instead (noted in methods).
```

The default annotator is hard-coded as a Bedrock model id:

```python
# src/annotation.py
ANNOTATION_MODEL = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
```

Transport is a thin POST to the lab's AWS API-Gateway proxy via
`CLAUDE_PROXY_URL` / `CLAUDE_PROXY_KEY`, `temperature=0.0`, `max_tokens=8192`.
Crucially `_extract_text` is provider-agnostic because the proxy returns
different response shapes per model family — Anthropic returns a list of content
blocks, while Qwen/Nova return a plain string:

```python
# src/annotation.py — _extract_text
content = payload.get("content")
if isinstance(content, list) and content and isinstance(content[0], dict):
    return content[0].get("text", "") or ""
if isinstance(content, str):
    return content
return ""
```

This polymorphism is what makes the **multi-annotator robustness** test possible
on the same code path: `--annotator-model` lets you swap Sonnet for
`qwen3-235b` or `nova-pro`, each writing to its own `--out` file. The three
annotated files are present on disk (`data/annotated_R1-1.5B.json`,
`…__qwen3-235b.json`, `…__nova-pro.json`, each ~60 MB), so R2.1 was actually run
on the full 1000 chains, not just designed.

**Defence of the choice:** GPT-4o genuinely is not reachable on the lab proxy, so
some substitution was forced. Using a *third* model family (Sonnet for the
primary pipeline; Qwen3 and Nova only as robustness probes) is the right
mitigation — it lets the thesis claim "the geometry survives changing the
annotator," which is a stronger external-validity statement than Venhoff made.

---

### C.3 Spans → token offsets: the occurrence-aware matcher

The annotator returns *text*, not character ranges. Mapping each span back to a
character offset in the original chain — and from there to token positions for
activation extraction — is done by the canonical helper `src/text_offsets.py`,
imported by Phase 4, the PCA scripts, `compare_annotators.py`, and `span_f1.py`
alike (single source of truth, deliberately numpy/torch-free).

The subtle, load-bearing detail is **occurrence-awareness**. R1 chains loop and
repeat sentences verbatim (especially truncated ones and stock backtracking
phrases like "Wait, that's wrong"). A naive `str.find` mapped every repeat to the
*first* occurrence, collapsing many annotations onto one token span and producing
"35–56% exact-duplicate activation rows (zero-distance neighbours in every
kNN-based estimator the geometry claims rest on)." The fix walks a forward cursor:

```python
# src/text_offsets.py — _find_from
idx = chain_text.find(needle, start)
if idx >= 0:
    return idx
if start > 0:
    idx = chain_text.find(needle)   # fall back to from-start (pre-fix behaviour)
    if idx >= 0:
        return idx
return None
```

`locate_annotation_offsets` calls this with a cursor advanced past each match, so
the i-th repeat binds to the i-th occurrence, with a 40-char-prefix fallback for
minor annotator truncation. This is logged as confound CF-13 in
`CONFOUNDS_AND_REMEDIATION.md` — it is one of the bugs that quarantined the
pre-fix results. **Both** the agreement metrics and the activation extraction use
this exact rule, so the annotator-agreement numbers below are computed on the
same offset semantics the geometry rests on.

---

### C.4 Chunking — why long chains are split

The AWS API Gateway has a ~29-second hard timeout. At ~80 tok/s output a single
annotation request maxes out around ~2300 tokens, so any chain over
`CHUNK_THRESHOLD_TOKENS = 1200` (estimated at 4 chars/token) is split on
paragraph boundaries with ~100-token overlap, annotated chunk-by-chunk, and
merged. Two non-obvious correctness details:

1. A **continuation prefix** is prepended to chunks 2+ because without it "Sonnet
   labels the first sentence of each continuation chunk as 'initializing' because
   it looks like a fresh response" — a seam artefact that would inflate the
   `initializing` count and corrupt span boundaries.
2. At merge time, leading spans of chunk N+1 whose text appears verbatim in the
   tail-overlap of chunk N are dropped, so the first chunk's labels win for
   overlapped sentences.

```python
# src/annotation.py — merge_chunk_annotations
for span in chunk_annotations[i]:
    # Drop span if its text appears verbatim in the overlap region
    if span["text"] in overlap_region:
        continue
    keep.append(span)
```

**Critique:** the dedup is a substring test on raw text, which can over-drop when
a short stock phrase (again, "Wait,") legitimately recurs across the seam, or
under-drop when the annotator lightly paraphrases the overlap. The token estimate
(`len(text)//4`) is crude; the real tokenizer is the R1 tokenizer, not a 4-char
heuristic, so threshold/overlap sizing is approximate. None of this is validated
against a held-out exact-tokenisation oracle.

---

### C.5 Parsing robustness and crash-safety

Parsing is a regex over the `["label"]…["end-section"]` delimiters, with
generous label normalisation (strips `"0. "`, `"1-"` prefixes, maps bare digits
to names). Unknown labels do **not** crash — they fall back to `deduction` with a
warning:

```python
# src/annotation.py — parse_annotation_response
if label not in VALID_LABELS:
    logger.warning(f"  Unknown label '{m.group(1).strip()}' → deduction")
    label = "deduction"
```

The batch loop checkpoints after every chain (`checkpoint_every=1` for the real
run), backs up the prior file before touching it, treats a chain as resumable
unless every chunk succeeded (`annotation_complete`), and swallows per-chain
exceptions so one bad record cannot abort a multi-thousand-chain run. This last
property is exactly what Phase 7 will rely on (annotate-with-checkpoint, resume on
credit exhaustion). `verify_annotation_completeness.py` is the post-run gate: it
fails on missing/duplicate task_ids, empty annotations, partial records, or any
label outside `VALID_LABELS`.

**Critique of the fallback:** routing unknown labels to `deduction` quietly
biases the majority class upward and hides annotator format drift. Because
`deduction` is the dominant label anyway (45–54% of spans), a few mis-parsed
target-behaviour spans demoted to `deduction` is a silent false-negative on
exactly the behaviours the thesis cares about. There is no counter logged for how
often this fires per annotator, so the rate is unaudited.

---

### C.6 Inter-annotator agreement — the real numbers (R2.1)

`compare_annotators.py` builds a per-character label code array for every chain
(0 = unlabelled "O"), accumulates pairwise confusion matrices over characters,
and computes Cohen's κ from them. `span_f1.py` complements this with span-level F1
(exact / IoU≥0.5 / any-overlap), which forgives boundary jitter. Both ran on
**1000 common chains**. The numbers actually on disk
(`results/robustness/cross_annotator_comparison.json`,
`results/robustness/span_f1.json`):

**Character-level Cohen's κ:**

| Pair | κ (6-label) | κ (target vs other) | agree (labeled) | agree (overall) |
|---|---|---|---|---|
| Sonnet vs Qwen3-235B | 0.436 | 0.352 | 0.408 | 0.579 |
| Sonnet vs Nova-Pro | 0.350 | 0.305 | 0.354 | 0.519 |
| Qwen3-235B vs Nova-Pro | 0.345 | 0.263 | 0.322 | 0.520 |

So the headline "κ = 0.35–0.44, fair-to-moderate" is exactly the 6-label diagonal.
On the **target-behaviour collapse** (the four behaviours-of-interest vs.
everything else) κ is *worse*, 0.26–0.35 — i.e. annotators agree slightly less on
precisely the labels the thesis hangs on.

**Span-level F1** (more forgiving; only "labelled span" populations, ~78k–89k
spans each):

| Pair | exact boundary | IoU ≥ 0.5 | any overlap |
|---|---|---|---|
| Sonnet vs Qwen3 | 0.211 | 0.306 | 0.413 |
| Sonnet vs Nova | 0.171 | 0.257 | 0.375 |
| Qwen3 vs Nova | 0.247 | 0.307 | 0.373 |

Per-label F1 at IoU≥0.5 for the targets is low across the board — e.g.
adding-knowledge 0.15–0.21, example-testing 0.17–0.25, backtracking 0.23–0.30.
The label distributions also diverge materially: Sonnet calls 21.6% of spans
`uncertainty-estimation` while Nova calls only 11.5% and Qwen3 7.0%; Sonnet labels
`example-testing` at 7.5% vs Nova's 3.9%. So the three annotators do not even
agree on *base rates*, let alone boundaries.

**Defence:** even the "any overlap" F1 (≤0.41) and these κ values are *in the
range Venhoff-style sentence-labelling tends to produce* — sentence-segmentation
plus a fuzzy 6-way semantic taxonomy is genuinely hard, and character-level κ is
an unusually harsh denominator because a one-token boundary slip counts as
disagreement at every off-by-one character (this is precisely why `span_f1.py`
exists). The thesis does not claim the labels are *correct*; it claims the
*geometry replicates* across annotators despite the disagreement — which is the
stronger and more defensible position.

---

### C.7 The manifold-replication tie-in is partly broken on disk

`compare_annotators.py` is supposed to also emit a `manifold_replication` block
(per-annotator intrinsic dim, geodesic curvature, PR-trough layer) so the thesis
can say "cdim/geo are similar across annotators despite differing label
distributions." But in the file currently on disk that block is **empty**:

```
"manifold_replication": { "Sonnet-4.5": {}, "Qwen3-235B": {}, "Nova-Pro": {} }
```

The code path keys the robustness JSONs by short name
(`results/robustness/{short}/geometry_robustness.json`) and only fills the entry
`if rp.exists()`. The empty dicts mean either those per-annotator robustness JSONs
were not present when this comparison last ran, or the schema (`keystone_cdim`,
etc.) did not match. **This matters:** the 3-way *geometry* replication that
MEMORY records as "R2.2 CLOSED, folded into thesis (κ=0.35–0.44)" is the citable
result — but the artefact that is supposed to carry the per-annotator geometry
numbers in *this* comparison file is blank, so the replication evidence lives in
some other artefact (the per-annotator `geometry_robustness.json` / thesis
`tab:geometry-replication`), not here. Anyone re-deriving the claim from
`cross_annotator_comparison.json` alone would find nothing. The
`run_multiannotator_pipeline.sh` orchestrator does run Phases 4/5/5b/5c/5d +
robustness per annotator before calling `compare_annotators.py`, so the intended
fill is there — but the committed JSON shows it did not populate, which is a
reproducibility gap worth closing before citing.

---

### C.8 Circularity, the single-annotator null, and what it limits

**The circularity concern.** The model under study is
`DeepSeek-R1-Distill-Qwen-1.5B`. The Qwen3-235B annotator is *the same model
family* as the base model's distillation teacher lineage. If an LLM annotator
shares inductive biases with the model whose activations are being labelled, the
behaviour boundaries it draws may track that family's own surface cues rather than
a model-independent notion of "backtracking." The mitigation in this repo is
breadth-of-family: Sonnet (Anthropic) is the primary annotator and is *not* in the
Qwen lineage, and Nova (Amazon) is a third independent family. The fact that the
geometry survives all three is the real de-circularising argument. MEMORY also
notes a future plan to use `--annotator-model` for a Qwen3 de-circularisation arm
whose id "is not yet located." So the circularity is *acknowledged and partially*
addressed, not closed.

**The single-annotator specificity null.** This is the sharper limitation. The
behaviour-*specificity* result (the claim that each behaviour has its own
subspace, and that add-knowledge in particular fails specificity) is, per MEMORY
and `CONFOUNDS_AND_REMEDIATION.md`, **single-annotator** — it rests on the Sonnet
labels only and has *not* been re-run across Qwen3/Nova. So the only result that
replicated 3-way is the *geometry* (low-dim subspace + curvature), not
*specificity*. A specificity claim built on labels with κ≈0.4 and per-label
target F1 of 0.15–0.30 is fragile: a behaviour whose boundary the annotators agree
on only ~25% of the time at IoU≥0.5 cannot strongly support "this behaviour
occupies a distinct, separable subspace" unless the subspace is robust to exactly
the boundary jitter the annotators exhibit. `adding-knowledge` is the worst case
on both axes (lowest per-label F1 *and* the behaviour MEMORY records as failing
specificity everywhere) — those two facts are plausibly the same fact: the label
is so unreliable that no clean subspace can be recovered.

---

### C.9 How κ≈0.4 constrains the steering decision you are about to make

This stage feeds the steering experiment directly, and the agreement numbers
should temper two specific choices:

- **Population purity.** Steering vectors are difference-of-means over the
  labelled token populations for each behaviour. With three annotators disagreeing
  on ~60% of character labels and on base rates by 2–3×, the Sonnet-only
  populations that define your steering vectors contain a substantial fraction of
  spans that another competent annotator would have labelled differently. The
  steering vector is therefore a vector toward "Sonnet's notion of backtracking,"
  not "backtracking." For the four-target behaviours specifically, the
  target-vs-other κ (0.26–0.35) is the relevant, and worse, number.

- **adding-knowledge is the riskiest arm.** Lowest inter-annotator F1, failed
  specificity, and the behaviour most likely to be contaminated by the
  unknown-label→deduction fallback. If budget forces trimming arms, this is the
  one whose null result is least interpretable — a flat steering response could be
  "no causal subspace" or "the labels were too noisy to build a clean vector."
  Energy-matched and label-shuffled control arms are essential here, not optional.

- **Layer choice is *not* set by this stage.** Annotation produces labels, not
  layers. The PR-trough layer per annotator (the geometry-derived layer candidate)
  is meant to come through the `manifold_replication` block — which is empty on
  disk (C.7) — so the cross-annotator file gives you *no* layer guidance right
  now. Layer selection must come from the Phase 5/5b/triangulation artefacts and
  the Venhoff mid-layer recipe (≈L11/14/16–19) plus L27, per the steering memo —
  not from anything in this annotation stage. The one thing this stage *does* tell
  you about layers is a caution: if you pick a layer by maximising behaviour
  separability on Sonnet labels, you are partially fitting to one annotator's
  idiosyncrasies; prefer a layer that the *3-way* geometry replication agrees on.

---

### C.10 Run / citable status summary

- **RUN:** Sonnet, Qwen3-235B, Nova-Pro annotations all exist for the full 1000
  R1-1.5B chains. Agreement (κ) and span-F1 ran on 1000 common chains; results in
  `results/robustness/cross_annotator_comparison.json` and `span_f1.json`.
- **CITABLE:** the geometry-replicates-across-annotators claim (R2.2) is folded
  into the thesis (`tab:geometry-replication`) and treated as closed.
- **NOT CITABLE / single-annotator:** behaviour *specificity* (Sonnet only;
  add-knowledge fails). κ≈0.4 caps how strongly any per-behaviour purity claim can
  be made.
- **REPRODUCIBILITY GAP:** the `manifold_replication` block in the committed
  comparison JSON is empty; the per-annotator geometry numbers must be read from
  elsewhere, and this file should be regenerated before being cited as the source
  of the replication evidence.
- **DEVIATION FROM SOURCE:** annotator is Sonnet 4.5 (Bedrock), not Venhoff's
  GPT-4o; must remain noted in methods.
