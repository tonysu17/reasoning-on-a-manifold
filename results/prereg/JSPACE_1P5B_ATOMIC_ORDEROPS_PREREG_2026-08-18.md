# PROTOCOL — 1.5B atomic-hop audit and order-of-operations J-Lens test

**Date:** 2026-08-18. **Status:** sealed automatically when the complete bundle commit described below
exists; approved by Tony in-session ("yes go ahead"). The commit that first contains this protocol, its
machine-readable input lock, the frozen atomic task file, and the tested runner is the seal. No
model-dependent output from either experiment may be generated or inspected before that commit exists.
Any change after the seal requires a dated amendment committed before the affected computation is rerun.

**Empirical evidence status:** prospective/unrun; exploratory follow-up. **Protocol marker:** amended
relative to Phase 1. **Thesis disposition if used:** retained–bounded. These experiments do not
reclassify the failed J-space Phase-1 validity gate and are not confirmatory thesis evidence. They may
refine the local explanation of the existing Qwen2.5-Math-1.5B / R1-Distill-Qwen-1.5B contrast. Thesis
use requires a validated result bundle, a derivation manifest, evidence-snapshot integration, and
hedge-preserving prose review.

At the pre-seal audit, no local order-of-operations model outputs were found. The official task
definitions were inspected only to freeze inputs, labels, and scoring. Experiment B runs regardless of
Experiment A; there are no interim efficacy or futility looks.

## 1. Questions

### A0 — atomic-hop behavioural competence audit

Does the checkpoint contrast on the existing 81 eligible multihop items primarily track unequal
knowledge of the two constituent hops, or does a residual contrast remain when the same items are asked
as composed questions under the same runtime and answer format?

This is a behavioural decomposition. It does not identify a hidden representation and its operational
composition contrast is not a mediation estimand.

### O1 — synthetic-arithmetic order-of-operations readout

Does the Math-base versus R1-distill J-Lens contrast recur on a fact-retrieval-light task where the
scored intermediate is an exactly specified arithmetic partial result rather than a retrieved factual
bridge? The registered directional expectation is lower middle-band readout in the distill checkpoint,
but all primary tests are two-sided.

The numeric partial result is the primary reasoning label. The pending operation is an official-suite
secondary because it is often lexically present in the prompt.

## 2. Fixed inputs

The companion JSON input lock is authoritative for full paths, hashes, and synonym lists.

| Role | Identity | Pin |
|---|---|---|
| Math-base model | `Qwen/Qwen2.5-Math-1.5B` | revision `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`; BF16 |
| R1-distill model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`; BF16 |
| Math-base lens | `data/jlens_local/qwen2.5-math-1.5b_wikitext100.pt` | SHA256 `ce4f034dd4eabc8d814913b4ed476aa1f9ea3f44e27ed92a408200aa7a1417c8`; 100 raw WikiText prompts; source layers 0–26 |
| R1-distill lens | `data/jlens_local/r1-distill-qwen-1.5b_wikitext100_merged.fp32.pt` | SHA256 `6b5f1043b3c3fa3fcd8d4d69919772a2d763c145f996477bf715fe4b9b89f672`; validated Phase-1 2×50 merged FP32 lens; source layers 0–26 |
| J-Lens code/data | `anthropics/jacobian-lens` | commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` |
| Order-ops suite | `data/evaluations/lens-eval-order-ops.json` at that commit | raw SHA256 `b203206d16ff628152cc86f3838604e06cb54776f3e14fa1c34f150db8bc7560`; 55 unique items; two intermediates/item |
| Evaluation README | `data/evaluations/README.md` at that commit | raw SHA256 `e061d9cce02a1cc651d58a81927833b760d3cef65bf4995126ecbe372a0ebe07` |
| Atomic source population | eligible multihop items in `results/prereg/jspace_r1_eval_eligibility_manifest.json` | file SHA256 `a193ce15ff18d1852a870703dba104763a01b06740c99ec8fd955e3a15927f7c`; exactly 81 items |
| Multihop source file | `lens-eval-multihop.json` at the pinned J-Lens commit | raw SHA256 `50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416` |
| Positive control | pinned `lens-eval-typo.json`, same common-eligible 96-item population | raw SHA256 `9d05e16b7234a57d0773d120a4e1c4e94fd3bc2235a8125d4200a70e60ab17aa` |

The two lens fits are not fully matched measurement replications: the distill lens was fitted as two
independent 50-prompt cells and merged after a held-out stability gate, whereas the base lens was one
100-prompt local MPS fit without the same split-half gate; their runtime/software environments also
differed. This uncertainty is not propagated by O1. A positive pair contrast remains checkpoint-local
and must be replicated with matched lens fits before any stronger cross-checkpoint or thesis claim.
Fit-compatible rendering is also asymmetric: the Math-base tokenizer/lens path has no BOS token, while
the distill fit used BOS ID 151646. O1 preserves those fit-time conventions rather than introducing a
new base-model BOS; this avoids an unvalidated lens-input change but further limits exact pair symmetry.

Before the first model forward in either experiment, one global sealed preflight must materialize and
verify the **complete snapshots of both checkpoints**. For each checkpoint this means the pinned
`config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, and complete
`model.safetensors`, each matching the SHA256 in the input lock. Required model, tokenizer, lens, task,
code, and requirements paths must resolve to regular files and must not themselves be symbolic links.
The model cache's resolved real path must be strictly outside the dedicated clean sealed checkout; a
cache path equal to, above, or below that checkout is invalid. The same file checks are repeated
immediately before each tokenizer or model load. Thus a per-cell download or an unverified second
tokenizer instance cannot silently redefine eligibility after computation has begun.

The campaign preflight also runs the sealed pure-code test selection. Its persisted receipt contains only
stable fields: the sealed test-file hash, fixed test selection, collected count, passed count, and pass
status. Timing, temporary paths, progress text, and raw `pytest` stdout/stderr are not receipt fields.
The stable receipt, complete-snapshot inventory, and execution fingerprint defined in Section 5 are
byte-compared on the sole permitted resume.

## 3. Experiment A0 — frozen atomic task construction

The frozen task file contains exactly one row per source item, in the sealed eligibility-manifest order:

- `hop1_question`: asks only the clue→bridge fact; it contains neither the target answer nor the
  bridge→target relation.
- `hop2_question`: states the bridge and asks only the bridge→target fact; it contains no original clue.
- `composite_question`: asks the source-intended two-hop question and contains neither the bridge answer
  nor the target answer.
- presealed exact aliases for the bridge and target.
- `fact_status` and `review_note`: a task-only pre-seal adjudication of stable versus ambiguous,
  disputed, or time-sensitive source facts.

Every row is manually authored from the source item, then checked before the seal for source-name/order
agreement, answer leakage, and whether each question isolates the intended relation. Where the source
relation is non-unique, anachronistic, or skips an annotated hop, pre-seal wording may disambiguate the
source-intended canonical path or explicitly supply the skipped mapping. Every such repair is recorded
in `fact_status`, `review_note`, and `review_flags`, and is excluded from the stable-only sensitivity.
No repair is permitted after model output is seen.

The 81-item source-compatible analysis is primary. A predeclared sensitivity excludes every row marked
non-stable before the seal (including the source’s now-obsolete “China is the most populous country”
items). That sensitivity cannot replace or rescue the primary result; it quantifies benchmark-fact
dependence. Report its paired effect estimates and 95% bootstrap intervals, but no additional p-values
or branch labels.

One wrapper is applied verbatim to all three question fields:

```text
Question: {question}
Give only the shortest correct answer; do not explain.
Answer:
```

Both checkpoints use raw text, not a chat template; greedy decoding (`do_sample=False`), no sampling
temperature or repetition penalty, `max_new_tokens=16`, and the same EOS/newline stopping rule. For A0
and every O1 behavioural or teacher-forced arm, tokenize with `add_special_tokens=False` and prepend no
manual BOS to either checkpoint. This gives the checkpoints identical ordinary raw-text prefix IDs;
the distill-only fit-compatible BOS is used only for J-Lens/logit-lens readout. Both tokenizers use EOS
ID 151643 and pad ID 151643. Generation is item-wise; the stored continuation excludes prompt IDs and a
terminal EOS ID. For A0 and Direct, check the decoded continuation after every generated token, stop at
the first LF (`\n`), and score only text preceding it. Model dtype is BF16. Seed `20260818` is recorded
even though greedy generation should be deterministic.

The same-runtime composed arm is required. A descriptive continuity arm also completes the original
unwrapped source prompt with greedy decoding and 16 new tokens; it is not part of the A0 primary
estimands.

### A0 scoring

Only the generated continuation is scored. Primary normalization is Unicode NFKC, first generated line,
whitespace collapse, removal of enclosing quotes and terminal punctuation, then case-folding. A primary
hit is exact equality to one of the presealed aliases. Bounded whole-label and substring matches are
reported as sensitivities and cannot change the primary verdict. Refusal, no match, or token-budget
exhaustion is incorrect, not missing.

For item `i` and checkpoint `m`:

- `H1_im`: hop-1 answer is correct.
- `H2_im`: hop-2 answer is correct.
- `M_im`: composed answer is correct.
- `K_im = H1_im × H2_im`: both atomic hops are correct.

Let `B` denote Math-base and `D` denote R1-distill. The two co-primary item-paired estimands, oriented so
positive values indicate a larger distill deficit, are:

```text
ΔK = mean_i(K_iB - K_iD)
ΔX = mean_i[(K_iD - M_iD) - (K_iB - M_iB)]
```

`ΔX` is the **operational excess-composition contrast**. It is not causal mediation and can also reflect
different prompt-format sensitivity between atomic and composed questions.

Supportive estimands are the paired hop-specific risk differences, the composed risk difference, each
checkpoint's full 2×2 atomic-state table (`both`, `hop1 only`, `hop2 only`, `neither`), and
`P(M=1 | K_B=K_D=1)` by checkpoint. The common-known conditional analysis is shown only when at least 20
items satisfy `K_B=K_D=1`; it never replaces the full-population estimands. For every 2×2 state cell,
report the checkpoint-specific proportion and the paired distill-minus-base proportion difference, each
with a percentile 95% interval from the same 20,000 paired item bootstrap. When the common-known gate is
met, report each checkpoint's composed-success rate and their paired risk difference, again with 95%
intervals. Each bootstrap draw resamples the full 81-item index vector `I_1,...,I_81` first and then
recomputes both the state cells and the draw-position mask
`S*={t:K_(I_t)B=K_(I_t)D=1}`; it must retain duplicate draws and must not reuse the observed common-known
membership. If any registered draw has an empty `S*`, withhold the common-known interval and report that
failure rather than imputing, deleting, or redrawing it.

### A0 inference and branch language

- Swap the complete `(H1,H2,M)` checkpoint outcome vectors within item, using 99,999 draws from
  NumPy `PCG64`, seed `20260818`; report +1-corrected two-sided p-values.
- Use 20,000 paired item bootstraps, seed `20260819`, for percentile 95% confidence intervals; recompute
  every derived subset and estimand inside each draw.
- Holm-adjust the two co-primary p-values.
- The practical-equivalence margin is ±0.10 (approximately eight of 81 items). Equivalence language is
  licensed only when a 90% paired-bootstrap interval lies wholly inside this range.

These permutation and bootstrap summaries resample this fixed prompt set. Shared construction families
and repeated answers preclude treating them as population uncertainty for factual reasoning generally.

Let a **material factual deficit** mean that the 95% interval for `ΔK` lies wholly above `+0.10`, and a
**material excess-composition deficit** mean that the 95% interval for `ΔX` lies wholly above `+0.10`.
Branch descriptions are deliberately conservative:

- **factual-dominant pattern:** material factual deficit, while the 90% interval for `ΔX` lies inside
  `[−0.10,+0.10]`;
- **composition-dominant pattern:** material excess-composition deficit, while the 90% interval for
  `ΔK` lies inside `[−0.10,+0.10]`;
- **mixed pattern:** both material-deficit criteria hold;
- **indeterminate:** everything else, including nonsignificance without equivalence.

These are patterns in this prompt set, not claims that a checkpoint globally knows or cannot compose
facts.

## 4. Experiment O1 — order-of-operations task and labels

The pinned upstream suite contains 55 prompts. Subject to the explicit annotation exception below,
`intermediates[0]` is the numeric partial result and `intermediates[1]` is the operation pending after
that result. `target` locates the completion and is not a primary scored intermediate. The readout
position is the joint-tokenization token immediately before the first target token, as specified below;
it is not assumed to equal the prompt-alone final token.

### Frozen synonym expansion

All variants receive exactly one leading ASCII space before tokenization. A variant is eligible only if
`add_special_tokens=False` produces exactly one non-special token whose ID lies in the registered scored
vocabulary. Variants are deduplicated by token ID within label family.

- numeric result: decimal digits and the lowercase English cardinal form;
- addition: `+`, `plus`, `add`, `addition`;
- subtraction: `-`, `minus`, `subtract`, `subtraction`;
- multiplication: `*`, `×`, `times`, `multiply`, `multiplication`;
- division: `/`, `÷`, `divide`, `division`;
- modulo: `%`, `mod`, `modulo`, `remainder`;
- squared: `^2`, `**2`, `square`, `squared`.

The English-cardinal function is deterministic for non-negative integers 0–999, omits “and,” and uses
lowercase tens hyphenation (for example, `one hundred twenty-three`). No variants may be added after
tokenization or after scores are seen.

Primary eligibility is strict, paired, and frozen before a model forward pass: the registered joint
prompt/target boundary locator must succeed and at least one numeric variant must be an eligible token in
**both** checkpoint tokenizers.
The common set of variant strings is used, mapped to checkpoint-specific token IDs. Behavioural
correctness never filters the primary population. A non-finite or absent model score is a technical
failure subject to Section 5, not an eligibility exclusion.

Eligibility and boundary records are computed only from the two locally materialized tokenizer files
that passed the complete-snapshot verification above, using `local_files_only` semantics. Before the
population is frozen, the ordinary (no-special-token) causal prefix ID sequence and boundary offsets
must be identical between the two registered tokenizers for every official prompt/target pair. A
difference is a pre-forward stop, not a checkpoint-specific eligibility rule. Checkpoint-specific
fit-compatible BOS handling occurs only after this identity check and is recorded separately.

The pre-seal task-only audit found that the partial result is already present as a standalone operand on
three items: `nested-sub-add-mult` (`5`), `add-add-add` (`3`), and `square-mult` (`4`). These are excluded
from the 52-item task-level non-lexically-present numeric-intermediate candidate set and retained in an explicitly
labelled common-token-eligible official-prompt sensitivity. The pending-operation family has an obvious synonym
literally present in all 55 prompts;
it is therefore a context/surface positive control, not clean evidence of latent computation.

A further result-blind arithmetic-tree audit found one upstream annotation mismatch. For
`mult-div-mult` (`2 * 6 / 4 * 3 =`, target `9`), the numeric label `12` is a real partial result but its
next operation is division, whereas the supplied `multiplication` label belongs after partial result
`3`. The pinned upstream bytes are not changed. Ranks may still be extracted for provenance, but this
row is excluded from every O1 readout estimand, null, numeric sensitivity, operation control, and paired
two-intermediate summary; the separate target-answer behavioural audit still covers all 55 source
prompts. This leaves 51 annotation-valid, non-lexically-present items before token
eligibility and 53 annotation-valid items for official-prompt sensitivities after the tokenizer gate.

A result-blind tokenizer preflight then found that neither registered numeric form for
`mult-div-left` (`24`, `twenty-four`) is a single token in both tokenizers. It is excluded by the fixed
eligibility rule. Together with the annotation exclusion above, this fixes the co-primary and label-null
population at 50 items before any model forward. Rank extraction still retains all 54 token-eligible rows so that the
exclusion remains auditable; the annotation-invalid row cannot enter an analysis denominator.

### O1 readout

- No chat template; maximum sequence length 128. Tokenize the exact concatenation `prompt + target`
  without special tokens and retain offset mappings. The first token whose source span overlaps the
  target-character interval is the first target token; the readout is its immediate predecessor. Pass
  only the already-tokenized ID prefix through that predecessor to the causal model. Match each lens's
  fit-time rendering: no BOS for Math-base (its tokenizer has no BOS ID), and prepend distill BOS ID
  151646. Do not re-tokenize the truncated text and do not pass target token IDs to the model.
- The boundary record stores the joint offsets, joint IDs, first-target index, BOS-adjusted readout
  index, and causal-prefix IDs for both tokenizers. Six word-target items are known to differ from the
  prompt-alone-final-token shortcut; they remain in scope.
- Source layers 0–26; final target block/layer 27 is not a source layer.
- Model forward in BF16; residuals and Jacobian transport in FP32. The primary unembedding is the D5
  path: deep-copy the final norm and LM head, cast both to FP32, and apply them to FP32 transported or
  untransported residuals. The ordinary BF16 final-norm/head top-25 path is retained as a precision
  sensitivity. This is an explicitly amended scorer, not byte-identical Phase-1 scoring.
- Ranked vocabulary is the contiguous tokenizer-ID prefix of the LM head, exactly as in the validated
  Phase-1 scorer; padded head rows are excluded, but special token IDs remain rank competitors. Special
  IDs are excluded only from eligibility as scored label variants.
- Deterministic one-indexed rank orders descending logit, with ascending token ID breaking equal-logit
  ties. A synonym-family rank is the minimum eligible-variant rank.
- Both J-Lens and untransported per-layer logit-lens ranks are retained, together with top-25 IDs and
  boundary-tie counts.

With 27 source layers, normalized depth is `u_j=(j+0.5)/27`. Frozen bands are:

| Band | Rule | Layers |
|---|---|---|
| early | `u ≤ 1/3` | 0–8 |
| middle | `1/3 < u ≤ 2/3` | 9–17 |
| late | `u > 2/3` | 18–26 |

Bands never move after scoring. Exact peak-layer plots are descriptive.

### O1 primary estimands

For method `q ∈ {J,L}` (J-Lens, logit lens), checkpoint `m`, item `i`, and the numeric partial-result
rank `r^q_imj`, define the middle-band layer persistence@25 score:

```text
A^q_im = (1/9) × Σ_{j=9..17} 1[r^q_imj ≤ 25]
```

The two co-primary paired estimands, oriented so negative values match the registered expectation of
lower distill readout, are:

```text
ΔJ  = mean_i(A^J_iD - A^J_iB)
DiD = mean_i[(A^J_iD - A^L_iD) - (A^J_iB - A^L_iB)]
```

`ΔJ` asks whether the checkpoint contrast recurs on a synthetic arithmetic intermediate. `DiD` asks
whether that contrast is specific to Jacobian transport rather than ordinary token prediction. Neither
estimand is a causal effect of distillation.

For both co-primary estimands, swap the complete checkpoint readout arrays within item, synchronously
across methods and layers; 99,999 `PCG64` draws, seed `20260820`; +1-corrected two-sided p-values. Use
20,000 paired item bootstraps, seed `20260821`, for percentile 95% intervals and Holm-adjust the two
co-primary p-values. The practical-equivalence margin is ±0.10; equivalence requires the corresponding
90% interval to lie wholly inside the margin.

These permutation and bootstrap intervals are prompt-resampling diagnostics for this fixed, templated
suite. Repeated labels and construction families mean they are not sampling uncertainty for a broad
population of reasoning tasks.

For branch language, a **material O1 J recurrence** requires the assay-presence gate below to pass and
the 95% interval for `ΔJ` to lie wholly below `−0.10`. It is **transport-specific** only if the 95%
interval for `DiD` also lies wholly below `−0.10`. **O1 practical equivalence** requires the assay gate
to pass and both co-primary 90% intervals to lie wholly within `[−0.10,+0.10]`. All other pair patterns
are indeterminate; a nonsignificant contrast is not equivalence. If material recurrence qualifies but
the `DiD` criterion does not, the registered label is
`material-recurrence-transport-specificity-not-established`; it does not establish that the recurrence
is transport-nonspecific.

### O1 assay-presence gate and label null

Before interpreting a checkpoint contrast as a readout contrast, the numeric-intermediate assay must
show signal above a synchronized label-assignment null. Its population is the fixed, fully complete
50-item annotation-valid, non-lexically-present primary used for `ΔJ` and `DiD`. Retain ranks on every
prompt for every frozen numeric label family represented in that population. Permute those whole families among all 50
items, preserving duplicates, and use the same item assignment for both checkpoints, methods, and all
layers. Any unresolved technical failure stops the stage rather than changing this population. For draw
`s`, recompute
`G_s = mean_i[(A^J_iB,s + A^J_iD,s)/2]` from the assigned-label ranks. Use 99,999 draws, seed
`20260822`. Secondary band-union nulls separately recompute their union statistic from the same assigned
ranks; they are not part of the primary gate.

The observed gate statistic is
`G_obs = mean_i[(A^J_iB + A^J_iD)/2]`. It must exceed the `G_s` null 95th percentile with
the percentile computed using NumPy `method="higher"`. The comparison is strict (`G_obs > q95`), and
the +1-corrected upper p-value is `(1 + count[G_s >= G_obs]) / 100000`; it must be ≤0.05. Failure does
not erase `ΔJ`; it changes its status to
**assay-inconclusive** and forbids a representation interpretation.

### O1 secondary and control outputs

All of the following are secondary and cannot rescue a failed middle-band assay:

- project band-union pass@25 on the 53 annotation-valid, common-token-eligible official prompts: per item, any eligible
  numeric synonym reaches rank≤25 at any layer in the band; also report an N=55 accounting table with
  `mult-div-left` marked not scoreable under the frozen single-token rule and `mult-div-mult` marked
  excluded for mismatched paired annotation;
- project two-intermediate score on those same 53 prompts, averaging the numeric and operation-family
  hits per item; never impute either excluded row;
- early, late, and all-layer pass@25 and layer persistence@25;
- band-best rank median/IQR and band mean reciprocal rank;
- operation-intermediate profile;
- J-minus-logit contrasts for every band.

No inferential “late-concentrated” claim is registered. The report may state that an observed maximum
occurred in the late band and provide its paired late-minus-middle interval, but cannot turn that maximum
into evidence of absence from the middle band.

### O1 behavioural competence

Behaviour is kept separate from readout and never filters the primary analysis.

- **Direct:** the exact raw upstream prompt, greedy, 16 new tokens.
- **CoT:** raw wrapper `Solve this arithmetic problem step by step and end with the final answer:
  "{prompt}"`, greedy, 128 new tokens.
- Direct generation stops at EOS, the first generated newline, or the 16-token cap. CoT stops at EOS or
  the 128-token cap; newline is not a CoT stop.
- Target aliases are frozen mechanically: a decimal or English-cardinal numeric target receives its
  decimal and canonical-cardinal forms; a nonnumeric target receives only its case-folded source form.
- Direct-arm primary matching uses the same first-line exact normalization as A0 against a frozen target
  synonym family. For CoT, normalize CRLF to LF, take the final non-empty generated line, apply Unicode
  NFKC, collapse whitespace, trim enclosing quotes and terminal punctuation, and case-fold. The primary
  is a match to any equally normalized alias under Unicode non-word boundaries
  (`(?<!\w)alias(?!\w)` with the alias regex-escaped). Apply the same matcher to the entire normalized
  continuation as a sensitivity. Teacher-forced target log-probability tokenizes the exact raw
  `prompt+target` with `add_special_tokens=False` and no manual BOS, identifies every token whose offset
  overlaps the target character interval, and reports both the sum and mean log-probability over those
  tokens; it is descriptive support.

Report full-population rates and a descriptive common-correct subset. If either checkpoint is below 0.80
on the CoT primary endpoint, claims about a reasoning-competent checkpoint are suspended; the readout
result remains instrument/task-bounded.

### Same-runtime typo control

Re-run the full 96-item common-eligible typo suite for both checkpoints through the same readout runtime:
tokenize the exact raw typo prompt with `add_special_tokens=False`, prepend only the checkpoint-specific
fit-compatible BOS used by its lens (none for Math-base, 151646 for distill), read at the final prompt
token, and use the same source layers, vocabulary domain, exact ranks, and D5-style FP32 readout as O1.
All 96 items must be paired technically complete after the single permitted byte-identical retry. For
checkpoint `m`, let `U_m` be the mean over items of the indicator that any source layer places any token
in that item's frozen typo-label family in the J-Lens top 25. Retain ranks for every frozen whole typo
family on every prompt. Permute the 96 families among prompts, preserving duplicates and using the same
assignment for both checkpoints and all layers; recompute `U_m,s` separately by checkpoint. Use 99,999
NumPy `PCG64` draws, seed `20260823`, the `method="higher"` 95th percentile, and the +1-corrected upper
p-value `(1 + count[U_m,s >= U_m]) / 100000`. The integrity rule requires `U_m ≥ 0.75`, `U_m` strictly
above its null 95th percentile, and upper p≤0.05 for each checkpoint. Failure marks O1
**instrument-invalid in this runtime**; it is not a negative order-operations result. The prior typo
results are context, not substitutes for this control.
Because the same checkpoints, lenses, and prompts were inspected previously, this is a
repeatability/runtime-integrity check, not independent instrument validation.

## 5. Completeness, failures, and stopping

- A0 requires all 81 items to have H1, H2, and M continuations and deterministic primary scores for both
  checkpoints before any A0 estimand is calculated. The continuity arm must also complete for its
  descriptive report but never enters the primary estimands.
- O1 requires rank extraction for all 54 common-token-eligible items in both checkpoints, including
  every registered donor-family rank at every source layer. All analysis summaries exclude the one
  annotation-invalid row, and the primary uses its frozen 50-item annotation-valid,
  non-lexically-present subset. Direct, CoT, and teacher-forced behaviour must complete for all 55
  official prompts. The fixed single-token ineligibility of `mult-div-left` remains an N/A readout row,
  not a failure.
- The typo repeatability control requires all 96 items in both checkpoints. There is no complete-case
  deletion, denominator adaptation, or missing-outcome imputation in A0, O1, or the typo control.
- Refusal, wrong text, no match, or generation truncation at the token cap is a completed but incorrect
  behavioural outcome, not technical missingness.
- A technical failure may be resumed once with byte-identical inputs and configuration. The original
  exception and the resume event are retained in the retry log. Resume is permitted only when the
  incomplete run contains a valid, atomically written `stage_exception` record binding the failed stage,
  seal commit, configuration digest, completed-prefix digest, execution fingerprint, exception type,
  traceback hash, and timestamp. An ordinary interruption, an orphan partial, or an incomplete directory
  without that record is not resumable. The exception record is consumed by the one resume event; if the
  resumed run remains incomplete, the affected stage stops without a scientific report and no second
  resume is permitted.
- Stop before any model forward on a model revision, lens hash, task hash, tokenizer identity,
  layer-count, vocabulary-domain, seal-commit, or pure-code-test mismatch. Non-finite logits stop the
  affected extraction stage. A failed same-runtime typo control stops O1 before checkpoint inference,
  bootstrapping, checkpoint permutations, behavioural interpretation, or pair-pattern assignment; the
  retained raw extraction is labelled instrument-invalid rather than a negative order-operations result.
- No effect-driven refit, item deletion, synonym addition, layer-band shift, prompt rewrite, or rerun.
- An engineering repair or endpoint change requires a dated amendment committed before rerunning every
  affected arm.

At the campaign preflight, freeze one execution fingerprint for the whole A0/O1 run: accelerator
backend (`cuda` or `mps`; CPU execution is not authorised), physical device identity and index,
platform, PyTorch and accelerator-runtime versions, accelerator capability where available, requested
and observed model dtype, and whether the registered BF16 execution path is supported. CUDA must pass
PyTorch's `is_bf16_supported()` gate; MPS must be built and available on macOS 14 or newer. Both must
also pass a deterministic BF16 allocation/arithmetic probe whose stable result digest is retained. This
is an operational execution-support gate, not a claim about a hardware-native arithmetic unit. A
single campaign authority is written before either A0 or O1 may forward; both stages, every partial,
and the sole resume must match it byte-for-byte. The authority's canonical UUID and resolved path are
bound inside both its manifest and ready marker, and an immutable parent-level
`CAMPAIGN_SELECTED.json` pointer selects the sole authority permitted under this sealed output root.
Creating a second sibling authority or relocating/copying the selected directory is a pre-forward
failure. The selection pointer and every selected-authority file are direct derivation inputs. Cells or
stages from different hardware or precision paths cannot be combined.

Each model-bearing arm is likewise single-shot within that selected campaign. Its first fresh
invocation creates one UUID run directory and atomically writes a sibling `STAGE_SELECTED.json` that
binds the stage, canonical run ID and path, seal, preflight, and campaign selection. The stage parent
must then contain exactly that pointer and that one run directory. A second fresh A0 or O1 invocation
fails; later access is limited to verification of the selected completed run, deterministic O1
analysis, or the sole technical resume of that exact selected run. The stage-selection pointer is
bound into run identity, every partial, and the derivation manifest.

Every partial is a self-describing envelope rather than an unchecked progress cache. Before use and
again before promotion, validate its schema/stage version; seal, input, configuration, preflight, device,
and completed-prefix digests; ordered item/model/method/layer/label identifiers; exact expected array
names, shapes, dtypes, and counts; finite score domains; rank bounds; top-25 cardinality, range, and
within-row uniqueness; behavioural record fields; and a SHA256 over the canonical payload. A partial
stores a non-recursive completed-prefix digest over its run-specific execution binding, ordered
completed semantic identifiers, and canonical payload digest; the run identity includes its UUID path
and campaign-authority digest, preventing transplantation between otherwise identical runs. A partial
that fails any check is retained for diagnosis but cannot be resumed or promoted. The same complete
structural and checksum validation is applied to final cell payloads both by their producer and by the
analysis consumer; filenames or marker presence alone never license loading.

## 6. Required output and provenance bundle

Each run writes to a new UUID directory under
`results/jspace_r1_pilot/followups/atomic_orderops/`; existing result directories are never reused or
overwritten. Partial files remain under a temporary run directory and are promoted only after the full
structural and checksum validation in Section 5.

Stage completion is transactional and idempotent. Final payloads are first written to temporary files,
atomically promoted, hashed, and re-read for validation. The stage manifest and derivation manifest are
then atomically written and validated. A completion marker is written last and binds the SHA256 of those
two manifests and every final evidence payload; only after that marker is durable may an incomplete
marker be removed. If both markers coexist after a crash, the stage is complete only if the completion
marker and all bound files revalidate. If no valid completion marker exists, unbound final-looking files
are preserved as recovery evidence and the deterministic finalization/analysis may be rerun without
changing validated extraction payloads. A valid completed stage is immutable and repeated invocation
only verifies it; it never overwrites it.

O1 rank extraction and O1 analysis share one run directory but retain separate, explicit output
namespaces. The rank derivation binds every rank-stage payload present at rank finalization while
excluding the later analysis-owned `report.json`, `REPORT.md`, and `derivation_manifest.json`. The
analysis derivation binds exactly those two report payloads and consumes the already validated
`RANKS_COMPLETE` marker as a direct input. Adding or recovering analysis files therefore cannot alter
the prior rank authority; both completion markers must continue to revalidate independently.

Required artefacts:

1. source task files exactly as fetched plus SHA256;
2. frozen atomic task file and input-lock SHA256;
3. per-model eligibility/synonym-token maps;
4. raw behavioural generations and deterministic per-item scores;
5. per-model, per-method, per-item, per-layer ranks/top-25 arrays;
6. permutation seeds and sufficient null summaries to reproduce p-values;
7. item-level bootstrap inputs and interval summaries;
8. machine-readable `manifest.json`, `report.json`, and human-readable `REPORT.md`;
9. a derivation manifest hashing every direct input, script, test, and output file, including
   `jspace_phase1_scoring.py`, the protocol and input lock, all task files, both lenses, and all five
   verified Hugging Face files for each checkpoint (`config.json`, `tokenizer.json`,
   `tokenizer_config.json`, `generation_config.json`, and `model.safetensors`), every sealed bundle
   entry including lens-fit provenance, all campaign-authority files, the selected arm's
   `STAGE_SELECTED.json`, and—for O1 analysis—the validated `RANKS_COMPLETE` authority consumed
   directly;
10. environment fingerprint, model revisions, lens hashes, seal commit, execution commit, dirty flag,
    device, dtype, BF16 probe, wall time, and retry log;
11. validated completion markers whose bound manifest/derivation/final-evidence hashes constitute the
    only completion authority.

The run must execute from the seal commit or a descendant in a dedicated clean worktree. Unrelated dirty
files in the main working tree are not part of this experiment and must not be staged, reset, or copied
into the run payload.

## 7. Interpretation matrix and claim boundaries

| A0 pattern | O1 pattern | Licensed local interpretation |
|---|---|---|
| factual-dominant, as defined | O1 practical equivalence plus both checkpoints behaviourally competent | Existing multihop contrast is more task/constituent-knowledge-specific than reasoning-general. |
| factual-dominant, as defined | material O1 recurrence (with transport specificity established or not established) | Both constituent competence and synthetic-arithmetic readout differ; evidence is mixed and checkpoint-local. |
| `ΔK` 90% interval within ±0.10 | material O1 recurrence plus both checkpoints behaviourally competent | The checkpoint contrast recurs on an exact symbolic intermediate, but remains observational and lens-fit-bounded. |
| both A0 co-primary 90% intervals within ±0.10 | O1 practical equivalence plus both checkpoints behaviourally competent | The prior pair difference is bounded to the factual multihop readout assay rather than demonstrated as reasoning-general. |
| any | behaviour succeeds but the assay or typo repeatability rule fails | Instrument ambiguity; no representation-absence claim. |
| any | observed maximum is late or J offers no advantage over the logit lens | Consistent with proto-output readout; not proof of temporal externalization. |

Always forbidden from these experiments alone:

- “distillation causes/removed/lost reasoning”;
- “distillation relocated composition out of a silent workspace”;
- “the model has/does not have a global workspace”;
- “no representation exists” after a J-Lens miss;
- causal-use claims from ranks, pass@25, layer persistence@25, or checkpoint comparisons;
- treating J-space as correlation dimension, participation ratio, PCA dimension, or manifold geometry;
- generalizing beyond these two checkpoint files, fitted lenses, prompts, positions, and layer bands.

## 8. Seal block

- User authorization: Tony, 2026-08-18, in-session: “yes go ahead.”
- Pre-seal leakage audit: no local order-of-operations model output located; only task definitions and
  provenance inspected.
- Seal commit: the commit introducing the complete protocol/input/task/runner/test bundle.
- First permissible model forward pass: after that seal commit exists and the one campaign preflight has
  verified both complete model snapshots, both lenses, all task/code/environment inputs, the stable
  pure-test receipt, verified-local-tokenizer identity checks, and the frozen execution fingerprint.
