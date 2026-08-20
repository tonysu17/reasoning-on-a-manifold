# PT-B1 — Safety-versus-control behavioural cell (pre-registration)

**Date sealed:** 2026-08-20, committed before any PT-B1 implementation, training,
generation, or annotation exists or runs.
**Owner authorisation:** Tony, in chat, 2026-08-20 ("yes let's run the
safety-vs-control behavioural cell, seal the spec first"), against a quoted
central estimate of roughly $40–45 total. Ceilings in §9 bind; exceeding a
ceiling is a stop, not a judgement call.
**Purpose:** RQ3 established a representation-level contrast between matched
safety and non-safety fine-tunes (directional recipe signature under matched
displacement magnitude) but explicitly deferred generated behaviour. PT-B1
supplies the missing behavioural cell: paired safety-minus-control differences
in annotated reasoning-behaviour prevalence in free generation, on the sealed
Phase-2 task battery, using the same three-seed adapter recipe pair RQ3
measured. This completes the training-time column of the thesis's
intervention-by-endpoint comparison; it is not a refusal, compliance, or
safety-benchmark study.

## 1. Fixed inputs (sha256)

| Input | Path | sha256 |
|---|---|---|
| Task battery (100 fresh tasks, A1-disjoint from corpus) | `results/prereg/phase2_task_manifest.json` | `69cbe32dc7991c42ee58c8b5fae683750abb6a8812444ab320e68e23b8214333` |
| Safety training set (STAR-1 1K) | `data/safety_star1_sft.json` | `e6fa7d158a9541faabfe7baa7ca68409fc7b0cb7b91857bd64c10c1fe3c665d8` |
| Control training set (size/length-matched own-corpus chains) | `data/control_generic_sft.json` | `3520bfdd157a2dc0374fb5aead090a2d2c5b1d6eff74254958e9a4040edc5774` |
| Training script | `pt02_train_safety_lora.py` | `a21278b9ac6b4654a435eba71820281a1080d62d65b90d91c1f2dc9f748d0ef9` |
| Recipe authority (original seed-replication command) | `spark_seedrep_remote.sh` | `a0d581a97f9b241a87f5c0270d2af21d761eaf3d779267bf88c1442c4c4e8c93` |
| Extraction module (span/pooling semantics) | `04_extract_activations.py` | `8461ecb30e947a3c645731605bb4b5f9de467a1e7a126857bbe40eebbb5dbcf5` |
| Authoritative pooling | `src/delta_floor.py` | `fbd9b98bd25ed3f5cd42d9c74756299638f0cf4d5afe708146aafc99f06f2285` |
| Coverage rule | `src/annotation_coverage.py` | `affd89cba2eced10cf7585ce188167f11c771440715a4de2f67fccd51ea87ae1` |
| Stored per-seed activations (identity-gate references) | `data/activations/R1-1.5B-lora-{safety,control}1000{,-s43,-s44}/` | metadata.json `d30eeb6ca4c6672c…` (identical config all six); the gate script binds the exact row-level `.npy` hashes it reads into its output manifest |
| Base vanilla baseline (reused, never regenerated or re-annotated) | `results/ph2/battery/base.json` (100 vanilla rows) + `results/ph2/annotation/base.json` (100 annotated vanilla rows) | already committed; hashes recorded in Phase-2 provenance |

Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` at the sealed Phase-2
revision recorded in the battery provenance.

## 2. Arms and adapter reconstruction

Six generation arms, one per adapter: `safety1000` and `control1000`, seeds
{42, 43, 44}. The original adapter weights were deleted on-host after
extraction (disk rule), so each adapter is **retrained under the pinned
original command**, verbatim except `--out-dir`:

```
pt02_train_safety_lora.py --data <data> --dose all --merge \
    --epochs 5 --lr 1e-5 --batch-size 4 --grad-accum 32 --max-len 4096 \
    --seed <seed>
```

A retrained adapter is a fresh draw of the *recipe*; PT-B1's claims are
recipe-level, and the identity gate below verifies the recipe was faithfully
reproduced before any generation counts.

### Identity gate (per arm, pre-generation)

Verification subset: 150 chain ids sampled with seed 20260820 (uniform,
without replacement) from the chains present in the stored row indices;
row universe = rows from those chains present in **all** stored arms and the
base extraction. Teacher-force the subset chains through the retrained merged
checkpoint; extract the same spans under the stored extraction semantics
(occurrence-aware matching, mean pooling, `n_preceding=1`, `n_execution=10`,
layers 12 and 16). Compute the pooled mean displacement vector
(arm minus base) over the subset rows at each layer.

- **G1 (identity):** cosine(retrained direction, stored same-seed direction)
  at BOTH layers ≥ 0.98 → PASS. In [0.95, 0.98) at either layer → proceed
  with protocol marker **amended: recipe-replicate**, disclosed wherever the
  arm is cited. < 0.95 at either layer → **STOP the arm** (recipe not
  reproduced; investigate environment before any rerun).
- **G2 (polarity sanity):** cosine(retrained direction, stored opposite-recipe
  same-seed direction) ≤ 0.5 at both layers; failure stops the arm.

Gate outcomes for all six arms are written and committed before generation.

## 3. Generation

- 6 arms × the 100 sealed tasks = 600 planned rows; **vanilla generation
  only** (no steering of any kind).
- Settings byte-matched to the Phase-2 battery vanilla arm: greedy
  (temperature 0.0), `max_new_tokens` 8192, generation seed 20260808, same
  prompt construction and record schema, so rows are directly consumable by
  `src.delta_floor.per_task_fraction` and the annotation pipeline.
- **Prompt-identity preflight (hard):** the driver reconstructs ≥3 base
  vanilla prompts and their token ids must match the stored battery rows
  exactly. **Chain-reproduction preflight (soft):** regenerate those rows on
  the base model; byte-identity is recorded but env-level greedy divergence is
  a disclosure, not a stop.
- The base arm is **reused** from Phase-2 (rows + annotations); it is a
  reference level only and is never regenerated or re-annotated.
- Resume-keyed on (arm, task); records store `expected_answer: null`
  explicitly (see §5 on correctness).

## 4. Annotation

Identical to the Phase-2 pipeline as amended through A5R10: Sonnet (A3
builder-annotator, caveat carried), six-label scheme, A4 ~3,000-token
paragraph-aligned annotation window, coverage rule
`ph2-annotation-coverage-4`, `MAX_COVERAGE_ATTEMPTS=2` then sealed-unresolved,
missing/empty is unresolved and never zero-filled, spend journal + guard
manifest under `results/eval/_annotation_spend_ledgers/`. 600 new rows;
central cost ≈ $38.

## 5. Endpoints

**Primary family (confirmatory, Holm over 4):** for each behaviour
b ∈ {backtracking, uncertainty-estimation, example-testing, adding-knowledge},
the paired per-task difference

D_b = mean over tasks t of [ mean over seeds s of prev_b(safety_s, t) −
mean over seeds s of prev_b(control_s, t) ],

with prev via the authoritative windowed six-label `behaviour_fraction` /
`per_task_fraction` semantics (coverage-gated; unresolved rows excluded). A
task's arm-class value requires ≥ 1 resolved row among its three seed rows; a
task enters the pair only if both arm-classes have a value; n reported.
Inference: task-cluster BCa bootstrap, B = 10,000, seed 20260820, two-sided;
Holm over the four behaviours.

**Missingness contract (binding):** the missingness rules sealed in
`results/prereg/PH2_MISSINGNESS_BOUNDING_SPEC_2026-08-19.md` apply verbatim
to this contrast (Manski worst-case, per-arm tipping points, behaviour-dense
q50/q75/q90 scenarios; same decision vocabulary). A primary endpoint may be
cited only as complete-case, with per-arm unresolved counts adjacent, plus
its robustness label; a Holm-significant endpoint labelled
missingness-fragile is reported as **not robust**, never as confirmed.

**Secondary (descriptive, no multiplicity claim):** `bt_per_1k` paired
difference (windowed numerator and denominator); the 3×3 per-seed-pair grid
with sign counts; base-referenced arm levels.

**Full-chain endpoints (annotation-free):** chain length, looped (rep4),
truncated-at-cap, and **boxed-emission rate** (whether a `\boxed{}` answer is
produced). Boxed **correctness is not computable in PT-B1 and will not be
reported**: the sealed battery tasks are open-ended generated reasoning
prompts with no gold answers (this supersedes the earlier chat remark about
"storing answer keys"; a correctness guard requires a different task set and
is out of scope).

**Damage-context rule (binding):** if |mean safety−control chain-length
difference| > 500 tokens, or the looped/truncated difference > 0.10, every
prevalence citation must carry the length/loop shift adjacent and co-report
`bt_per_1k`.

## 6. Wording ceilings and non-claims

- Claims are **recipe-level** (three seeds per class), on this battery, under
  builder annotation, for one response-distilled 1.5B model. No item may
  claim: refusal/compliance/safety-benchmark behaviour; task performance;
  generalisation beyond the battery; a causal mechanism; or seed-population
  inference beyond the 3v3 design.
- The control-corpus asymmetry is disclosed wherever PT-B1 is cited: the
  control adapter was trained on the model's own corpus chains; battery tasks
  are A1-disjoint from corpus tasks, but format familiarity may still
  transfer, direction unknown at the behavioural level.
- The juxtaposition with the E8 steering effect is **qualitative only**
  ("same estimand family, both controlled"); the task sets differ and **no
  pooled or cross-experiment test will be computed**. This is pre-committed.
- PT-B1 does not modify, upgrade, or reinterpret any RQ3 representation-level
  result or any Phase-2 transport verdict.

## 7. Outputs

`results/ptb1/` with: `identity_gate/` (per-arm gate reports + bound npy
hashes), `battery/` (per-arm generation records), `annotation/` (annotated
rows + status), `analysis/` (`ptb1_analysis.json`, `PTB1_REPORT.md`,
missingness bounds), `provenance/` (per-stage provenance in the house schema,
including pod environment, model revision, and full input hashes). Thesis
disposition is decided by Tony after results exist; nothing in this spec
admits PT-B1 to the thesis.

## 8. Stop conditions

Stop immediately (fail closed, preserve partial artefacts, never delete) if:
G1 < 0.95 or G2 fails on any arm; the prompt-identity preflight fails; the
annotation guard projects a ceiling breach; any pod or API ceiling is
reached; a credential appears in any artefact; or the executed protocol
needs a change — protocol changes require a dated amendment committed before
the affected stage runs.

## 9. Budget ceilings

| Item | Central | Ceiling |
|---|---|---|
| Pod (train 6 adapters + 6 identity gates + 600 generations, 4090-class) | ~$5–8 | **$15** |
| Annotation (600 windowed rows, Sonnet via lab proxy) | ~$38 | **$60** |
| **Total** | **~$45** | **$75** |

Ceilings are hard; the annotation guard manifest enforces the API ceiling
fail-closed. No other spend is authorised by this document.
