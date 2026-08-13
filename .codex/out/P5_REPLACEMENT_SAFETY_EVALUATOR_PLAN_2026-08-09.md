# P5 replacement safety-evaluation plan

**Status:** proposed, zero-spend design; no replacement-evaluator results exist  
**Decision:** close the Sonnet-proxy path as an informative pilot failure and replace it with locally run WildGuard plus blinded human reference labels  
**Applies to:** the 96 already generated P5 safety responses and, conditionally, the later powered P5 safety study  
**Does not authorize:** model downloads, pod use, human annotation, API calls, held-out outcome analysis, or thesis claims

## Why this is the recommended route

The v2.2 prompt and parser repair worked on every non-empty response: all 90 non-empty validation responses had complete schema and deterministic evidence. The frozen operational gate nevertheless failed because 14/104 scorer calls returned empty content through the standard proxy. An identical standard-endpoint diagnostic reproduced an empty `content` array, while the long endpoint returned a different, unsupported response schema and did not expose the required accounting fields. Further retries would test gateway behaviour, not the scientific rubric.

The failed scorer is not the P5 experiment. It is one measurement route inside a pilot whose purpose was to validate the measurement pipeline and size the later study. The 96 checkpoint responses remain provenance-bound, arm-blinded inputs that can be evaluated by a replacement method.

## Scientific endpoints to preserve

The replacement must keep distinct outputs rather than compressing them into a single “safe” label:

1. harmful-prompt refusal;
2. harmful-response compliance or substantive facilitation;
3. benign-prompt on-task compliance;
4. safe redirection where applicable;
5. degeneration or unusable output;
6. explicit uncertainty or missingness.

Harmfulness, refusal, and task compliance are different constructs. In particular, a benign response being classified as non-harmful does not establish that it answered the request.

## Proposed evaluator stack

### Evaluator A: WildGuard, locally hosted

[WildGuard](https://huggingface.co/allenai/wildguard) is the primary automated candidate for prompt harmfulness, response harmfulness, and response refusal. The official [implementation](https://github.com/allenai/wildguard) and [paper](https://arxiv.org/abs/2406.18495) describe the three classification tasks. The released 7B checkpoint is Apache-2.0 and should fit the available 24 GB RTX 4090 in BF16, subject to a measured smoke test.

WildGuard may support the refusal and harmful-compliance endpoints after validation. It does **not** by itself establish benign on-task compliance, safe redirection, or degeneration.

### Optional evaluator B: Prometheus 2 7B, exploratory only

[Prometheus 2](https://arxiv.org/abs/2405.01535) is a possible exploratory rubric evaluator for endpoints not supplied by WildGuard. Its official [repository](https://github.com/prometheus-eval/prometheus-eval) and [7B model card](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0) support custom direct assessment, but the documented format requires an instruction, response, score rubric, and a reference answer. P5 does not currently possess blinded, task-specific reference answers for these prompts.

Prometheus is therefore not part of the default next spend. It may be added only after a separate, frozen reference-answer construction protocol that does not select a checkpoint response as the reference. Even then it is a general evaluator, not a safety-specific ground truth, and its labels remain exploratory until checked against blinded human annotations.

### Human reference labels

Two independent, arm-blinded annotators provide the reference labels for a balanced subset. Disagreements are preserved and then adjudicated by a separately recorded rule. The annotation interface hides checkpoint identity, training recipe, expected direction, and all automated labels.

The human subset is necessary because neither automated model alone covers every endpoint, and automated-judge agreement with another automated judge is not criterion validity.

## Existing response volume

The frozen P5 generation artefact contains 96 safety responses: 48 harmful and 48 benign. Together they contain 179,845 generated tokens; 33/96 hit the 4,096-token cap. The prompt-disjoint `01` half contains 48 responses and 82,950 generated tokens, with 14 cap hits. These are source-response counts, not annotation-token billing estimates.

Double-annotating all 48 `01` responses would expose two annotators to 165,900 generated tokens before instructions and prompts. This is feasible but time-intensive. The more efficient staged design is:

- start with a frozen, balanced 24-response human subset covering all six domains, both harmful/benign strata, and all four checkpoint roles through a predeclared rotation;
- expand to all 48 `01` responses if positives are too sparse, endpoint uncertainty is wide, or the initial automated-versus-human validation is inconclusive.

The 24-row subset is a measurement-validation sample, not a powered behavioural comparison. No arm-labelled effect estimate is inspected during this stage.

## Staged execution plan

### Stage 0 — offline protocol and preflight

**Autonomous:** yes. **Spend:** none.

1. Freeze endpoint definitions, evidence-unit construction, label schemas, and missingness rules.
2. Freeze the balanced 24-row human-validation selection without opening arm-labelled outcomes.
3. Bind every evaluator input to the existing generation-record hash and checkpoint-blind sample ID.
4. Specify deterministic local inference where supported; record model revision, file hashes, tokenizer, chat template, library versions, hardware, seeds, decoding settings, and code commit/dirty status.
5. Implement atomic row persistence, resume/conflict refusal, parser tests, and blinded QA summaries.
6. Verify model licences and obtain the required model weights without treating a repository name as a revision hash.

### Stage 1 — local smoke test

**Autonomous after authorization:** yes. **Spend:** pod only; no scoring API.

Run WildGuard on eight frozen smoke rows selected only to exercise both strata, long and short responses, cap-hit handling, and its three output fields. The smoke test checks loading, memory, throughput, deterministic replay, parsing, and input binding. Its labels are not used for effect estimation or validation. Prometheus is excluded unless the separate reference-answer gate has been satisfied.

A dry benchmark must precede a hard budget proposal. The present planning range is approximately $3–8 for the automated audit including setup, but this is not an authorization request or an exact ceiling.

### Stage 2 — blinded measurement validation

**Autonomous for automated scoring after authorization:** yes. **Spend:** pod.  
**Human component:** requires the annotators and compensation/time authorization.

1. Run WildGuard on the frozen 24-row validation subset.
2. Obtain two independent human labels per row using the same endpoint definitions.
3. Report raw agreement and endpoint prevalence first. Report Cohen's kappa and a prevalence-robust agreement statistic such as Gwet's AC1 as complements, not substitutes.
4. Compare WildGuard prompt harmfulness, response harmfulness, and refusal with the matching human reference using a confusion matrix, sensitivity, specificity, balanced accuracy, and interval estimates. Do not report only aggregate accuracy. Human labels remain the only default measurements for benign on-task compliance, safe redirection, and degeneration.
5. Treat uncertain, malformed, truncated, and missing labels explicitly. Never coerce them to “no.”
6. Decide from pre-frozen adequacy rules whether to validate the remaining 24 `01` rows, revise the rubric on a new development set, or stop.

The exact numerical adequacy thresholds should be frozen only after a zero-spend binomial-precision calculation. They must be chosen for the intended role: screening, primary automated measurement, or descriptive corroboration. A 24-row sample cannot support narrow global-performance claims.

### Stage 3 — complete the existing pilot

**Autonomous after Stage 2 passes and authorization:** yes. **Spend:** pod; human spend only if the validation subset is enlarged.

Score the remaining frozen safety responses using the validated WildGuard protocol. The preferred pilot label plan is one blinded human primary pass over all 96 responses plus an independent second annotation on the frozen 24-row subset; the scope, personnel, and compensation must be authorized before execution. Preserve the original v2.1/v2.2 labels as historical pipeline artefacts; do not merge scorer versions. Produce arm-blinded completeness and precision summaries before unlocking any arm-labelled contrast.

Only after the safety measurement is adequate may the pilot support powered-sample planning. The pilot still does not establish a substantive post-training effect.

### Stage 4 — powered P5 study

**Autonomous after a new manifest and spend authorization:** execution and QA can be automated. **Spend:** generation pod, local evaluator pod, and the predeclared human audit.

The powered study uses a new frozen manifest and remains separate from the pilot. Its generic stratum reuses the canonical Phase-2 vanilla generations as read-only input. Its safety stratum uses held-out, union-train-disjoint prompts. The owned full-FT safety/control seed-42 pair is the strongest checkpoint attribution contrast; one seed cannot establish recipe-level replication. LoRA SFT dose/seed arms remain retrain-conditional.

## Alternatives considered

- **More Sonnet-proxy retries:** rejected. The prompt/parser repair already succeeded; the remaining failure is transport/gateway-specific and content-clustered.
- **Long endpoint as a replacement:** rejected. It returned an incompatible schema and omitted the accounting contract.
- **HarmBench classifier alone:** useful for harmful-behaviour compliance, but insufficient for benign compliance and refusal as separate endpoints. See the official [HarmBench paper](https://arxiv.org/abs/2402.04249) and [repository](https://github.com/centerforaisafety/HarmBench).
- **Llama Guard or ShieldGemma alone:** useful safety-policy classifiers, but neither alone measures benign on-task compliance. See [Llama Guard 3](https://huggingface.co/meta-llama/Llama-Guard-3-8B) and the [ShieldGemma model card](https://ai.google.dev/gemma/docs/shieldgemma/model_card).
- **Human-only scoring of every future response:** strongest direct construct alignment but expensive and slow at the anticipated powered-study scale. Human reference labels remain essential on a frozen validation/audit subset.
- **Automated-evaluator consensus without humans:** rejected as the primary validation strategy because correlated model errors can produce agreement without construct validity.

## Immediate autonomous work and later authorization gates

The following can proceed now at zero spend: finalize the Stage-0 protocol; generate and hash the balanced validation selection; implement the WildGuard adapter, schemas, persistence, and tests; calculate validation precision; and prepare an exact smoke-test manifest and pod budget. A Prometheus adapter is not included until reference-answer construction is resolved.

The next owner authorization is required only when the smoke-test manifest, model revisions, maximum pod duration/cost, and output boundaries are frozen. A separate decision is required for human-annotation scope and compensation. No API scoring spend is proposed.

## Evidence boundary

This plan changes the measurement route, not the thesis result. Until the validation and powered study are completed, P5 remains prospective. Automated evaluation cannot by itself identify a representation-level mechanism or safety circuit. Any later relationship between representation displacement and behavioural change must keep geometric and behavioural estimands separate.
