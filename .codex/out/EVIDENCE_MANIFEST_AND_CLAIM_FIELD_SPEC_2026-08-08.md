# Evidence-manifest and claim-ledger field specification

**Date:** 8 August 2026

**Audience:** Claude/Phase-0 and Phase-2 artifact producers

**Purpose:** emit enough metadata natively that the later thesis evidence-snapshot refresh and claim-ledger integration require no provenance reconstruction.

**Scope:** field contract only. This document does not refresh the evidence snapshot, register artifacts, or change any result, ledger, methodology, preregistration, or thesis file.

## The short contract

For every result family, preserve three distinct layers:

1. **Snapshot manifest:** repository/snapshot metadata plus `path`, rendered size, and full SHA-256 for each copied or size-excluded artifact. The refresh script computes this layer; it does not replace result provenance.
2. **Result provenance:** immutable input lineage, checkpoint/tokenizer identity, protocol and amendment lineage, execution configuration, unit/count definitions, code/environment identity, and unresolved fields. Claude should emit this with the result.
3. **Claim-ledger row:** the bounded thesis claim, exact estimand, model/annotator/unit/N, representation/layer rule, evidence paths/configuration, three separate status dimensions, statistical scope, and thesis action. This is claim-level metadata, not another run log.

These layers are complementary. A snapshot hash proves which bytes were copied; it does not prove how those bytes were produced. A provenance file establishes execution lineage; it does not by itself license a thesis claim. A claim-ledger row bounds the permitted claim; it does not substitute for machine-readable evidence.

## 1. Evidence-snapshot manifest: exact current fields

`thesis/_planning/review/refresh_evidence.sh` currently regenerates `thesis/_planning/review/evidence/MANIFEST.md` from scratch. Its exact snapshot-level fields are:

| Scope | Field | Current representation | Producer action |
|---|---|---|---|
| Snapshot | Source repository | Markdown scalar; currently `tonysu17/reasoning-on-a-manifold` | None; refresh script supplies it. |
| Snapshot | Source commit at snapshot time | Short Git commit from `git rev-parse --short HEAD`, or `unknown` | Ensure the closure state is committed before refresh. Do not supply a narrative substitute. |
| Snapshot | Source working tree dirty at snapshot time | Boolean `true`/`false` | Leave source state inspectable. A clean closure commit is preferred; dirty state must not be hidden. |
| Snapshot | Snapshot taken | Human-readable local date | None; refresh script supplies it. |
| Copied artifact | Snapshot path | Repository-relative path | Put the final artifact at a stable repository-relative path and arrange explicit inclusion in `ARTEFACTS`; refresh is not automatic discovery. |
| Copied artifact | Size | Human-readable `du -h` value | None; refresh script computes it. |
| Copied artifact | sha256 | Full 64-hex SHA-256 of the copied bytes | None; refresh script computes it. Preserve bytes after closure. |
| Size-excluded artifact | File | Repository-relative path plus rendered size | Arrange explicit inclusion in `EXCLUDED_BY_SIZE`. |
| Size-excluded artifact | sha256 | Full 64-hex SHA-256 of the source bytes | None; refresh script computes it. |
| Size-excluded artifact | Why | Human-readable exclusion reason | Supply a specific reason when requesting the later refresh-script update. |

Important operational consequences:

- New artifacts are **not** picked up automatically. A later authorised thesis pass must add their paths to `ARTEFACTS` or `EXCLUDED_BY_SIZE` before running the refresh.
- Snapshot entries are copies, not artifacts of record; the analysis repository remains authoritative.
- Large source arrays may be recorded by hash/size, but a hash alone does not make a numerical table recomputable. A compact machine-readable result must be copied when recomputation without the large arrays is required.
- The manifest uses full SHA-256 values. Producer-side provenance should also use full 64-hex hashes; legacy 16-character prefixes are insufficient for a zero-rework evidence chain.
- Do not put an artifact's own hash inside that same artifact: this creates a circular digest. Record output hashes in a companion artifact manifest or let the snapshot manifest compute them.

## 2. Result provenance: producer-side required fields

### 2.1 Phase-2 compatibility keys

The current `ph2_executor.py` declares these exact top-level keys:

```text
git_commit
git_dirty
prereg_sha256
manifest_sha256
checkpoint_revisions
tokenizer_gate
frame_source
counts
seeds
annotator
achieved_mde
authorised
amended
```

Keep those names for executor compatibility. Populate them rather than leaving a key absent. Use `null` only when the fact is genuinely unavailable, and preserve that unresolved state through thesis use.

Two current names have narrower semantics than their names suggest:

- `prereg_sha256` is currently a 16-character prefix. Preserve it for the current executor schema and add `prereg_sha256_full` with the full 64-hex digest.
- `manifest_sha256` is currently populated from the manifest document's `ids_sha256`; it is **not** a hash of the manifest file bytes. Preserve that compatibility value and add both `manifest_ids_sha256` and `manifest_file_sha256`. Do not silently change the meaning of an existing field without a schema-version change.

### 2.2 Required companion fields for thesis-grade lineage

The compatibility keys alone do not contain every field required by the thesis contract and claim ledger. The result-family provenance sidecar should contain the following fields as well:

| Group | Exact field | Required content |
|---|---|---|
| Schema | `schema_version` | Stable provenance-schema identifier. |
| Artifact family | `run_id` | Immutable run identifier shared by the family's outputs. |
| Artifact family | `stage` | Producing stage, e.g. `phase0_closure`, `generate_battery`, `annotate`, or `analyse`. |
| Source | `git_commit` | Full source commit, or explicit `null`. Do not use file mtime as a run record. |
| Source | `git_dirty` | Boolean. |
| Source | `dirty_paths` | Exact paths dirty at execution, empty list if clean; required whenever `git_dirty=true`. |
| Protocol | `preregistration_path` | Repository-relative path to the governing sealed preregistration. |
| Protocol | `prereg_sha256` | Existing compatibility prefix of the governing preregistration hash. |
| Protocol | `prereg_sha256_full` | Full SHA-256 of the governing preregistration bytes. |
| Protocol | `amended` | List of amendment IDs, including A1/A2 as applicable. This is a protocol marker, never an evidence status. |
| Protocol | `authorised` | Boolean launch/spend authorisation recorded by the executor. |
| Protocol | `authorisation_record` | Durable reference or identifier for the authorising record; no credentials or secrets. |
| Inputs | `input_manifest` | Map from every consumed repository-relative input path to its full SHA-256. Include prompts, annotations, activations, row indices, vectors/frames, configs, schemas, and upstream result files actually consumed. |
| Inputs | `manifest_path` | `results/prereg/phase2_task_manifest.json` when the Phase-2/P5 shared generic set is involved. |
| Inputs | `manifest_sha256` | Compatibility alias currently containing the manifest document's `ids_sha256`, not a file-byte hash. |
| Inputs | `manifest_ids_sha256` | The manifest's canonical ordered-ID/task-content digest; equal to the current compatibility value above. |
| Inputs | `manifest_file_sha256` | Full SHA-256 of the manifest file bytes. Keep this separate from the canonical ID/task digest. |
| Checkpoints | `checkpoint_revisions` | Per role/arm: model ID or local path, immutable revision/commit if available, and complete/partial identity status. |
| Checkpoints | `checkpoint_weight_sha256` | Per role/arm full weights hash or hash-manifest digest. Use `null` plus an unresolved-provenance note if unavailable. |
| Tokenizer | `tokenizer_gate` | Tokenizer alias/source; tokenizer, tokenizer-config and chat-template hashes; exact input-ID manifest hash; gate result; mismatch reason if failed. |
| Generation | `generation_config` | Chat-template source, system/user formatting, `do_sample`, temperature, top-p/top-k if applicable, seed, max-new-token cap, EOS/stop conditions, batch size, and library generation options. |
| Representation | `frame_source` | Frame/vector source paths and hashes; builder script/hash; behaviour; layer; layer indexing convention; pooling/window; rank/direction; fit/split rule; any control construction. |
| Design | `scientific_unit` | Independent unit, normally task or chain. |
| Design | `observation_unit` | Nested row/sentence/generation unit. Never collapse this into `scientific_unit`. |
| Design | `pairing_key` | Key used for paired estimates, or explicit `null` if unpaired. |
| Design | `estimands` | Exact named estimands, orientations, denominators, and aggregation order. Keep correlation dimension, participation ratio, PCA threshold dimension, and variance concentration separate. |
| Counts | `counts` | Planned and observed units by arm/cell: generated, annotated/scored, paired complete, missing, empty, unresolved, excluded, and final analysis N. Include both row and independent-unit counts where relevant. |
| Missingness | `missingness_policy` | Explicit rule. Missing/empty annotation is `unresolved`, never coerced to zero. |
| Randomness | `seeds` | Named seed per stochastic component, not one ambiguous scalar. |
| Annotation | `annotator` | Provider/model/revision, prompt/rubric path and hash, sampling parameters, parser/schema version, retry policy, and annotation run ID. |
| Analysis | `analysis_config` | Script path/hash, configuration path/hash, control/floor construction, pooling/weighting, resampling/permutation method, multiplicity family/rule, alpha, and invalid-resample policy. |
| Sensitivity | `achieved_mde` | Achieved minimum detectable effect/power by primary endpoint or explicit `null` with reason. |
| Outputs | `output_manifest` | Companion entries for each produced artifact: repository-relative path, full SHA-256, byte count, media/schema type, and role. Exclude the provenance file itself to avoid circular hashing, or generate a separate manifest last. |
| Runtime | `argv` | Exact invoked arguments. |
| Runtime | `utc_start` / `utc_end` | UTC timestamps. |
| Runtime | `wall_time_seconds` | Execution wall time. |
| Runtime | `host` | Hostname, OS/system, machine/architecture, CPU count, and accelerator identity where applicable. |
| Runtime | `process` | PID and worker count. |
| Environment | `python_version` | Exact interpreter version. |
| Environment | `package_versions` | Versions of packages material to the computation. |
| Status | `empirical_evidence_status` | One controlled label: `current non-confirmatory`, `current resource record`, `provisional`, `exploratory`, `prospective/unrun`, or `superseded (do not cite)`. |
| Status | `provenance_status` | `resolved` or `unresolved provenance`, with a reason list. |
| Status | `protocol_markers` | Amendment markers only; do not put evidence status here. |

Analysis-specific result fields remain mandatory in the result itself. For example, a resampling result needs every draw, valid/invalid draw counts and reasons, exact p-value construction, multiplicity handling, random seed, and the observed statistic. The provenance sidecar identifies how to interpret those values; it need not duplicate every result cell.

### 2.3 Recommended machine-readable shape

This is a concrete shape, not a request to rename the executor's existing compatibility keys:

```json
{
  "schema_version": "rom-result-provenance-v1",
  "run_id": "<immutable-id>",
  "stage": "<stage>",
  "git_commit": "<full-commit-or-null>",
  "git_dirty": false,
  "dirty_paths": [],
  "preregistration_path": "results/prereg/<file>.md",
  "prereg_sha256": "<current-16-hex-prefix>",
  "prereg_sha256_full": "<64-hex>",
  "manifest_path": "results/prereg/phase2_task_manifest.json",
  "manifest_sha256": "<compatibility-alias-of-ids-sha256>",
  "manifest_ids_sha256": "<canonical-task-digest>",
  "manifest_file_sha256": "<64-hex-file-hash>",
  "checkpoint_revisions": {},
  "checkpoint_weight_sha256": {},
  "tokenizer_gate": {},
  "frame_source": {},
  "input_manifest": {"path": "<64-hex>"},
  "generation_config": {},
  "scientific_unit": "task",
  "observation_unit": "generated response",
  "pairing_key": "task_id",
  "estimands": [],
  "counts": {},
  "missingness_policy": "missing_or_empty_is_unresolved_never_zero",
  "seeds": {},
  "annotator": {},
  "analysis_config": {},
  "achieved_mde": {},
  "authorised": true,
  "authorisation_record": "<durable-reference>",
  "amended": ["A1", "A2"],
  "argv": [],
  "utc_start": "<ISO-8601-Z>",
  "utc_end": "<ISO-8601-Z>",
  "wall_time_seconds": 0,
  "host": {},
  "process": {},
  "python_version": "<version>",
  "package_versions": {},
  "output_manifest": [],
  "empirical_evidence_status": "<controlled-label>",
  "provenance_status": "resolved",
  "provenance_unresolved_reasons": [],
  "protocol_markers": ["amended:A1", "amended:A2"]
}
```

## 3. Claim-artifact ledger: exact row fields

The current thesis claim ledger uses these exact eight columns:

| Column | Required content |
|---|---|
| `ID` | Stable claim ID (`Cxx`); allocate during ledger integration, not in the result executor. |
| `Source / RQ` | Authoritative thesis source locations and the research question(s) addressed. |
| `Claim requiring audit` | One bounded, falsifiable claim. Do not combine different estimands or positive and negative outcomes into one omnibus sentence. |
| `Model, annotator, unit/N` | Exact checkpoint/arm, annotator, independent scientific unit and N, nested observation rows and N, and per-cell attrition where relevant. |
| `Data / representation / layer rule` | Dataset/prompt manifest, split, extraction version, representation site, pooling/window, exact layer(s), zero-based/fractional-depth convention, and how the layer was selected. |
| `Analysis / evidence / configuration` | Producing script, machine-readable artifact paths, provenance/config/prereg paths, exact estimand, principal values/intervals/tests, control construction, seed/resampling/multiplicity details. |
| `Protocol marker; evidence status; thesis disposition; statistical scope` | Four explicitly separated elements: amendment marker(s); one empirical evidence status; provenance status if unresolved; one thesis disposition; and the local model/layer/representation/control/power scope. |
| `Route A action` | Concrete thesis action: keep/rewrite/move/cut/rerun, including the wording boundary or missing gate. For new work this becomes the current integration action. |

The producer should supply a claim-handoff record with the same semantic fields even though the final `Cxx` ID and thesis line numbers are assigned later:

```json
{
  "rq": ["RQ3"],
  "claim_candidate": "<bounded claim>",
  "models_and_arms": [],
  "annotator": {},
  "scientific_unit": {"name": "task", "n": 0},
  "observation_unit": {"name": "response", "n": 0},
  "cell_counts": {},
  "data_and_split": {},
  "representation_and_layer_rule": {},
  "estimand": {},
  "analysis_and_controls": {},
  "artifact_paths": [],
  "configuration_paths": [],
  "protocol_markers": [],
  "empirical_evidence_status": "<controlled-label>",
  "provenance_status": "resolved",
  "thesis_disposition": "retained-current",
  "statistical_scope": "<exact local scope>",
  "integration_action": "<keep/rewrite/move/cut/rerun>"
}
```

Ledger rules that must travel with the handoff:

- The chain/task is the independent scientific unit; a sentence row is a nested observation.
- State the estimand before the number.
- Never merge correlation dimension, participation ratio, PCA variance-threshold dimension, or fixed-top-ten variance concentration.
- Negative results are local to the tested statistic, layer, representation, control, and achieved power.
- `amended` is a protocol marker, not an evidence status.
- `git_commit: null`, `git_dirty: true`, a missing input hash, or uncertain execution lineage requires explicit `unresolved provenance`; uncertainty travels with every use.
- `failed`, `unrun`, `provisional`, `exploratory`, and `superseded` are not synonyms.
- A narrative report may locate evidence but cannot replace a machine-readable result plus provenance.

## 4. Claude handoff checklist

Before announcing a Phase-0 or Phase-2 artifact family ready for thesis integration, provide:

- stable repository-relative artifact paths;
- a full SHA-256 and byte count for every output in a non-circular companion manifest;
- the populated Phase-2 compatibility keys and the thesis-grade companion fields above;
- full hashes for every consumed input, governing preregistration/amendment, task manifest, script, schema, and configuration;
- checkpoint/weight and tokenizer/input-ID identities by arm;
- exact independent-unit, observation-row, paired-complete, missing, empty, unresolved, and analysed counts;
- exact estimands, orientations, control floors, multiplicity family, alpha, seeds, and achieved sensitivity;
- one bounded claim-handoff record per claim family, with the three status dimensions kept separate;
- an explicit list of unresolved provenance facts rather than inferred replacements;
- confirmation that no output was written into an authoritative result directory until promoted through the analysis ledger/methodology process.

For the forthcoming shared vanilla set, additionally record both the task-manifest file hash and canonical task-ID/content digest, the exact E8 sealed generation configuration, the common tokenizer/input-ID gate, and a statement that the generation array is the single shared input artifact for Phase 2 and P5. P5 must consume it and must not regenerate a divergent vanilla set.

## Sources inspected

- `thesis/AGENTS.md`
- `thesis/_planning/review/refresh_evidence.sh`
- `thesis/_planning/review/evidence/MANIFEST.md`
- `thesis/_planning/THESIS_CLAIM_ARTIFACT_LEDGER_2026-07-26.md`
- representative frozen provenance sidecars under `thesis/_planning/review/evidence/`
- `src/config.py::provenance`
- `ph2_executor.py::PROVENANCE_KEYS` and `build_provenance`

The sources remain authoritative if this field specification and an executable schema later diverge.
