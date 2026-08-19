# Reporting addendum — 7B matched-pair v2 correction

**Plan date:** 2026-08-18 (filename follows the integration plan)
**Executed:** 2026-08-19
**Record type:** post-execution reporting/provenance correction, made after the
7B results were visible. This is **not** a scientific amendment: no estimand,
statistic, threshold, eligibility rule, or scored artefact changed. Every
primary 7B value in v2 equals v1 exactly (verified programmatically, 23/23
fields).

## Authority

- Execution run: `jspace7b-20260818T075715Z-fa3a59ab`, status `SUCCEEDED`
  2026-08-18T10:46:49Z, source commit
  `fa3a59ab3817b705ff8af3e7bbeae3d56202dbe1`.
- Sealed parent: `results/prereg/JSPACE_7B_PAIR_SHEET_2026-08-17.md`.
- Pre-execution amendments:
  `results/prereg/JSPACE_7B_PAIR_AMENDMENT_1_2026-08-18.md`,
  `results/prereg/JSPACE_7B_PAIR_AMENDMENT_2_2026-08-18.md`.
- Implementation runbook: `JSPACE_7B_THESIS_INTEGRATION_PLAN_2026-08-18.md`.

## Preserved v1 outputs

The v1 report remains the execution output of record and is byte-identical to
its preservation commit:

| File | SHA256 |
|---|---|
| `pair7b/report.json` | `6bc2714a85b8a03a560c43b16c5ffeaaf5edc8fad8324ffe39878cb1273f3813` |
| `pair7b/REPORT.md` | `864f957ba75cd6854b1e20d3a0b7c86901e2052ddf2ab20c8ad6e05de593e42e` |
| Math-7B raw `report.json` | `a578e13e8a05038dc692534a9aec0cc6ea3d8e38da6da05dab9312f95876a6d9` |
| Distill-7B raw `report.json` | `e89913d125050f23eb17110fae30314f6b37e6a95a820b8da1eb97487283fdb1` |
| `remote_manifest.txt` | `b41abf4f9ff0d74f7bd840ad646c174b85876f35a1ebf1c0b302c397e19b7390` |

All 37 remote-manifest entries were re-verified against local files by SHA256
before the preservation commit.

## Complete list of v1 → v2 changes

All changes are reporting or provenance additions; none alters a 7B estimand.

1. **Exact reference values replace rounded literals.** v1 hard-coded the
   1.5B comparison as `0.4074`/`0.1852` and the E3 references as
   `-0.056`/`+0.037`. v2 loads them from committed artefacts:
   - Math-1.5B multihop union `0.4074074074074074` and Δ(J−logit)
     `-0.05555555555555558` from
     `diagnostics/d3/d3-qwen2.5-math-1.5b-base-2008de267117/report.json`;
   - validated Phase-1 R1-1.5B multihop union `0.18518518518518517`
     (BF16, recomputed from the stored Phase-1 arrays by the sealed D5
     precision diagnostic) from `diagnostics/d5/d5-c480b291f5aa/report.json`;
   - Qwen3-1.7B Δ(J−logit) `0.03703703703703698` from
     `diagnostics/d3/d3-qwen3-1.7b-3bd7b4c16291/report.json`;
   - Qwen2.5-7B-Instruct values from
     `diagnostics/d2/d2-qwen2.5-7b-it-f1b6cdfdff03/report.json` (v1 already
     loaded these live; unchanged).
   Consequence: the derived 1.5B ratio corrects from `0.4545900834560629`
   (computed from the rounded literals) to `0.45454545454545453` (= 15/33
   exactly). No 7B value is affected.
2. **Complete E2 layer profiles.** v2 emits the full 27-layer
   `layer_profile_pass_at_25` for both models alongside the fixed-band maxima,
   so "sparse L12–L17, late-dominated" is verifiable from the report itself.
3. **E5 retained names.** v2 emits `retained_in_distill` (8 items) in addition
   to v1's `lost_in_distill` (44) and `gained_in_distill` (0).
4. **Sealed documents recorded.** v2 records the sealed sheet and both
   pre-execution amendments with SHA256; v1 recorded only Amendment 2.
5. **Provenance block.** v2 records execution source commit, evaluator v1/v2
   paths + SHA256 + commit, run UUIDs, model revisions, anchor identity,
   generation UTC, schema and output version, and the preserved v1 hashes.
6. **Explicit arguments.** The run ID and output destination are explicit
   arguments; v1 discovered the run ID from `data/jlens_local/RUN_ID.txt`
   (a gitignored path).
7. **`supported` semantics stated.** v2 states in-band that E1
   `supported: true` records only the registered directional inequality, is
   not a paired model-difference test, and that per-cell permutation p-values
   test labels within a cell.
8. **No post-result thresholds.** v2 applies no numerical threshold to E2, E3,
   or E5; the v1 protocol notes are retained.

## Derivation

`DERIVATION_MANIFEST_v2.json` binds every input (both 7B raw bundles' report,
manifest, eligibility, and multihop arrays; the four reference artefacts; the
sealed sheet and both amendments; both evaluator sources) and both v2 outputs
by SHA256, together with the exact invocation. A repeated identical invocation
reproduces all three v2 files byte-identically; `--generated-utc` is the only
operator-supplied field.

## Interpretation boundary (unchanged from the sealed sheet)

Protocol marker **amended**; evidence status **exploratory**; thesis
disposition **retained–bounded**. The result is a checkpoint-pair-associated,
predominantly late token-readout contrast. It does not identify distillation
as a cause, does not establish a lost or relocated workspace, and licenses no
claim about reasoning ability in general.
