# J-space 7B thesis-integration implementation plan

**Date:** 2026-08-18  
**Status:** EXECUTED 2026-08-19 except Stage 7 (claim-ledger addendum, awaiting
scope approval) — see completion log  
**Purpose:** preserve, correct, provenance-bind, and incorporate the completed
7B matched-pair J-space result without changing its scientific estimands or
overstating its interpretation.  
**Record type:** operational runbook and audit aid. This document is not an
empirical result, preregistration amendment, or thesis claim.

## Authority and fixed run identity

- Execution run: `jspace7b-20260818T075715Z-fa3a59ab`
- Execution source commit: `fa3a59ab3817b705ff8af3e7bbeae3d56202dbe1`
- Sealed parent: `results/prereg/JSPACE_7B_PAIR_SHEET_2026-08-17.md`
- Pre-execution evaluator/provenance amendment:
  `results/prereg/JSPACE_7B_PAIR_AMENDMENT_1_2026-08-18.md`
- Pre-execution lifecycle amendment:
  `results/prereg/JSPACE_7B_PAIR_AMENDMENT_2_2026-08-18.md`
- J-lens implementation commit:
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`

The raw D3 bundles and the first derived pair report are immutable. Corrections
must be additive and versioned. Never silently regenerate, format, rename, or
overwrite a v1 result.

## Definition of complete

Integration is complete only when:

1. the verified v1 execution outputs are committed unchanged;
2. a versioned v2 report is reproducible and hash-bound to all inputs;
3. the analysis memo and ledger preserve the registered negative and mixed
   findings;
4. the thesis evidence snapshot names a clean, full analysis commit and
   contains every artefact needed to verify the cited J-space numbers;
5. a dated post-charter claim-ledger addendum maps every claim to that snapshot;
6. the thesis prose reports E1–E5 with their limitations adjacent to E1;
7. the authoritative thesis build succeeds and its page count and any
   compensating cut are recorded; and
8. both repositories finish clean, with unrelated user work excluded from all
   commits in this plan.

## Locked scientific result

The following values have been recomputed independently from the verified raw
reports and item-level arrays. They are not to be changed by the reporting pass.

| Endpoint | Qwen2.5-Math-7B | R1-Distill-Qwen-7B | Disposition |
|---|---:|---:|---|
| Multihop any-layer union pass@25 | 52/81 = 0.641975 | 8/81 = 0.098765 | E1 direction held; descriptive checkpoint-pair contrast |
| Mid-band maximum, L10–L18 | 4/81 = 0.049383 at L13 | 0/81 | E2 mixed; no registered Boolean threshold |
| Late-band maximum, L19–L26 | 40/81 = 0.493827 at L24 | 8/81 = 0.098765 at L24/L25 | Both late-dominated |
| J-minus-logit multihop union | +0.074074 | +0.012346 | E3 consistent with scale-account direction; descriptive only |
| Association union | 1/98 = 0.010204, p=0.180819 | 0/98, p=1 | E4 nonqualifying |
| Typo union | 75/96 = 0.78125 | 78/96 = 0.8125 | Positive controls pass |
| Common-eligible multihop hit items | 52 | 8, all contained in base set | 44 lost, none gained; E5 type-skew not cleanly supported |

The distill/base multihop ratio is `0.153846`, or an 84.6% descriptive
reduction. Each cell's `p=1/1001` is against its own label-permutation null; it
is not a paired test of the model difference.

## Status vocabulary

- **Phase 1:** current non-confirmatory. The fitted lens did not pass the
  prespecified validity gate.
- **Phase 2:** prospective/unrun, later shelved. Never call it failed.
- **D1–D5:** exploratory diagnostics.
- **7B pair:** amended; exploratory; thesis disposition retained–bounded.
- **Amended:** protocol marker only, never an evidence status.
- **E1:** the registered directional inequality held.
- **E2:** mixed continuous result; no registered categorical verdict.
- **E3:** descriptively consistent with the registered scale account.
- **E4:** current non-confirmatory/nonqualifying for this statistic.
- **E5:** qualitative expectation not cleanly supported; do not relabel it as
  failed or statistically refuted.
- **v1 pair report:** preserved execution output; superseded for reporting only
  after v2 passes all gates.
- **Raw bundles:** current artefacts of record.
- **Before snapshot completion:** thesis-ineligible because provenance is not
  yet complete, even though the local execution bundles are hash-verified.

## Explicit non-claims

No implementation step may claim that:

- distillation causally removed, destroyed, or relocated a workspace;
- either checkpoint lacks internal multihop computation;
- reduced readout means reduced reasoning ability;
- Math-7B is literally free of mid-stack signal;
- scale caused the format change or defines a universal scaling law;
- math continued pretraining caused association narrowing;
- the 7B E5 category-selectivity prediction replicated;
- the per-cell permutation p-values test the base–distill difference;
- one checkpoint per condition establishes seed robustness;
- hosted 466–485-prompt lenses are fit-matched to 100-prompt self-fitted
  lenses;
- D4 proved that prompt-specific signal never exists or that corpus averaging
  cannot matter elsewhere;
- J-space is correlation dimension, participation ratio, a linear subspace,
  or a manifold estimate; or
- the 7B run reclassifies Phase 1 or implies Phase 2 ran.

## Current worktree protection

At plan creation, the analysis repository contains unrelated user changes in
the Venhoff/steering workstream, and the thesis repository contains six dirty
tracked files plus an untracked handoff. They must not be staged, stashed,
reset, reformatted, or included in J-space commits.

Before every commit:

```sh
git status --short --untracked-files=all
git diff --cached --name-only
git diff --check
```

Stage only exact J-space pathspecs. For the evidence snapshot, prefer a clean
temporary worktree at the final analysis commit rather than disturbing the
user's main worktree.

## Stage 1 — preserve v1 exactly

### Scope

- `results/jspace_r1_pilot/diagnostics/d3/d3-math-7b-base-1c070326af0b/`
- `results/jspace_r1_pilot/diagnostics/d3/d3-r1-distill-7b-429801a55af0/`
- `results/jspace_r1_pilot/diagnostics/pair7b/report.json`
- `results/jspace_r1_pilot/diagnostics/pair7b/REPORT.md`
- run-specific `pair7b.status` and `remote_manifest.txt`

Known preservation hashes:

| File | SHA256 |
|---|---|
| pair report JSON | `6bc2714a85b8a03a560c43b16c5ffeaaf5edc8fad8324ffe39878cb1273f3813` |
| pair report Markdown | `864f957ba75cd6854b1e20d3a0b7c86901e2052ddf2ab20c8ad6e05de593e42e` |
| Math raw report | `a578e13e8a05038dc692534a9aec0cc6ea3d8e38da6da05dab9312f95876a6d9` |
| Distill raw report | `e89913d125050f23eb17110fae30314f6b37e6a95a820b8da1eb97487283fdb1` |
| Remote manifest | `b41abf4f9ff0d74f7bd840ad646c174b85876f35a1ebf1c0b302c397e19b7390` |

### Acceptance gates

- All 37 remote-manifest entries map to local files and match SHA256.
- Run status is `SUCCEEDED` with the fixed run ID and source commit.
- No `.pt` lens or checkpoint is staged.
- No unrelated path is staged.
- A dedicated preservation commit exists before any evaluator/report change.
- All v1 result files are tracked after the commit.

Suggested commit:

`J-space 7B: preserve verified v1 execution outputs`

## Stage 2 — create a transparent reporting correction

Create additive files; do not replace the top-level v1 files:

- `results/jspace_r1_pilot/diagnostics/pair7b/REPORTING_ADDENDUM_2026-08-18.md`
- `results/jspace_r1_pilot/diagnostics/pair7b/report_v2.json`
- `results/jspace_r1_pilot/diagnostics/pair7b/REPORT_v2.md`
- `results/jspace_r1_pilot/diagnostics/pair7b/DERIVATION_MANIFEST_v2.json`

The addendum must state that it is a post-execution reporting/provenance
correction made after results were visible, not a scientific amendment. It
must list every v1→v2 change and preserve the v1 hashes above.

### Required evaluator/report changes

1. Load exact 1.5B and hosted reference values from committed artefacts instead
   of rounded literals.
2. Emit the complete E2 layer profiles as well as fixed-band maxima.
3. Emit the eight retained E5 item names, plus lost and gained sets.
4. Record the sealed sheet and both amendments.
5. Record execution source commit, evaluator commit and SHA256, run UUIDs,
   input paths and hashes, anchor identity, generation UTC, schema, and output
   version.
6. Accept an explicit run ID and versioned output destination.
7. Explain that `supported: true` means only that the registered E1 inequality
   held.
8. Apply no post-result numerical threshold to E2, E3, or E5.

### Derivation-manifest inputs

Bind at minimum:

- both 7B raw `report.json`, `manifest.json`, `eligibility.json`, and multihop
  NPZ files;
- the D2 Qwen2.5-7B-Instruct anchor;
- the exact 1.5B Math, validated Phase-1 R1, and Qwen3 reference artefacts;
- evaluator source and source commit;
- sealed sheet and Amendments 1–2;
- v1 report/output hashes; and
- v2 output hashes and exact invocation.

### Acceptance gates

- A deterministic rerun produces byte-identical v2 output, excluding any
  deliberately documented timestamp field.
- All primary 7B values equal v1 and the locked table above.
- Any changed reference number differs only because v2 uses exact precision.
- E2 has no Boolean verdict.
- E5 includes retained, lost, and gained names.
- v1 remains byte-identical.

## Stage 3 — preserve small fit and execution provenance

Copy byte-for-byte into a committed run-specific provenance directory:

- `math-7b-base_fit_meta.json`
- `r1-distill-7b_fit_meta.json`
- `PIP_FREEZE.txt`
- `PAYLOAD_MANIFEST.sha256`
- `RUN_STATUS.json`
- `RUN_ID.txt`
- `SOURCE_COMMIT.txt`
- `NVIDIA_SMI.txt`
- `DISK_LAYOUT.txt`
- `pair7b.status`
- `remote_manifest.txt`

Inspect logs for credentials before force-adding or copying them. Do not commit
lens weights or resumable checkpoints. Their retained hashes are:

- Math lens: `686ab0df2f403b22fe33c4f3f4183c524c9b010d28e369541cc3d56b9803a31e`
- Distill lens: `8ecce094d474b57efa9092ad37dc5362baef41e3d90b9d479ac4f3bfa0f1508a`

Acceptance requires every copied file to match its remote-manifest hash and
the fit metadata to bind model revision, fit-manifest hash, J-lens commit,
source commit, environment, `dim_batch_used`, prompt count, and lens hash.

## Stage 4 — correct the analysis synthesis

### Decision memo

Add a dated 18 August addendum. Preserve earlier recommendations as historical.
The addendum must report E1–E5 and both typo controls; mark E2 mixed and E5 not
cleanly supported; replace the simple association scale-gradient account with
a confounded scale/recipe/breadth comparison; and preserve the Phase-1 stop and
Phase-2 unrun status.

Bound D4 to the tested endpoint:

> On R1-Distill-1.5B multihop at the tested prompt position, the position-local,
> skip-16/windowed, and corpus-merged estimators each produced 0/81 top-25 hits
> at L17; at L25 the merged estimator produced 11/81 and the local estimator
> 6/81. D4 therefore did not support averaging loss as the explanation of the
> registered L17 failure. It does not establish that prompt-specific signal is
> absent at other positions, layers, tasks, representations, or models, or that
> averaging cannot matter elsewhere.

Remove or qualify “nothing existed for averaging to destroy.” Record the
one-token position difference between the exact local map and the Phase-1
final-prompt-token readout.

### Results Ledger §B6

- Extend the section through 18 August.
- Add a separate amended/exploratory 7B row.
- Correct the 1.5B base-control row: both any-layer multihop unions exceeded
  permutation; both are zero at L17.
- State that Math-1.5B had sparse L13–L16 readout and Math-7B had sparse
  L12–L17 readout.
- Bound D4 as above.
- Do not claim that composition occurs only in emitted tokens.
- Distinguish the rejected generic-small-model account from E3's separate
  descriptive format/scale account.
- Do not report an exact 7B cost without a billing artefact.
- Point to v2 and its derivation manifest.

Acceptance requires every number to resolve to a machine-readable artefact and
every evidence/protocol/thesis status to remain separate.

## Stage 5 — finish the analysis-side commit chain

Recommended logical commits:

1. v1 preservation;
2. v2 reporting correction plus small provenance; and
3. decision memo and ledger correction.

Before snapshot work, record the full final analysis commit. Confirm no lens
weights, credentials, unrelated files, or generated scratch outputs entered
the commits. Because unrelated user changes are already present, create a clean
temporary worktree at that commit for the snapshot rather than stashing or
resetting the main worktree.

## Stage 6 — harden and refresh the thesis evidence snapshot

Update `thesis/_planning/review/refresh_evidence.sh` in a dedicated evidence
maintenance pass so that it:

1. accepts an explicit clean analysis-source path, enabling a temporary
   worktree;
2. detects staged, unstaged, and untracked files with full porcelain status;
3. refuses a dirty analysis source;
4. records the full source commit;
5. validates all required source artefacts before touching the old snapshot;
6. builds into a validated temporary directory;
7. fails non-zero on any missing artefact; and
8. replaces the old snapshot only after all copies and hashes pass.

The recursive-delete target must be resolved and validated before replacement.
A deliberate negative test must show that a missing artefact aborts without
destroying the previous snapshot.

Add the complete J-space evidence chain:

- Phase-1 gate, independent validation, and execution manifests;
- D1–D5 reports/manifests and the arrays needed for retained claims;
- 1.5B base-control report and item-level inputs;
- 7B sheet and Amendments 1–2;
- both 7B raw reports/manifests/eligibility/multihop arrays;
- v1, reporting addendum, v2, and v2 derivation manifest; and
- run success status, remote manifest, and small fit provenance.

The refreshed `MANIFEST.md` must name the full clean analysis commit, report
`dirty: false`, and contain enough evidence to reproduce every J-space number
used in the thesis.

## Stage 7 — add a post-charter claim-ledger addendum

Do not rewrite the frozen 26 July ledger. With explicit scope approval, create
a dated addendum such as:

`thesis/_planning/THESIS_CLAIM_ARTIFACT_LEDGER_ADDENDUM_2026-08-18.md`

Use a distinct claim ID such as `PC-J01` and record:

- both checkpoint pairs and exact revisions;
- item denominators: multihop 81, association 98, typo 96;
- lens-fit unit: one 100-prompt aggregate lens per self-fit model, noting the
  1.5B split/merge and execution asymmetry;
- representation and estimand: prompt-only final-token readout, any-layer
  union pass@25, and fixed layer profiles;
- E1–E5 exact outcomes;
- protocol marker `amended`;
- evidence status `exploratory`;
- thesis disposition `retained–bounded`;
- evidence-snapshot paths and hashes;
- Phase-1 and Phase-2 statuses separately; and
- every explicit non-claim in this plan.

The addendum must acknowledge that the appendix is post-charter and must not
treat the ledger itself as empirical evidence.

## Stage 8 — correct the thesis prose and table

### `chapters/v2/safety.tex`

- Change “one matched-pair measurement” to two tested pairs, one per scale.
- Define a hit operationally as the registered union pass@25 endpoint before
  interpreting it as readable.
- Replace “differing only in distillation” with an architecture-matched,
  lineage-based checkpoint contrast; disclose the 1.5B fit-execution
  asymmetry.
- Replace causal substrate language with checkpoint-pair-associated
  token-readout language.
- Correct Math-1.5B's sparse L13–L16 values; say both 1.5B cells are zero at
  L17 and strongly late-dominated.
- State the bounded 7B result, `52/81` versus `8/81`.
- State that Math-7B nevertheless has a sparse L12–L17 run.
- State that E5 did not cleanly reproduce the item-type skew.
- State that no 7B behavioural audit or paired model-difference test was run.
- Remove the stale “7B replication prepared” sentence.
- Keep detailed E2–E5 results in the appendix; use at most one compact,
  bounded replication paragraph in the main chapter.

### `chapters/v2/appendix_exploratory.tex`

- Remove “base qualifies where the distill did not.” Both 1.5B any-layer
  unions exceeded their permutation controls; both are zero at L17.
- Replace “entire signal L20–L26” with the exact sparse-mid/late-dominated
  profile.
- Bound D4 to its model, task, position, layers, representation, and top-25
  endpoint.
- Phrase D1 as evidence consistent with externalisation, not proof of where
  computation occurs.
- Verify or remove the novelty claim.
- Correct hosted fit size from `457–485` to `466–485` prompts.
- Add the following table rows:

| Model | Multihop | MH @L17 | Association | Typo |
|---|---:|---:|---:|---:|
| Qwen2.5-Math-7B | .642 | .012 | .010† | .781 |
| R1-Distill-Qwen-7B | .099 | .000 | .000† | .813 |

- Report E2 fixed-band maxima in prose.
- Update E3 with Math-7B `+.074` and Distill-7B `+.012`.
- Separate the post-hoc 1.5B item pattern from the non-clean registered 7B E5
  result.
- Replace the universal association scale account with a confounded,
  descriptive comparison.
- Update the boundary from one pair to two pairs, with no training seeds,
  causal intervention, paired model-difference test, or 7B behavioural audit.

Do not promote the 7B result into the abstract or conclusion.

### Required stale-phrase audit

Before commit, search for and remove or explicitly mark as historical:

- `both.*zero.*mid-stack`
- `entire.*signal`
- `base qualifies where`
- `nothing existed for averaging`
- `relocated.*workspace`
- `pruning silent knowledge composition`
- `compositions.*performs at all`
- `scale-robust`

Counts, decimals, layer indices, prompt counts, and dagger markers must match
the v2 evidence snapshot.

## Stage 9 — build, page budget, and thesis commits

Before editing, create a fresh authoritative baseline:

```sh
git -C thesis status --short
cd thesis && tectonic ucl_msc.tex
```

Do not rely on the stale 63-page repository note or an old 68-page PDF. Record
the fresh baseline and final page counts. Any increase must name the exact
compensating cut. Prefer deleting duplicated exposition between the safety
chapter and appendix rather than cutting negative results or provenance
qualifications.

Acceptance gates:

- the authoritative build succeeds;
- citations and cross-references resolve;
- `git diff --check` passes;
- generated files are not committed;
- baseline/final page counts and any cut are recorded; and
- unrelated dirty thesis changes are not lost or silently folded into a
  J-space-only commit.

Recommended logical thesis commits:

1. `Evidence snapshot: bind J-space chain to <analysis-commit> (<N>pp)`
2. `Post-charter ledger: register bounded J-space evidence (<N>pp)`
3. `Safety/appendix: incorporate amended 7B J-space result (<N>pp)`

## Completion log

| Stage | Status | Analysis commit | Thesis commit | Acceptance evidence | Notes |
|---|---|---|---|---|---|
| 1. Preserve v1 | **done 2026-08-19** | `2f23a2f` | — | 37/37 remote-manifest hashes verified pre-commit; 5 preservation hashes match; 24 files, no `.pt`, no unrelated paths | — |
| 2. Reporting v2 | **done 2026-08-19** | `6103588` (evaluator) + `0a6893f` (outputs) | — | rerun byte-identical (3/3 hashes); 23/23 primary values equal v1; references exact-precision only (1.5B ratio 0.45459→0.454545); E2 no Boolean; E5 retained/lost/gained named; v1 byte-identical | `jspace_pair_report_v2.py`; `REPORTING_ADDENDUM_2026-08-18.md` |
| 3. Small provenance | **done 2026-08-19** | `0a6893f` | — | 11 files copied byte-for-byte into `pair7b/provenance/`, all match remote manifest; fit metas bind revision/fit-manifest/jlens-commit/git-commit/dim_batch/n_prompts/lens hash; credential scan clean; run logs force-added | no lens weights/checkpoints |
| 4. Memo and ledger | **done 2026-08-19** | `6865b44` | — | memo 2026-08-19 addendum (E1–E5 + typo, E2 mixed, E5 not cleanly supported, confounded association account, D4 bounded + seq_len−2 note, A0/O1 noted, historical text preserved); ledger B6 rows 7B/A0/O1 + base-control/D4/synthesis corrections + §B Venhoff row; no 7B dollar figure | — |
| 5. Analysis clean commit | **done 2026-08-19** | `6865b44` | — | worktree at HEAD porcelain-clean (Tony's in-flight `venhoff_official_pca_alignment` files remain untracked in the main tree, excluded from all commits) | snapshot taken from temp worktree |
| 6. Evidence snapshot | **done 2026-08-19** | `6865b44` | `cdeb3f5` | hardened script refuses dirty sources (full porcelain), validates before replacement, temp-dir build + hash verify, atomic swap, non-zero on missing; both negative tests aborted with old snapshot intact; 154 files; MANIFEST names full commit, dirty=false | — |
| 7. Claim-ledger addendum | **pending — scope approval required** | — | — | — | awaiting Tony (decision 2 in `thesis/SESSION_HANDOFF_2026-08-19.md` §9) |
| 8. Thesis prose | **done 2026-08-19** | — | `db42060` (chronology) + `f791ee7` (J-Lens replacement) | stale-phrase audit clean; safety = one compact bounded paragraph; appendix corrected + 7B rows (dagger markers verified against v2); 7B absent from abstract/conclusion; stale prose never committed | — |
| 9. Build and commits | **done 2026-08-19** | — | `f791ee7` | `tectonic ucl_msc.tex` clean; final-pass log zero undefined refs, zero missing characters (glyph fixed); **68 physical pages, page-neutral vs pre-rewrite worktree — no compensating cut**; `git diff --check` clean; no generated files committed | baseline was 68pp uncommitted / 63pp committed |

## Stop conditions

Stop the integration pass immediately if:

- a v1 hash changes;
- a remote and local hash disagree;
- a required artefact is missing;
- v2 unexpectedly changes a primary 7B result;
- the evidence source is dirty;
- a thesis number cannot be mapped to the refreshed snapshot;
- page growth occurs without an identified cut;
- a credential appears in a proposed provenance file; or
- unrelated analysis or thesis changes cannot be cleanly excluded.

