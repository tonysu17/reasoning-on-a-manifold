# BUILD COMPLETE — Knowledge-creation extension scaffolded end-to-end

**Date**: 2026-05-27
**Spec**: `empirical_plan_synthesis.md` build-now block (§9, 10 actions).
**Repo state**: every milestone branch merged into `main`.
**Status**: BUILD PHASE COMPLETE. Halting per synthesis §9 instruction:
> "When the sentinel is written, halt. Scientific runs come later, when
> Phase 4 finishes on the full corpus."

---

## Branches (all merged into main, --no-ff)

| Step | Branch | Merge commit |
|---|---|---|
| 1. P0.3 scaffolding | `cbs/p0-scaffold` | 7944dc3 |
| 2. M1 annotation | `cbs/m1-annotation` | c7a02dd |
| 3. P0.4 truncation policy | `cbs/p0-truncation-policy` | dcdc873 |
| 4. P0.2 anchor candidates | `cbs/p0-anchor-candidates` | 95f2e52 |
| 5. M2 geometry | `cbs/m2-geometry` | c9057ee |
| 6. M3 trajectory | `cbs/m3-trajectory` | 9f39337 |
| 7. M4 analysis | `cbs/m4-analysis` | b56d248 |
| 8. M4.5 chain_gen temperature | `cbs/chain-gen-temperature` | 2e05af3 |
| 9. M5 ablation | `cbs/m5-ablation` | 73458c0 |
| 10. M6 baseline orchestration | `cbs/m6-baseline` | 349b62b |
| (this report) | `cbs/build-complete` | _to be merged_ |

## Completion reports

| File | Status |
|---|---|
| `results/cbs/M_P03_completion_report.md` | OK |
| `results/cbs/M1_completion_report.md` | OK |
| `results/cbs/R1-1.5B/truncation_policy_decision.md` | (b) stratify by truncated:bool |
| `results/cbs/P02_completion_report.md` | candidates emitted; awaiting Tony's curation |
| `results/cbs/M2_completion_report.md` | OK + smoke regression artefact |
| `results/cbs/M3_completion_report.md` | OK + smoke regression artefact |
| `results/cbs/M4_completion_report.md` | OK + smoke group comparisons + blocked stubs |
| `results/cbs/M4_5_completion_report.md` | OK |
| `results/cbs/M5_completion_report.md` | OK (build only — intervention deferred) |
| `results/cbs/M6_completion_report.md` | OK (orchestration only — no smoke) |

## Smoke regression artefacts (NOT paper-grade)

* `results/cbs/R1-1.5B-smoke/geometry_results.json` — 576 records
  (192 main + 192 shuffle_control + 192 reversal_control) + 32 plots.
* `results/cbs/R1-1.5B-smoke/v_cbs_construction_blocked.json` —
  status:blocked stub (M5 build path).
* `results/trajectory/R1-1.5B-smoke/layer{17,27}/*.json` — per-chain
  trajectories for the first 20 chains.
* `results/trajectory/R1-1.5B-smoke/layer{17,27}_summary.parquet` —
  per-layer summary tables.
* `results/trajectory/R1-1.5B-smoke/group_comparisons.json` — M4 group
  comparisons end-to-end.
* `results/trajectory/R1-1.5B-smoke/matched_pair_results.json`,
  `verification_gradient.json` — status:blocked stubs.
* `results/cbs/cross_model/cross_model_blocked.json` — M6 build-now
  stub naming Extension A prereqs.

Every smoke artefact is tagged `"smoke-only, not paper-grade"` in its
own `note` field.

## Tests

```
$ python -m pytest src/cbs/tests/
98 passed
```

Test counts per module:
* schemas (CBSResult validation): 17 (test_annotation.py)
* annotation: 17
* geometry: 28
* trajectory: 16
* matching: 11
* chain_gen temperature: 5
* ablation: 14
* comparison: 7

All hard fail-stops have both passing and failing-path tests:
* M1 annotation: malformed JSON / invalid tier raise.
* M2 geometry: shuffle / reversal sanity tests.
* M3 trajectory: synthetic helix within 5%; straight-line zero curvature;
  degenerate T < 3 → all NaN.
* M5 ablation: cosine-similarity failstop; probe-accuracy failstop;
  unit-norm assertion.

## Outstanding decisions / blockers

### Tony-side actions (offline, no code change needed)

1. **Anchor curation (P0.2 step 1)**: pick 15 anchors (5 per tier) from
   `results/cbs/anchor_candidates.csv` (60 candidates, 30/30 behaviour
   balanced across 10 task categories). Write a locked anchor-block
   text file. Re-run `08_annotate_cbs.py` with
   `--anchor-block-path <file>` for the pilot.

2. **Tier ranking re-emission (optional)**: with `CLAUDE_PROXY_URL` /
   `CLAUDE_PROXY_KEY` set, re-run
   `python 08_annotate_cbs.py --build-anchors --behaviour-balance`
   to get Sonnet-pre-classified tier estimates in the
   `anchor_candidates.csv`. Costs ~$0.06 API.

### Pipeline-side blockers

3. **Phase 4 full-corpus extraction**: per PROGRESS.md still queued.
   Once complete, the existing `*_layer{N}.npy` files at
   `data/activations/R1-1.5B/` unblock:

   * P0.1 subspace-cleanliness verification (gate to paper-grade runs).
   * M1 full-corpus annotation (gated on P0.2 anchor lock first).
   * M2 paper-grade rerun → `results/cbs/R1-1.5B/`.
   * M3 paper-grade rerun → `results/trajectory/R1-1.5B/`.
   * M5 v_CBS construction + validation.

4. **Phase 7 answer-checker**: gates M4 matched-pair + verification-
   gradient + M5 task-set construction + M6 trajectory-Wasserstein +
   M6 cross-model classifier. Reuse Phase 7's evaluation infrastructure
   per synthesis §M4.4.

5. **Multi-seed re-generation**: gated on (3) and (4). Now unblocked
   from the code side via the M4.5 `chain_gen.py` temperature + seed
   change (commit 1ce87df). Run command:

   ```bash
   python 02_generate_chains.py \
       --model 1.5b --temperature 0.7 \
       --seeds 0,1,2,...,19 \
       --tasks-subset data/m4_100_task_ids.json \
       --max-tokens 16384
   ```

   → `data/chains_R1-1.5B_multiseed.json`. ~10 cluster GPU-hours per
   synthesis §M4.6.

6. **Extension A pipeline** on Qwen-2.5-Math-1.5B (M6 prereqs):
   Phase 2b → Phase 3 → Phase 4 → M1 → M2 → M3 → M4 with
   `--model-suffix QwenMath-1.5B`. See M6 completion report for the
   exact command chain.

### What is NOT blocked (build phase confirmed)

* Every `src/cbs/*.py` module's function signatures match synthesis
  spec verbatim and have passing unit tests.
* Every runner script (`08_*.py`–`13_*.py`) has its CLI, defaults, and
  output schema in place.
* Hard fail-stops fire (and have tests) at the right thresholds:
  - M5: `|cos| < 0.5 AND cv_acc_mean >= 0.7 AND cv_acc_std <= 0.15`.
  - M2: shuffle / reversal sanity drops `|Cliff's δ| < 0.20` in tests.
  - M1 pilot: κ < 0.5 OR tier-3 rate < 0.05 emits
    `results/cbs/{model}/FAILSTOP_M1.md` with three options.

## Workflow notes for the run phase

* Truncation policy across M2/M3/M4 is **(b) stratify by `truncated`**
  per Tony's P0.4 decision. The flag is computed by
  `src/cbs/cohort.py::is_truncated(chain)` and propagated through
  `src/cbs/trajectory.py::build_trajectory`.
* The CBS prompt is locked at P0.3; the only post-build edit is
  substituting the curated anchor block. Synthesis §12.4 — do not
  improve the prompt mid-run.
* Smoke output dirs (`*-smoke/`) and full output dirs (no suffix) must
  never mix. Every runner accepts `--model-suffix` and the
  completion-report convention disambiguates.

## Build-only deviations summary

| Where | Deviation | Why |
|---|---|---|
| P0.3 | scaffolding written via Bash heredocs / Python rewrite | Write/Edit hook IPC returned a JSON validation error; the artefacts are identical to what `Write` would have produced |
| M1 | `_extract_json_object` helper not in synthesis | Sonnet occasionally returns code-fenced JSON; helper added so the parser tolerates it |
| M2 | `--synthetic-tiers` mode added | Required for build-now smoke because CBS annotations are gated on P0.2 |
| M3 | `build_trajectory` skips `initializing` / `deduction` sentences | Phase 4 did not save activations for non-target behaviours; the resulting trajectories are sparser than chain length |
| M4 | matched-pair + verification-gradient runners emit `blocked` stubs | Both depend on Phase 7 answer-checker; stub format includes named blockers + exact unblocking commands |
| M5 | `dry-run-validate-only` flag added | Lets the runner exercise the validation step on real-but-incomplete data without running the cluster intervention |
| M6 | no smoke run | Synthesis §9 step 10 calls explicitly for "no smoke run" |

## Final state

```
src/cbs/                     8 modules, all importable, all unit-tested
src/cbs/tests/               8 test modules, 98 passed, 0 skipped
08_annotate_cbs.py
09_cbs_geometry.py
10_trajectory_build.py
11_trajectory_analysis.py
12_cbs_ablation.py
13_baseline_replication.py   all with synthesis-spec CLIs and main() bodies

results/cbs/                 10 completion reports + smoke artefacts +
                             3 blocker stubs (v_cbs, matched-pair,
                             cross_model) + anchor_candidates.csv

results/trajectory/          smoke per-chain + per-layer parquet +
                             group_comparisons + blocked stubs
```

## Halting

Per synthesis §9: "When the sentinel is written, halt." Build phase
complete. Awaiting Tony's anchor curation + the run-phase pipeline
(Phase 4 full / Phase 7 / multi-seed re-gen / Extension A) before any
paper-grade run.
