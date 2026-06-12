# `data/` manifest

What each file in `data/` is, and whether it is **canonical** (use this),
**intermediate** (a step toward a canonical file; kept for provenance), or
**deprecated / do-not-use**. This is documentation only — nothing here has been
moved or deleted.

`data/` is gitignored. All files share a common record schema within each
category (see key lists below).

Quick rule of thumb:
- Tasks → use **`tasks_final.json`**.
- Chains (R1-Distill) → use **`chains_R1-1.5B.json`**.
- Annotations → use **`annotated_R1-1.5B.json`**.
- Activations → use **`activations/R1-1.5B/`**.

---

## Tasks

Record keys: `id`, `category`, `difficulty`, `prompt`. 10 categories.

| File | Status | Notes |
|------|--------|-------|
| `tasks_final.json` | **CANONICAL** | The task corpus: 1000 tasks, 100/category, after lateral-thinking regeneration + final dedup (Phase 1.5, `04_cleanup_tasks.py`). 2026-05-23. |
| `tasks.json` | **DEPRECATED (stale)** | Older 1000-task corpus (2026-05-21), pre-cleanup. **Do not use.** (The old bug where `07_evaluate_steering.py` read this file was fixed in the 2026-06-05 audit — 07 now reads `tasks_final.json`; `00_pilot_gate.py` is the only remaining reader and is guarded as HISTORICAL.) |
| `tasks_deduped.json` | Intermediate | 901 tasks after the first dedup pass, before lateral-thinking regeneration/top-up. Input to `04_cleanup_tasks.py`. |
| `tasks_500balanced.json` | Intermediate | 500-task balanced subset (50/category), 2026-05-28. |
| `tasks_pilot.json` | Intermediate | 20-task stratified pilot (2/category) for the pilot gate (`00_pilot_gate.py`). |
| `dedup_removed.json` | Provenance log | 99 records of `{kept_id, removed_id, category}` removed during dedup. Not a task corpus. |

---

## Chains

Record keys: `task_id`, `category`, `instruction`, `prompt`, `chain`,
`full_text`, `n_tokens`.

### R1-Distill-Qwen-1.5B (primary model)

| File | Status | Notes |
|------|--------|-------|
| `chains_R1-1.5B.json` | **CANONICAL** | The reasoning-chain corpus: 1000 chains, max 8192 tokens (greedy). 2026-05-27. NB: ~50% hit the token cap / are truncated mid-`<think>` — see PROGRESS.md "Known limitations". |
| `chains_R1-1.5B_BAD_2048cap.json` | **DEPRECATED — DO NOT USE** | Generated with the wrong 2048-token cap; only 200 chains (2 categories). Kept solely as a record of the cap bug (commit `251e6c1`). |
| `chains_R1-1.5B_checkpoint100.json` | Intermediate | Mid-run checkpoint: first 100 chains (mathematical_logic only). Byte-identical to `chains_math_logic_preview.json`. |
| `chains_math_logic_preview.json` | Intermediate / preview | 100 mathematical_logic chains (identical md5 to `chains_R1-1.5B_checkpoint100.json`). |
| `chains_math_logic_readable.txt` | Human-readable dump | Plain-text rendering of the 100 math-logic chains (tokens + cap flag per chain). Not machine-read. |
| `chains_pilot.json` | Intermediate | 20-chain pilot output (all 10 categories) from the pilot gate. |
| `chains_pilot_readable.txt` | Human-readable dump | Plain-text rendering of the pilot chains. Not machine-read. |

### Qwen-2.5-Math-1.5B (non-reasoning baseline, Phase 2b)

| File | Status | Notes |
|------|--------|-------|
| `chains_QwenMath-1.5B_smoke.json` | Smoke / intermediate | 20 baseline Q/A-scaffold responses (mathematical_logic), smoke test of `02b_generate_baseline_chains.py`. No full baseline chain corpus exists in `data/` yet (awaits cluster run — see PROGRESS.md). |

---

## Annotated chains

Record keys: chain keys above plus `annotations`, `annotation_complete`.
Sentence-level behaviour labels from Phase 3.

| File | Status | Notes |
|------|--------|-------|
| `annotated_R1-1.5B.json` | **CANONICAL** | Behaviour-annotated version of `chains_R1-1.5B.json`: 1000 chains. 2026-05-27. |
| `annotated_R1-1.5B-smoke.json` | **Symlink → `annotated_R1-1.5B.json`** | Not a separate file. Lets phases addressed at the `R1-1.5B-smoke` model short-name run against the full annotated corpus (PROGRESS.md). |
| `annotated_pilot.json` | Intermediate | Annotated 20-chain pilot, from the pilot gate. |

---

## Activations

Per-behaviour residual-stream activation matrices from Phase 4
(`04_extract_activations.py`). Each directory holds
`<behaviour>_layer<N>.npy` for the 4 target behaviours
(`backtracking`, `uncertainty-estimation`, `example-testing`,
`adding-knowledge`) × 28 layers (0–27), plus a `metadata.json`
(behaviours, layers, per-behaviour extracted counts, pooling settings).

| Directory | Status | Notes |
|-----------|--------|-------|
| `activations/R1-1.5B/` | **CANONICAL** | 113 files (4 behaviours × 28 layers + metadata). Extracted counts: backtracking 10267, uncertainty-estimation 16728, example-testing 5829, adding-knowledge 5027. 2026-05-27. |
| `activations/R1-1.5B-smoke/` | Smoke (same filenames) | Smoke-run activations addressed by the `R1-1.5B-smoke` short-name; same file structure as the canonical dir. 2026-05-25. |

---

*Generated as documentation; verify counts with `ls -la data/` if in doubt.
No files were moved, renamed, or deleted in producing this manifest.*
