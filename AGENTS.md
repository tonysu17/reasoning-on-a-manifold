# AGENTS.md — repository contract

Research repository for *The Geometry of Machine Reasoning* (UCL MSc). Python
analysis pipeline plus a LaTeX dissertation.

## Critical facts an agent must know before touching anything

1. **`thesis/` is a nested, SEPARATE git repository, and it is gitignored here**
   (`.gitignore:76`). The two live on GitHub as separate repos:
   `reasoning-on-a-manifold` (this one, with `results/`) and
   `reasoning-on-a-manifold-thesis` (the prose). `git status` at this root does
   not show thesis changes and `git commit` here does not commit them. Use
   `git -C thesis <cmd>`. Never `git add thesis/` from this root.
2. **Evidence lives here; prose lives in `thesis/`.** Because neither repository
   contains both, the artefacts backing the thesis's quantitative claims are
   mirrored into `thesis/_planning/review/evidence/` for single-repository
   audits. That snapshot is a **copy**; the files under `results/` here are the
   artefacts of record. Refresh the copy with
   `thesis/_planning/review/refresh_evidence.sh` whenever a result changes.
3. **Never invent, round, or "tidy" a number.** Every quantitative claim in the
   thesis is auditable to a result file. If a number cannot be located in
   `results/`, report it as unverified. Do not substitute a plausible value and
   do not silently reconcile two numbers that disagree.

## Layout

| Path | What it is |
|---|---|
| `thesis/` | LaTeX dissertation (separate git repo). See `thesis/AGENTS.md`. |
| `results/` | Machine-readable outputs. The evidence of record. |
| `data/` | Corpora, annotations, activation metadata, `row_index.json` provenance. |
| `configs/` | Run configuration; `configs/config.yaml` holds the model spec. |
| `src/`, `NN_*.py` | Analysis pipeline, numbered by stage. |

## Canonical trackers — read before citing any result

- `METHODOLOGY.md` — how things are measured.
- `RESULTS_LEDGER.md` — what was found and its trust status, including the
  do-not-cite and needs-rebuild lists.
- `CONFOUNDS_AND_REMEDIATION.md` — why a given result is wrong or bounded.
- `thesis/_planning/THESIS_CLAIM_ARTIFACT_LEDGER_2026-07-26.md` — the
  claim-to-artefact registry for the thesis specifically. This is the ground
  truth for any claim audit.

## Known non-defects — do not "fix" these

These look like inconsistencies and are not. Flagging them wastes a review pass.

- **`RESULTS_LEDGER.md` reports correlation dimensions of ~5.9 / 6.2 / 6.0 / 7.7
  while the thesis reports 7.208 / 7.364 / 7.040 / 8.252.** Both are correct.
  The ledger figures are the older behaviour-specific layer chronology
  (L14/L14/L27/L17); the thesis primary is the preregistered equal-chain
  common-L27 analysis. Different preprocessing contracts. They must not be
  interchanged, averaged, or reconciled.
- **Row counts differ across documents (37,851 / 37,436 / 37,324 / 37,380).**
  These correspond to the raw extraction, the older within-label dedup rule, the
  preregistered symmetric-collision audit, and the exploratory cross-annotation
  arm respectively. Each is correct for its own contract.
- **Different layer sets appear for different analyses** (common L27; five-depth
  L11/L14/L17/L20/L27; curvature L16; steering L17/L15/L15/L17). None is
  evidence that another is wrong or "best".
- **`configs/config.yaml` is largely STALE and is not the executing
  configuration.** Its `tasks:` block claims `generation_model: "gpt-4o"` and
  `batch_size: 25`; the code that actually ran, `src/task_gen.py`, uses
  `claude-sonnet-4-5` (line 38), `temperature 0.8` (line 85), and
  `batch_size 5` (line 172), which is what the thesis reports. Verify protocol
  constants against the executing script, not this file.
- **The chain-level sign-flip null has three values for three different
  comparisons.** Quoting one against another is a contract error:
  - full-checkpoint contrast: $0.0075$–$0.0142$
    (`spillover_gated_full.json`, coherence $0.699$–$0.783$)
  - low-rank adapter arms: $0.0074$–$0.0143$
    (`spillover_gated_safety*.json`, `spillover_gated_control1000.json`)
  - annotator-swap sensitivity: $\approx0.035$
    (`pt04c_annotator_swap.json`, coherence $0.709$–$0.760$, $p=0.002$)

  Match by the coherence value, not by the layer: the full-checkpoint and
  annotator-swap coherences overlap, so the null alone will not disambiguate
  them.

## Working rules

- Long jobs run under a background watcher that reports success, error, and
  process death. Never block-poll a running job.
- Prefer `rg` over `grep -r`. Result JSON is often large; grep it, do not bulk
  read it.
- Do not regenerate or overwrite anything under `results/` during a review or
  writing task. Those files are the audit trail.
