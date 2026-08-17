# D4 — prompt-specific vs corpus-averaged Jacobian (two arms)

Run `d4-195b5eb7f2d2`; sealed §6 + A2 + A3; arms: skip16, local. Descriptive, non-gating.

## Arm skip16 (A2) — 8/8 fitted

| Item | tokens | valid pos | merged L17 | skip16 L17 | merged L25 | skip16 L25 |
|---|---:|---:|---:|---:|---:|---:|
| nhop-guitar-planet | 37 | 20 | 26 | 26 | 26 | 26 |
| planet-3-moons | 18 | 1 | 26 | 26 | 26 | 26 |
| nhop-primary-planet | 36 | 19 | 26 | 26 | 26 | 26 |
| nhop-volleyball-planet | 39 | 22 | 26 | 26 | 26 | 26 |
| func-filters-count | 18 | 1 | 26 | 26 | 26 | 26 |
| chem-organic-Z | 18 | 1 | 26 | 26 | 26 | 26 |
| func-pumps-chambers | 20 | 3 | 26 | 26 | 26 | 26 |
| etym-janus-monthnum | 19 | 2 | 26 | 26 | 26 | 26 |

Summary: `{"17": {"n": 8, "skip16_rank_median": 26.0, "merged_median": 26.0, "n_prompt_specific_better": 0, "n_merged_better": 0, "n_tied": 8, "n_prompt_specific_in_top25": 0, "n_merged_in_top25": 0}, "25": {"n": 8, "skip16_rank_median": 26.0, "merged_median": 26.0, "n_prompt_specific_better": 0, "n_merged_better": 0, "n_tied": 8, "n_prompt_specific_in_top25": 0, "n_merged_in_top25": 0}}`

## Arm local (A3) — all 81 items, exact single-position Jacobian

Summary: `{"17": {"n": 81, "local_rank_median": 26.0, "merged_median": 26.0, "n_prompt_specific_better": 0, "n_merged_better": 0, "n_tied": 81, "n_prompt_specific_in_top25": 0, "n_merged_in_top25": 0}, "25": {"n": 81, "local_rank_median": 26.0, "merged_median": 26.0, "n_prompt_specific_better": 3, "n_merged_better": 10, "n_tied": 68, "n_prompt_specific_in_top25": 6, "n_merged_in_top25": 11}}`

Per-item table in report.json. Rank 26 = outside top-25 (censored).

_descriptive only; no threshold. Registered directions: averaging-destroys (account d) => prompt-specific ranks materially better than merged (esp. L25); never-existed => both poor. A3 asymmetry: the local map is exact, so poor local ranks are strong evidence AGAINST (d)._

