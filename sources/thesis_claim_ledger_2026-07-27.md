# Thesis claim-verification ledger

Status: completed ledger for the examiner-facing pass of 27 July 2026.
The ledger records the controlling evidence for claims that recur across the
abstract, chapters, tables, and conclusion. It is not an additional analysis.

## Evidence hierarchy

1. Frozen preregistration and amendment:
   `results/prereg/THESIS_CORE_HARDENING_PREREG_2026-07-26.md`.
2. Serialized outputs produced by the frozen analysis harness.
3. Analysis reports generated directly from those outputs.
4. Repository provenance, configuration, and source code.
5. Project logs and planning documents, used only when immutable execution
   records are absent.
6. Primary literature and official model or proceedings pages for external
   claims.

If levels conflict, the lower-numbered available source controls. A conflict
that changes an RQ answer is material and is not silently resolved by editing.

## RQ1: intrinsic-dimensional occupancy

| Claim | Controlling evidence | Status at start of pass |
|---|---|---|
| Common-L27 equal-chain cdim estimates are 7.040--8.252 | Frozen confirmatory report and serialized cdim outputs cited in Chapter 4 | Verified |
| All four behaviours pass the descriptive H1 threshold | Frozen report, H1 decision rule | Verified |
| Direct within-chain specificity H2 fails for all four after Holm correction | Direct-null family report | Verified |
| Composite chain/truncation H3 fails for all four | Chain and truncation sensitivity report | Verified |
| Matched pooling/window H4 is unrun because six aligned inputs are absent | Preregistration, input inventory, harness status | Verified |
| The supported claim is low estimated occupancy, not a linear or behaviour-specific subspace | Synthesis of the preceding decision rules | Wording control |

## RQ2: specificity, linear concentration, and curvature

| Claim | Controlling evidence | Status at start of pass |
|---|---|---|
| Direct five-depth cdim family has 0/20 Holm-adjusted passes | Five-depth family output | Verified |
| Direct three-annotation cdim family has 0/24 Holm-adjusted passes | Three-annotation family output | Verified |
| Fixed-top-ten PCA is supported at all tested depths for backtracking and uncertainty estimation | PCA permutation-family output | Verified |
| PCA support for example testing is late-only and adding knowledge is unsupported | PCA permutation-family output | Verified |
| Common-L16 curvature is mixed: bounded negative criterion for uncertainty estimation and example testing; nonconforming for backtracking and adding knowledge | Curvature report and registered criterion | Verified |
| PCA, cdim, and curvature are distinct estimands and cannot validate one another by agreement | Method definition and literature | Wording control |

## RQ3: steering

| Claim | Controlling evidence | Status at start of pass |
|---|---|---|
| Backtracking single and rank-five operators beat their matched floors on count and per-1,000-token endpoints | Matched-floor report and deconfounding report | Verified |
| Target specificity is partial because uncertainty estimation is also suppressed | Cross-behaviour specificity output | Verified |
| Example-testing fraction effect does not survive the count and per-token sensitivity endpoints | Count/rate sensitivity output | Verified |
| Adding knowledge does not beat its matched floor; uncertainty estimation is unresolved at the executed power | Matched-floor report and power diagnostic | Verified |
| No consistent projection-over-single advantage is detected | Arm comparison output | Verified |
| The answer is provisional, within annotator, single dose, and lacks an executed task-accuracy guard | Preregistration, execution manifest, result report | Verified |

## RQ4: post-training geometry

| Claim | Controlling evidence | Status at start of pass |
|---|---|---|
| Full-checkpoint mean paired row displacement is 4.7--6.4% of mean activation norm; with coherence 0.70--0.78, centroid displacement is 3.5--4.6% | Full-checkpoint translation output and direct calculation from stored magnitude and coherence | Verified; wording corrected |
| Translation coherence is 0.70--0.78 and cross-behaviour cosine is 0.95--0.99 | Full-checkpoint output | Verified |
| Rotation excess is +0.26 to +0.45 degrees and resolves above split noise only for uncertainty estimation | Recalibrated rotation output | Verified |
| Matched safety and non-safety adapters have similar translation magnitudes and no resolved rotation | Matched-adapter report | Verified |
| Recipe-associated directions differ, but the three-seed exact result is exploratory: one-sided p=0.05, two-sided p=0.10 | Per-seed output and exact permutation calculation | Verified |
| The supported description is translation rather than strong reshaping; no universal safety axis or general causal attribution is established | Synthesis of the executed comparisons | Wording control |

## Corpus, annotation, and representation claims

| Claim | Controlling evidence | Status at start of pass |
|---|---|---|
| Canonical task corpus contains 1,000 tasks, 100 in each of ten categories | `data/tasks_final.json` and corpus audit | Verified |
| Extracted target rows total 37,851 across 993 chains before analysis-side deduplication | Row index and extraction audit | Verified |
| The preregistered symmetric collision and exact-vector rule retains 37,324 rows | Per-behaviour row inventory | Verified; distinguished from the earlier 37,436 within-label total |
| The generation corpus has 502 chains at the token cap and 499 without a closing think tag | Chain-quality report | Verified |
| Annotation agreement and partial-parse counts match Table 3.3 | Annotation comparison output | Verified |
| Layers are zero-based and R1-1.5B has 28 layers of width 1,536 | Model configuration | Verified |

## External claims

Every cited claim will be assigned one of four outcomes during the pass:

- verified against the cited primary source;
- narrowed to match the source;
- recited to a more appropriate primary source; or
- removed because the citation does not support it.

Bibliographic metadata was checked for cited entries. Uncited legacy entries
were left unchanged unless they caused a build or consistency error.

## Final document validation

- All citation keys resolve against `thesis/references.bib`.
- All `\Cref`, `\cref`, and `\ref` targets resolve, with no duplicate labels in
  the compiled document.
- `tectonic ucl_msc.tex --keep-logs --keep-intermediates` completed with exit
  code 0.
- The rebuilt PDF contains 63 A4 pages and no unresolved-reference, missing
  citation, overfull-box, or TeX error warning.
- Remaining diagnostics are underfull-box notices and bibliography warnings
  for conference/preprint entries without page fields; neither changes content
  or page bounds.
