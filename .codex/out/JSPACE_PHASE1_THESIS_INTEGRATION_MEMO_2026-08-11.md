# J-space Phase-1 thesis-integration memo

**Date:** 2026-08-11  
**Status:** proposed writing disposition; no thesis source edited  
**Source run:** `jspace-p1-20260810T224338Z-996077031021`  
**Independent validation:** `jspace-p1v-20260810T231952Z-6ad3de6f3754`; all twelve integrity/recomputation checks passed, including recomputation of the failed overall scientific gate.  
**Current generated thesis length:** 62 pages (`ucl_msc.pdf`); this memo does not alter it

**Application gate:** no thesis empirical claim or quantitative value from this pilot may be inserted until the validated Phase-1 reports, independent-validation record, and any retained post-hoc diagnostic are frozen into the thesis evidence snapshot with matching hashes. Separate claim-ledger entries must record the model, unit, N, representation, layer rule, statistic, provenance and evidence status, preserving **current non-confirmatory** for the registered result and **exploratory** for the post-hoc layer profile.

## Recommendation

Treat the pilot as a stopped instrument-validation study, not as evidence for or against J-space in R1-1.5B. Preserve the registered consequence verbatim:

> The fitted lens did not pass the prespecified validity gate.

**Empirical evidence status:** current non-confirmatory. **Record type:** instrument validation. **Protocol marker:** amended. The J-space decomposition and causal component comparison are **prospective/unrun**, not failed. Nothing in this pilot changes the thesis's four geometric estimands or its RQs.

In this staged protocol, passing the numerical, serialization and split-fit stability checks was insufficient to establish semantic criterion validity; the external gate prevented the mechanistic decomposition from proceeding.

## Evidence summary

The two independently fitted 50-prompt lenses passed numerical, merge, serialization, and held-out split-half stability checks. The external gate required at least two of three suites to qualify in the same registered evaluation domains. Only typo qualified.

| External suite | Merged any-layer union | Merged L17 | Disposition |
|---|---:|---:|---|
| Association | 4/98 | 1/98 | non-qualifying; neither registered permutation criterion passed |
| Typo | 86/96 | 62/96 | qualifying; all four registered criteria passed |
| Multihop | 0.1852 across 81 items (82 eligible labels) | 0.0000 across 81 items | non-qualifying; the all-layer permutation criterion passed, but the L17 criterion failed |

“Any-layer union” asks whether eligible labels appear anywhere in the union of all 27 source-layer top-25 sets; it is not an average across layers. All six registered A/B-difference checks were within 0.10; those threshold checks therefore did not cause the external-gate failure. A separate exploratory, non-gating layer profile localized most multihop hits to L21--L26; this did not alter the registered outcome.

## Literature boundary

Gurnee et al. define the Jacobian lens as a corpus-averaged linearized map from a source-layer residual activation to later output effects. The token-indexed J-lens directions form an overcomplete set and may span the entire residual stream. For a fixed sparsity bound \(k\), J-space is the set of nonnegative combinations supported on at most \(k\) directions---a union of sparsity-bounded cones, not a single low-dimensional linear subspace. The paper typically uses \(k\leq25\) as an activity/capacity convention, not an intrinsic-dimension estimate. See the [official paper](https://transformer-circuits.pub/2026/workspace/index.html#the-j-space), [methods](https://transformer-circuits.pub/2026/workspace/index.html#the-jacobian-lens), and [pinned reference implementation](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e).

Consequently, J-space must not be merged with correlation dimension, participation ratio, PCA variance-threshold dimension, fixed-top-ten variance concentration, curvature, or “manifold structure.” The legitimate connection is methodological: both projects interrogate layer-indexed residual representations, but they measure different objects.

This pilot is not an attempted exact replication of Anthropic's paper. The reference-implementation README states that the paper's lenses used 1,000 sequences of 128 tokens; the paper's quantitative comparison summarizes normalized area under the pass@\(k\) curve and studies primarily fully trained production LLMs of considerable size. The R1 pilot fitted 100 generic prompts and registered pass@25 against label-permutation controls both across all source layers and separately at the prespecified L17. Anthropic explicitly leaves scaling to smaller models unresolved and notes single-token coverage and inconsistent readout interpretability as limitations. The defensible label is therefore **an R1-specific criterion-validity test using released prompts**.

## Recommended thesis placement

Do not change the abstract, RQs, headline results, or main geometry interpretation. Because this pilot is post-charter and stopped before the thesis-relevant causal estimand, the safest empirical placement is a compact subsection in `chapters/v2/appendix_exploratory.tex`, plus at most one sentence in the methodological discussion or conclusion.

### Proposed appendix prose

```tex
\subsection{Stopped Jacobian-lens validity pilot}
\label{sec:appendix-jspace-pilot}

The Jacobian lens maps a source-layer residual activation through a
corpus-averaged linearisation of its downstream effect, yielding
vocabulary-indexed readout directions \citep{gurnee2026verbalizable}.
For a fixed sparsity level, the associated J-space is a union of sparse
nonnegative cones over an overcomplete token-indexed set; it is not a
low-dimensional linear subspace or a manifold estimate. It therefore does
not reinterpret the correlation-dimension estimate, participation ratio, PCA
variance-threshold dimension, or fixed-top-ten variance concentration used
elsewhere in this thesis.

A preregistered exploratory pilot fitted two Jacobian lenses to disjoint
50-prompt WikiText samples for R1-1.5B. The A, B and merged lens artefacts
passed the registered matrix, serialization, merge and held-out A-versus-B
stability checks. On the released external prompt families, only typo
satisfied all four suite-level qualification criteria. Association satisfied
neither registered permutation criterion. Multihop satisfied the all-layer
criterion but not the L17 criterion. Because qualification required both
readouts within the same suite, one of three suites qualified, below the
prespecified requirement of two.

The fitted lens consequently did not pass the prespecified validity gate.
Under the registered stopping rule, no J-space decomposition or causal
component comparison was run. Empirical evidence status is current
non-confirmatory for this fitted R1 lens and protocol. This is not evidence
that R1 lacks J-space, that Anthropic's result failed to replicate, or that
any thesis geometry statistic has been explained.

A separate exploratory, post-hoc, non-gating layer profile found
tokenizer-eligible intermediate labels mainly in layers 21--26 for multihop;
this descriptive localization did not alter the registered result.
```

### Proposed main-text sentence

```tex
A preregistered exploratory attempt to validate a Jacobian-lens readout on
R1-1.5B stopped at its external criterion-validity gate, so no J-space
decomposition or causal comparison was undertaken.
```

### Bibliography entry required if applied

```bibtex
@article{gurnee2026verbalizable,
  author  = {Gurnee, Wes and Sofroniew, Nicholas and Pearce, Adam and
             Piotrowski, Mateusz and Kauvar, Isaac and Chen, Runjin and
             Soligo, Anna and Bogdan, Paul and Ong, Euan and Wang, Rowan and
             Thompson, T. Ben and Abrahams, David and Kantamneni, Subhash and
             Ameisen, Emmanuel and Batson, Joshua and Lindsey, Jack},
  title   = {Verbalizable Representations Form a Global Workspace in Language Models},
  journal = {Transformer Circuits Thread},
  year    = {2026},
  url     = {https://transformer-circuits.pub/2026/workspace/index.html}
}
```

## Page and scope disposition

The current generated PDF is 62 pages. The proposed appendix text is deliberately compact and contains no table or figure; the full layer plot remains in the analysis record. Applying the text still requires a rebuild and an exact new page count. If it increases the PDF to 63 pages, the page-neutral choices are:

1. retain only the one-sentence main-text methodological lesson and keep the empirical pilot in the analysis record; or
2. replace comparable post-charter exploratory appendix material, but only after an explicit scope decision—do not silently cut a negative or provenance qualification.

The first choice is the safest default. No automatic cut is recommended because the appendix's existing hedges are load-bearing.

## Next experimental decision

The completed CPU-only failure map cannot reclassify Phase 1. Before any refit, seal a separate diagnostic protocol that:

1. applies the same preregistered eligibility and pass@\(k\) definitions to the prefitted Qwen lens linked from the pinned Anthropic walkthrough, with model-specific tokenizer and layer conventions sealed before readout;
2. audits, independently of lens ranks, R1's behavioural task competence on each registered positive-control item, using a prospectively sealed endpoint and scoring rule;
3. compares a small hash-selected set of exact prompt-specific Jacobians with the averaged lens; and
4. tests FP32 versus the registered BF16 unembedding while measuring rank-25/26 margins and boundary ties.

These exploratory diagnostics may distinguish scorer/reference incompatibility from hypotheses involving model task competence, averaging, corpus/model mismatch or precision, but cannot reclassify Phase 1. Any fresh fit requires a new preregistration and previously uninspected confirmatory controls. Phase 2 remains prospective/unrun and is not licensed under the current protocol.

## Artefact links

- Validated scientific report: `/Users/tonysu/.codex/jspace_phase1_watch_root/results/jspace_r1_pilot/phase1/jspace-p1-20260810T224338Z-996077031021-validated-a5/phase1_report.json`
- Authoritative validation: `/Users/tonysu/.codex/jspace_phase1_watch_root/results/jspace_r1_pilot/phase1/jspace-p1-20260810T224338Z-996077031021-validated-a5/validation_attempts/jspace-p1v-20260810T231952Z-6ad3de6f3754/INDEPENDENT_VALIDATION.json`
- Post-hoc failure-map report: `.codex/out/jspace_phase1_posthoc_failure_map_2026-08-11/REPORT.md`
- Post-hoc diagnostic manifest: `.codex/out/jspace_phase1_posthoc_failure_map_2026-08-11/DIAGNOSTIC_MANIFEST.json`
