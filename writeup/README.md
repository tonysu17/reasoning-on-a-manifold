# Reasoning on a Manifold — research summary (NeurIPS format)

Files:

- `main.tex` — the summary: original per-behaviour-geometry project + safety/post-training
  extension + a unified literature review.
- `references.bib` — 37 verified references (each arXiv ID checked against the live record
  in June 2026).

## Build

No LaTeX engine was available on the machine where this was written, so the PDF was **not**
compiled here. It uses only standard packages and passed a static check (all cite keys
resolve, braces/math/environments balanced, no stray specials). Compile on Overleaf or with
a full TeX install:

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## Styling: two options

`main.tex` defaults to a **self-contained** NeurIPS-like style (Times, 10pt, 5.5×9 in block)
that needs no extra files. For an *official* NeurIPS PDF, follow the commented `Option B`
block at the top of `main.tex`: drop the official `neurips_2024.sty` next to the file,
uncomment the two `Option B` lines, comment out the `Option A` block, and delete the
`\usepackage[numbers]{natbib}` line (the official style loads natbib itself).

## Two honesty notes for the author

1. **Which numbers are results.** Only the within-chain temporal ordering (init 0.31 →
   knowledge 0.39 → uncertainty 0.52 → backtracking 0.60) is presented as an empirical
   result, from the 20-chain pilot. The geometry, steering, and all safety experiments are
   written as *proposed apparatus and falsifiable predictions*. Per the internal audit, the
   earlier curvature / TwoNN numbers came from a confounded estimator and are deliberately
   **not** cited as findings — regenerate them on the fixed estimators first.

2. **The literature pass corrected several items in `safety_reasoning_extension.md`.**
   Notably: the "latents beat text" claim belongs to Chan et al. (2507.12428), *not*
   2603.22582 (which is open-weight thinking-vs-answer text); Venhoff's "layer 17" is
   behaviour-specific (backtracking 17, knowledge/uncertainty ~18, example-testing ~15), not
   a global layer; the "refusal cliff" (2510.06036) is a final-token collapse, not a
   chain-length effect; and 2602.11096 (early-steering recovery) is a *multimodal*,
   prefix-injection result, not activation steering. These corrections are reflected in the
   prose. Author lists marked "and others" in `references.bib` should be filled in before
   submission.
