# Thesis scientific style sheet

Status: controlling editorial convention for the examiner-facing pass begun
27 July 2026.

## Voice and register

- Use British English: behaviour, labelled, modelling, centre, artefact.
- Prefer a neutral, direct scientific voice. Use "this thesis" only when the
  document itself is the relevant subject; otherwise name the study, analysis,
  estimator, result, or evidence.
- Do not introduce an artificial first-person plural. Retain first person only
  where authorship or a deliberate analytical choice is clearer than a passive
  construction.
- Prefer concrete subjects and verbs. Avoid nominalisations, inflated
  importance claims, rhetorical questions, and conversational asides.
- Preserve evidential asymmetry. Positive, negative, unresolved, unrun, and
  exploratory results must not be made rhetorically equivalent.

## Tense

- Established knowledge and definitions: present tense.
- Actions performed in this study: past tense.
- Observed results: past tense, except when directing the reader to a display
  ("Table 4.1 reports").
- Supported interpretation: present tense, with a verb calibrated to the
  evidence ("supports", "suggests", "is consistent with").
- Planned or unexecuted work: future or conditional tense, confined to
  limitations, outlook, or the exploratory appendix.

## Claims and uncertainty

- Separate observation from interpretation, preferably in separate sentences.
- Reserve "significant" for a stated statistical criterion. Report the
  comparison family and multiplicity adjustment where material.
- Do not use "prove", "demonstrate", "robust", "causal", "specific", or
  "replicate" unless the design supports that exact term.
- Use "descriptive" for estimates without an inferential comparison;
  "exploratory" for analyses not protected by the frozen confirmatory design;
  "unresolved" when the analysis lacks power or identification; and "unrun"
  only when required inputs or execution are absent.
- State the scientific unit whenever row-level observations could be mistaken
  for independent samples.
- Do not use a failed null test as evidence for absence.

## Terminology

- "Reasoning trace": generated textual sequence. Do not treat it as a direct
  transcript of latent computation.
- "Behaviour": an operational annotation label, not a natural kind.
- "Behaviour-indexed activation cloud": the extracted, pooled representation
  associated with an annotation label.
- "Intrinsic-dimensional occupancy": correlation-dimension estimand used for
  RQ1. Do not replace it with "subspace dimension".
- "Variance concentration": fixed-top-ten PCA estimand. Do not call it
  intrinsic dimension.
- "Direction": a one-dimensional linear operator. "Projection" or "subspace"
  is reserved for rank greater than one.
- "Translation": displacement of a cloud mean between specified checkpoints.
- "Rotation excess": estimated subspace-angle change after subtraction of the
  within-model split-noise floor.
- "Safety-specific": permitted only for a contrast against the matched
  non-safety post-training condition; not for the unmatched public checkpoint.

## Model, layer, and behaviour names

- DeepSeek-R1-Distill-Qwen-1.5B is abbreviated as R1-1.5B after first use.
- Layer indices are zero-based and written L11, L12, L14, L15, L16, L17, L20,
  and L27 without spaces.
- In prose use backtracking, uncertainty estimation, example testing, and
  adding knowledge. Hyphenated annotation strings are used only for literal
  labels or filenames.
- Use "Sonnet 4.5", "Qwen3-235B", and "Nova-Pro" consistently after their
  checkpoints or proxy identifiers have been defined.

## Numbers and statistical notation

- Use a leading zero for ordinary decimals and omit it only where the chosen
  bibliography/table style already does so consistently.
- Write exact values to the precision supported by the stored result. Avoid
  silently increasing precision during prose revision.
- Report ranges with en dashes. Use non-breaking or mathematical formatting
  for percentages, sample sizes, confidence intervals, and \(p\)-values.
- Distinguish raw and Holm-adjusted \(p\)-values. State one-sided tests
  explicitly.
- "Confidence interval" is not used for resampling bands that lack coverage;
  call those stability bands or partition intervals as defined by the method.

## Paragraph and sentence form

- Each paragraph has one principal claim and normally begins with that claim.
- Keep causal qualifications close to the sentence they qualify.
- Avoid repeated templates such as "This thesis...", "Taken together...",
  "The answer is...", "It is important to note...", and serial "First,
  second, third" constructions.
- Vary sentence length, but split sentences that carry more than one
  inferential step.
- Use transitions only when they express a real logical relation.
- Do not repeat a table row in prose; state the pattern and the decisive
  values.

## Section functions

- Introduction: problem, gap, research questions, approach, contribution.
- Background and literature review: prior evidence, construct boundaries, and
  unresolved inferential gaps; no preview of this study's results.
- Methods: executed design and prespecified distinctions; results appear only
  where necessary to document sample formation or an executed gate.
- Results: observations and decision-rule outcomes; minimal interpretation.
- Steering and post-training chapters: design, results, and a bounded answer
  to their respective research question.
- Discussion: synthesis, comparison with prior work, alternative
  explanations, limitations, and implications; no new empirical result.
- Abstract: standalone question--method--result--conclusion account, revised
  last and containing no citations.

