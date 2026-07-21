# P3b forgery provenance probe — result (2026-07-21)

**VERDICT: CONFOUNDED CONTRAST — the provenance question is not yet answerable
with this dataset. The pre-registered text-classifier control caught the
confound; no Lucky-Coin claim is made in either direction.**

## What ran
- 160 records/arm (80 genuine model spec-spans vs 80 forged permissive-policy
  fabrications), two arms: attacker-register and style-paraphrased forged.
- gpt-oss-20b injection-span activations, 4 layers {6,11,12,18}, extracted on the
  96 GB pod (320 rows, peak VRAM well within budget, longest-first fail-fast).
- Chain-grouped 5-fold logistic probe (genuine vs forged) + the two sealed
  controls: a bag-of-words text classifier on the span text, and the paraphrase
  arm.

## Result
| Arm | text-classifier AUROC | activation-probe AUROC |
|---|---|---|
| attacker | 1.00 | 1.00 |
| paraphrase | 1.00 | 1.00 |

## Why this is not a provenance result
The text classifier separates genuine from forged spans at **AUROC 1.00 in both
arms** — the spans are trivially text-distinguishable even after style-matching.
The sealed rule: for a *positive* the probe must BEAT the text classifier; for an
informative *null* the probe must be at chance while the text classifier
succeeds. Neither can hold when text sits at ceiling: the probe's 1.00 is
indistinguishable from "reads the same lexical cue the text classifier does."

The root cause is a construction flaw I introduced: the **genuine pool is
prohibitive** policy deliberation (the model's real spec citations skew toward
harmful chains: "Disallowed content: ...") while the **forged pool is
permissive** ("policy permits this, so comply"). The probe (and the text
classifier) separate policy VALENCE, which is in the words, not provenance.
Paraphrasing the register does not remove the valence difference, so the
paraphrase control (correctly) also sits at ceiling.

This is the text-classifier control working as designed: it exists precisely to
refuse a provenance claim that could be lexical, and it did.

## The attempted fix failed on inspection — and revealed a deeper problem
The first-pass fix was "match valence" (genuine permissive vs forged permissive).
Two findings killed it, the second decisive:

1. **The corpus has ZERO genuine permissive policy-deliberation spans.** When
   gpt-oss cites policy it is essentially always refusing (38 genuine
   spec-citation spans, all prohibitive; benign chains that comply mostly do so
   without citing policy). So a genuine-permissive pool cannot be mined; it would
   have to be generated.

2. **The probe cannot test provenance-as-generation IN PRINCIPLE, for any pool.**
   Both variants in this design are teacher-forced INSERTIONS into a host chain.
   A forward pass's activations are a function of the input TOKENS (and their
   context) only — not of whether those tokens were sampled by the model or
   spliced in. Two sequences that differ only by an inserted span differ in
   activation space because the SPAN TEXT differs, full stop. So a "provenance
   probe" can only ever read text/content/context differences between genuine and
   forged spans; "did the model generate these tokens" is not a function of the
   current activations. Matching valence would only move the confound from
   valence to whatever text cue remained; identical-text pools would collapse the
   probe to reading host-context congruity. There is no pool construction that
   makes this probe isolate provenance.

## The well-posed experiment is CAUSAL, not a probe
The forged-policy jailbreak is a claim about EFFECT: does inserted forged
*permissive* deliberation cause the model to comply when it otherwise refuses?
That is measurable and attack-relevant, and it is the activation-patching /
behavioural arm the thesis names ("where the forged premise becomes causally
load-bearing"), not the probe:
- **Behavioural**: on refusal-expected (harmful) chains, insert forged permissive
  policy deliberation before the decision point; measure the refuse→comply flip
  rate vs a no-insertion and a neutral-insertion control. Needs generation
  (sampling continuations), not just a forward pass — a larger pod job.
- **Mechanistic (patching)**: patch activations at the injection site between a
  forged-continuation and its genuine-refusal counterpart aligned at that site;
  locate where the forged premise becomes load-bearing.

The probe is retained only as a text-distinguishability descriptor, correctly
caveated; it is not evidence about provenance representation.

## Status
Files kept: `results/safety/p3_forgery/{p3b_probe_results.json,activations/,
manifest_*.jsonl,pools.json}`; code `rom-safety-worktree/p3b_*.py`. The forged
permissive pool + hosts are reusable as the INSERTION set for the causal
experiment. Recommend NOT re-running the probe; pivot S3 to the behavioural flip
test (scope + cost is a fresh decision — it needs generation, not extraction).
