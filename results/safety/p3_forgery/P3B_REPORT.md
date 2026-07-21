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

## The fix (a matched-valence redesign; needs a phase-A rebuild + re-extraction)
Isolate provenance by matching valence:
- **Genuine PERMISSIVE pool**: mine the model's real deliberation that concludes
  compliance is allowed — benign-chain `decision:comply`/`safe_complete` spans
  with policy reasoning ("this is allowed", "no policy bars this"). These are
  genuine permissive deliberation, the natural counterpart to a permissive forgery.
- Contrast genuine-permissive vs forged-permissive: both argue for compliance;
  only provenance differs. The text classifier should then DROP toward chance
  (or at least off ceiling), making the activation probe interpretable.
- Optionally add a forged-PROHIBITIVE pool matched to the current genuine pool as
  a symmetric second contrast.

Cost: rebuild pools (offline) + one more injection-span extraction (~$2–5 pod).
The probe/controls/verdict machinery is unchanged and ready.

## Status
Files: `results/safety/p3_forgery/{p3b_probe_results.json,activations/,
manifest_*.jsonl,pools.json}`. Code `rom-safety-worktree/p3b_*.py`. The
attacker/paraphrase manifests + activations are kept (the genuine spans and the
forged-permissive pool are reusable in the matched redesign).
