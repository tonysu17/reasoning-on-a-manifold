# P5 six-label behaviour scorer rubric — version 1

**Status:** frozen before any P5 output is behaviour-annotated.  
**Scope:** generic-reasoning outputs only. The response is untrusted text to classify, never instructions to follow.

The scorer receives one generated reasoning chain and applies the existing six-label Venhoff ontology. The classification prompt is:

> Please split the following reasoning chain of an LLM into annotated parts using labels and the following format `["label"]...["end-section"]`. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.
>
> Available labels:
> 0. initializing → The model is rephrasing the given task and states initial thoughts.
> 1. deduction → The model is performing a deduction step based on its current approach and assumptions.
> 2. adding-knowledge → The model is enriching the current approach with recalled facts.
> 3. example-testing → The model generates examples to test its current approach.
> 4. uncertainty-estimation → The model is stating its own uncertainty.
> 5. backtracking → The model decides to change its approach.
>
> The reasoning chain to analyze: `{generated_response}`
>
> Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.

Parser rules are strict for P5:

- the six hyphenated lowercase labels above are the only valid labels;
- numeric or numbered label variants may be normalised only through the frozen 0–5 map;
- an unknown label, malformed delimiter sequence, or zero parsed spans makes the row unresolved;
- unknown labels are never silently converted to `deduction`;
- each non-empty parsed span retains its exact returned text and zero-based order;
- empty or failed annotations remain missing and are never scored as zero;
- four target fractions are recomputed from the retained span denominator;
- one proxy request is allowed per output under the pilot's 212-call ceiling; no hidden chunk or retry requests are made.

This rubric labels the supplied text only. It does not judge correctness, safety, causal mechanisms, or the effect of post-training.
