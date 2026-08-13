# P5 powered-generic annotation validation plan

**Status:** prospective and non-executable  
**Spend authorized:** none  
**Independent scientific unit:** task; checkpoint rows are repeated observations

## Decision

Validate the final generic annotation protocol on 20 of the 100 Phase-2 tasks—two per category and all five checkpoint roles—before annotating the remaining 80 tasks. The 100 validation rows become part of the final study if the unchanged protocol passes, so the gate adds no duplicate scoring.

Tasks are selected by a frozen hash ranking within category after the Phase-2 manifest is bound and before any annotation result is opened. The held-out 400 rows cannot start until the operational gate passes.

## Operational gate

- 100% atomic terminal persistence;
- at least 98/100 globally valid rows;
- at least 19/20 valid rows for every checkpoint role;
- 100% generation-lineage and 4,096-token-prefix hash matching;
- 100% source reassembly and endpoint recomputation among valid rows;
- complete response-envelope and accounting records;
- missing annotations remain missing, never negative labels.

Transport failures may receive one identical retry. A non-empty parser/schema failure is terminal for the validation protocol rather than being repeatedly sampled until it parses. If the gate fails, the held-out rows stop and any repair requires a new versioned validation.

## Phase-2 reuse boundary

The canonical Phase-2 generations are always read-only inputs. Phase-2 annotations are reusable only if they cover the complete P5 first-4,096-token-or-EOS prefix and match the exact source-unit segmentation, ontology, model, prompt, parser, evidence, and missingness hashes. An annotation of only the Phase-2 A4/window subset is not interchangeable with P5's full-prefix sentence-fraction estimand.

If these conditions fail, P5 reuses the raw generation—not the annotation—and requires a separately authorized scoring pass. It never regenerates the shared vanilla rows.

## Human audit

A prospective 20-row checkpoint-blind human audit is recommended to measure annotator dependence. It is not currently authorized. Without it, every resulting behavioural rate must retain the “builder-annotator scored” qualifier; same-Sonnet repeats do not constitute inter-annotator agreement.

## Remaining gates

Execution still needs the final Phase-2 and owned-row hashes, exact scorer/envelope hashes, the selected validation IDs, a calibrated request count and cost ceiling, and explicit authorization for that exact package.
