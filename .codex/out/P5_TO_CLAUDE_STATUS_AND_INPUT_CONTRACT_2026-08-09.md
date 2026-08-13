# P5 → Claude status and Phase-2 input contract

## Action for Claude

No P5 action blocks Phase 2. Continue the authorized Phase-2 route. When the canonical shared vanilla generation artefact is finalized, hand P5 the hashes and applied generation facts listed below. Do not generate a second P5 “vanilla” set.

## P5 status

- The 176-generation pilot is complete.
- The Sonnet safety-scoring route is formally closed after its frozen validation gate failed. This is a P5 measurement-pipeline disposition and does not alter Phase 2.
- Generic behavioural annotation resolved 78/80 pilot rows. The 97.5% rate is below P5's frozen 98% operational gate, so a powered run needs a fresh prospective annotation validation.
- The generic pilot's independent N is 20 tasks. Arm-blinded sizing is complete; no signed or arm-labelled pilot effect was inspected.
- The powered generic protocol skeleton remains non-executable until the shared Phase-2 artefact and owner decisions are bound.

## Shared vanilla contract

P5 treats the Phase-2 vanilla set as a read-only input artefact:

1. generate it once under the Phase-2 protocol;
2. preserve the exact response bytes and per-row hashes;
3. never regenerate a missing or incompatible row inside P5;
4. keep P5 observational endpoints separate from the Phase-2 causal family;
5. do not expose Phase-2 causal-family outcomes to P5 before the injection-recovery gate permits it.

P5's current primary recommendation is an analytic prefix ending at the first 4,096 generated token IDs or earlier EOS. This is an analysis rule, not permission to rewrite a sealed Phase-2 decoding contract. If Phase 2 applies a shorter generation cap, especially 2,048 tokens, report that exact fact: the P5 pilot's variance evidence does not transfer directly because 34/80 generic pilot rows already hit 4,096 tokens.

## Handoff fields required when the shared artefact lands

- Phase-2 task-manifest path, file SHA-256, internal/ID SHA-256, and source commit;
- shared vanilla artefact path and file/shard SHA-256 values;
- exact row count and stable row identifiers;
- per-row prompt hash, response hash, status, stop reason, generated-token count, and row hash;
- model identifier and revision/weight hash where available;
- tokenizer, chat-template, and generation-config hashes;
- actual maximum-new-token setting, EOS/pad IDs, stop conditions, sampling flags, temperature semantics, and seed;
- code commit/dirty status, environment hash, hardware, and timestamps;
- injection-recovery gate status, without disclosing gated causal-family outcomes prematurely.

## No collision boundary

Claude retains ownership of `ph2_*`, `results/`, sealed preregistrations, the pod, `RESULTS_LEDGER.md`, and `METHODOLOGY.md`. P5 work remains under `.codex/out/` and the thesis repository until promotion through the evidence ledger. The replacement safety-evaluator plan does not write into Phase-2 or analysis result paths.
