# PT-B1 Amendment 2 — preflight gate replacement after a 0/3 stop

**Date:** 2026-08-20, committed before any PT-B1 arm has been trained or
generated. **This amendment is POST-HOC**: it was written *after* the
Amendment-1 preflight gate failed and after its failure mode was diagnosed.
That ordering is disclosed here and travels with every PT-B1 citation.

**Amends:** Amendment 1's chain-reproduction gate only. No estimand, arm,
recipe, endpoint, ceiling, or identity-gate (G1/G2) band is modified.

## What happened

The Amendment-1 preflight ran on the execution pod (RTX 4090, torch
2.6.0+cu124 / transformers 5.15.0, matching the committed 7B pod environment)
and returned **0/3 identical chain reproductions**, which Amendment 1 defines
as a STOP: "prompt drift cannot be excluded."

## Why the gate failed — diagnosed, not assumed

The stop was investigated before any change was proposed. Regenerating the two
shortest stored base vanilla rows produced:

| Probe | stored chars | new chars | identical leading chars |
|---|---:|---:|---:|
| `CAUS_120` | 3,615 | 3,497 | **514** |
| `CAUS_118` | 4,047 | 4,063 | **429** |

Both continuations open with several hundred characters of *byte-identical,
on-task* reasoning about the correct problem (`CAUS_120`: Chemical X in 60% of
industrial workplaces; `CAUS_118`: the ice-cream/drowning confound), then
diverge mid-sentence at a plausible near-tie in the token distribution and
never re-converge. A drifted prompt cannot produce a 429–514-character
identical on-task prefix. The signature is the documented hardware/library
sensitivity of greedy decoding: a single flipped logit tie under different
fp16 kernels separates the chains irreversibly.

## Direct evidence that replaces the proxy

Amendment 1's chain-reproduction test was a *proxy* for prompt identity —
sufficient, never necessary. Prompt identity is now established directly and
more strongly:

- `results/prereg/phase2_task_manifest.json` is byte-identical on the pod and
  locally: sha256
  `69cbe32dc7991c42ee58c8b5fae683750abb6a8812444ab320e68e23b8214333`.
  (Note: the manifest's internal `ids_sha256` covers task **ids only**, not
  prompt text, so the whole-file hash is the load-bearing check.)
- The concatenated prompt text over all 100 tasks, id-sorted, hashes
  identically on both hosts: sha256
  `f6b91123016196596f265a4a910e780508f7c7277325c1c87344dc63e7252211`.
- `results/ph2/battery/base.json` is byte-identical on both hosts: sha256
  `fab5a0d1366a8158b77c2260147a8daa2928827d3752be784e452759d6983832`.
- Prompts reach the engine through the same hash-bound code path
  (`manifest_tasks()` → `task["prompt"]` → the Phase-2 battery engine).

## Replacement gate (binding from this amendment)

The preflight now records the manifest and prompt-text hashes above and
applies a **prefix-agreement** criterion to the same three probes:

- **PASS:** at least 2 of 3 probes share a leading identical run of **≥ 200
  characters** with their stored chain. Prompt drift is excluded; greedy
  divergence beyond the shared prefix is recorded, not penalised.
- **STOP:** fewer than 2 of 3 probes reach 200 characters, or either hash
  above fails to match.

Exact chain reproduction is retained as a reported diagnostic
(`n_identical`), never as a gate.

## Scientific consequence — recorded, not argued away

The execution environment of the PT-B1 arms is **not** byte-equivalent to the
environment that produced the July Phase-2 battery. This has one real effect
and one non-effect, and both are binding on how PT-B1 is cited:

1. **Primary contrast unaffected.** All six PT-B1 arms are generated on this
   pod, in this environment, on the same tasks, and compared *to each other*
   as paired safety-minus-control differences. The environment is common to
   both arm-classes and cancels in the contrast.
2. **Base-referenced comparisons are downgraded.** The reused Phase-2 base
   vanilla baseline (§5 "base reference levels") was generated in a different
   environment. Any PT-B1-versus-base statement is therefore **descriptive
   with an environment confound disclosed adjacent**, and must never be
   reported as a matched contrast. This supersedes any reading of §5 that
   treated the base level as environment-matched.

No other section of the pre-registration is modified.
