# Amendment 1 — Phase-0 per-prompt resource instrumentation

**Sealed:** 2026-08-10T21:12:55Z, before the corrective execution.  
**Protocol marker:** amended.  
**Evidence status of the first execution:** current resource record for
compatibility only; not the final protocol-conforming Phase-0 record.  
**Authorization boundary:** Tony explicitly authorized the Phase-0 five-prompt
benchmark on a fresh 24 GB CUDA GPU. This corrective rerun remains within that
authorization. It does not authorize Phase 1, a reusable J-lens fit, generation,
annotation, or a thesis claim change.

## Discovery requiring amendment

The first execution completed its controller gate, but independent validation
found that `jspace_pilot_benchmark.py` recorded GPU and CPU peak memory only at
worker scope. The sealed manifest and preregistration require those three
fields for every prompt. Consequently, the first execution is retained
unchanged at `results/jspace_r1_pilot/benchmark/`, but it is not used as the
final protocol-conforming Phase-0 record.

Immutable first-execution hashes:

- `benchmark_report.json`:
  `f1127d557025d05e0d9e4b6a09b47a6f6a792a29a36a99dbce8a7df9c48bfe6a`
- `five_prompt_dim8.json`:
  `e6be98cc26c1b4b039bdf9562e2345415c130104ab7b464b62954740ffda9e70`
- `probe_dim8.json`:
  `11aa5a01222e83536f7efda319d904671e5a8340cc9938eccb2dda90f75c3a2c`

## Prospective correction

No model, prompt, layer, estimator, precision, seed, threshold, or
`dim_batch` rule changes. The amended runner only:

1. synchronizes CUDA and resets CUDA peak statistics immediately before each
   Jacobian call;
2. records per-prompt peak allocated and reserved GPU bytes after the call;
3. on Linux, writes `5` to `/proc/self/clear_refs` immediately before each
   call and reads `VmHWM` afterward, recording the measurement basis; if that
   interface is unavailable, it records the unresettable process-level
   `ru_maxrss` fallback explicitly;
4. defines worker-level GPU peaks as the maximum over setup, each prompt, and
   the registered repeat; and
5. adds a gate check requiring all three per-prompt resource fields to be
   positive integers for all five prompts.

The amended runner SHA-256 is
`445fe62cbf74be6e12b7dcc534455c9350cf8b461a3f675a974a859fa2eae298`.
Eight offline tests pass. The corrective execution writes only to the new,
non-overwriting output root
`results/jspace_r1_pilot/benchmark_amendment1/`. The first execution is never
deleted, renamed, resumed, or overwritten.

