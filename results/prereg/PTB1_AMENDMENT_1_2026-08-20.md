# PT-B1 Amendment 1 — preflight operationalisation (pre-execution)

**Date:** 2026-08-20, committed before any PT-B1 pod stage has run.
**Amends:** `PTB1_SAFETY_CONTROL_BEHAVIOUR_PREREG_2026-08-20.md` §3 only.

The sealed prompt-identity preflight reads: "the driver reconstructs ≥3 base
vanilla prompts and their token ids must match the stored battery rows
exactly." The stored Phase-2 battery rows do not persist prompt token ids
(they store the generated chain and its token count), so the hard form as
written is not computable against the stored artefacts. It is operationalised,
without weakening intent, as:

1. **Manifest identity (hard):** the task manifest's `ids_sha256` must
   re-derive exactly (`ph2_manifest.manifest_hash`), and prompts are taken
   from the same manifest records by the same code path as the Phase-2
   battery (`manifest_tasks()` → `task["prompt"]` → the battery engine).
2. **Chain-reproduction gate (hard STOP at 0/3):** the three stored base
   vanilla rows with the smallest `n_tokens` are regenerated greedily on the
   pinned base checkpoint with `max_new_tokens` set to each row's stored
   `n_tokens`. Byte-identity of each regenerated chain with its stored chain
   is recorded. 3/3 identical ⇒ prompt construction AND generation
   environment are jointly verified. 1–2/3 ⇒ proceed with an env-divergence
   disclosure (greedy reproduction is hardware/library-sensitive; prompt
   identity is still evidenced by any exact match). **0/3 ⇒ STOP** — prompt
   drift cannot be excluded; investigate before any arm runs.

No other section of the pre-registration is modified.
