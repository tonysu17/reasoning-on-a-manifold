# AMENDMENT 1 — J-space diagnostic protocol (venue + D5 clarification)

**Date:** 2026-08-17. **Amends:** `JSPACE_R1_DIAGNOSTIC_PROTOCOL_2026-08-16.md` (seal commit `752dee7`).
**Recorded before any affected stage runs.** D1 is unaffected (already executed locally, run
`d1-20260816T212205Z-ccdae99be0fd`, committed `0c9d852`).

## 1. Execution venue and spend (amends §"Spend boundary", §4, §5, §6, §7)

D2–D5 execute on a **RunPod pod** instead of local M4 / DGX Spark. Authorized by Tony in-session
2026-08-17: *"i am ready to run the next experiments. i want to run it on runpod instead."*
Envelope: one RTX-4090-class community pod, estimated 1–3 h, **estimated $1–3**; any materially larger
spend (bigger GPU class, >6 h) requires fresh sign-off. No change to endpoints, wrappers, thresholds,
seeds, populations, execution order (D2 gates D3 interpretation), stop rules, or licensed wording.
Adapter/harness code is still committed before the runs (§4).

## 2. D5 mechanics clarification (amends §7)

The sealed §7 reads "From the validated `external_readout_arrays.npz`, recompute all external readout
ranks…". Inspection (2026-08-17) shows the npz stores the resulting **top-25 token-ID sets**
(`[n_items, 27, 25] int32` per lens) and null distributions, not activations or logits. FP32
recomputation therefore requires re-extracting final-prompt-position residuals for the eval items with
the pinned R1 model (BF16 forward, as Phase 1), then applying an FP32 lens copy and FP32 unembedding.
**Endpoint unchanged:** per suite, count of (label, layer) hits crossing the 25/26 boundary relative to
the stored BF16 top-25 sets, and any change to any-layer union counts. Registered expectation unchanged.

## 3. D4 comparator transfer (mechanical note under §6)

The Phase-1 merged-lens comparator slices at source layers 17 and 25 are extracted locally from the
validated bundle (`…-validated-a5/lenses/`), SHA-256 recorded at extraction, and transferred to the pod.
Slice extraction is mechanical and does not alter the comparator bytes' provenance chain.
