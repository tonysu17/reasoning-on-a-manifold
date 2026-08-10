# J-space R1 pilot — Phase-1 pre-execution amendment 2

**Protocol marker:** amended  
**Evidence status:** prospective/unrun at sealing  
**Sealed UTC:** 2026-08-10T22:06:24Z  
**Parent amendment:** `results/prereg/JSPACE_R1_PHASE1_AMENDMENT_1_2026-08-10.md` (`ef9ccda7bf4fc41532f52b052f9cf18411c57deacc82ca8e35bd275135a8bf02`)  
**Source commit:** `6f5e23a165060b516a27f1e17ba83bc3025a9d5a`  
**Run UUID:** `jspace-p1-20260810T220624Z-1c2c4230f713`

This amendment freezes the final execution and integrity implementation and resolves four small conventions identified by the independent synthetic-test and code-review passes. No Phase-1 fit, held-out readout, external-evaluation readout, or rank existed when it was sealed.

## Frozen executable files

| File | SHA-256 |
|---|---|
| `jspace_phase1_scoring.py` | `eeb7a65ae4b6555d2659af02975cf92c8ed217c9c316c82bad11a0d22de68e15` |
| `jspace_phase1_run.py` | `e2022bf7ea417dbbcbb80b6bd6b2698fd88ca539a0400ec874eefc18b6083f74` |
| `jspace_phase1_validate.py` | `06d77f251d2952cda984153772d0c800d57e53f8fa9a0f40a414ff110ba100a1` |
| `jspace_phase1_job.py` | `e99238344cc11a39600c9027a1481ca4930369d1512fd0512776d40705385a85` |
| `tests/test_jspace_phase1_scoring.py` | `2dcc92e82dec51a3efd8e5982523829420f8972e229d44a156e1ca6a0d12fd64` |
| `.codex/out/jspace_phase1_finish_watch.py` | `c9d023aaf820c978620e612c6187a95bba91000ddd98cb372a2cdb8eb645560d` |

The 17 synthetic/unit checks passed under both the remote experiment environment and the laptop watcher's `/usr/bin/python3`. A separate agent code review returned GO with no remaining code-level launch blocker. Those checks did not load the model or inspect held-out/external ranks.

## Convention clarifications

1. A vocabulary-null bijection is applied as `mapped_id = permutation[old_id]`, to B only. In the external null, recipient prompt `i` receives the complete eligible label-entry list `labels[permutation[i]]`. One permutation is shared by the all-layer and L17 statistics for that draw.
2. Duplicate eligible intermediate labels, if any, remain separate entries in an item's denominator; they are not deduplicated after the frozen eligibility manifest. The registered manifests contain the authoritative lists.
3. Per-layer relative Frobenius error uses `0/0 = 0`. A nonzero numerator over a zero reference norm is positive infinity and fails. FP32 bitwise identity compares raw float32 bit patterns, so `+0.0` and `-0.0` are not identical.
4. The 10,000-row-bootstrap interval uses NumPy's default linear quantile interpolation. It remains descriptive rather than a pass/fail threshold.

Observed and null Jaccards are both computed in float64 before strict `>`/`>=` comparisons. Top-25 readouts fail on any nonfinite logit, unexpected vocabulary size, duplicate ID, or wrong registered shape.

## Independent validation and terminal semantics

After the runner completes, a fresh validator process reloads every lens, recomputes merge and serialization errors, recomputes all registered held-out and external report fields, regenerates all 1,000 external permutations in fixed order, and regenerates all 1,000 stability permutations on CUDA. Stored and regenerated null arrays must be exactly equal. It also verifies the overall scientific gate and its licensed consequence. A hash allowlist is published only after this validation succeeds.

`DONE` means an integrity-valid execution completed. It contains `scientific_gate_pass: true|false`. A prespecified scientific negative is therefore a completed result, not an execution failure: it is synchronized and may terminate the exact Pod, but it stops before Phase 2 and licenses only the bounded wording in amendment 1. An execution, protocol, serialization, null-replay, hash, or local-sync failure produces no valid `DONE`; the finish watcher holds the Pod and alerts instead.

The laptop watcher pulls into a fresh staging directory, requires exact path-set and SHA-256 equality before and after transfer, reruns the CPU-appropriate semantic checks, verifies the remote CUDA-null attestation, and only then promotes the result locally. Immediately before the sole `podTerminate` mutation it rechecks the literal Pod ID, run UUID, terminal hashes, and the API's unique SSH endpoint binding. There is no stop/kill/fallback terminator and no automatic mutation retry after an ambiguous response.

## Runtime interpretation

Phase 0 measured 604.353 seconds for the 100 Jacobian calls. A production-shape synthetic stability-null replay measured 1.266 seconds for 10 draws on this RTX 4090, projecting about 127 seconds for 1,000; it runs once in the analysis and again independently. These figures do not time the 275 eligible external prompt forwards/readouts, network-volume serialization, or local transfer, so 30--60 minutes remains an operational estimate rather than a measured bound.
