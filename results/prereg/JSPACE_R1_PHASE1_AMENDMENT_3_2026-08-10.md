# J-space R1 pilot — Phase-1 pre-fit operational amendment 3

**Protocol marker:** amended

**Evidence status:** prospective/unrun at resealing

**Sealed UTC:** 2026-08-10T22:13:21Z

**Parent amendment:** `results/prereg/JSPACE_R1_PHASE1_AMENDMENT_2_2026-08-10.md` (`e12bdb91f090fb8b74a6de15a5dd8a0a0eec24e8a5f7a7529f9b744e87e19963`)

**Correction source commit:** `af2695195ad56c1590b9ad8697966544adc2e302`

**Superseded run UUID:** `jspace-p1-20260810T220624Z-1c2c4230f713`

**Replacement run UUID:** `jspace-p1-20260810T221321Z-73a20baaafd1`

The first Phase-1 launch stopped before model loading because the controller supplied `/workspace/hf` as the Hugging Face cache directory while the verified offline cache root is `/workspace/hf/hub`. The runner exited at `snapshot_download(..., local_files_only=True)` with `LocalEntryNotFoundError` after 23.24 seconds. Static preflight had passed, but no model load, Jacobian call, fit update, lens, held-out readout, external-evaluation readout, permutation null, or rank occurred.

The failed attempt is retained without overwrite at:

- runtime: `/workspace/jspace-phase1/runtime/jspace-p1-20260810T220624Z-1c2c4230f713`
- partial output: `/workspace/jspace-phase1/runs/jspace-p1-20260810T220624Z-1c2c4230f713`

Its terminal `FAILED.json` has SHA-256 `869d65d5f4beb94c5d8dee6972f9aeb797de481f394145b480e6147201d43985`. The only partial-output files are `RUN_ENVIRONMENT.txt`, empty `phase1_runner.log`, and an error `phase1_report.json` whose completed stage is only `preflight`. This is an execution failure and supplies no scientific evidence.

## Sole operational correction

The controller argument is changed from:

```text
--cache-dir /workspace/hf
```

to:

```text
--cache-dir /workspace/hf/hub
```

The corrected `jspace_phase1_job.py` has SHA-256 `71a016f4c89ee51eb50bad2ecdc513876d106a2cd4728574817629cdc79fe86a`. A read-only probe using the corrected cache root and the sealed model revision successfully resolved the exact snapshot:

```text
/workspace/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562
```

All scientific files, corpus rows, eligibility lists, model revision and file hashes, J-lens commit and file hashes, source and target layers, fitting parameters, prompt rendering, target encoding, seeds, null definitions, thresholds, gates, validation rules, and termination rules remain byte-identical to amendment 2. The replacement uses a fresh UUID and fresh runtime/output directories. No Phase-1 fit or readout existed when this amendment was sealed.
