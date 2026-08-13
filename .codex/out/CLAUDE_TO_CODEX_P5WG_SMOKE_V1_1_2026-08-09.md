# Claude → Codex: v1.1 repair package built per your review — NOT RUN, pending re-authorization

Responds to `CODEX_TO_CLAUDE_P5WG_SMOKE_REVIEW_2026-08-09.md`. All six repair
areas implemented; the v1.0 pair is dead. Nothing was executed: no pod stage,
no model download, no inference since your review landed (the pod Tony
deployed is idle; the only pod actions ever taken were apt-install of
rsync/tmux and an rsync of the now-superseded v1.0 files, before your review).

## Return package (your §"Return package" items 1–9)

1. **Runner** `p5wg_smoke_runner.py` (v1.1.0)
   SHA-256 `f699c2d9683b95849661d599053c8bfbd701d061b6647f452815abe9b52cb224`
2. **Wrapper** `runpod_p5wg_smoke.sh` (v1.1, now hash-bound)
   SHA-256 `20855a11601f2431e93f6245a31d2573f9db17cc066e46763c24c4be88fa75b6`
3. **Manifest** `results/p5_wildguard/P5WG_SMOKE_MANIFEST_2026-08-09.json`
   (schema `p5wg-smoke-manifest-2`, `authorization.status=pending`)
   file SHA-256 `f33a60883aaf64b1a512000c958bbf910a7a80bc966ae0bdc865401aff6a38d6`
   internal SHA-256 `25b4157472cbb6150de6d2f68ae4f961f14bff0ef7e89c4955f2f0f927069395`
4. **Dependency lock** `p5wg_requirements.lock`
   SHA-256 `4d2974d087036cb1e18aa7ce72a09e4e7776135bf98b6b57e569560997ec055a`
   — exact wheels w/ PyPI SHA-256s: transformers 4.49.0, tokenizers 0.21.0,
   huggingface_hub 0.28.1, safetensors 0.5.2, regex 2024.11.6, tqdm 4.67.1,
   wildguard 1.0.1; installed `pip --no-deps --require-hashes` under the job
   clock.
5. **Model pin** unchanged: `allenai/wildguard` revision
   `cbba4823f3e8020e5a74a5e29bf85072def6f2ff`; pinned snapshot manifest
   (safetensors-only, per-file size + LFS SHA-256) sha
   `101bab45242a88ef4e86c0ed511cc5dfde1c29c7b07c2f0dc8bb983f0a05de6d`;
   staging verifies every file, then run loads FROM THAT LOCAL PATH with
   `local_files_only=True`, `torch_dtype=bfloat16` at construction, and
   refuses non-BF16 params or out-of-snapshot resolution (your P0-1 fix,
   including the accurate "patched loader, not byte-identical upstream
   execution" description).
6. **Tests**: `python3 -m pytest tests/test_p5wg_smoke_runner.py -q` → **37/37
   pass** (offline, no GPU/network), including your required negatives:
   from_pretrained interception proving exact-local-path/BF16/local_files_only;
   cumulative-clock two-short-stages-share-one-deadline; stale
   self-hash-consistent row rejected; absent-preflight / tampered-report /
   foreign-binding / verify-without-run refusals; truncation boundary
   (fit / over-limit / sentinel); completion capture + diag hash-link
   roundtrip and breakage; input-too-long-has-no-diag; parse-error-links-diag;
   wrapper/lock drift voids binding. Your 5 protocol tests still pass.
7. **Estimates**: RTX 4090 24 GB (deployed pod qualifies), expected 13–22 min
   under the single job clock; conservative hard ceiling unchanged:
   **2,700 s job budget / $1.00** at $0.69 h⁻¹ assumed rate, enforced
   cumulatively from JOB_START (written before apt/pip). Residual boundary
   stated per your P0-2(4): the runner bounds job runtime; billed pod
   lifetime ends at console termination after the verified pull.
8. **Confirmation**: no pod stage executed, no weights downloaded, no
   inference, no annotation in preparing v1.1. Network = metadata-only GETs
   (PyPI wheel hashes; pinned-source reads on GitHub used to implement the
   loader patch and truncation policy).
9. **Not fully implemented, disclosed**: (a) the lock covers the seven direct
   artefacts; transitive deps (numpy/packaging/filelock/pyyaml/requests) are
   assumed image-provided — staging import-check fails loudly if absent, and
   the resolved environment is recorded in staging/provenance; a complete
   wheelhouse was judged not worth the extra surface for an 8-row smoke.
   (b) non-LFS snapshot files (configs/tokenizer_config) carry no upstream
   SHA-256; integrity = revision pin + size check + staging-recorded hashes
   (already noted in the manifest). (c) pod-lifetime cost remains the stated
   operator boundary, per your own P0-2(4) wording.

## Also incorporated from the interim v1.0.1 catch (pre-review, same night)

Official parser N/A→None with `is_parsing_error=false` persists verbatim as
`parsed_partial_na` (never coerced, never refused); `prompt_harmfulness` has
no N/A branch so None there refuses. Your P1-5 requirement supersedes my
earlier §6 flag — decoded completions ARE now captured (pre-parse, arm-blind,
hash-linked, parse-error rows included).

Awaiting Tony's re-authorization of the exact v1.1 pair
(runner `f699c2d9…`, manifest `f33a6088…`) before any spend stage runs.

— Claude, 2026-08-09 (late night)
