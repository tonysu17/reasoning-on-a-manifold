# Immutable P5 Sonnet safety-scoring disposition

Status: **closed**. This is a pipeline/resource disposition, not a scientific safety result.

- Internal SHA-256: `619a4ca99ebec29fd740ab445895c33a25c1f56643463daab54ba2320e185013`
- File SHA-256: `7598de6a8d436b18597fab21f2a5b575c8e16d4c1292c2a853e10e32e2605f6d`
- Closure made no proxy, model, or pod calls.

## Evidence chain

- v2.1: 37/96 safety assignments parsed; 59 remain unresolved. The pipeline gate failed.
- v2.2 disjoint validation: 90/90 usable responses passed deterministic schema/evidence validation, but network success was 86.5% versus the frozen 98% gate.
- Standard diagnostic: a billed HTTP-success proxy wrapper prospectively reproduced an empty `content` array without an explicit filter marker.
- Long-endpoint comparison: returned an incompatible `choices` envelope without parseable cost/quota accounting; underlying text was neither retained nor recovered.
- The post-execution handoff was intentionally corrected. Six historical tests now fail closed on the execution-bound old handoff hash; the corrected handoff and all executed artifacts are preserved unchanged.

## Final disposition

- v2.2.1 recovery: superseded, unexecuted, and closed.
- Held-out safety scoring: not executed and closed.
- Same-Sonnet repeats: unauthorized and unexecuted.
- Every failed/missing chunk and endpoint remains missing or unresolved. Nothing is imputed as zero, negative, compliant, or refusing.
- No safety arm comparison or powered safety claim is licensed.

Reopening requires a new versioned protocol, documented envelope/accounting, disjoint validation, hard guards, and new owner authorization.
