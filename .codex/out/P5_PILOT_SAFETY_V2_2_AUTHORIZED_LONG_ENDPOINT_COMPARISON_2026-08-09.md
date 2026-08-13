# Authorized P5 v2.2 official long-endpoint comparison

Tony directly authorized this deterministic one-call construction and execution
before its hashes were frozen. No additional hash repetition is required.

- Exact authorization: “One official long-endpoint comparison. Send the identical diagnostic once through the 120-second endpoint supplied. Expected cost about $0.006; hard ceiling $0.025; diagnostic-only.”
- Internal manifest SHA-256: `d4d14169e34c840cc16146226c9d45024531fc5190396f814097b33155bb67f0`
- File SHA-256: `a6681b29a111f2c79a98b8698e1c00932b9163a734ed94f90e65f67ed8e6fdca`
- Request-body SHA-256: `8c075e5defcebb0eba555aab4a0241a3883c20394a195906876649d0c0dcf0be`
- Long endpoint URL SHA-256: `2e9bb6979f6ac76b368a67332ede97eab840112146ec0904b1c74e34c847b7a8`
- Model: `anthropic.claude-sonnet-4-5-20250929-v1:0`
- Attempts / retries: 1 / 0
- Output tokens / client timeout: 400 / 25s
- Long endpoint service limit: 120s
- Total and per-call ceiling: $0.025
- Quota floor: $5.00
- Use: diagnostic only; excluded from annotation, gates, held-out, and recovery.

The endpoint URL itself is not persisted in this manifest/report; it is loaded
from the exact approved handoff and verified by SHA-256 before execution.
