# P5 v2.1 primary scoring — proposed hash-bound authorization

**Proposal status:** `proposed_awaiting_owner_exact_hash_approval`  
**Proxy/scoring calls made while building:** `0`  
**Proposed manifest internal SHA-256:** `af8754ae13658c5b82ea7ec1360e25f6fd39d674549f5512c80c338d0e6be803`  
**Proposed manifest file SHA-256:** `988348a549480cbc63aa1437e493c7ed048ed428028246e9517a16a208e45592`

## Bound execution limits

- Primary assignments only: 176
- Initial planned requests: 490
- Total network-attempt ceiling: 600
- Global retry-attempt capacity: 110
- Per-chunk retry ceiling: 3, still subject to the 600-attempt global ceiling
- Spend ceiling: $15.00
- Approved maximum cost per request: $0.025
- Remaining-quota stop floor: $5.00
- Same-Sonnet repeats: absent and unauthorized

The ceiling product is `600 × $0.025 = $15.00`; `490 ≤ 600`. Calibration is bound by result SHA `1d79ececf4b2548b1bfc2e0d4b4c6a4d1044a220d8168415b872b04ffada3988` and journal SHA `23efb6c5cea32ebd021d14038a8510837bad621063d5da5a03afdce3edf91d09`. It observed two successful zero-retry calls costing `$0.01692` total, `$0.01095` maximum, with minimum remaining quota `$135.65018`.

## Decision-ready owner authorization line

> I authorize P5 v2.1 primary scoring under proposed manifest internal SHA `af8754ae13658c5b82ea7ec1360e25f6fd39d674549f5512c80c338d0e6be803` and file SHA `988348a549480cbc63aa1437e493c7ed048ed428028246e9517a16a208e45592`, with exactly the bound limits recorded there: 176 primary assignments, 490 initial requests, at most 600 total attempts (therefore at most 110 retries globally and no more than 3 per chunk), $15.00 total spend, $0.025 maximum cost per request, and $5.00 remaining-quota stop floor; same-Sonnet repeats remain unauthorized.

Until the owner supplies this exact-hash approval, do not pass `--authorised` and do not execute scoring.
