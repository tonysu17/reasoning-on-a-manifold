# P5 v2.1 two-call cost calibration — authorization report

**Status:** `awaiting_hash_bound_authorization`  
**Proxy calls made during planning:** `0`  
**Internal manifest SHA-256:** `f8595d94bf24da7162bd107770947a1af7601552fab91594f94ba8f5a8b0d310`  
**Manifest file SHA-256:** `f3de379d9902f1a1e4d870d1938a8248b46451e441d3702eed631984b8254a4b`

## Deterministic selections

- Generic: `primary:owned_fullft_safety_s42:SCIE_000` chunk 2; 6676 canonical bytes; request SHA `b1d336bf42eee483a1315049277d0b11ff4751be0a69d3fb48fa487a75f46631`.
- Safety: `primary:owned_fullft_safety_s42:p5sp_fraud_b01` chunk 3; 6168 canonical bytes; request SHA `90e367cb4b51ed21cbd35c738d95738e5a15744c4b1f980cbd567aa2621afe67`.

Both calls use exact Sonnet, `max_tokens=800`, a 25-second timeout, temperature zero, sequential execution, and zero retries. Returned text is hashed but not persisted and is excluded from every annotation/scoring output.

## Unset authorization fields

- Approved request ceiling: `unset` (must equal 2)
- Approved spend ceiling: `unset`
- Approved maximum cost per call: `unset`
- Remaining-quota stop floor: `unset`

## Decision-ready authorization line

> I authorize the two calibration-only requests selected by P5 cost-calibration manifest internal SHA `f8595d94bf24da7162bd107770947a1af7601552fab91594f94ba8f5a8b0d310` and file SHA `f3de379d9902f1a1e4d870d1938a8248b46451e441d3702eed631984b8254a4b`, with request ceiling `2`, total spend ceiling `$____`, maximum cost per call `$____`, and remaining-quota stop floor `$____`; exact Sonnet, 800 output tokens, 25-second timeout, sequential execution, zero retries, and exclusion of returned text from all pilot annotations/results are mandatory.

Execution remains disabled until the blank ceilings are supplied together with both exact hashes and the explicit authorization flag.
