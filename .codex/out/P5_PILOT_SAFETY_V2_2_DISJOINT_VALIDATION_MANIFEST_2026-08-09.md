# P5 safety v2.2 — prompt-disjoint validation manifest

**Status:** `dry_run_non_executable`  
**API/model/pod calls made:** `0`  
**Internal SHA-256:** `133dc700edcd2ecc44999e0f183153035fccae0a774b3ce3484ad95095134736`  
**File SHA-256:** `e61a13d4efdb85602ef5414cda9efc990908d206aac71a7a856ea989835c35a2`

This non-executable manifest contains only the 01 harmful/benign prompt pairs across all six safety domains and all four checkpoint roles. Its 48 assignments require exactly 104 initial requests. The held-out 02 prompts have zero overlap and are bound separately by assignment and request-plan hashes; they are not executable from this manifest.

Projected validation cost at the conservative observed envelope is `$0.8958`; this is not a ceiling.

## Operational gate

- Atomic terminal persistence: at least 100%
- Network success: at least 98%
- Complete schema-plus-evidence parsing among successful responses: at least 98%
- No duplicate chunk keys and no guard breach
- Endpoint/effect values are not used in the gate

Held-out commitment: 48 assignments, 119 initial requests, assignment SHA `b1006f1734b45548d7ce1d7acec955bec77449b81861733fe1fe269c63188a4d`, request-plan SHA `086379a11f9a3183fc7238e6908a72624c2cc782589fe099b7644ca34aee35c7`.

This gate is an early-stop rule, not proof that the population parse rate is at least 98%.
