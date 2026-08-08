# Exact-vector duplicate provenance audit

- Status: `COMPLETE`
- Root-cause classification: `same_extraction_window_reused_across_annotation_aliases`
- Raw rows checked: `37851`
- Rows after occurrence collapse: `37851`
- Repeated-occurrence groups: `0`
- Exact-vector duplicate groups: `313`
- Cross-label groups: `56`
- Cross-chain groups: `0`
- Zero-vector groups: `0`
- Constant-vector groups: `0`
- Same-location multilabel groups: `0`
- Same-window multilabel groups: `56`

## Classifications

- `same_extraction_window_multilabel_annotation_alias`: `56`
- `within_label_exact_match`: `257`

## Remedy feasibility

- Proposed rule: `drop_all_members_of_same_extraction_window_cross_label_groups`
- Applicable groups: `56`
- Unsupported cross-label groups: `0`
- Rows removed by proposed rule: `136`
- Target chains before/after: `705` / `705`
- Target chains lost: `0`
- All chains lost: `0`
- Mixed-label chain fraction after: `0.962739`
- Sample minimum met: `TRUE`
- Mixed-chain gate met: `TRUE`
- Positive: `TRUE`

## Boundaries

- Vector values are excluded; only cryptographic identities and occurrence provenance are recorded.
- No estimator, permutation, pilot retry, data modification, GPU, or API operation was performed.
- Group-level occurrence details are in the companion JSON.
