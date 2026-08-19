# 7B matched-pair registered evaluation — v2 reporting correction

Reporting/provenance correction over the preserved v1 report; all primary 7B values are unchanged. See REPORTING_ADDENDUM_2026-08-18.md.

## E1_reduction_replicates
```json
{
 "base_union": 0.6419753086419753,
 "comparison_1p5b": {
  "base": 0.4074074074074074,
  "distill": 0.18518518518518517,
  "ratio": 0.45454545454545453,
  "sources": {
   "base": "results/jspace_r1_pilot/diagnostics/d3/d3-qwen2.5-math-1.5b-base-2008de267117/report.json",
   "distill": "results/jspace_r1_pilot/diagnostics/d5/d5-c480b291f5aa/report.json (union_pass_at_25_bf16; validated Phase-1 arrays)"
  }
 },
 "distill_union": 0.09876543209876543,
 "ratio": 0.15384615384615385,
 "supported": true,
 "supported_meaning": "supported=true records only that the registered E1 directional inequality (distill any-layer union < base any-layer union) held. It is not a paired base-vs-distill model-difference test, not an effect-size threshold, and not a causal claim about distillation. Each cell's permutation p tests token labels within that cell only."
}
```
## E2_location_late_band
```json
{
 "math-7b-base": {
  "late_max": 0.49382716049382713,
  "layer_profile_pass_at_25": [
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.012345679012345678,
   0.04938271604938271,
   0.024691358024691357,
   0.012345679012345678,
   0.024691358024691357,
   0.012345679012345678,
   0.0,
   0.04938271604938271,
   0.07407407407407407,
   0.24691358024691357,
   0.32098765432098764,
   0.43209876543209874,
   0.49382716049382713,
   0.4567901234567901,
   0.4444444444444444
  ],
  "mid_max": 0.04938271604938271
 },
 "protocol_note": "The sealed sheet registered a directional expectation, not a numerical threshold. Report both continuous maxima and the full per-layer profiles; no Boolean operationalises 'confined to late' or 'approximately zero'.",
 "r1-distill-7b": {
  "late_max": 0.09876543209876543,
  "layer_profile_pass_at_25": [
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.012345679012345678,
   0.08641975308641975,
   0.09876543209876543,
   0.09876543209876543,
   0.08641975308641975
  ],
  "mid_max": 0.0
 }
}
```
## E3_format_delta
```json
{
 "distill_7b_delta": 0.012345679012345678,
 "math_7b_delta": 0.07407407407407407,
 "protocol_note": "The sealed sheet registered competing directions, not categorical thresholds; report the continuous deltas.",
 "reference": {
  "math_1p5b": -0.05555555555555558,
  "qwen2p5_7b_it": 0.09259259259259256,
  "qwen3_1p7b": 0.03703703703703698,
  "sources": {
   "math_1p5b": "results/jspace_r1_pilot/diagnostics/d3/d3-qwen2.5-math-1.5b-base-2008de267117/report.json",
   "qwen2p5_7b_it": "results/jspace_r1_pilot/diagnostics/d2/d2-qwen2.5-7b-it-f1b6cdfdff03/report.json",
   "qwen3_1p7b": "results/jspace_r1_pilot/diagnostics/d3/d3-qwen3-1.7b-3bd7b4c16291/report.json"
  }
 }
}
```
## E4_association
```json
{
 "distill_7b_union": 0.0,
 "general_7b_it_reference": 0.061224489795918366,
 "math_7b_p": 0.18081918081918083,
 "math_7b_qualifies": false,
 "math_7b_union": 0.01020408163265306
}
```
## E5_selectivity
```json
{
 "base_hit_items_common_eligible": 52,
 "base_only_eligible_items": [],
 "common_eligible_items": 81,
 "distill_hit_items_common_eligible": 8,
 "distill_only_eligible_items": [],
 "gained_in_distill": [],
 "lost_in_distill": [
  "atomic-26-symbol",
  "atomic-29-symbol",
  "atomic-79-symbol",
  "basketball-players",
  "birthstone-emerald-month",
  "chem-bones-Z",
  "chem-organic-Z",
  "chem-photosynthesis-Z",
  "christmas-season",
  "double-dice-faces",
  "etym-caesar-monthnum",
  "etym-frigg-position",
  "etym-janus-monthnum",
  "etym-saturn-position",
  "etym-wargod-month",
  "firstletter-paris-country",
  "func-filters-count",
  "func-pumps-chambers",
  "holiday-christmas-monthnum",
  "holiday-halloween-monthnum",
  "holiday-independence-monthnum",
  "holiday-valentines-monthnum",
  "inv-antarctica-opposite",
  "inv-balloon-opposite",
  "inv-roots-opposite",
  "inv-sunrise-opposite",
  "letterpos-carbon-symbol",
  "letterpos-nitrogen-symbol",
  "letterpos-oxygen-symbol",
  "letterpos-water-symbol",
  "mars-color",
  "month-1-godof",
  "month-3-godof",
  "month-7-namedafter",
  "planet-3-moons",
  "rhyme-door-doubled",
  "rhyme-fix-halved",
  "rhyme-hive-plusone",
  "rhyme-shoe-doubled",
  "rhyme-spoon-orbit",
  "roman-rings-olympic",
  "spider-legs",
  "succ-halloween-nextmonth",
  "topeka-west"
 ],
 "note": "common-eligible item names only; qualitative post-hoc reading; no inferential weight",
 "overlap": 8,
 "retained_in_distill": [
  "chem-atmosphere-Z",
  "dbl-altitude-antonym",
  "dbl-armistice-antonym",
  "dbl-obituary-antonym",
  "dual-stars-visible-opposite",
  "firstletter-halloween-month",
  "firstletter-valentines-month",
  "pred-valentines-prevmonth"
 ]
}
```
## provenance
```json
{
 "anchor_identity": {
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
  "run_uuid": "d2-qwen2.5-7b-it-f1b6cdfdff03"
 },
 "correction_type": "post-execution reporting/provenance correction; estimands unchanged; see REPORTING_ADDENDUM_2026-08-18.md",
 "evaluator": {
  "v1": {
   "path": "jspace_pair_report.py",
   "sha256": "8abb01802673600d66bc33b4e29d7536f7e809e90a23860fd7db4010311342e7"
  },
  "v2": {
   "git_commit": "61035880e3fc863367d1158654138dcc9c8992d4",
   "path": "jspace_pair_report_v2.py",
   "sha256": "9b52c72e35d63f1dba2f364b75685b57ba8edad0bb553d7c74854e09f7c32081"
  }
 },
 "execution_run_id": "jspace7b-20260818T075715Z-fa3a59ab",
 "execution_source_commit": "fa3a59ab3817b705ff8af3e7bbeae3d56202dbe1",
 "generated_utc": "2026-08-19T00:00:00Z",
 "model_revisions": {
  "math-7b-base": "b101308fe89651ea5ce025f25317fea6fc07e96e",
  "r1-distill-7b": "916b56a44061fd5cd7d6a8fb632557ed4f724f60"
 },
 "output_version": "v2",
 "run_uuids": {
  "math-7b-base": "d3-math-7b-base-1c070326af0b",
  "r1-distill-7b": "d3-r1-distill-7b-429801a55af0"
 },
 "schema_version": "rom-jspace-pair7b-report-v2",
 "sealed_documents": {
  "results/prereg/JSPACE_7B_PAIR_AMENDMENT_1_2026-08-18.md": "3b669e9604bb3443bcdc93690d0b7a8f1eb84e4bb4268c95e20363c334a14b0c",
  "results/prereg/JSPACE_7B_PAIR_AMENDMENT_2_2026-08-18.md": "11224d519c0735af62ae1919a3859e18d0e6e6edcfb12a060d8ccd8e69b23c41",
  "results/prereg/JSPACE_7B_PAIR_SHEET_2026-08-17.md": "88b24eba916bffe594d12d090eabc262d63bce734342dfcd1f9efdb69a76256b"
 },
 "v1_outputs_preserved": {
  "results/jspace_r1_pilot/diagnostics/pair7b/REPORT.md": "864f957ba75cd6854b1e20d3a0b7c86901e2052ddf2ab20c8ad6e05de593e42e",
  "results/jspace_r1_pilot/diagnostics/pair7b/report.json": "6bc2714a85b8a03a560c43b16c5ffeaaf5edc8fad8324ffe39878cb1273f3813"
 }
}
```
## typo_internal_control
```json
{
 "math-7b-base": {
  "qualifies": true,
  "union": 0.78125
 },
 "r1-distill-7b": {
  "qualifies": true,
  "union": 0.8125
 }
}
```
