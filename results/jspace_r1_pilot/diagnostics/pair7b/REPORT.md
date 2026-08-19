# 7B matched-pair registered evaluation (sealed sheet E1–E5)

## E1_reduction_replicates
```json
{
 "distill_union": 0.09876543209876543,
 "base_union": 0.6419753086419753,
 "supported": true,
 "ratio": 0.15384615384615385,
 "comparison_1p5b": {
  "base": 0.4074,
  "distill": 0.1852,
  "ratio": 0.4545900834560629
 }
}
```
## E2_location_late_band
```json
{
 "math-7b-base": {
  "mid_max": 0.04938271604938271,
  "late_max": 0.49382716049382713
 },
 "r1-distill-7b": {
  "mid_max": 0.0,
  "late_max": 0.09876543209876543
 },
 "protocol_note": "The sealed sheet registered a directional expectation, not a numerical threshold. Report both continuous maxima; no Boolean operationalises 'confined to late' or 'approximately zero'."
}
```
## E3_format_delta
```json
{
 "math_7b_delta": 0.07407407407407407,
 "distill_7b_delta": 0.012345679012345678,
 "reference": {
  "math_1p5b": -0.056,
  "qwen3_1p7b": 0.037,
  "qwen2p5_7b_it": 0.09259259259259256
 },
 "protocol_note": "The sealed sheet registered competing directions, not categorical thresholds; report the continuous deltas."
}
```
## E4_association
```json
{
 "math_7b_union": 0.01020408163265306,
 "math_7b_p": 0.18081918081918083,
 "math_7b_qualifies": false,
 "distill_7b_union": 0.0,
 "general_7b_it_reference": 0.061224489795918366
}
```
## E5_selectivity
```json
{
 "common_eligible_items": 81,
 "base_only_eligible_items": [],
 "distill_only_eligible_items": [],
 "base_hit_items_common_eligible": 52,
 "distill_hit_items_common_eligible": 8,
 "overlap": 8,
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
 "gained_in_distill": [],
 "note": "common-eligible item names only; qualitative post-hoc reading; no inferential weight"
}
```
## typo_internal_control
```json
{
 "math-7b-base": {
  "union": 0.78125,
  "qualifies": true
 },
 "r1-distill-7b": {
  "union": 0.8125,
  "qualifies": true
 }
}
```
## protocol_amendment
```json
"results/prereg/JSPACE_7B_PAIR_AMENDMENT_2_2026-08-18.md"
```
## execution_run_id
```json
"jspace7b-20260818T075715Z-fa3a59ab"
```
