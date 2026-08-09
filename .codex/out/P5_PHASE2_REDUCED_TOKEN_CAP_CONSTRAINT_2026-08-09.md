# P5 / Phase-2 reduced-token-cap constraint

**Recorded:** 9 August 2026  
**Owner instruction:** future generation must be costed and frozen in the 2,048–4,096
new-token range rather than assuming 8,192.  
**Status:** binding planning constraint; exact cap not yet frozen; not authority to amend a
sealed preregistration or launch generation

## Immediate disposition

- The running P5 pipeline pilot remains at its frozen 4,096 cap. It is not changed mid-run.
- No powered P5 or Phase-2 generation may launch on an 8,192 assumption.
- Claude must pause before the one permitted Phase-2 generation until Tony selects an exact
  cap and, if required, seals a prospective amendment to the E8-compatible settings.
- All model arms within an admitted comparison use the same cap. No checkpoint-specific cap
  is permitted.
- Truncation remains an explicit endpoint and missing terminal evidence remains unresolved;
  a shorter cap is not treated as equivalent to natural completion.

## Live P5 pilot evidence for the choice

The planning snapshot contained 154 successful 4,096-capped generations, zero terminal
errors. Re-capping the observed decoded token counts gives:

| Candidate cap | Rows that would reach cap | Rate | Observed-token reduction versus 4,096 |
|---:|---:|---:|---:|
| 2,048 | 58 / 154 | 37.7% | 36.1% |
| 3,072 | 53 / 154 | 34.4% | 17.9% |
| 4,096 | 53 / 154 | 34.4% | 0% |

For the generic stratum specifically, the cap-exposure rates are 39/80 (48.8%) at 2,048
and 34/80 (42.5%) at both 3,072 and 4,096 in this snapshot. Harmful/benign safety rows have
19/74 (25.7%) exposure at 4,096 and 19/74 (25.7%) at 3,072; 2,048 raises this to 19/74 in
this particular snapshot as well. These are pipeline-planning quantities, not checkpoint
effects or thesis results.

The 3,072 option reduces token volume without reducing the number of exposed rows relative
to 4,096 because current chain lengths are strongly split between early EOS and the 4,096
ceiling. It still preserves more observed content in those exposed chains.

## Scientific consequence

A reduced cap changes the generated-response estimand. In particular:

- behaviour fractions describe the observed prefix up to the common cap;
- correctness and task completion may be unavailable for cap-hit rows;
- safety endpoints may be resolved only from a decisive stance already present in observed
  text; otherwise they remain unresolved;
- arm-specific truncation differences are a damage/missingness result, not a nuisance to
  discard;
- direct comparability with historical E8 8,192-capped generation is limited and must be
  stated explicitly if Phase 2 is amended.

## Freeze decision still required

Before powered or Phase-2 generation, freeze exactly one value in `{2048, 3072, 4096}` with:

1. the estimated generation budget at that cap;
2. the expected scoring-request budget after chunking;
3. the cap-hit and terminal-correctness handling rules;
4. the Phase-2 preregistration amendment status; and
5. the exact manifest/config hash.

