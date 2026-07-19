# Human labelling tasks — the annotator-validity anchors

Three blind labelling tasks. Every LLM-annotator layer in this project has been
checked against *other LLMs* and never against a human; these close that gap.
Each task is a self-contained HTML app — no dependencies, autosaves as you go.

## Run

```bash
./human_labelling/serve.sh          # then open the printed URLs
```

Label → **Export JSON** (lands in `~/Downloads`) → score:

```bash
python3 human_labelling/score_human.py ~/Downloads/H*_human.json
```

Serving over localhost rather than opening the files directly is deliberate:
`file://` origins restrict `localStorage` in some browsers, and losing an hour of
labelling to that would be miserable. Progress is keyed per task, so you can stop
and resume, and label the three tasks in any order across several sittings.

## The tasks

| | What you label | n | Time | What it decides |
|---|---|---|---|---|
| **H1** | gpt-oss safety sentences, 4 DSR labels (multi-select) | 150 | ~45 min | Whether the P1 kill criterion actually fires |
| **H2** | R1-1.5B reasoning sentences, 6 behaviour labels (single) | 150 | ~40 min | Whether the geometry/steering/spillover spans are valid |
| **H3** | R3 solution chains, primary strategy (single) | 56 | ~50 min | Whether the headline R3 result survives its keyword classifier |

**H1 — gpt-oss deliberative-safety gold anchor.** P1 ran and the three LLM judges
disagree badly: `decision` κ = 0.22 and `adjudication` κ = 0.14, both under the sealed
0.4 floor, and `decision` failing is the pre-registered kill criterion for the whole
gpt-oss H1 fingerprint campaign. But LLM-vs-LLM κ cannot tell you *why*. The judges
differ ~4× in how much text they mark as `decision` (Nova 0.060 of chars vs Qwen 0.014),
which is the signature of miscalibrated span extent rather than an unlearnable
distinction. A human anchor separates the two: if you agree well with at least one
judge, the schema is fine and the judges need calibrating; if you agree with none of
them, the schema is genuinely not identifiable and the kill stands.

**H2 — behaviour-span validity.** This is the largest standing exposure in the thesis.
The geometry, steering, and spillover chapters all rest on sentence spans labelled with
the six-label scheme, and the three annotators disagree by up to 3× in label frequency
(`example-testing`: 5,831 / 11,341 / 3,672 spans). κ was only ever computed
annotator-vs-annotator; no human has ever checked whether the labels mean what they say.
The four load-bearing behaviours are `backtracking`, `uncertainty-estimation`,
`example-testing`, `adding-knowledge` — the scorer reports each separately, so a
failure localises to a behaviour instead of condemning the whole scheme.

**H3 — R3 strategy replication.** The R3 headline (steering retains more value than
temperature at matched strategy entropy, +0.016 CI [0.013, 0.019]) is labelled by a
frozen keyword matcher, declared a range-finder only (CF-T), and is recorded as
PROVISIONAL for exactly that reason. The scorer reports agreement *per cell*: if the
classifier is systematically better on `pump` than on `vanilla` arms, the +0.016 gap is
partly a labelling artefact rather than a real effect, and that shows up as a large
`cell_skew_pump_minus_thermostat`.

## Design notes (they matter for how the numbers may be reported)

- **Blind.** No LLM label is shown in the UI. The held-back labels live in `_key.json`,
  which the scorer reads and you should not.
- **Stratified, not random.** Positives are rare (DSR labels cover 1–6% of characters),
  so a uniform sample would contain almost no signal. H1 and H2 deliberately over-sample
  contested items — H1 is 70 contested / 45 unanimous-positive / 35 negative; H2 is 60
  three-way-disagreement / 55 two-vs-one / 35 unanimous. **κ on the enriched sample is a
  lower bound on corpus κ, not an estimate of it**, because contested items are
  over-represented. The scorer also reports κ within each stratum so the enrichment is
  visible and the sample can be re-weighted if a corpus-level number is needed.
- **One sentence per chain in H2**, so items are independent rather than 150 correlated
  draws from a handful of chains.
- **Verbatim definitions.** The guidelines shown in each app are copied from the prompts
  the LLM annotators received, so disagreement measures validity rather than the two of
  you having been told different things.
- **"Genuinely ambiguous" flag** (`u`) on every item. Use it freely — a high ambiguity
  rate is itself a finding about the schema, and the scorer counts it.

## Files

| File | Role |
|---|---|
| `build_tasks.py` | Regenerates the apps (seed 20260718, recorded in `_manifest.json`) |
| `H1/H2/H3_*.html` | The labelling apps |
| `_key.json` | Held-back LLM labels + sampling strata |
| `_manifest.json` | Sample sizes, strata, arm/cell balance |
| `score_human.py` | Human-vs-LLM κ, per-label gates, verdicts |
| `serve.sh` | Local server |

Re-running `build_tasks.py` reshuffles the sample and invalidates in-progress labelling,
so don't, once you've started.
