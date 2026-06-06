# M4.5 Completion Report — chain_gen.py temperature + seed support

**Branch**: `cbs/chain-gen-temperature`
**Date**: 2026-05-27
**Synthesis-plan reference**: §M4.5
**Status**: COMPLETE

## What changed

### `src/chain_gen.py`

* New helper `_seed_torch(seed)` — `torch.manual_seed` + CUDA RNG seed.
* `generate_chain(...)` now takes a `seed: int = 0` parameter; calls
  `_seed_torch(seed)` before `model.generate`. Returns `seed` and
  `temperature` fields on the chain record.
* `generate_chains(...)` takes `seed: int = 0` and `dedup_keys: tuple =
  ("task_id",)`. Multi-seed runs pass `dedup_keys=("task_id", "seed")`
  so resume logic does not deduplicate across seeds.

### `02_generate_chains.py`

New CLI flags per synthesis §M4.5:

```
--temperature TEMPERATURE   (default 0.0)
--seeds SEEDS               (default "0", comma-separated)
--tasks-subset PATH         (JSON list of task_ids)
```

Behaviour:

* Single seed (default): writes `data/chains_{model}.json` (legacy path).
* Multiple seeds: writes `data/chains_{model}_multiseed.json` with a
  `seed` field per chain. Dedup keys = `("task_id", "seed")`.
* `--tasks-subset` filters to a curated list (e.g. the 100 selected tasks
  for M4 matched-pair re-generation).

## Validation

`src/cbs/tests/test_chain_gen_temperature.py` — 5 tests:

* `_seed_torch` is deterministic (same seed → identical `torch.randn`).
* `_seed_torch` survives int-cast from string (argparse-style input).
* `generate_chain` calls `_seed_torch(seed)` (verified via mocked
  fake model + tokenizer; post-generation RNG state matches a direct
  `_seed_torch(seed)` consumption).
* Two seeds produce different post-generation RNG states (the basis on
  which downstream sampling diverges).
* The runner helper `_parse_seeds` accepts comma-separated, whitespace-
  tolerant input.

Full suite (across the build): `79 passed, 4 skipped`.

## How to use post-M4.5 for the multi-seed re-gen (M4 prerequisite)

```bash
# Generate 100-task × 20-seed multi-seed corpus at T=0.7.
python 02_generate_chains.py \
    --model 1.5b --temperature 0.7 \
    --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --tasks-subset data/m4_100_task_ids.json \
    --max-tokens 16384

# Output: data/chains_R1-1.5B_multiseed.json
```

100 tasks × 20 seeds = 2000 chains, ~10 cluster GPU-hours per synthesis
§M4.6. Phase 3 re-annotation follows on the multi-seed file.

## Next milestone

M5 — confirm scope with Tony, then implement ablation (if approved).
