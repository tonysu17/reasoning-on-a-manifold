# P5 / Phase-2 cap-fork resolution — 2026-08-09

## Owner decision

Tony selected **Option A** in chat on 2026-08-09 and separately instructed
Claude to proceed on that basis.

## Operational consequence

- Phase-2 generation retains the sealed E8 maximum of **8,192 new tokens**.
- Phase-2 Sonnet annotation uses the sealed Amendment A4 paragraph-aligned
  window of approximately **3,000 tokens**.
- Full-chain fields needed for correctness, damage, length, repetition, and
  truncation remain available as specified by A4.
- Phase-2 does not require an A5 generation-cap amendment.
- P5 consumes `results/ph2/battery/base_vanilla_shared.json` read-only and
  never regenerates overlapping vanilla rows.
- Any reduced-cap P5 comparison is an explicitly declared analytical prefix
  (one value selected prospectively from 2,048, 3,072, or 4,096 tokens), not a
  second generation run and not a mutation of the shared artifact.
- P5 does not request the optional early stand-alone vanilla slice; it waits
  for the canonical Phase-2 battery export.

## Supersession

This owner decision supersedes the launch prohibition in
`P5_PHASE2_REDUCED_TOKEN_CAP_CONSTRAINT_2026-08-09.md`. That document remains a
historical record of the surfaced fork and pilot recapping calculations; it no
longer controls Phase-2 generation.

## Remaining independent gates

This decision does not authorize P5 scoring, the powered P5 study, Phase-2
compute or annotation spend, sample enlargement, optional LoRA retraining, or
steering rescoring.
