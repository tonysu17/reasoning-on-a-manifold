#!/usr/bin/env python3
"""
Pilot length validation — run after Phase 2 pilot completes.

Decision rule:
  P95 < 6500 AND <=2/20 hit ceiling  → proceed to full Phase 2 at cap 8192
  P95 >= 6500 OR  3+/20 hit ceiling  → stop and report before any further action
"""
import json, sys
import numpy as np
from pathlib import Path

path = Path("data/chains_pilot.json")
if not path.exists():
    print("chains_pilot.json not found — run 00_pilot_gate.py --generate-chains first")
    sys.exit(1)

chains = json.load(open(path))
lengths = [c["n_tokens"] for c in chains]
hit_ceiling = [c for c in chains if c["n_tokens"] >= 8192]
terminated  = [c for c in chains if "</think>" in c["chain"]]

print(f"Chains                        : {len(chains)}/20")
print(f"Naturally terminated </think> : {len(terminated)}/20")
print(f"Hit 8192 ceiling              : {len(hit_ceiling)}/20")

print(f"\nLength distribution:")
print(f"  Mean   : {np.mean(lengths):.0f}")
print(f"  Median : {np.median(lengths):.0f}")
print(f"  P90    : {np.percentile(lengths, 90):.0f}")
print(f"  P95    : {np.percentile(lengths, 95):.0f}")
print(f"  Max    : {max(lengths)}")

if hit_ceiling:
    print(f"\nChains hitting ceiling:")
    for c in hit_ceiling:
        print(f"  {c['task_id']:10s} {c['category']:30s} {c['difficulty']:8s}  {c['n_tokens']} tok")

p95 = np.percentile(lengths, 95)
n_ceiling = len(hit_ceiling)

print(f"\n--- DECISION ---")
if p95 < 6500 and n_ceiling <= 2:
    print(f"PASS: P95={p95:.0f} < 6500 and {n_ceiling}/20 hit ceiling")
    mean_len = np.mean(lengths)
    phase3_cost = (mean_len * 2.3 / 1e6 * 1.25 + mean_len * 2.6 / 1e6 * 5.00) * 1000
    print(f"Proceed to full Phase 2 at cap 8192.")
    print(f"Updated Phase 3 cost estimate (1000 chains, Batch API): ~${phase3_cost:.2f}")
else:
    print(f"STOP: P95={p95:.0f}, {n_ceiling}/20 hit ceiling — report before proceeding")
    sys.exit(1)
