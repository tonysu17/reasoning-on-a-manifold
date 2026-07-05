#!/usr/bin/env python3
"""Safety check before batching the headline run: confirm GREEDY batched
generation produces byte-identical chains to batch=1 (so batching is a pure
throughput win with no batch-position confound). Tests a few prompts × a couple
arms with the real pooled L27 vectors. Prints IDENTICAL / DIVERGENCE per case."""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from src.config import model_tuple
from src.chain_gen import load_model
from src.steering import load_steering_vectors
from src.steered_inference import SteeredModel
from src.task_gen import load_tasks, stratified_eval_split

MAXTOK = 256          # short: if the first 256 greedy tokens match, the path is sound
N_PROMPTS = 4
BATCH = N_PROMPTS

model_id, short, dtype = model_tuple("1.5b")
model, tok = load_model(model_id, dtype=dtype, use_4bit=False, cache_dir=None)
vecs = load_steering_vectors(Path("results/steering_vectors/R1-1.5B__L27_pooled"))
tasks, _ = stratified_eval_split(load_tasks(Path("data/tasks_final.json")), 50)
prompts = [t["prompt"] for t in tasks[:N_PROMPTS]]

all_ok = True
for beh, arm, key in [("backtracking", "single", "single_direction"),
                      ("backtracking", "manifold_k3", ("manifold_projected", 3))]:
    if isinstance(key, tuple):
        vec = np.asarray(vecs[beh][key[0]][key[1]])
    else:
        vec = np.asarray(vecs[beh][key])
    layer = vecs[beh]["layer"]
    sm = SteeredModel(model, tok, vec, layer, alpha=1.0, mode="subtract", energy_scale=1.0)
    b1 = [sm.generate(p, max_new_tokens=MAXTOK, temperature=0.0)["chain"] for p in prompts]
    bb = [r["chain"] for r in sm.generate_batch(prompts, max_new_tokens=MAXTOK)]
    print(f"\n=== {beh} / {arm} (L{layer}) ===")
    for i in range(N_PROMPTS):
        same = b1[i] == bb[i]
        all_ok = all_ok and same
        tag = "IDENTICAL" if same else "DIVERGENCE"
        print(f"  prompt {i}: {tag}  (len b1={len(b1[i])} batched={len(bb[i])})")
        if not same:
            m = min(len(b1[i]), len(bb[i]))
            j = next((k for k in range(m) if b1[i][k] != bb[i][k]), m)
            print(f"    first diff @char {j}: {b1[i][max(0,j-25):j+25]!r}")
            print(f"                      vs : {bb[i][max(0,j-25):j+25]!r}")
# vanilla via a zero-vector α=0 SteeredModel. THE batching check is α=0 batch-1 vs
# α=0 batched (must be identical → batching is exact). The generate_chain comparison
# is informational only (a different wrapper; both are valid unsteered baselines).
from src.chain_gen import generate_chain
van = SteeredModel(model, tok, np.zeros(model.config.hidden_size, dtype=np.float32),
                   0, alpha=0.0, mode="subtract", energy_scale=0.0)
v1 = [van.generate(p, max_new_tokens=MAXTOK, temperature=0.0)["chain"] for p in prompts]
vb = [r["chain"] for r in van.generate_batch(prompts, max_new_tokens=MAXTOK)]
gc = [generate_chain(model, tok, p, MAXTOK, temperature=0.0)["chain"] for p in prompts]
print("\n=== vanilla α=0: batch-1 vs batched (THE batching check — must match) ===")
for i in range(N_PROMPTS):
    same = v1[i] == vb[i]
    all_ok = all_ok and same
    print(f"  prompt {i}: {'IDENTICAL' if same else 'DIVERGENCE'}  "
          f"(len b1={len(v1[i])} batched={len(vb[i])})")
print("=== vanilla: α=0 batched vs generate_chain (wrapper diff — informational) ===")
for i in range(N_PROMPTS):
    print(f"  prompt {i}: {'identical' if gc[i] == vb[i] else 'wrapper-diff'}  "
          f"(gc={len(gc[i])} batched={len(vb[i])})")

print("\n" + ("ALL IDENTICAL — batching is exact, safe to batch" if all_ok else
              "DIVERGENCE in a batching check — do NOT batch; investigate"))
