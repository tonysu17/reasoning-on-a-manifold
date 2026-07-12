#!/bin/zsh
# R0 restart-loop runner around 29_r0_entropy_ladder.py.
#
# Why: the MPS caching allocator accumulates freed blocks across variable-length
# chains within one process (2026-07-11 cascade: latency 67→670 s/chain, then
# 280 OOM failures). Stages are checkpointed and model load is ~10 s, so bounding
# each process to --limit items and restarting is the robust allocator reset;
# in-process torch.mps.empty_cache() per chain (patched same day) is the first
# line of defence, this loop is the backstop.
set -u
cd "$(dirname "$0")"

BASE_LIMIT=20     # 8k-token loop chains dominate; keep per-process footprint small
T06_LIMIT=60      # T06 chains are short (~1-3k tokens)

remaining() {
python3 - <<'EOF'
import json, pathlib
base = pathlib.Path('results/r0_entropy_ladder/R1-1.5B')
s = json.load(open(base / 'sample.json'))
d = base / 'ent_shards'
n_base = sum(1 for t in s['loop'] + s['clean'] if not (d / f'{t}.npz').exists())
rows = [r for r in json.load(open('results/eval/R1-1.5B__E9_1_T06/steering_results.json'))
        if r['method'] == 'vanilla']
done = set()
p = base / 't06_ent.json'
if p.exists():
    done = {r['task_id'] for r in json.load(open(p))}
n_t06 = sum(1 for r in rows if r['task_id'] not in done)
print(n_base + n_t06)
EOF
}

prev=999999
for i in {1..60}; do
  rem=$(remaining)
  echo "R0-RUNNER round $i: $rem items remaining"
  if [[ $rem -eq 0 ]]; then
    break
  fi
  if [[ $rem -ge $prev ]]; then
    echo "R0-RUNNER: no progress last round ($prev -> $rem) — stopping; persistent failures are excluded+counted downstream"
    break
  fi
  prev=$rem
  python3 -u 29_r0_entropy_ladder.py --stage extract-base --limit $BASE_LIMIT || true
  python3 -u 29_r0_entropy_ladder.py --stage extract-t06 --limit $T06_LIMIT || true
done

python3 -u 29_r0_entropy_ladder.py --stage analyse
echo "R0-RUNNER: DONE"
