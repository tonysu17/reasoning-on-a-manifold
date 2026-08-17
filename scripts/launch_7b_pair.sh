#!/usr/bin/env bash
# One-command launcher for the sealed 7B J-lens pair on a fresh RunPod pod.
#
# Why this exists: cloning the repo on the pod stalled twice (2026-08-17) —
# ~1 GB of committed results over a slow datacenter link. The pod only needs
# ~840 KB of code + the fit manifest, so we push a tar payload over ssh
# instead (instant), then launch fully detached via setsid so the job
# survives the ssh session closing (a non-detached launch was SIGHUP-killed
# the same evening and silently ran nothing).
#
# Usage:  scripts/launch_7b_pair.sh --host H --port P [--key ~/.ssh/id_ed25519]
# Then:   RUNPOD_API_KEY via ~/.runpod_env (see scripts/pod_sync_and_terminate.sh)
#         scripts/pod_sync_and_terminate.sh --host H --port P
set -euo pipefail

HOST=""; PORT=""; KEY="$HOME/.ssh/id_ed25519"
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --key)  KEY="$2";  shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
[ -n "$HOST" ] && [ -n "$PORT" ] || { echo "need --host and --port"; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
SSH="ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -p $PORT -i $KEY root@$HOST"

echo "== preflight =="
$SSH 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -BG / | tail -1; python --version 2>&1 | head -1'
free_gb=$($SSH "df -BG --output=avail / | tail -1 | tr -dc '0-9'")
[ "$free_gb" -ge 60 ] || { echo "FATAL: container disk ${free_gb}G < 60G required"; exit 1; }

echo "== pushing payload (code + fit manifest only; no repo clone) =="
tar czf - \
  jspace_diag_common.py jspace_phase1_scoring.py jspace_d2d3_readout.py \
  jspace_lens_fit.py runpod_jspace_7b_pair.sh \
  results/prereg/jspace_r1_fit_manifest.json \
  results/prereg/JSPACE_7B_PAIR_SHEET_2026-08-17.md \
| $SSH 'rm -rf /root/reasoning-on-manifold && mkdir -p /root/reasoning-on-manifold && cd /root/reasoning-on-manifold && tar xzf - 2>/dev/null; du -sh /root/reasoning-on-manifold'

echo "== launching detached (setsid; survives ssh disconnect) =="
$SSH 'cat > /root/boot.sh <<"EOF"
#!/usr/bin/env bash
exec >> /root/bootstrap.log 2>&1
echo "[$(date -u +%H:%M:%S)] boot: payload pre-staged, starting job"
exec bash /root/reasoning-on-manifold/runpod_jspace_7b_pair.sh
EOF
chmod +x /root/boot.sh
setsid nohup /root/boot.sh < /dev/null > /dev/null 2>&1 &
disown
echo launched'

echo "== verifying (45s) =="
sleep 45
$SSH 'tail -4 /root/pair7b.log 2>/dev/null; echo "--- detached procs (PPID 1 == good) ---"; ps -eo pid,ppid,etime,args | awk "/[r]unpod_jspace_7b_pair/ {print}"'
echo
echo "Launched. Watch:  ssh -p $PORT -i $KEY root@$HOST 'tail -f /root/pair7b.log'"
echo "When done:        scripts/pod_sync_and_terminate.sh --host $HOST --port $PORT"
