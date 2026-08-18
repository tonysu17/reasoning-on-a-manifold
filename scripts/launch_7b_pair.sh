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
SOURCE_COMMIT="$(git rev-parse HEAD)"
[ -z "$(git status --porcelain --untracked-files=normal)" ] || {
  echo "FATAL: local analysis repo is dirty; commit the exact payload first"; exit 1;
}
RUN_ID="jspace7b-$(date -u +%Y%m%dT%H%M%SZ)-${SOURCE_COMMIT:0:8}"
SSH="ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -p $PORT -i $KEY root@$HOST"

echo "== preflight =="
$SSH 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -BG / | tail -1; python --version 2>&1 | head -1; command -v setsid tar sha256sum'
gpu_count=$($SSH "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -dc '0-9'")
[ "$gpu_count" -eq 1 ] || { echo "FATAL: expected exactly one GPU, got $gpu_count"; exit 1; }
gpu_name=$($SSH "nvidia-smi --query-gpu=name --format=csv,noheader | head -1")
gpu_mib=$($SSH "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9'")
case "$gpu_name" in
  *L40S*|*A6000*) ;;
  *) echo "FATAL: GPU '$gpu_name' is not the sealed L40S/A6000 class"; exit 1;;
esac
[ "$gpu_mib" -ge 45000 ] || { echo "FATAL: GPU memory ${gpu_mib}MiB < 45000MiB"; exit 1; }
total_gb=$($SSH "df -BG --output=size / | tail -1 | tr -dc '0-9'")
free_gb=$($SSH "df -BG --output=avail / | tail -1 | tr -dc '0-9'")
[ "$total_gb" -ge 100 ] || { echo "FATAL: container allocation ${total_gb}G < 100G required"; exit 1; }
[ "$free_gb" -ge 60 ] || { echo "FATAL: container disk ${free_gb}G < 60G required"; exit 1; }
python_version=$($SSH "python -c 'import sys; print(str(sys.version_info.major)+\".\"+str(sys.version_info.minor))'")
[ "$python_version" = "3.11" ] || { echo "FATAL: Python $python_version; expected 3.11"; exit 1; }
stale=$($SSH 'for p in /root/reasoning-on-manifold /root/lenses /root/hf /root/venv-jlens /root/pair7b.log /root/bootstrap.log /root/boot.sh /root/pair7b.pid /root/pair7b.status /root/jspace_7b_pair_results.tar.gz; do [ -e "$p" ] && echo "$p"; done; true')
[ -z "$stale" ] || { echo "FATAL: pod is not fresh; refusing stale state:"; echo "$stale"; exit 1; }

echo "== pushing payload (code + fit manifest only; no repo clone) =="
tar czf - \
  jspace_diag_common.py jspace_phase1_scoring.py jspace_d2d3_readout.py \
  jspace_lens_fit.py runpod_jspace_7b_pair.sh \
  results/prereg/jspace_7b_requirements_lock.txt \
  results/prereg/jspace_r1_fit_manifest.json \
  results/prereg/JSPACE_7B_PAIR_AMENDMENT_1_2026-08-18.md \
  results/prereg/JSPACE_7B_PAIR_AMENDMENT_2_2026-08-18.md \
  results/prereg/JSPACE_7B_PAIR_SHEET_2026-08-17.md \
| $SSH 'mkdir -p /root/reasoning-on-manifold && cd /root/reasoning-on-manifold && tar xzf - 2>/dev/null; du -sh /root/reasoning-on-manifold'
$SSH "printf '%s\n' '$SOURCE_COMMIT' > /root/reasoning-on-manifold/.source_commit; printf '%s\n' '$RUN_ID' > /root/reasoning-on-manifold/.run_id; cd /root/reasoning-on-manifold; find . -type f ! -name PAYLOAD_MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > /root/PAYLOAD_MANIFEST.sha256; mv /root/PAYLOAD_MANIFEST.sha256 ."

echo "== launching detached (setsid; survives ssh disconnect) =="
$SSH 'cat > /root/boot.sh <<"EOF"
#!/usr/bin/env bash
exec >> /root/bootstrap.log 2>&1
export ROM_GIT_COMMIT="$(cat /root/reasoning-on-manifold/.source_commit)"
export ROM_RUN_ID="$(cat /root/reasoning-on-manifold/.run_id)"
export ROM_PROTOCOL_AMENDMENT="results/prereg/JSPACE_7B_PAIR_AMENDMENT_2_2026-08-18.md"
echo "[$(date -u +%H:%M:%S)] boot: payload pre-staged, starting job"
exec bash /root/reasoning-on-manifold/runpod_jspace_7b_pair.sh
EOF
chmod +x /root/boot.sh
setsid nohup /root/boot.sh < /dev/null > /dev/null 2>&1 &
echo $! > /root/pair7b.pid
disown
echo launched'

echo "== verifying (45s) =="
sleep 45
$SSH 'tail -6 /root/pair7b.log 2>/dev/null; echo "--- detached process (PPID 1 == good) ---"; pid=$(cat /root/pair7b.pid); kill -0 "$pid"; ps -o pid,ppid,etime,args -p "$pid"; [ "$(ps -o ppid= -p "$pid" | tr -d " ")" = 1 ]'
echo
echo "Run ID:   $RUN_ID"
echo "Commit:   $SOURCE_COMMIT"
echo "Launched. Watch:  ssh -p $PORT -i $KEY root@$HOST 'tail -f /root/pair7b.log'"
echo "When done:        scripts/pod_sync_and_terminate.sh --host $HOST --port $PORT"
