#!/usr/bin/env bash
# Wait for a pod job to finish, sync EVERY artifact locally, verify byte-for-byte,
# and only then terminate the pod.
#
# HARD SAFETY CONTRACT — the pod is terminated ONLY if all of these hold:
#   1. the job's completion marker is present in the remote log;
#   2. every remote artifact transferred and its local sha256 matches the remote sha256;
#   3. the expected result bundles exist locally and their report.json files parse;
#   4. a terminate mechanism is available (RUNPOD_API_KEY in the caller's environment).
# Any failure => NO TERMINATION, loud message, exit non-zero. Data safety beats billing.
#
# Usage:
#   RUNPOD_API_KEY=... scripts/pod_sync_and_terminate.sh --host H --port P --pod-id ID [--dry-run]
# Without RUNPOD_API_KEY the script still syncs + verifies, then tells you to stop the pod
# manually (it never terminates blind).
set -uo pipefail

HOST=""; PORT=""; POD_ID=""; DRY_RUN=0
KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_LOG="${REMOTE_LOG:-/root/pair7b.log}"
DONE_MARKER="${DONE_MARKER:-ALL 7B PAIR STAGES COMPLETE}"
REMOTE_REPO="${REMOTE_REPO:-/root/reasoning-on-manifold}"
REMOTE_LENSES="${REMOTE_LENSES:-/root/lenses}"
POLL_SECONDS="${POLL_SECONDS:-180}"
MAX_HOURS="${MAX_HOURS:-12}"

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --pod-id) POD_ID="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
[ -n "$HOST" ] && [ -n "$PORT" ] || { echo "need --host and --port"; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_LOG="$REPO_ROOT/logs/pod_sync_${STAMP}.log"
STAGE="$REPO_ROOT/logs/pod_sync_${STAMP}_staging"
mkdir -p "$(dirname "$LOCAL_LOG")" "$STAGE"
SSH="ssh -o ConnectTimeout=15 -o BatchMode=yes -p $PORT -i $KEY root@$HOST"

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOCAL_LOG"; }
fail() { say "ABORT: $*"; say "POD LEFT RUNNING (by design). Investigate, then stop it yourself."; exit 1; }

say "watching $HOST:$PORT for marker '$DONE_MARKER' in $REMOTE_LOG"

# ---------- 1. wait for completion ----------
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
while :; do
  state=$($SSH "grep -q '$DONE_MARKER' '$REMOTE_LOG' 2>/dev/null && echo DONE; \
                pgrep -f 'bash /root/reasoning-on-manifold/runpod_jspace_7b_pair.sh' >/dev/null && echo ALIVE" 2>/dev/null)
  case "$state" in
    *DONE*) say "completion marker found"; break;;
  esac
  if ! printf '%s' "$state" | grep -q ALIVE; then
    sleep 20
    state2=$($SSH "grep -q '$DONE_MARKER' '$REMOTE_LOG' 2>/dev/null && echo DONE" 2>/dev/null)
    printf '%s' "$state2" | grep -q DONE && { say "completion marker found (late)"; break; }
    fail "job process is gone and no completion marker — probable crash. Remote tail:
$($SSH "tail -15 '$REMOTE_LOG' 2>/dev/null" || true)"
  fi
  [ "$(date +%s)" -lt "$deadline" ] || fail "exceeded MAX_HOURS=$MAX_HOURS"
  sleep "$POLL_SECONDS"
done

# ---------- 2. remote manifest of everything we intend to keep ----------
say "building remote checksum manifest"
$SSH "cd / && { find '$REMOTE_REPO/results/jspace_r1_pilot/diagnostics' -type f 2>/dev/null; \
                find '$REMOTE_LENSES' -type f 2>/dev/null; \
                ls '$REMOTE_LOG' 2>/dev/null; } | sort | xargs -r sha256sum" \
  > "$STAGE/remote_manifest.txt" 2>>"$LOCAL_LOG" || fail "could not build remote manifest"
n_remote=$(wc -l < "$STAGE/remote_manifest.txt" | tr -d ' ')
[ "$n_remote" -gt 0 ] || fail "remote manifest is empty — nothing to sync"
say "remote files to sync: $n_remote"

# ---------- 3. sync ----------
say "syncing to $STAGE"
rsync -az --info=stats2 -e "ssh -p $PORT -i $KEY -o BatchMode=yes" \
  "root@$HOST:$REMOTE_REPO/results/jspace_r1_pilot/diagnostics/" "$STAGE/diagnostics/" >>"$LOCAL_LOG" 2>&1 \
  || fail "rsync of diagnostics failed"
rsync -az -e "ssh -p $PORT -i $KEY -o BatchMode=yes" \
  "root@$HOST:$REMOTE_LENSES/" "$STAGE/lenses/" >>"$LOCAL_LOG" 2>&1 || fail "rsync of lenses failed"
scp -q -P "$PORT" -i "$KEY" "root@$HOST:$REMOTE_LOG" "$STAGE/" >>"$LOCAL_LOG" 2>&1 || fail "log copy failed"

# ---------- 4. verify every checksum ----------
say "verifying checksums"
missing=0; mismatched=0
while read -r sum path; do
  case "$path" in
    "$REMOTE_REPO"/results/jspace_r1_pilot/diagnostics/*) local_path="$STAGE/diagnostics/${path#"$REMOTE_REPO"/results/jspace_r1_pilot/diagnostics/}";;
    "$REMOTE_LENSES"/*) local_path="$STAGE/lenses/${path#"$REMOTE_LENSES"/}";;
    "$REMOTE_LOG") local_path="$STAGE/$(basename "$REMOTE_LOG")";;
    *) continue;;
  esac
  if [ ! -f "$local_path" ]; then missing=$((missing+1)); say "  MISSING: $path"; continue; fi
  got=$(shasum -a 256 "$local_path" | awk '{print $1}')
  [ "$got" = "$sum" ] || { mismatched=$((mismatched+1)); say "  MISMATCH: $path"; }
done < "$STAGE/remote_manifest.txt"
[ "$missing" -eq 0 ] || fail "$missing remote file(s) did not arrive"
[ "$mismatched" -eq 0 ] || fail "$mismatched file(s) failed checksum"
say "checksums OK for all $n_remote files"

# ---------- 5. structural verification ----------
say "verifying expected result bundles"
python3 - "$STAGE" <<'PY' >>"$LOCAL_LOG" 2>&1 || fail "structural verification failed (see $LOCAL_LOG)"
import glob, json, os, sys
stage = sys.argv[1]
ok = True
for label in ("math-7b-base", "r1-distill-7b"):
    hits = glob.glob(os.path.join(stage, "diagnostics", "d3", f"*{label}*", "report.json"))
    if not hits:
        print(f"FAIL: no report.json for {label}"); ok = False; continue
    d = json.load(open(hits[-1]))
    suites = d.get("results", {})
    need = {"lens-eval-association", "lens-eval-typo", "lens-eval-multihop"}
    if not need <= set(suites):
        print(f"FAIL: {label} missing suites {need - set(suites)}"); ok = False; continue
    mh = suites["lens-eval-multihop"]["any_layer_union_pass_at_25"]
    print(f"OK: {label} multihop union={mh:.4f}")
    for s in need:
        p = os.path.join(os.path.dirname(hits[-1]), f"top25__{s.replace('-', '_')}.npz")
        if not os.path.exists(p):
            print(f"FAIL: {label} missing {os.path.basename(p)}"); ok = False
sys.exit(0 if ok else 1)
PY
say "structural verification passed"

# ---------- 6. install into the repo ----------
say "installing artifacts into repo"
mkdir -p "$REPO_ROOT/results/jspace_r1_pilot/diagnostics" "$REPO_ROOT/data/jlens_local"
rsync -a "$STAGE/diagnostics/" "$REPO_ROOT/results/jspace_r1_pilot/diagnostics/" >>"$LOCAL_LOG" 2>&1 \
  || fail "install of diagnostics into repo failed"
rsync -a "$STAGE/lenses/" "$REPO_ROOT/data/jlens_local/" >>"$LOCAL_LOG" 2>&1 || fail "install of lenses failed"
cp "$STAGE/$(basename "$REMOTE_LOG")" "$REPO_ROOT/results/jspace_r1_pilot/diagnostics/pod_logs/" 2>/dev/null || true
say "artifacts installed; staging copy retained at $STAGE"

# ---------- 7. terminate (only now) ----------
if [ "$DRY_RUN" = "1" ]; then
  say "DRY RUN: all gates passed; NOT terminating."; exit 0
fi

# Credential handling: the key is read from a file the OPERATOR creates
# (default ~/.runpod_env, mode 600). It is never passed on a command line,
# never placed in a URL, and never echoed to any log.
KEY_FILE="${RUNPOD_ENV_FILE:-$HOME/.runpod_env}"
if [ -z "${RUNPOD_API_KEY:-}" ] && [ -f "$KEY_FILE" ]; then
  # shellcheck disable=SC1090
  set -a; . "$KEY_FILE"; set +a
fi
if [ -z "${RUNPOD_API_KEY:-}" ]; then
  say "DATA IS SAFE LOCALLY AND VERIFIED, but no API key found ($KEY_FILE absent/empty)."
  say "  POD LEFT RUNNING — stop it in the RunPod console."
  exit 0
fi

api() {  # api <graphql-query-json>  -- Bearer header, key never in the URL
  curl -s --max-time 30 https://api.runpod.io/graphql \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H 'Content-Type: application/json' -d "$1"
}

if [ -z "$POD_ID" ]; then
  say "no --pod-id given; discovering pod by public IP/port"
  pods=$(api '{"query":"query { myself { pods { id name desiredStatus runtime { ports { ip publicPort } } } } }"}')
  POD_ID=$(printf '%s' "$pods" | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
for p in (d.get('data',{}).get('myself',{}) or {}).get('pods',[]) or []:
    rt=p.get('runtime') or {}
    for prt in rt.get('ports') or []:
        if prt.get('ip')=='$HOST' and str(prt.get('publicPort'))=='$PORT':
            print(p['id']); sys.exit(0)
" 2>/dev/null)
  [ -n "$POD_ID" ] || { say "could not discover pod id — DATA IS SAFE; stop the pod manually."; exit 0; }
  say "discovered pod id: $POD_ID"
fi

say "all gates passed — terminating pod $POD_ID"
resp=$(api "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$POD_ID\\\"}) { id } }\"}")
say "terminate response: $resp"
if printf '%s' "$resp" | grep -qi '"errors"'; then
  say "TERMINATION FAILED — data is safe locally; stop the pod manually in the console."
  exit 1
fi
sleep 20
if $SSH 'true' 2>/dev/null; then
  say "WARNING: pod still reachable after terminate call — verify in the console."
  exit 1
fi
say "pod unreachable (expected after termination). Billing stopped. Done."
exit 0
