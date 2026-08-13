#!/bin/bash
# Keep the Mac awake and run the fail-closed Phase-2 completion check until the
# exact pod termination has been accepted and recorded locally.
set -u

REPO="/Users/tonysu/Documents/Reasoning on a Manifold/reasoning-on-manifold"
RECORD="$REPO/.codex/out/PH2_POD_TERMINATION_2026-08-09.json"

cd "$REPO" || exit 1
while [[ ! -f "$RECORD" ]]; do
  python3 .codex/out/ph2_verified_pull_terminate.py --check
  [[ -f "$RECORD" ]] && exit 0
  sleep 120
done
