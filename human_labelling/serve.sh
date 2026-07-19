#!/usr/bin/env bash
# Serve the labelling apps over localhost so browser localStorage (autosave) works
# reliably. Open the printed URL, label, then hit "Export JSON".
cd "$(dirname "$0")"
PORT="${1:-8777}"
echo "Labelling apps:"
for f in H*.html; do echo "  http://127.0.0.1:$PORT/$f"; done
echo
python3 -m http.server "$PORT" --bind 127.0.0.1
