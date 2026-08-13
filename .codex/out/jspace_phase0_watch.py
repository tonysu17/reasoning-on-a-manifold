#!/usr/bin/env python3
"""Detached Mac-side watcher for the J-space Phase-0 RunPod job."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--remote-root", default="/workspace/jspace-phase0")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    ssh = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
        "-o", "StrictHostKeyChecking=accept-new", "-i", str(args.key),
        "-p", str(args.port), f"root@{args.host}",
    ]
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    while True:
        command = f'''root={args.remote_root!r}; status=$(cat "$root/JSPACE_PHASE0_STATUS" 2>/dev/null || echo MISSING); pid=$(cat "$root/JSPACE_PHASE0_DRIVER_PID" 2>/dev/null || true); if test -f "$root/JSPACE_PHASE0_DONE.marker"; then marker=DONE; elif test -f "$root/JSPACE_PHASE0_FAILED.marker"; then marker=FAILED; else marker=NONE; fi; if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then driver=RUNNING; else driver=ABSENT; fi; printf '%s|%s|%s\n' "$status" "$marker" "$driver"'''
        completed = subprocess.run(ssh + [command], text=True, capture_output=True)
        if completed.returncode:
            failures += 1
            event = {"utc": utc(), "event": "ssh_error", "count": failures,
                     "stderr": completed.stderr.strip()}
        else:
            failures = 0
            fields = completed.stdout.strip().split("|")
            if len(fields) != 3:
                event = {"utc": utc(), "event": "malformed_snapshot",
                         "stdout": completed.stdout}
            else:
                status, marker, driver = fields
                event = {"utc": utc(), "event": "snapshot", "status": status,
                         "marker": marker, "driver": driver}
        with args.audit.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        marker = event.get("marker")
        driver = event.get("driver")
        if marker in {"DONE", "FAILED"}:
            return 0 if marker == "DONE" else 1
        if driver == "ABSENT" and event.get("status", "MISSING") != "MISSING":
            with args.audit.open("a") as handle:
                handle.write(json.dumps({"utc": utc(), "event": "process_death",
                                         "status": event.get("status")}, sort_keys=True) + "\n")
            return 2
        if failures >= 10:
            return 3
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
