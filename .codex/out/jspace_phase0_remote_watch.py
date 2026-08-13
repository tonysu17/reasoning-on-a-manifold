#!/usr/bin/env python3
"""Pod-resident status watcher for the detached J-space Phase-0 driver."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def append(path: Path, payload: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--prefix", default="JSPACE_PHASE0")
    args = parser.parse_args()
    root = args.root.resolve()

    while True:
        status_path = root / f"{args.prefix}_STATUS"
        pid_path = root / f"{args.prefix}_DRIVER_PID"
        status = status_path.read_text().strip() if status_path.is_file() else "MISSING"
        try:
            pid = int(pid_path.read_text().strip())
        except (OSError, ValueError):
            pid = 0
        marker = (
            "DONE" if (root / f"{args.prefix}_DONE.marker").is_file()
            else "FAILED" if (root / f"{args.prefix}_FAILED.marker").is_file()
            else "NONE"
        )
        driver = "RUNNING" if process_exists(pid) else "ABSENT"
        append(args.audit, {"utc": utc(), "event": "snapshot", "status": status,
                            "marker": marker, "driver": driver, "driver_pid": pid})
        if marker == "DONE":
            return 0
        if marker == "FAILED":
            return 1
        if driver == "ABSENT" and status != "MISSING":
            append(args.audit, {"utc": utc(), "event": "process_death", "status": status,
                                "driver_pid": pid})
            (root / f"{args.prefix}_WATCH_PROCESS_DEATH.marker").touch()
            return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
