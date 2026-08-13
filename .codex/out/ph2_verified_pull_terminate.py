#!/usr/bin/env python3
"""Completion-only Phase-2 pull, verification, and RunPod termination.

This watcher is intentionally conservative. It only targets the exact Phase-2
pod recorded below, never terminates a RUNNING job, and refuses termination if
any required remote artifact is absent or differs from the local pulled copy.
The pod's self-scoped RunPod key remains on the pod and is never printed or
persisted locally.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POD_ALIAS = "runpod"
EXPECTED_POD_ID = "oit3b06a7l9u7o"
REMOTE_ROOT = "/workspace/reasoning-on-manifold"
AUDIT_LOG = HERE / "PH2_VERIFIED_FINISH_WATCH_2026-08-09.jsonl"
TERMINATION_RECORD = HERE / "PH2_POD_TERMINATION_2026-08-09.json"
LOCK_PATH = HERE / "PH2_VERIFIED_FINISH_WATCH.lock"

REQUIRED_DONE = (
    "results/ph2/markers/discovery_extract.done",
    "results/ph2/markers/target_gates.done",
    "results/ph2/markers/refit.done",
    "results/ph2/discovery/discovery_stats.json",
    "results/ph2/gates/target_gates.json",
    "results/ph2/refit/refit_report.json",
    "results/ph2/provenance/discovery_extract.json",
    "results/ph2/provenance/target_gates.json",
    "results/ph2/provenance/refit.json",
)

# Trees needed for the Phase-2/Phase-0/P5 analysis handoff. Optional trees are
# verified when present remotely; critical trees must exist.
TREE_MAPPINGS = (
    ("results/ph2", "results/ph2", True),
    ("results/p5_wildguard/smoke", "results/p5_wildguard/smoke", True),
    ("results/safety_posttrain/ph0_s3", "results/safety_posttrain/ph0_s3", False),
    (
        "data/activations/R1-1.5B-dpo-control-f5",
        "data/activations/R1-1.5B-dpo-control-f5",
        False,
    ),
)

# Analysis-relevant F5 artifacts are retained, while multi-gigabyte merged
# model copies are deliberately excluded: the adapters are sufficient to
# reconstruct them and are the artifact needed for later analysis.
FILTERED_TREE_MAPPINGS = (
    (
        "results/safety_posttrain/rl/dpo_control_f5",
        "results/safety_posttrain/rl/dpo_control_f5",
        ("merged",),
    ),
)

FILE_MAPPINGS = (
    (
        "results/safety_posttrain/rl/pt08_R1-1.5B-dpo-control-f5.json",
        "results/safety_posttrain/rl/pt08_R1-1.5B-dpo-control-f5.json",
    ),
)


class FinishError(RuntimeError):
    pass


def run(args: list[str], *, check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=ROOT, text=True, capture_output=True, check=check, timeout=timeout
    )


def ssh(command: str, *, check: bool = True, timeout: int = 60) -> str:
    result = run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", POD_ALIAS, command],
        check=check,
        timeout=timeout,
    )
    return result.stdout


def audit(event: str, **fields: Any) -> None:
    record = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **fields,
    }
    with AUDIT_LOG.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def is_terminal(status: str) -> bool:
    return status == "DONE" or status.startswith("FAILED:")


def remote_snapshot() -> dict[str, Any]:
    script = f"""
set -eu
cd {REMOTE_ROOT}
. /etc/rp_environment
printf '%s\\n' "$RUNPOD_POD_ID"
cat PH2_STATUS
[ -f PH2_DONE.marker ] && echo MARKER_PRESENT || echo MARKER_ABSENT
if tmux has-session -t ph2 2>/dev/null; then echo DRIVER_PRESENT; else echo DRIVER_ABSENT; fi
if [ -f results/p5_wildguard/smoke/SMOKE_DONE.marker ]; then echo P5_DONE;
elif [ -f results/p5_wildguard/smoke/SMOKE_FAILED.marker ]; then echo P5_FAILED;
else echo P5_NONTERMINAL; fi
"""
    lines = ssh(script).splitlines()
    if len(lines) != 5:
        raise FinishError(f"unexpected remote status envelope: {lines!r}")
    pod_id, status, marker, driver, p5 = lines
    if pod_id != EXPECTED_POD_ID:
        raise FinishError(f"pod identity mismatch: {pod_id!r}")
    return {
        "pod_id": pod_id,
        "status": status,
        "completion_marker": marker == "MARKER_PRESENT",
        "driver_present": driver == "DRIVER_PRESENT",
        "p5_status": p5,
    }


def remote_tree_exists(relative: str) -> bool:
    out = ssh(f"test -d {REMOTE_ROOT}/{relative} && echo yes || echo no")
    return out.strip() == "yes"


def rsync_tree(
    remote_relative: str, local_relative: str, *, excluded_parts: tuple[str, ...] = ()
) -> None:
    local = ROOT / local_relative
    local.mkdir(parents=True, exist_ok=True)
    exclusions: list[str] = []
    for part in excluded_parts:
        exclusions.extend((f"--exclude={part}/", f"--exclude=*/{part}/"))
    run(
        [
            "rsync",
            "-rltz",
            "--checksum",
            "--partial",
            "--timeout=600",
            *exclusions,
            f"{POD_ALIAS}:{REMOTE_ROOT}/{remote_relative}/",
            str(local) + "/",
        ],
        timeout=1800,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_manifest(
    relative: str, *, excluded_parts: tuple[str, ...] = ()
) -> dict[str, str]:
    code = (
        "import hashlib,json,pathlib; "
        f"b=pathlib.Path({str(REMOTE_ROOT + '/' + relative)!r}); "
        f"excluded=set({list(excluded_parts)!r}); "
        "o={}; "
        "\nfor p in sorted(x for x in b.rglob('*') "
        "if x.is_file() and not excluded.intersection(x.relative_to(b).parts)):\n"
        " h=hashlib.sha256();\n"
        " with p.open('rb') as f:\n"
        "  for c in iter(lambda:f.read(1048576),b''): h.update(c)\n"
        " o[str(p.relative_to(b))]=h.hexdigest()\n"
        "print(json.dumps(o,sort_keys=True))"
    )
    # A quoted `python -c` argument turns embedded newlines into literal `\\n`
    # on some remote shells. A single-quoted heredoc transports the exact code
    # bytes without interpolating paths or shell metacharacters.
    return json.loads(ssh(f"python3 - <<'PY'\n{code}\nPY", timeout=1800))


def local_manifest(
    relative: str, *, excluded_parts: tuple[str, ...] = ()
) -> dict[str, str]:
    base = ROOT / relative
    return {
        str(path.relative_to(base)): sha256_file(path)
        for path in sorted(item for item in base.rglob("*") if item.is_file())
        if "runtime" not in path.relative_to(base).parts
        and not set(excluded_parts).intersection(path.relative_to(base).parts)
    }


def verify_tree(
    remote_relative: str,
    local_relative: str,
    *,
    excluded_parts: tuple[str, ...] = (),
) -> dict[str, Any]:
    remote = remote_manifest(remote_relative, excluded_parts=excluded_parts)
    local = local_manifest(local_relative, excluded_parts=excluded_parts)
    if not remote:
        raise FinishError(f"remote tree is empty: {remote_relative}")
    missing = sorted(set(remote) - set(local))
    mismatched = sorted(key for key in remote.keys() & local.keys() if remote[key] != local[key])
    if missing or mismatched:
        raise FinishError(
            f"pull verification failed for {remote_relative}: "
            f"missing={missing[:5]}, mismatched={mismatched[:5]}"
        )
    return {
        "remote": remote_relative,
        "local": local_relative,
        "n_files": len(remote),
        "excluded_path_parts": list(excluded_parts),
        "manifest_sha256": hashlib.sha256(
            json.dumps(remote, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def remote_file_exists(relative: str) -> bool:
    out = ssh(f"test -f {REMOTE_ROOT}/{relative} && echo yes || echo no")
    return out.strip() == "yes"


def pull_and_verify_file(remote_relative: str, local_relative: str) -> dict[str, Any]:
    local = ROOT / local_relative
    local.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "rsync",
            "-ltz",
            "--checksum",
            "--partial",
            "--timeout=600",
            f"{POD_ALIAS}:{REMOTE_ROOT}/{remote_relative}",
            str(local),
        ],
        timeout=1800,
    )
    remote_hash = ssh(
        f"sha256sum {REMOTE_ROOT}/{remote_relative} | cut -d' ' -f1", timeout=1800
    ).strip()
    local_hash = sha256_file(local)
    if remote_hash != local_hash:
        raise FinishError(f"pull verification failed for file {remote_relative}")
    return {
        "remote": remote_relative,
        "local": local_relative,
        "n_files": 1,
        "manifest_sha256": remote_hash,
    }


def pull_runtime_files() -> None:
    phase_runtime = ROOT / "results/ph2/runtime"
    p5_runtime = ROOT / "results/p5_wildguard/smoke/runtime"
    phase_runtime.mkdir(parents=True, exist_ok=True)
    p5_runtime.mkdir(parents=True, exist_ok=True)
    run(
        [
            "rsync",
            "-rltz",
            "--checksum",
            f"{POD_ALIAS}:{REMOTE_ROOT}/ph2.log",
            f"{POD_ALIAS}:{REMOTE_ROOT}/PH2_STATUS",
            f"{POD_ALIAS}:{REMOTE_ROOT}/PH2_DONE.marker",
            str(phase_runtime) + "/",
        ]
    )
    run(
        [
            "rsync",
            "-rltz",
            "--checksum",
            f"{POD_ALIAS}:{REMOTE_ROOT}/p5wg_run.log",
            f"{POD_ALIAS}:{REMOTE_ROOT}/P5WG_STATUS",
            str(p5_runtime) + "/",
        ]
    )


def verify_semantics(status: str) -> None:
    runtime_status = (ROOT / "results/ph2/runtime/PH2_STATUS").read_text().strip()
    if runtime_status != status:
        raise FinishError("local PH2_STATUS differs from terminal remote status")
    if not (ROOT / "results/ph2/runtime/PH2_DONE.marker").is_file():
        raise FinishError("local PH2_DONE.marker missing")

    p5 = ROOT / "results/p5_wildguard/smoke"
    if not (p5 / "SMOKE_DONE.marker").is_file():
        raise FinishError("WildGuard smoke is not successfully complete")
    if len(list((p5 / "rows").glob("*.json"))) != 8:
        raise FinishError("WildGuard rows are incomplete")
    if len(list((p5 / "rows_diag").glob("*.json"))) != 8:
        raise FinishError("WildGuard diagnostic rows are incomplete")

    if status == "DONE":
        for relative in REQUIRED_DONE:
            if not (ROOT / relative).is_file():
                raise FinishError(f"required Phase-2 completion artifact missing: {relative}")
        gates = json.loads((ROOT / "results/ph2/gates/target_gates.json").read_text())
        refit = json.loads((ROOT / "results/ph2/refit/refit_report.json").read_text())
        if set(gates) != {"base", "star1", "deepscaler"}:
            raise FinishError("target-gate role set is incomplete")
        if not all(gates[role].get("complete") for role in gates):
            raise FinishError("target-gate report has incomplete role")
        if set(refit) != {"star1", "deepscaler"}:
            raise FinishError("refit role set is incomplete")
        if not all(refit[role].get("complete") for role in refit):
            raise FinishError("refit report has incomplete role")


def pull_and_verify(status: str) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for remote_relative, local_relative, critical in TREE_MAPPINGS:
        exists = remote_tree_exists(remote_relative)
        if not exists:
            if critical:
                raise FinishError(f"critical remote tree absent: {remote_relative}")
            continue
        rsync_tree(remote_relative, local_relative)
        verified.append(verify_tree(remote_relative, local_relative))

    for remote_relative, local_relative, excluded_parts in FILTERED_TREE_MAPPINGS:
        if not remote_tree_exists(remote_relative):
            continue
        rsync_tree(remote_relative, local_relative, excluded_parts=excluded_parts)
        verified.append(
            verify_tree(
                remote_relative,
                local_relative,
                excluded_parts=excluded_parts,
            )
        )

    for remote_relative, local_relative in FILE_MAPPINGS:
        if remote_file_exists(remote_relative):
            verified.append(pull_and_verify_file(remote_relative, local_relative))

    # STAR1 staging uses a deliberate local alias.
    if remote_tree_exists("data/activations/STAR1-1.5B"):
        rsync_tree("data/activations/STAR1-1.5B", "data/activations/STAR1-1.5B-6label")
        verified.append(
            verify_tree("data/activations/STAR1-1.5B", "data/activations/STAR1-1.5B-6label")
        )

    # F5 result files excluding merged model weights are pulled by the existing
    # project routine. They are not a Phase-2 completion precondition, but the
    # routine preserves the planned backlog before termination.
    env = dict(os.environ)
    env["POD"] = POD_ALIAS
    subprocess.run(
        ["bash", "runpod_ph2.sh", "pull"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    pull_runtime_files()
    verify_semantics(status)
    return verified


def terminate_exact_pod() -> dict[str, Any]:
    command = f"""
set -eu
. /etc/rp_environment
[ "$RUNPOD_POD_ID" = "{EXPECTED_POD_ID}" ]
curl -sS --max-time 30 "https://api.runpod.io/graphql?api_key=${{RUNPOD_API_KEY}}" \
  -H 'Content-Type: application/json' \
  -d '{{"query":"mutation{{podTerminate(input:{{podId:\"{EXPECTED_POD_ID}\"}})}}"}}'
"""
    response_text = ssh(command, timeout=45)
    try:
        response = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise FinishError("RunPod termination response is not JSON") from error
    if response.get("errors") or "podTerminate" not in response.get("data", {}):
        raise FinishError(f"RunPod termination was not accepted: {response}")
    return response


def write_termination_record(
    snapshot: dict[str, Any], verified: list[dict[str, Any]], response: dict[str, Any]
) -> None:
    record = {
        "schema_version": "ph2-verified-pull-termination-1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pod_alias": POD_ALIAS,
        "pod_id": EXPECTED_POD_ID,
        "terminal_status": snapshot["status"],
        "completion_marker": snapshot["completion_marker"],
        "p5_status": snapshot["p5_status"],
        "verified_trees": verified,
        "termination_api_accepted": True,
        "termination_response": response,
    }
    record["record_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    TERMINATION_RECORD.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


def check_once(*, dry_run: bool = False) -> dict[str, Any]:
    if TERMINATION_RECORD.is_file():
        return {"status": "already_terminated", "record": str(TERMINATION_RECORD)}
    snapshot = remote_snapshot()
    audit("snapshot", **snapshot)
    if not is_terminal(snapshot["status"]):
        return {
            "status": "running",
            "phase2_status": snapshot["status"],
            **{key: value for key, value in snapshot.items() if key != "status"},
        }
    if not snapshot["completion_marker"]:
        raise FinishError("terminal status without PH2_DONE.marker")
    if snapshot["driver_present"]:
        raise FinishError("terminal status while Phase-2 tmux driver still exists")
    if snapshot["p5_status"] != "P5_DONE":
        raise FinishError("WildGuard output on the shared pod is not complete")
    if dry_run:
        return {
            "status": "terminal_dry_run",
            "phase2_status": snapshot["status"],
            **{key: value for key, value in snapshot.items() if key != "status"},
        }

    verified = pull_and_verify(snapshot["status"])
    audit("pull_verified", terminal_status=snapshot["status"], trees=verified)
    response = terminate_exact_pod()
    write_termination_record(snapshot, verified, response)
    audit("termination_accepted", pod_id=EXPECTED_POD_ID, terminal_status=snapshot["status"])
    return {
        "status": "termination_accepted",
        "pod_id": EXPECTED_POD_ID,
        "terminal_status": snapshot["status"],
        "verified_trees": verified,
        "record": str(TERMINATION_RECORD),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="perform one guarded check")
    parser.add_argument("--dry-run", action="store_true", help="never pull or terminate")
    args = parser.parse_args()
    if not args.check:
        parser.error("--check is required")
    LOCK_PATH.touch(exist_ok=True)
    with LOCK_PATH.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "another_check_is_running"}))
            return 0
        try:
            result = check_once(dry_run=args.dry_run)
        except Exception as error:
            audit("error", error_type=type(error).__name__, message=str(error))
            print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
            return 2
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
