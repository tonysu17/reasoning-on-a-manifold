#!/usr/bin/env python3
"""Sync, independently verify, and terminate one exact J-space Phase-1 Pod."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK_PATH = HERE / "JSPACE_PHASE1_FINISH_WATCH.lock"
AUDIT_LOG = HERE / "JSPACE_PHASE1_FINISH_WATCH.jsonl"


class NotReady(RuntimeError):
    pass


class HoldPod(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def audit(event: str, **fields: Any) -> None:
    record = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **fields,
    }
    with AUDIT_LOG.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def notify(title: str, message: str) -> None:
    script = (
        f"display notification {json.dumps(message)} with title {json.dumps(title)}"
    )
    completed = subprocess.run(
        ["osascript", "-e", script], text=True, capture_output=True, check=False
    )
    audit("notification", title=title, message=message, exit_code=completed.returncode)


def load_launch_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    required = {
        "run_uuid",
        "pod_id",
        "ssh_host",
        "ssh_port",
        "ssh_identity",
        "remote_root",
        "run_output_relative",
        "local_output_relative",
        "execution_manifest_relative",
        "execution_manifest_sha256",
    }
    if not required <= set(record):
        raise HoldPod(f"launch record lacks {sorted(required-set(record))}")
    if record["pod_id"] != "bghqqjkco3q7r5":
        raise HoldPod("launch record does not target the authorised literal Pod ID")
    execution = ROOT / record["execution_manifest_relative"]
    if not execution.is_file() or sha256_file(execution) != record["execution_manifest_sha256"]:
        raise HoldPod("local execution manifest missing or hash-mismatched")
    return record


def ssh_args(record: dict[str, Any]) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-i",
        os.path.expanduser(record["ssh_identity"]),
        "-p",
        str(record["ssh_port"]),
        f"root@{record['ssh_host']}",
    ]


def ssh(record: dict[str, Any], command: str, *, timeout: int = 60) -> str:
    completed = subprocess.run(
        [*ssh_args(record), command],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise NotReady(
            f"SSH exited {completed.returncode}: {completed.stderr.strip()[:400]}"
        )
    return completed.stdout


def remote_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    runtime = f"{record['remote_root']}/runtime/{record['run_uuid']}"
    run_root = f"{record['remote_root']}/{record['run_output_relative']}"
    code = f"""
import hashlib,json,os,pathlib
runtime=pathlib.Path({runtime!r})
run_root=pathlib.Path({run_root!r})
def load(path):
    try: return json.loads(path.read_text())
    except Exception: return None
def digest(path):
    if not path.is_file(): return None
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1048576),b''): h.update(c)
    return h.hexdigest()
pid_doc=load(runtime/'DRIVER_PID.json') or {{}}
pid=pid_doc.get('pid')
active=False
if isinstance(pid,int):
    try: os.kill(pid,0); active=True
    except ProcessLookupError: active=False
    except PermissionError: active=True
inventory=[]
symlinks=[]
if run_root.is_dir():
    for p in sorted(run_root.rglob('*')):
        if p.is_symlink(): symlinks.append(str(p.relative_to(run_root)))
        elif p.is_file(): inventory.append(str(p.relative_to(run_root)))
print(json.dumps({{
 'pod_id':os.environ.get('RUNPOD_POD_ID'),
 'status':load(runtime/'STATUS.json'),
 'failed':load(runtime/'FAILED.json'),
 'done':load(run_root/'DONE.json'),
 'driver_pid':pid,
 'driver_active':active,
 'inventory':inventory,
 'symlinks':symlinks,
 'artifact_manifest_sha256':digest(run_root/'ARTIFACT_MANIFEST.json'),
 'done_sha256':digest(run_root/'DONE.json'),
}},sort_keys=True))
"""
    command = (
        "set -eu; . /etc/rp_environment; "
        f"python3 - <<'PY'\n{code}\nPY"
    )
    snapshot = json.loads(ssh(record, command, timeout=120))
    if snapshot.get("pod_id") != record["pod_id"]:
        raise HoldPod("remote /etc/rp_environment Pod ID differs from launch record")
    return snapshot


def validate_done_snapshot(record: dict[str, Any], snapshot: dict[str, Any]) -> None:
    status = snapshot.get("status") or {}
    if snapshot.get("failed") is not None or status.get("state") == "FAILED":
        raise HoldPod(f"remote Phase 1 failed: {snapshot.get('failed') or status}")
    if status.get("state") != "DONE" or snapshot.get("done") is None:
        if not snapshot.get("driver_active") and status.get("state") not in {
            "STARTING",
            "RUNNING_PHASE1",
            "VALIDATING_REMOTE",
        }:
            raise HoldPod(f"driver absent in ambiguous state: {status}")
        raise NotReady(f"Phase 1 state is {status.get('state', 'MISSING')}")
    if snapshot.get("driver_active"):
        raise NotReady("DONE published; waiting for remote controller to exit")
    done = snapshot["done"]
    if (
        done.get("state") != "DONE"
        or done.get("integrity_valid_complete") is not True
        or done.get("run_uuid") != record["run_uuid"]
        or done.get("pod_id") != record["pod_id"]
        or done.get("execution_manifest_sha256")
        != record["execution_manifest_sha256"]
        or done.get("artifact_manifest_sha256")
        != snapshot.get("artifact_manifest_sha256")
    ):
        raise HoldPod("DONE identity/integrity envelope is invalid")
    if snapshot.get("symlinks"):
        raise HoldPod(f"remote run contains symlinks: {snapshot['symlinks']}")


def validate_manifest_paths(files: dict[str, Any]) -> None:
    for relative, metadata in files.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or not relative or relative != str(path):
            raise HoldPod(f"unsafe artifact path: {relative!r}")
        if set(metadata) != {"sha256", "size_bytes"}:
            raise HoldPod(f"malformed artifact metadata: {relative}")
        digest = metadata["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise HoldPod(f"malformed artifact hash: {relative}")
        if not isinstance(metadata["size_bytes"], int) or metadata["size_bytes"] < 0:
            raise HoldPod(f"malformed artifact size: {relative}")


def scp_file(record: dict[str, Any], remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "scp",
            "-q",
            "-P",
            str(record["ssh_port"]),
            "-i",
            os.path.expanduser(record["ssh_identity"]),
            f"root@{record['ssh_host']}:{remote_path}",
            str(local_path),
        ],
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0:
        raise HoldPod(
            f"scp failed for {remote_path}: {completed.stderr.strip()[:400]}"
        )


def pull_and_verify(
    record: dict[str, Any], before: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    remote_run = f"{record['remote_root']}/{record['run_output_relative']}"
    staging_parent = ROOT / "results/jspace_r1_pilot"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = staging_parent / f".phase1_staging_{record['run_uuid']}"
    if staging.exists():
        quarantine = staging.with_name(
            staging.name + ".partial." + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        )
        os.replace(staging, quarantine)
    staging.mkdir(exist_ok=False)

    scp_file(record, f"{remote_run}/ARTIFACT_MANIFEST.json", staging / "ARTIFACT_MANIFEST.json")
    scp_file(record, f"{remote_run}/DONE.json", staging / "DONE.json")
    if sha256_file(staging / "ARTIFACT_MANIFEST.json") != before["artifact_manifest_sha256"]:
        raise HoldPod("pulled artifact-manifest hash differs from remote pre-pull hash")
    manifest = json.loads((staging / "ARTIFACT_MANIFEST.json").read_text())
    if (
        manifest.get("run_uuid") != record["run_uuid"]
        or manifest.get("pod_id") != record["pod_id"]
        or manifest.get("execution_manifest_sha256")
        != record["execution_manifest_sha256"]
    ):
        raise HoldPod("artifact manifest identity differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise HoldPod("artifact manifest has no files")
    validate_manifest_paths(files)
    expected_remote_inventory = set(files) | {"ARTIFACT_MANIFEST.json", "DONE.json"}
    if set(before["inventory"]) != expected_remote_inventory:
        raise HoldPod("remote inventory is not exact artifact-manifest + terminal envelope")

    for relative in sorted(files):
        scp_file(record, f"{remote_run}/{relative}", staging / relative)
    actual = {
        str(path.relative_to(staging))
        for path in staging.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if any(path.is_symlink() for path in staging.rglob("*")):
        raise HoldPod("local staging contains a symlink")
    if actual != expected_remote_inventory:
        raise HoldPod("local staging inventory is not exact")
    for relative, metadata in files.items():
        path = staging / relative
        if path.stat().st_size != metadata["size_bytes"]:
            raise HoldPod(f"local size mismatch: {relative}")
        if sha256_file(path) != metadata["sha256"]:
            raise HoldPod(f"local hash mismatch: {relative}")
    done = json.loads((staging / "DONE.json").read_text())
    if done != before["done"]:
        raise HoldPod("pulled DONE differs from pre-pull remote DONE")

    after = remote_snapshot(record)
    validate_done_snapshot(record, after)
    if (
        after["artifact_manifest_sha256"] != before["artifact_manifest_sha256"]
        or after["done_sha256"] != before["done_sha256"]
        or after["inventory"] != before["inventory"]
    ):
        raise HoldPod("remote terminal bundle changed during pull")

    execution = ROOT / record["execution_manifest_relative"]
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "jspace_phase1_validate.py"),
            "--root",
            str(ROOT),
            "--run-root",
            str(staging),
            "--execution-manifest",
            str(execution),
            "--require-remote-attestation",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0:
        raise HoldPod(
            "local semantic validator failed: "
            + (completed.stdout + completed.stderr)[-2000:]
        )
    validation = json.loads(completed.stdout)
    if validation.get("ok") is not True:
        raise HoldPod("local semantic validator returned ok=false")

    final = ROOT / record["local_output_relative"]
    if final.exists():
        raise HoldPod(f"local final output already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    receipt = {
        "schema_version": "rom-jspace-r1-phase1-sync-verified-v1",
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_uuid": record["run_uuid"],
        "pod_id": record["pod_id"],
        "remote_root": remote_run,
        "local_root": str(final),
        "scientific_gate_pass": bool(before["done"]["scientific_gate_pass"]),
        "execution_manifest_sha256": record["execution_manifest_sha256"],
        "artifact_manifest_sha256": before["artifact_manifest_sha256"],
        "done_sha256": before["done_sha256"],
        "local_semantic_validation_ok": True,
    }
    receipt_path = HERE / f"JSPACE_PHASE1_SYNC_VERIFIED_{record['run_uuid']}.json"
    atomic_json(receipt_path, receipt)
    audit("sync_verified", **receipt)
    return final, receipt


def runpod_api(query: str) -> dict[str, Any]:
    shell = r'''
set -eu
. "$HOME/.rom_runpod_env"
key="${RUNPOD_API_KEY:-${RUNPOD_KEY:-}}"
[ -n "$key" ]
curl -sS --max-time 30 "https://api.runpod.io/graphql?api_key=${key}" \
  -H 'Content-Type: application/json' --data-binary @-
'''
    completed = subprocess.run(
        ["bash", "-c", shell],
        input=json.dumps({"query": query}),
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    if completed.returncode != 0:
        raise HoldPod(f"RunPod API transport failed: {completed.stderr.strip()[:300]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HoldPod("RunPod API response was not JSON") from exc


def query_exact_pod(record: dict[str, Any]) -> dict[str, Any] | None:
    pod_id = record["pod_id"]
    query = (
        "query { pod(input:{podId:\""
        + pod_id
        + "\"}){id desiredStatus runtime{ports{ip isIpPublic privatePort publicPort type}}} }"
    )
    response = runpod_api(query)
    if response.get("errors"):
        messages = " ".join(str(row.get("message", "")) for row in response["errors"])
        if "POD_NOT_FOUND" in messages:
            return None
        raise HoldPod(f"RunPod API query errors: {messages[:300]}")
    return response.get("data", {}).get("pod")


def validate_api_identity(record: dict[str, Any], pod: dict[str, Any] | None) -> None:
    if pod is None or pod.get("id") != record["pod_id"]:
        raise HoldPod("exact Pod absent or API identity mismatch before termination")
    ports = (pod.get("runtime") or {}).get("ports") or []
    matches = [
        row
        for row in ports
        if row.get("ip") == record["ssh_host"]
        and int(row.get("publicPort", -1)) == int(record["ssh_port"])
        and int(row.get("privatePort", -1)) == 22
        and row.get("type") == "tcp"
    ]
    if len(matches) != 1:
        raise HoldPod("RunPod API endpoint does not uniquely match sealed SSH endpoint")
    if pod.get("desiredStatus") != "RUNNING":
        raise HoldPod(f"Pod is not RUNNING before termination: {pod.get('desiredStatus')}")


def terminate_exact_pod(record: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    # Reconfirm both the remote execution envelope and API endpoint immediately
    # before the one authorised mutation.
    snapshot = remote_snapshot(record)
    validate_done_snapshot(record, snapshot)
    if snapshot["artifact_manifest_sha256"] != receipt["artifact_manifest_sha256"]:
        raise HoldPod("remote manifest differs immediately before termination")
    pod = query_exact_pod(record)
    validate_api_identity(record, pod)
    pod_id = record["pod_id"]
    mutation = (
        "mutation { podTerminate(input:{podId:\"" + pod_id + "\"}) }"
    )
    response = runpod_api(mutation)
    if response.get("errors") or "podTerminate" not in response.get("data", {}):
        raise HoldPod(f"RunPod termination was not accepted: {response}")

    confirmed = False
    terminal_observation: Any = None
    for _ in range(12):
        time.sleep(5)
        try:
            observed = query_exact_pod(record)
        except HoldPod as exc:
            terminal_observation = {"ambiguous_error": str(exc)}
            continue
        terminal_observation = observed
        if observed is None or observed.get("desiredStatus") != "RUNNING":
            confirmed = True
            break
    result = {
        "schema_version": "rom-jspace-r1-phase1-pod-termination-v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pod_id": pod_id,
        "run_uuid": record["run_uuid"],
        "sync_receipt": receipt,
        "mutation_accepted": True,
        "termination_confirmed": confirmed,
        "terminal_observation": terminal_observation,
        "mutation_response": response,
    }
    result_path = HERE / f"JSPACE_PHASE1_POD_TERMINATION_{record['run_uuid']}.json"
    atomic_json(result_path, result)
    if not confirmed:
        raise HoldPod("termination mutation accepted but terminal state was not confirmed")
    audit("termination_confirmed", pod_id=pod_id, run_uuid=record["run_uuid"])
    return result


def one_cycle(record: dict[str, Any]) -> dict[str, Any]:
    termination_path = HERE / f"JSPACE_PHASE1_POD_TERMINATION_{record['run_uuid']}.json"
    if termination_path.is_file():
        result = json.loads(termination_path.read_text())
        if result.get("termination_confirmed") is True:
            return {"status": "already_complete", "termination": result}
        raise HoldPod("an unconfirmed termination record already exists; no automatic retry")
    snapshot = remote_snapshot(record)
    audit(
        "snapshot",
        state=(snapshot.get("status") or {}).get("state"),
        driver_active=snapshot.get("driver_active"),
        done=bool(snapshot.get("done")),
        failed=bool(snapshot.get("failed")),
    )
    validate_done_snapshot(record, snapshot)
    _, receipt = pull_and_verify(record, snapshot)
    termination = terminate_exact_pod(record, receipt)
    return {
        "status": "synced_verified_terminated",
        "scientific_gate_pass": receipt["scientific_gate_pass"],
        "termination": termination,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-record", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        try:
            record = load_launch_record(args.launch_record.resolve())
            while True:
                try:
                    result = one_cycle(record)
                    audit("watch_complete", result=result)
                    if result["status"] == "synced_verified_terminated":
                        outcome = "passed" if result["scientific_gate_pass"] else "did not pass"
                        notify(
                            "J-space Phase 1 complete",
                            f"Results synced and verified; scientific gate {outcome}; Pod terminated.",
                        )
                    print(json.dumps(result, indent=2, sort_keys=True))
                    return 0
                except NotReady as exc:
                    audit("not_ready", message=str(exc))
                    if not args.watch:
                        print(json.dumps({"status": "not_ready", "message": str(exc)}))
                        return 0
                    time.sleep(max(10, args.interval))
                except HoldPod as exc:
                    hold = {
                        "status": "HOLD_POD",
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "run_uuid": record.get("run_uuid"),
                        "pod_id": record.get("pod_id"),
                        "message": str(exc),
                    }
                    atomic_json(
                        HERE / f"JSPACE_PHASE1_HOLD_{record.get('run_uuid','unknown')}.json",
                        hold,
                    )
                    audit("hold_pod", **hold)
                    notify("J-space Phase 1 needs attention", str(exc)[:180])
                    print(json.dumps(hold, indent=2, sort_keys=True))
                    return 1
        except Exception as exc:
            audit("watch_fatal", type=type(exc).__name__, message=str(exc))
            notify("J-space Phase 1 watcher error", str(exc)[:180])
            raise


if __name__ == "__main__":
    raise SystemExit(main())
