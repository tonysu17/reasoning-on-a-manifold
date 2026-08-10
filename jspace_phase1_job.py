#!/usr/bin/env python3
"""Fail-closed remote controller for the sealed J-space Phase-1 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


ARTIFACT_PATHS = (
    "RUN_ENVIRONMENT.txt",
    "phase1_runner.log",
    "phase1_report.json",
    "numerical_validation.json",
    "heldout_stability.json",
    "heldout_stability_arrays.npz",
    "external_positive_controls.json",
    "external_readout_arrays.npz",
    "lenses/fit_a.fp32.pt",
    "lenses/fit_a.fp16.pt",
    "lenses/fit_b.fp32.pt",
    "lenses/fit_b.fp16.pt",
    "lenses/merged.fp32.pt",
    "lenses/merged.fp16.pt",
    "INDEPENDENT_VALIDATION.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any, *, overwrite: bool = True) -> None:
    if not overwrite and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
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
        if overwrite:
            os.replace(temporary_name, path)
        else:
            os.link(temporary_name, path)
            Path(temporary_name).unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def status(runtime: Path, state: str, **extra: Any) -> None:
    atomic_json(
        runtime / "STATUS.json",
        {
            "state": state,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **extra,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    execution_path = args.execution_manifest.resolve()
    python_path = Path(os.path.abspath(args.python))
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise FileNotFoundError(f"Python interpreter is not executable: {python_path}")
    execution = json.loads(execution_path.read_text())
    run_uuid = execution["run_uuid"]
    runtime = root / "runtime" / run_uuid
    run_root = root / execution["run_output_relative"]
    if runtime.exists() or run_root.exists():
        raise FileExistsError("run-scoped runtime or output already exists")
    runtime.mkdir(parents=True, exist_ok=False)
    atomic_json(runtime / "DRIVER_PID.json", {"pid": os.getpid()}, overwrite=False)
    status(runtime, "STARTING", run_uuid=run_uuid)

    try:
        remote_pod = os.environ.get("RUNPOD_POD_ID")
        if remote_pod != execution["pod_id"]:
            raise ValueError(f"remote Pod ID {remote_pod!r} != sealed {execution['pod_id']!r}")
        runner_log = runtime / "RUNNER_DRIVER.out"
        status(runtime, "RUNNING_PHASE1", run_uuid=run_uuid)
        with runner_log.open("xb") as handle:
            completed = subprocess.run(
                [
                    str(python_path),
                    str(root / "jspace_phase1_run.py"),
                    "--authorised",
                    "--execution-manifest",
                    str(execution_path),
                    "--root",
                    str(root),
                    "--jlens-checkout",
                    str(root / "_external/jacobian-lens"),
                    "--cache-dir",
                    "/workspace/hf",
                    "--output-root",
                    str(run_root),
                    "--scratch-root",
                    "/tmp/jspace-phase1",
                    "--run-uuid",
                    run_uuid,
                    "--device",
                    "cuda:0",
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        if completed.returncode != 0:
            raise RuntimeError(f"Phase-1 runner exited {completed.returncode}")

        status(runtime, "VALIDATING_REMOTE", run_uuid=run_uuid)
        validator_log = runtime / "VALIDATOR_DRIVER.out"
        with validator_log.open("xb") as handle:
            completed = subprocess.run(
                [
                    str(python_path),
                    str(root / "jspace_phase1_validate.py"),
                    "--root",
                    str(root),
                    "--run-root",
                    str(run_root),
                    "--execution-manifest",
                    str(execution_path),
                    "--write-report",
                    str(run_root / "INDEPENDENT_VALIDATION.json"),
                    "--recompute-stability-null",
                    "--stability-device",
                    "cuda:0",
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        if completed.returncode != 0:
            raise RuntimeError(f"independent validator exited {completed.returncode}")
        validation = json.loads((run_root / "INDEPENDENT_VALIDATION.json").read_text())
        if validation.get("ok") is not True:
            raise RuntimeError("independent validation report is not ok=true")
        for required_check in (
            "stability_null_regenerated_from_seed",
            "external_nulls_regenerated_in_fixed_rng_order",
            "overall_scientific_gate_recomputed",
        ):
            if validation.get("checks", {}).get(required_check) is not True:
                raise RuntimeError(
                    f"independent validation omitted/failed {required_check}"
                )

        actual_files = {
            str(path.relative_to(run_root))
            for path in run_root.rglob("*")
            if path.is_file()
        }
        if actual_files != set(ARTIFACT_PATHS):
            raise ValueError(
                f"artifact path set mismatch: missing={sorted(set(ARTIFACT_PATHS)-actual_files)} "
                f"extra={sorted(actual_files-set(ARTIFACT_PATHS))}"
            )
        artifact_manifest = {
            "schema_version": "rom-jspace-r1-phase1-artifact-manifest-v1",
            "run_uuid": run_uuid,
            "pod_id": execution["pod_id"],
            "execution_manifest_sha256": sha256_file(execution_path),
            "files": {
                relative: {
                    "sha256": sha256_file(run_root / relative),
                    "size_bytes": (run_root / relative).stat().st_size,
                }
                for relative in ARTIFACT_PATHS
            },
        }
        artifact_manifest_path = run_root / "ARTIFACT_MANIFEST.json"
        atomic_json(artifact_manifest_path, artifact_manifest, overwrite=False)
        report = json.loads((run_root / "phase1_report.json").read_text())
        done = {
            "schema_version": "rom-jspace-r1-phase1-done-v1",
            "state": "DONE",
            "integrity_valid_complete": True,
            "scientific_gate_pass": bool(report["scientific_gate"]["pass"]),
            "run_uuid": run_uuid,
            "pod_id": execution["pod_id"],
            "execution_manifest_sha256": sha256_file(execution_path),
            "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        # DONE is the final write into the authoritative run directory.
        atomic_json(run_root / "DONE.json", done, overwrite=False)
        status(
            runtime,
            "DONE",
            run_uuid=run_uuid,
            scientific_gate_pass=done["scientific_gate_pass"],
            done_sha256=sha256_file(run_root / "DONE.json"),
        )
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "rom-jspace-r1-phase1-failure-v1",
            "state": "FAILED",
            "run_uuid": run_uuid,
            "pod_id": execution.get("pod_id"),
            "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(runtime / "FAILED.json", failure, overwrite=False)
        status(runtime, "FAILED", run_uuid=run_uuid, message=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
