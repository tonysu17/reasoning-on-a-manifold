#!/usr/bin/env python3
"""Validation-only recovery controller for one immutable Phase-1 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path, PurePosixPath
from typing import Any


REQUIRED_VALIDATION_CHECKS = {
    "fixed_inputs_match_execution_manifest",
    "validation_only_source_artifacts_immutable",
    "required_run_files_present",
    "report_identity_and_complete_status",
    "registered_vocabulary_domains_match",
    "lens_inventory_shapes_dtypes_finite_counts",
    "merge_and_fp16_recomputed",
    "stability_null_regenerated_from_seed",
    "heldout_stability_recomputed",
    "external_nulls_regenerated_in_fixed_rng_order",
    "external_positive_controls_recomputed",
    "overall_scientific_gate_recomputed",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str, *, first_part: str | None = None) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value or str(path) != value:
        raise ValueError(f"unsafe relative path: {value!r}")
    if first_part is not None and (not path.parts or path.parts[0] != first_part):
        raise ValueError(f"relative path is outside {first_part}/: {value!r}")
    return path


def atomic_json_new(path: Path, payload: Any) -> None:
    if path.exists():
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


def atomic_status(runtime: Path, state: str, **extra: Any) -> None:
    path = runtime / "STATUS.json"
    payload = {
        "state": state,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".STATUS.json.", suffix=".tmp", dir=runtime
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


def verify_source_artifacts(
    run_root: Path,
    expected: dict[str, str],
    *,
    allowed_additions: set[str] | None = None,
) -> None:
    actual = {
        str(path.relative_to(run_root))
        for path in run_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if any(path.is_symlink() for path in run_root.rglob("*")):
        raise ValueError("source run contains a symlink")
    allowed = set(expected) | (allowed_additions or set())
    if actual != allowed:
        raise ValueError(
            f"source artifact inventory mismatch: missing={sorted(allowed-actual)} "
            f"extra={sorted(actual-allowed)}"
        )
    for relative, expected_hash in expected.items():
        path = run_root / relative
        if sha256_file(path) != expected_hash:
            raise ValueError(f"source artifact hash mismatch: {relative}")


def require_exact_validation_checks(checks: dict[str, Any]) -> None:
    if set(checks) != REQUIRED_VALIDATION_CHECKS or not all(checks.values()):
        raise RuntimeError(
            "corrected validation check inventory is incomplete or failed: "
            f"observed={sorted(checks)} expected={sorted(REQUIRED_VALIDATION_CHECKS)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = args.validation_manifest.resolve()
    python_path = Path(os.path.abspath(args.python))
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise FileNotFoundError(f"Python interpreter is not executable: {python_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != (
        "rom-jspace-r1-phase1-validation-continuation-v1"
    ):
        raise ValueError("unexpected validation-continuation schema")
    recovery = manifest["validation_only"]
    run_uuid = manifest["run_uuid"]
    attempt_uuid = recovery["validation_attempt_uuid"]
    run_relative = safe_relative(manifest["run_output_relative"], first_part="runs")
    source_runtime_relative = safe_relative(
        recovery["source_runtime_relative"], first_part="runtime"
    )
    validation_runtime_relative = safe_relative(
        recovery["validation_runtime_relative"], first_part="runtime"
    )
    run_root = root / Path(*run_relative.parts)
    source_runtime = root / Path(*source_runtime_relative.parts)
    runtime = root / Path(*validation_runtime_relative.parts)
    attestation_relative = recovery["authoritative_attestation"]
    attestation_parts = safe_relative(attestation_relative)
    if attestation_parts.parts != (
        "validation_attempts",
        attempt_uuid,
        "INDEPENDENT_VALIDATION.json",
    ):
        raise ValueError("authoritative attestation path differs from attempt UUID")
    attestation_path = run_root / Path(*attestation_parts.parts)
    artifact_manifest_path = run_root / "ARTIFACT_MANIFEST.json"
    done_path = run_root / "DONE.json"

    if runtime.exists():
        raise FileExistsError(f"validation runtime already exists: {runtime}")
    if not run_root.is_dir() or not source_runtime.is_dir():
        raise FileNotFoundError("source run/runtime is missing")
    runtime.mkdir(parents=True, exist_ok=False)
    atomic_json_new(runtime / "DRIVER_PID.json", {"pid": os.getpid()})
    atomic_status(runtime, "STARTING", run_uuid=run_uuid, validation_attempt_uuid=attempt_uuid)

    try:
        if os.environ.get("RUNPOD_POD_ID") != manifest["pod_id"]:
            raise ValueError("remote Pod ID differs from validation manifest")
        for relative, expected_hash in manifest["fixed_inputs_sha256"].items():
            fixed_relative = safe_relative(relative)
            fixed_path = root / Path(*fixed_relative.parts)
            if not fixed_path.is_file() or sha256_file(fixed_path) != expected_hash:
                raise ValueError(f"fixed input missing/hash mismatch: {relative}")
        source_manifest_relative = safe_relative(
            recovery["source_execution_manifest_relative"], first_part="results"
        )
        source_manifest = root / Path(*source_manifest_relative.parts)
        if sha256_file(source_manifest) != recovery[
            "source_execution_manifest_sha256"
        ]:
            raise ValueError("source execution manifest hash differs")
        failed_path = source_runtime / "FAILED.json"
        if sha256_file(failed_path) != recovery["source_runtime_failed_sha256"]:
            raise ValueError("source runtime FAILED.json hash differs")
        failed = json.loads(failed_path.read_text())
        if failed.get("state") != "FAILED" or failed.get("run_uuid") != run_uuid:
            raise ValueError("source runtime failure identity differs")
        source_pid = json.loads((source_runtime / "DRIVER_PID.json").read_text()).get(
            "pid"
        )
        if isinstance(source_pid, int):
            try:
                os.kill(source_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("source Phase-1 controller is still active")
        expected = recovery["source_artifacts_sha256"]
        for relative in expected:
            safe_relative(relative)
        verify_source_artifacts(run_root, expected)
        for terminal in (attestation_path, artifact_manifest_path, done_path):
            if terminal.exists():
                raise FileExistsError(f"recovery terminal path already exists: {terminal}")

        atomic_status(
            runtime,
            "VALIDATING_REMOTE",
            run_uuid=run_uuid,
            validation_attempt_uuid=attempt_uuid,
        )
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
                    str(manifest_path),
                    "--write-report",
                    str(attestation_path),
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
            raise RuntimeError(f"corrected independent validator exited {completed.returncode}")
        attestation = json.loads(attestation_path.read_text())
        if attestation.get("ok") is not True:
            raise RuntimeError("corrected independent attestation is not ok=true")
        if (
            attestation.get("execution_manifest_sha256") != sha256_file(manifest_path)
            or attestation.get("validation_attempt_uuid") != attempt_uuid
        ):
            raise RuntimeError("corrected attestation identity differs")
        require_exact_validation_checks(attestation.get("checks", {}))

        verify_source_artifacts(
            run_root, expected, allowed_additions={attestation_relative}
        )
        report = json.loads((run_root / "phase1_report.json").read_text())
        scientific_gate_pass = bool(report["scientific_gate"]["pass"])
        files = [*sorted(expected), attestation_relative]
        artifact_manifest = {
            "schema_version": "rom-jspace-r1-phase1-artifact-manifest-v2",
            "run_uuid": run_uuid,
            "validation_attempt_uuid": attempt_uuid,
            "pod_id": manifest["pod_id"],
            "execution_manifest_sha256": sha256_file(manifest_path),
            "source_execution_manifest_sha256": recovery[
                "source_execution_manifest_sha256"
            ],
            "authoritative_attestation": attestation_relative,
            "files": {
                relative: {
                    "sha256": (
                        expected[relative]
                        if relative in expected
                        else sha256_file(run_root / relative)
                    ),
                    "size_bytes": (run_root / relative).stat().st_size,
                }
                for relative in files
            },
        }
        atomic_json_new(artifact_manifest_path, artifact_manifest)
        verify_source_artifacts(
            run_root,
            expected,
            allowed_additions={attestation_relative, "ARTIFACT_MANIFEST.json"},
        )
        done = {
            "schema_version": "rom-jspace-r1-phase1-done-v2",
            "state": "DONE",
            "integrity_valid_complete": True,
            "scientific_gate_pass": scientific_gate_pass,
            "run_uuid": run_uuid,
            "validation_attempt_uuid": attempt_uuid,
            "pod_id": manifest["pod_id"],
            "execution_manifest_sha256": sha256_file(manifest_path),
            "source_execution_manifest_sha256": recovery[
                "source_execution_manifest_sha256"
            ],
            "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
            "authoritative_attestation": attestation_relative,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json_new(done_path, done)
        atomic_status(
            runtime,
            "DONE",
            run_uuid=run_uuid,
            validation_attempt_uuid=attempt_uuid,
            scientific_gate_pass=scientific_gate_pass,
            done_sha256=sha256_file(done_path),
        )
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "rom-jspace-r1-phase1-validation-failure-v1",
            "state": "FAILED",
            "run_uuid": run_uuid,
            "validation_attempt_uuid": attempt_uuid,
            "pod_id": manifest.get("pod_id"),
            "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json_new(runtime / "FAILED.json", failure)
        atomic_status(
            runtime,
            "FAILED",
            run_uuid=run_uuid,
            validation_attempt_uuid=attempt_uuid,
            message=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
