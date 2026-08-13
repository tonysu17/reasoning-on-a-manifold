#!/usr/bin/env python3
"""Independent validator for the amended J-space Phase-0 output package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    manifest = json.loads(manifest_path.read_text())
    controller = json.loads((output_root / "benchmark_report.json").read_text())
    probe = json.loads((output_root / "probe_dim8.json").read_text())
    full = json.loads((output_root / "five_prompt_dim8.json").read_text())

    checks: dict[str, bool] = {}
    checks["exact_output_file_set"] = {
        path.name for path in output_root.iterdir() if path.is_file()
    } == {"benchmark_report.json", "probe_dim8.json", "five_prompt_dim8.json"}
    checks["amendment_marker"] = manifest.get("protocol_amendment", {}).get("marker") == "amended"
    checks["manifest_hash_controller"] = controller.get("manifest_sha256") == sha256(manifest_path)
    checks["manifest_hash_worker"] = full.get("manifest_sha256") == sha256(manifest_path)
    checks["controller_complete"] = controller.get("status") == "complete"
    checks["controller_gate_pass"] = controller.get("gate", {}).get("pass") is True
    checks["selected_dim_batch_8"] = controller.get("selected_dim_batch") == 8
    checks["only_probe_attempt_dim8"] = controller.get("attempts") == [
        {"dim_batch": 8, "returncode": 0, "status": "complete"}
    ]
    checks["probe_complete"] = probe.get("status") == "complete"
    checks["full_worker_complete"] = full.get("status") == "complete"
    checks["no_worker_exceptions"] = full.get("exceptions") == []
    checks["five_prompts"] = len(full.get("prompts", [])) == 5
    checks["prompt_order_and_hashes"] = [
        (row.get("task_id"), row.get("prompt_sha256")) for row in full.get("prompts", [])
    ] == [(row["task_id"], row["prompt_sha256"]) for row in manifest["prompts"]]

    expected_layers = {str(layer) for layer in range(27)}
    checks["all_layer_matrices_valid"] = all(
        set(prompt.get("layers", {})) == expected_layers
        and all(
            value.get("shape") == [1536, 1536]
            and value.get("dtype") == "torch.float32"
            and value.get("finite") is True
            for value in prompt["layers"].values()
        )
        for prompt in full.get("prompts", [])
    )
    resource_fields = (
        "peak_gpu_allocated_bytes",
        "peak_gpu_reserved_bytes",
        "peak_cpu_rss_bytes",
    )
    checks["per_prompt_resource_metrics_present"] = all(
        isinstance(prompt.get(field), int) and prompt[field] > 0
        for prompt in full.get("prompts", [])
        for field in resource_fields
    )
    checks["cpu_peak_measurement_primary"] = all(
        prompt.get("peak_cpu_rss_measurement") == "linux_vm_hwm_after_clear_refs_5"
        for prompt in full.get("prompts", [])
    )
    coordinate = full.get("coordinate_identity", {})
    checks["coordinate_module_identity"] = coordinate.get("same_module_object") is True
    checks["coordinate_tensor_identity"] = (
        coordinate.get("allclose_rtol_1e-5_atol_1e-6") is True
        and coordinate.get("max_abs_difference") == 0.0
    )
    repeat_max = full.get("repeat_first", {}).get("relative_frobenius", {}).get("max")
    checks["repeat_relative_frobenius_le_1e6"] = repeat_max is not None and repeat_max <= 1e-6
    headroom = full.get("hardware", {}).get("total_memory_bytes", 0) - full.get(
        "peak_gpu_reserved_bytes", 0
    )
    checks["gpu_headroom_at_least_2gib"] = headroom >= 2 * 1024**3
    prompt_seconds = sum(prompt["wall_seconds"] for prompt in full.get("prompts", []))
    projected = prompt_seconds / len(full.get("prompts", [])) * 100
    checks["projection_recomputed"] = math.isclose(
        projected,
        controller.get("projected_100_prompt_wall_seconds", math.nan),
        rel_tol=0,
        abs_tol=1e-9,
    )
    gate_checks = controller.get("gate", {}).get("checks", {})
    checks["amended_resource_gate_present_and_true"] = (
        gate_checks.get("per_prompt_resource_metrics_recorded") is True
    )
    checks["all_controller_subchecks_true"] = bool(gate_checks) and all(gate_checks.values())

    errors = [name for name, passed in checks.items() if not passed]
    warnings: list[str] = []
    if full.get("git_commit") is None:
        warnings.append(
            "git_commit is null because the network-volume execution bundle is not a git checkout; "
            "exact runner, manifest, preregistration, amendment, and J-lens hashes remain recorded"
        )

    report = {
        "schema_version": "rom-jspace-phase0-independent-validation-v1",
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "artifact_hashes": {
            path.name: sha256(path)
            for path in sorted(output_root.iterdir())
            if path.is_file()
        },
        "metrics": {
            "selected_dim_batch": controller.get("selected_dim_batch"),
            "prompt_wall_seconds": {
                prompt["task_id"]: prompt["wall_seconds"] for prompt in full["prompts"]
            },
            "per_prompt_peak_gpu_reserved_bytes": {
                prompt["task_id"]: prompt["peak_gpu_reserved_bytes"] for prompt in full["prompts"]
            },
            "per_prompt_peak_cpu_rss_bytes": {
                prompt["task_id"]: prompt["peak_cpu_rss_bytes"] for prompt in full["prompts"]
            },
            "worker_peak_gpu_allocated_bytes": full.get("peak_gpu_allocated_bytes"),
            "worker_peak_gpu_reserved_bytes": full.get("peak_gpu_reserved_bytes"),
            "gpu_headroom_bytes": headroom,
            "worker_peak_cpu_rss_bytes": full.get("peak_cpu_rss_bytes"),
            "repeat_relative_frobenius_max": repeat_max,
            "projected_100_prompt_wall_seconds": projected,
            "full_worker_total_wall_seconds": full.get("total_wall_seconds"),
        },
        "provenance": {
            "manifest_sha256": sha256(manifest_path),
            "runner_sha256": sha256(root / "jspace_pilot_benchmark.py"),
            "amendment_sha256": sha256(
                root / "results/prereg/JSPACE_R1_BENCHMARK_AMENDMENT_1_2026-08-10.md"
            ),
            "jlens_commit": full.get("jlens_commit"),
            "model_revision": full.get("model_revision"),
            "git_commit": full.get("git_commit"),
            "git_dirty_paths": full.get("git_dirty_paths"),
        },
    }
    atomic_json(args.validation_output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
