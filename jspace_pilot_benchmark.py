#!/usr/bin/env python3
"""Authorisation-gated five-prompt benchmark for the sealed R1 J-lens pilot.

The default controller validates the sealed manifest, tries dim_batch values in
fresh worker processes, and runs the five-prompt benchmark at the first setting
that fits. Workers fit no reusable scientific lens and generate no continuation.

Nothing runs without the explicit ``--authorised`` flag. The sealed manifest
also keeps ``execution_authorised=false``; launch authority is recorded by the
command/provenance report rather than by mutating that immutable input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import jspace_pilot_preflight as preflight


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    ROOT / "results/prereg/JSPACE_R1_BENCHMARK_MANIFEST_2026-08-10.json"
)
DEFAULT_PREREG = (
    ROOT / "results/prereg/JSPACE_R1_STEERING_PILOT_PREREG_2026-08-10.md"
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_peak_rss_bytes() -> int:
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def reset_per_prompt_cpu_peak() -> str:
    """Reset Linux VmHWM to current RSS so the next prompt has its own peak."""
    clear_refs = Path("/proc/self/clear_refs")
    try:
        clear_refs.write_text("5\n")
        return "linux_vm_hwm_after_clear_refs_5"
    except OSError:
        return "process_ru_maxrss_fallback_not_resettable"


def per_prompt_cpu_peak_rss_bytes(basis: str) -> int:
    if basis == "linux_vm_hwm_after_clear_refs_5":
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    return cpu_peak_rss_bytes()


def require_launch_authorisation(authorised: bool) -> None:
    if not authorised:
        raise SystemExit(
            "REFUSAL: model execution requires --authorised; the sealed manifest "
            "alone grants no launch authority"
        )


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _prompt_rows(manifest: dict[str, Any], prompt_id: str | None) -> list[dict[str, Any]]:
    rows = manifest["prompts"]
    if prompt_id is None:
        return rows
    selected = [row for row in rows if row["task_id"] == prompt_id]
    if len(selected) != 1:
        raise ValueError(f"prompt id {prompt_id!r} is not unique in the manifest")
    return selected


def _tokenizer_file_hashes(snapshot: Path) -> dict[str, str]:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
        "generation_config.json",
    )
    return {name: sha256_file(snapshot / name) for name in names if (snapshot / name).is_file()}


def _coordinate_identity(model: Any, lens_model: Any, prompt: str) -> dict[str, Any]:
    """Pin the existing steering and J-lens L17 hook to the same block output."""
    import torch
    from jlens.hooks import ActivationRecorder

    steering_module = model.model.layers[17]
    lens_module = lens_model.layers[17]
    direct: dict[str, Any] = {}

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        tensor = output if torch.is_tensor(output) else output[0]
        direct["value"] = tensor.detach().clone()

    handle = steering_module.register_forward_hook(capture)
    input_ids = lens_model.encode(prompt, max_length=128)
    try:
        with torch.no_grad(), ActivationRecorder(lens_model.layers, at=[17]) as recorder:
            lens_model.forward(input_ids)
        observed = recorder.activations[17].detach()
    finally:
        handle.remove()
    expected = direct["value"]
    difference = (expected.float() - observed.float()).abs()
    return {
        "same_module_object": steering_module is lens_module,
        "same_shape": list(expected.shape) == list(observed.shape),
        "max_abs_difference": float(difference.max().item()),
        "allclose_rtol_1e-5_atol_1e-6": bool(
            torch.allclose(expected.float(), observed.float(), rtol=1e-5, atol=1e-6)
        ),
    }


def _run_one_prompt(
    lens_model: Any,
    row: dict[str, Any],
    *,
    source_layers: list[int],
    target_layer: int,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
) -> tuple[dict[str, Any], dict[int, Any]]:
    import torch
    from jlens.fitting import jacobian_for_prompt

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    cpu_peak_basis = reset_per_prompt_cpu_peak()
    started = time.perf_counter()
    jacobians, seq_len, n_valid = jacobian_for_prompt(
        lens_model,
        row["prompt"],
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_gpu_allocated = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    peak_gpu_reserved = (
        int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
    )
    peak_cpu_rss = per_prompt_cpu_peak_rss_bytes(cpu_peak_basis)
    layer_summary = {
        str(layer): {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "finite": bool(torch.isfinite(matrix).all().item()),
        }
        for layer, matrix in jacobians.items()
    }
    return (
        {
            "task_id": row["task_id"],
            "prompt_sha256": row["prompt_sha256"],
            "token_count": int(seq_len),
            "valid_positions": int(n_valid),
            "backward_calls": math.ceil(lens_model.d_model / dim_batch),
            "wall_seconds": elapsed,
            "peak_gpu_allocated_bytes": peak_gpu_allocated,
            "peak_gpu_reserved_bytes": peak_gpu_reserved,
            "peak_cpu_rss_bytes": peak_cpu_rss,
            "peak_cpu_rss_measurement": cpu_peak_basis,
            "layers": layer_summary,
        },
        jacobians,
    )


def _repeat_relative_frobenius(first: dict[int, Any], second: dict[int, Any]) -> dict[str, Any]:
    import torch

    per_layer: dict[str, float] = {}
    for layer in sorted(first):
        numerator = torch.linalg.vector_norm(first[layer] - second[layer])
        denominator = torch.linalg.vector_norm(first[layer]).clamp_min(1e-30)
        per_layer[str(layer)] = float((numerator / denominator).item())
    return {"per_layer": per_layer, "max": max(per_layer.values())}


def worker(args: argparse.Namespace) -> int:
    require_launch_authorisation(args.authorised)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    manifest_path = args.manifest.resolve()
    checkout = args.jlens_checkout.resolve()
    check = preflight.validate(manifest_path, root=ROOT, jlens_checkout=checkout)
    if not check["ok"]:
        atomic_json(args.worker_output, {"status": "preflight_failed", "preflight": check})
        return 1

    sys.path.insert(0, str(checkout))
    manifest = load_manifest(manifest_path)
    settings = manifest["estimator"]
    rows = _prompt_rows(manifest, None if args.worker_all else args.worker_prompt_id)
    report: dict[str, Any] = {
        "schema_version": "rom-jspace-r1-benchmark-worker-v1",
        "status": "running",
        "authorised_flag": True,
        "manifest_sha256": sha256_file(manifest_path),
        "prereg_sha256": sha256_file(DEFAULT_PREREG),
        "jlens_commit": manifest["jlens"]["commit"],
        "model_revision": manifest["model"]["revision"],
        "dim_batch": args.dim_batch,
        "prompt_scope": "all" if args.worker_all else args.worker_prompt_id,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_paths": (git_value("status", "--porcelain") or "").splitlines(),
        "prompts": [],
        "exceptions": [],
    }

    try:
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from jlens import from_hf

        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("sealed benchmark requires an available CUDA device")
        torch.manual_seed(settings["seed"])
        torch.cuda.manual_seed_all(settings["seed"])
        torch.use_deterministic_algorithms(bool(settings["deterministic_algorithms"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        total_started = time.perf_counter()

        model_spec = manifest["model"]
        common = {
            "revision": model_spec["revision"],
            "cache_dir": args.cache_dir,
            "local_files_only": args.local_files_only,
        }
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_spec["id"], **common)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_spec["id"],
            torch_dtype=torch.bfloat16,
            attn_implementation=settings["attention_backend"],
            **common,
        ).to(args.device)
        lens_model = from_hf(model, tokenizer, compile=False, force_bos=True)
        if lens_model.d_model != model_spec["d_model"]:
            raise RuntimeError(f"d_model mismatch: {lens_model.d_model}")
        if lens_model.n_layers != model_spec["n_layers"]:
            raise RuntimeError(f"n_layers mismatch: {lens_model.n_layers}")

        snapshot = Path(
            snapshot_download(
                model_spec["id"],
                revision=model_spec["revision"],
                cache_dir=args.cache_dir,
                local_files_only=True,
            )
        )
        report["tokenizer_hashes"] = _tokenizer_file_hashes(snapshot)
        report["software_versions"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        properties = torch.cuda.get_device_properties(torch.device(args.device))
        report["hardware"] = {
            "device": args.device,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "cuda_runtime": torch.version.cuda,
        }
        report["load_dtype"] = str(next(model.parameters()).dtype)
        report["coordinate_identity"] = _coordinate_identity(model, lens_model, rows[0]["prompt"])
        setup_gpu_allocated = int(torch.cuda.max_memory_allocated())
        setup_gpu_reserved = int(torch.cuda.max_memory_reserved())

        first_jacobians = None
        for index, row in enumerate(rows):
            prompt_report, jacobians = _run_one_prompt(
                lens_model,
                row,
                source_layers=settings["source_layers"],
                target_layer=settings["target_layer"],
                dim_batch=args.dim_batch,
                max_seq_len=settings["max_seq_len"],
                skip_first=settings["skip_first"],
            )
            report["prompts"].append(prompt_report)
            if index == 0 and args.repeat_first:
                first_jacobians = jacobians
                repeat_report, repeated = _run_one_prompt(
                    lens_model,
                    row,
                    source_layers=settings["source_layers"],
                    target_layer=settings["target_layer"],
                    dim_batch=args.dim_batch,
                    max_seq_len=settings["max_seq_len"],
                    skip_first=settings["skip_first"],
                )
                report["repeat_first"] = {
                    "task_id": row["task_id"],
                    "wall_seconds": repeat_report["wall_seconds"],
                    "peak_gpu_allocated_bytes": repeat_report["peak_gpu_allocated_bytes"],
                    "peak_gpu_reserved_bytes": repeat_report["peak_gpu_reserved_bytes"],
                    "peak_cpu_rss_bytes": repeat_report["peak_cpu_rss_bytes"],
                    "peak_cpu_rss_measurement": repeat_report[
                        "peak_cpu_rss_measurement"
                    ],
                    "relative_frobenius": _repeat_relative_frobenius(
                        first_jacobians, repeated
                    ),
                }
                del repeated, first_jacobians
            del jacobians

        torch.cuda.synchronize()
        measured_phases = [*report["prompts"]]
        if "repeat_first" in report:
            measured_phases.append(report["repeat_first"])
        report["peak_gpu_allocated_bytes"] = max(
            [setup_gpu_allocated]
            + [row["peak_gpu_allocated_bytes"] for row in measured_phases]
        )
        report["peak_gpu_reserved_bytes"] = max(
            [setup_gpu_reserved]
            + [row["peak_gpu_reserved_bytes"] for row in measured_phases]
        )
        report["peak_cpu_rss_bytes"] = cpu_peak_rss_bytes()
        report["total_wall_seconds"] = time.perf_counter() - total_started
        report["status"] = "complete"
        atomic_json(args.worker_output, report)
        return 0
    except Exception as exc:
        oom = type(exc).__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"} or (
            "out of memory" in str(exc).lower()
        )
        report["status"] = "oom" if oom else "error"
        report["exceptions"].append(
            {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        )
        report["peak_cpu_rss_bytes"] = cpu_peak_rss_bytes()
        atomic_json(args.worker_output, report)
        return 42 if oom else 1


def evaluate_gate(report: dict[str, Any]) -> dict[str, Any]:
    expected_layers = {str(i) for i in range(27)}
    prompt_checks = []
    for prompt in report.get("prompts", []):
        layers = prompt.get("layers", {})
        prompt_checks.append(
            set(layers) == expected_layers
            and all(
                value.get("shape") == [1536, 1536]
                and value.get("dtype") == "torch.float32"
                and value.get("finite") is True
                for value in layers.values()
            )
        )
    hardware = report.get("hardware", {})
    headroom = hardware.get("total_memory_bytes", 0) - report.get("peak_gpu_reserved_bytes", 0)
    coordinate = report.get("coordinate_identity", {})
    resource_metrics_recorded = len(report.get("prompts", [])) == 5 and all(
        isinstance(prompt.get(field), int) and prompt[field] > 0
        for prompt in report.get("prompts", [])
        for field in (
            "peak_gpu_allocated_bytes",
            "peak_gpu_reserved_bytes",
            "peak_cpu_rss_bytes",
        )
    )
    repeat_max = (
        report.get("repeat_first", {}).get("relative_frobenius", {}).get("max", float("inf"))
    )
    checks = {
        "worker_complete": report.get("status") == "complete",
        "five_prompts": len(report.get("prompts", [])) == 5,
        "all_layer_matrices_valid": len(prompt_checks) == 5 and all(prompt_checks),
        "gpu_headroom_at_least_2gib": headroom >= 2 * 1024**3,
        "coordinate_module_identity": coordinate.get("same_module_object") is True,
        "coordinate_tensor_identity": coordinate.get("allclose_rtol_1e-5_atol_1e-6") is True,
        "per_prompt_resource_metrics_recorded": resource_metrics_recorded,
        "repeat_relative_frobenius_le_1e-6": repeat_max <= 1e-6,
    }
    return {"pass": all(checks.values()), "checks": checks, "gpu_headroom_bytes": headroom}


def _worker_command(args: argparse.Namespace, *, dim_batch: int, output: Path,
                    prompt_id: str | None, repeat_first: bool) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--authorised",
        "--manifest",
        str(args.manifest.resolve()),
        "--jlens-checkout",
        str(args.jlens_checkout.resolve()),
        "--device",
        args.device,
        "--dim-batch",
        str(dim_batch),
        "--worker-output",
        str(output),
    ]
    if prompt_id is None:
        command.append("--worker-all")
    else:
        command.extend(["--worker-prompt-id", prompt_id])
    if repeat_first:
        command.append("--repeat-first")
    if args.cache_dir:
        command.extend(["--cache-dir", args.cache_dir])
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def controller(args: argparse.Namespace) -> int:
    require_launch_authorisation(args.authorised)
    check = preflight.validate(
        args.manifest.resolve(), root=ROOT, jlens_checkout=args.jlens_checkout.resolve()
    )
    if not check["ok"]:
        print(json.dumps(check, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    manifest = load_manifest(args.manifest)
    output_root = ROOT / manifest["output_root"]
    if output_root.exists():
        raise SystemExit(
            f"REFUSAL: output root already exists; no overwrite/resume is sealed: {output_root}"
        )
    output_root.mkdir(parents=True)

    attempts: list[dict[str, Any]] = []
    selected: int | None = None
    for dim_batch in manifest["estimator"]["dim_batch_attempt_order"]:
        path = output_root / f"probe_dim{dim_batch}.json"
        command = _worker_command(
            args, dim_batch=dim_batch, output=path, prompt_id="MATH_000", repeat_first=False
        )
        completed = subprocess.run(command, check=False)
        payload = json.loads(path.read_text()) if path.is_file() else {"status": "missing_report"}
        attempts.append(
            {"dim_batch": dim_batch, "returncode": completed.returncode, "status": payload.get("status")}
        )
        if completed.returncode == 0 and payload.get("status") == "complete":
            selected = dim_batch
            break
        if completed.returncode != 42:
            break

    controller_report: dict[str, Any] = {
        "schema_version": "rom-jspace-r1-benchmark-controller-v1",
        "authorised_flag": True,
        "manifest_sha256": sha256_file(args.manifest),
        "prereg_sha256": sha256_file(DEFAULT_PREREG),
        "attempts": attempts,
        "selected_dim_batch": selected,
    }
    if selected is None:
        controller_report["status"] = "no_compatible_dim_batch"
        controller_report["gate"] = {"pass": False}
        atomic_json(output_root / "benchmark_report.json", controller_report)
        return 1

    full_path = output_root / f"five_prompt_dim{selected}.json"
    command = _worker_command(
        args, dim_batch=selected, output=full_path, prompt_id=None, repeat_first=True
    )
    completed = subprocess.run(command, check=False)
    full = json.loads(full_path.read_text()) if full_path.is_file() else {"status": "missing_report"}
    gate = evaluate_gate(full)
    prompt_seconds = sum(row.get("wall_seconds", 0.0) for row in full.get("prompts", []))
    controller_report.update(
        {
            "status": "complete" if completed.returncode == 0 else "worker_failed",
            "full_worker_returncode": completed.returncode,
            "full_worker_report": str(full_path.relative_to(ROOT)),
            "projected_100_prompt_wall_seconds": (
                prompt_seconds / len(full.get("prompts", [])) * 100
                if full.get("prompts")
                else None
            ),
            "projection_basis": "mean measured Jacobian call time; excludes model load and repeat",
            "gate": gate,
        }
    )
    atomic_json(output_root / "benchmark_report.json", controller_report)
    return 0 if completed.returncode == 0 and gate["pass"] else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorised", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--jlens-checkout", type=Path, required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-all", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-prompt-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repeat-first", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dim-batch", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.worker_output is None or args.dim_batch is None:
            parser.error("worker mode requires --worker-output and --dim-batch")
        if args.worker_all == (args.worker_prompt_id is not None):
            parser.error("worker mode requires exactly one of --worker-all/--worker-prompt-id")
    return args


def main() -> int:
    args = parse_args()
    return worker(args) if args.worker else controller(args)


if __name__ == "__main__":
    sys.exit(main())
