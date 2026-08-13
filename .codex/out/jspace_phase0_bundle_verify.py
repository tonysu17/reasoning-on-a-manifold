#!/usr/bin/env python3
"""Verify the complete network-volume bundle before the J-space Phase-0 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_BUNDLE_FILES = (
    "jspace_pilot_preflight.py",
    "jspace_pilot_benchmark.py",
    "jspace_phase0_bundle_verify.py",
    "jspace_phase0_pod_job.sh",
    "jspace_phase0_watch.py",
    "results/prereg/JSPACE_R1_BENCHMARK_MANIFEST_2026-08-10.json",
    "results/prereg/JSPACE_R1_STEERING_PILOT_PREREG_2026-08-10.md",
    "data/tasks_final.json",
    "results/steering_vectors/R1-1.5B__E1_pooled/backtracking_single.npy",
    "results/steering_vectors/R1-1.5B__E1_pooled/metadata.json",
    "results/eval/R1-1.5B__E1/eval_task_ids.json",
    "_external/jacobian-lens/pyproject.toml",
    "_external/jacobian-lens/uv.lock",
    "_external/jacobian-lens/jlens/fitting.py",
    "_external/jacobian-lens/jlens/hf.py",
    "_external/jacobian-lens/jlens/lens.py",
    "ENVIRONMENT_FREEZE.txt",
)

MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
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


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    cache_dir = args.cache_dir.resolve()
    manifest_path = root / "results/prereg/JSPACE_R1_BENCHMARK_MANIFEST_2026-08-10.json"
    manifest = json.loads(manifest_path.read_text())
    revision = manifest["model"]["revision"]
    model_root = cache_dir / "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
    snapshot = model_root / "snapshots" / revision

    errors: list[str] = []
    required_hashes: dict[str, str] = {}
    for relative in REQUIRED_BUNDLE_FILES:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing bundle file: {relative}")
        else:
            required_hashes[relative] = sha256_file(path)

    sys.path.insert(0, str(root))
    import jspace_pilot_preflight as preflight

    preflight_report = preflight.validate(
        manifest_path,
        root=root,
        jlens_checkout=root / "_external/jacobian-lens",
    )
    if not preflight_report["ok"]:
        errors.extend(f"preflight: {message}" for message in preflight_report["errors"])

    ref_path = model_root / "refs/main"
    ref_value = ref_path.read_text().strip() if ref_path.is_file() else None
    if ref_value != revision:
        errors.append(f"model ref mismatch: {ref_value!r} != {revision!r}")

    model_hashes: dict[str, str] = {}
    for name in MODEL_FILES:
        path = snapshot / name
        if not path.is_file():
            errors.append(f"missing cached model file: {name}")
        else:
            model_hashes[name] = sha256_file(path)

    jlens_head = git_head(root / "_external/jacobian-lens")
    if jlens_head != manifest["jlens"]["commit"]:
        errors.append(
            f"J-lens commit mismatch: {jlens_head!r} != {manifest['jlens']['commit']!r}"
        )

    output_root = root / manifest["output_root"]
    if output_root.exists():
        errors.append(f"sealed output root already exists: {output_root}")

    import torch
    import transformers
    import huggingface_hub

    if transformers.__version__ != "5.5.0":
        errors.append(f"unexpected transformers version: {transformers.__version__}")
    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable")

    report = {
        "schema_version": "rom-jspace-phase0-bundle-verification-v1",
        "ok": not errors,
        "errors": errors,
        "root": str(root),
        "network_volume": str(root).startswith("/workspace/"),
        "preflight": preflight_report,
        "required_bundle_file_count": len(REQUIRED_BUNDLE_FILES),
        "required_bundle_hashes": required_hashes,
        "model": {
            "id": manifest["model"]["id"],
            "revision": revision,
            "ref_main": ref_value,
            "snapshot": str(snapshot),
            "file_hashes": model_hashes,
        },
        "jlens_commit": jlens_head,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "sealed_output_absent": not output_root.exists(),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
