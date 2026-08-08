#!/usr/bin/env python3
"""Build checkpoint-weight interpolation organisms for the sealed pt19 alpha grid.

    theta_alpha = theta_R1 + alpha * (theta_STAR1 - theta_R1)

The implementation is shard-at-a-time and CPU-only. It never loads a whole checkpoint into
memory. `--dry-run` resolves both safetensors manifests and validates three probe tensors
without creating an output directory. This tool only constructs checkpoints; it does not run
extraction, generation, evaluation, or any Phase-1 hypothesis test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_ALPHAS = (0.25, 0.5, 0.75, 1.0)
INDEX_NAME = "model.safetensors.index.json"


@dataclass(frozen=True)
class TensorLocation:
    filename: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class CheckpointManifest:
    root: Path
    tensors: dict[str, TensorLocation]
    shard_order: tuple[str, ...]
    index: dict | None


def interpolate_tensor(base: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Interpolate one tensor, preserving endpoint values and the base dtype exactly."""
    if base.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(base.shape)} vs {tuple(target.shape)}")
    if base.dtype != target.dtype:
        raise ValueError(f"dtype mismatch: {base.dtype} vs {target.dtype}")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be finite and within [0, 1], got {alpha}")
    if alpha == 0.0:
        return base.clone()
    if alpha == 1.0:
        return target.clone()
    if not (base.is_floating_point() or base.is_complex()):
        if torch.equal(base, target):
            return base.clone()
        raise ValueError("non-floating endpoint tensors differ at an intermediate alpha")
    # Float32 accumulation avoids fp16/bfloat16 subtraction overflow while retaining the
    # checkpoint's original storage dtype in the emitted state dict.
    work_dtype = torch.complex64 if base.is_complex() else torch.float32
    merged = base.to(work_dtype).lerp(target.to(work_dtype), alpha)
    return merged.to(base.dtype)


def interpolate_state_dict(base: Mapping[str, torch.Tensor],
                           target: Mapping[str, torch.Tensor],
                           alpha: float) -> dict[str, torch.Tensor]:
    """Tiny-state-dict reference implementation used by unit tests and probes."""
    if set(base) != set(target):
        missing = sorted(set(base) - set(target))
        extra = sorted(set(target) - set(base))
        raise ValueError(f"tensor-key mismatch; target missing={missing[:5]}, extra={extra[:5]}")
    return {key: interpolate_tensor(base[key], target[key], alpha) for key in sorted(base)}


def _single_or_index(root: Path) -> tuple[dict[str, str], dict | None, tuple[str, ...]]:
    index_path = root / INDEX_NAME
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path} has no non-empty weight_map")
        order = tuple(dict.fromkeys(str(name) for name in weight_map.values()))
        return {str(key): str(value) for key, value in weight_map.items()}, index, order

    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors or {INDEX_NAME} under {root}")
    weight_map: dict[str, str] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in weight_map:
                    raise ValueError(f"duplicate tensor {key!r} across safetensors shards")
                weight_map[key] = shard.name
    return weight_map, None, tuple(shard.name for shard in shards)


def load_manifest(root: Path) -> CheckpointManifest:
    root = root.resolve()
    weight_map, index, shard_order = _single_or_index(root)
    tensors: dict[str, TensorLocation] = {}
    by_shard: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        by_shard.setdefault(filename, []).append(key)
    for filename in shard_order:
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"manifest names missing shard {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual = set(handle.keys())
            declared = set(by_shard.get(filename, []))
            if index is not None and actual != declared:
                raise ValueError(
                    f"index/shard key mismatch in {filename}: "
                    f"missing={sorted(declared-actual)[:5]}, extra={sorted(actual-declared)[:5]}")
            for key in sorted(actual):
                tensor = handle.get_slice(key)
                tensors[key] = TensorLocation(
                    filename=filename,
                    shape=tuple(tensor.get_shape()),
                    dtype=str(tensor.get_dtype()),
                )
    return CheckpointManifest(root=root, tensors=tensors, shard_order=shard_order, index=index)


def validate_manifests(base: CheckpointManifest, target: CheckpointManifest) -> None:
    if set(base.tensors) != set(target.tensors):
        missing = sorted(set(base.tensors) - set(target.tensors))
        extra = sorted(set(target.tensors) - set(base.tensors))
        raise ValueError(f"tensor-key mismatch; target missing={missing[:10]}, extra={extra[:10]}")
    mismatches = []
    for key in sorted(base.tensors):
        left, right = base.tensors[key], target.tensors[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            mismatches.append((key, left.shape, left.dtype, right.shape, right.dtype))
    if mismatches:
        raise ValueError(f"shape/dtype mismatch (first 5): {mismatches[:5]}")


def _open_tensor(manifest: CheckpointManifest, key: str, handles: dict[str, object],
                 stack: ExitStack) -> torch.Tensor:
    filename = manifest.tensors[key].filename
    if filename not in handles:
        handles[filename] = stack.enter_context(
            safe_open(manifest.root / filename, framework="pt", device="cpu"))
    return handles[filename].get_tensor(key)  # type: ignore[union-attr]


def dry_run_probe(base: CheckpointManifest, target: CheckpointManifest,
                  alphas: Sequence[float], n_tensors: int = 3) -> dict:
    validate_manifests(base, target)
    keys = sorted(base.tensors)[:n_tensors]
    if len(keys) < n_tensors:
        raise ValueError(f"dry-run requested {n_tensors} tensors but manifest has {len(keys)}")
    rows = []
    with ExitStack() as stack:
        base_handles: dict[str, object] = {}
        target_handles: dict[str, object] = {}
        for key in keys:
            left = _open_tensor(base, key, base_handles, stack)
            right = _open_tensor(target, key, target_handles, stack)
            probes = {}
            for alpha in alphas:
                value = interpolate_tensor(left, right, alpha)
                probes[str(alpha)] = {
                    "finite": bool(torch.isfinite(value).all()) if value.is_floating_point() else True,
                    "norm": float(value.float().norm()),
                    "matches_base_exactly": bool(torch.equal(value, left)),
                    "matches_target_exactly": bool(torch.equal(value, right)),
                }
            rows.append({
                "tensor": key,
                "shape": list(left.shape),
                "dtype": str(left.dtype),
                "base_target_delta_norm": float((right.float() - left.float()).norm()),
                "alphas": probes,
            })
    return {"status": "dry-run only; no output checkpoint written", "probe_tensors": rows}


def _alpha_dirname(alpha: float) -> str:
    return "alpha_" + f"{alpha:.8g}".replace("-", "m").replace(".", "p")


def _copy_non_weight_files(source: Path, destination: Path) -> None:
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".safetensors" or path.name == INDEX_NAME:
            continue
        shutil.copy2(path, destination / path.name)


def merge_one_alpha(base: CheckpointManifest, target: CheckpointManifest, alpha: float,
                    output_root: Path) -> Path:
    validate_manifests(base, target)
    final_dir = output_root / _alpha_dirname(alpha)
    partial_dir = output_root / (_alpha_dirname(alpha) + ".partial")
    if final_dir.exists() or partial_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing alpha output: {final_dir} or {partial_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir()
    _copy_non_weight_files(base.root, partial_dir)

    keys_by_base_shard: dict[str, list[str]] = {}
    for key, location in base.tensors.items():
        keys_by_base_shard.setdefault(location.filename, []).append(key)

    total_size = 0
    for filename in base.shard_order:
        shard_tensors: dict[str, torch.Tensor] = {}
        with ExitStack() as stack:
            base_handles: dict[str, object] = {}
            target_handles: dict[str, object] = {}
            for key in sorted(keys_by_base_shard[filename]):
                left = _open_tensor(base, key, base_handles, stack)
                right = _open_tensor(target, key, target_handles, stack)
                merged = interpolate_tensor(left, right, alpha).contiguous()
                shard_tensors[key] = merged
                total_size += merged.numel() * merged.element_size()
            base_metadata = base_handles[filename].metadata()  # type: ignore[union-attr]
        temporary = partial_dir / (filename + ".partial")
        save_file(shard_tensors, temporary, metadata=base_metadata)
        os.replace(temporary, partial_dir / filename)
        del shard_tensors

    if base.index is not None:
        index = json.loads(json.dumps(base.index))
        index.setdefault("metadata", {})["total_size"] = total_size
        (partial_dir / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")

    manifest = {
        "script": "pt19_alpha_dial.py",
        "status": "checkpoint construction only; no evaluation run",
        "formula": "theta_alpha = theta_R1 + alpha * (theta_STAR1 - theta_R1)",
        "alpha": alpha,
        "base": str(base.root),
        "target": str(target.root),
        "n_tensors": len(base.tensors),
        "output_shards": list(base.shard_order),
        "endpoint_exactness": "alpha=0 copies base tensors; alpha=1 copies target tensors",
    }
    (partial_dir / "pt19_alpha_dial_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    partial_dir.rename(final_dir)
    return final_dir


def parse_alphas(values: Sequence[float]) -> tuple[float, ...]:
    alphas = tuple(float(value) for value in values)
    if not alphas:
        raise ValueError("at least one alpha is required")
    if len(set(alphas)) != len(alphas):
        raise ValueError("duplicate alpha values are not allowed")
    for alpha in alphas:
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be finite and within [0, 1], got {alpha}")
    return alphas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--star1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate manifests and exactly three tensors; write nothing")
    args = parser.parse_args()

    alphas = parse_alphas(args.alphas)
    base = load_manifest(args.base_dir)
    target = load_manifest(args.star1_dir)
    validate_manifests(base, target)
    if args.dry_run:
        print(json.dumps(dry_run_probe(base, target, alphas, n_tensors=3), indent=2))
        return

    outputs = [str(merge_one_alpha(base, target, alpha, args.output_dir))
               for alpha in alphas]
    print(json.dumps({"outputs": outputs, "status": "constructed; not evaluated"}, indent=2))


if __name__ == "__main__":
    main()
