#!/usr/bin/env python3
"""Run the authorised, sealed R1 Jacobian-lens Phase-1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import jspace_phase1_scoring as scoring


SOURCE_LAYERS = list(range(27))
TARGET_LAYER = 27
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
DIM_BATCH = 8
MAX_SEQ_LEN = 128
SKIP_FIRST = 16
CHECKPOINT_EVERY = 10


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


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_lens_save(lens: Any, path: Path, dtype: Any) -> tuple[Any, dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        lens.save(str(temporary), dtype=dtype)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        loaded = type(lens).load(str(temporary))
        temporary_hash = sha256_file(temporary)
        size_bytes = temporary.stat().st_size
        os.replace(temporary, path)
        final_hash = sha256_file(path)
        if final_hash != temporary_hash:
            raise RuntimeError(f"lens hash changed across atomic rename: {path}")
        return loaded, {"sha256": final_hash, "size_bytes": size_bytes}
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}")


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, stderr=subprocess.DEVNULL
    ).decode().strip()


def relative_frobenius(reference: Any, candidate: Any) -> float:
    import torch

    denominator = torch.linalg.vector_norm(reference.float())
    numerator = torch.linalg.vector_norm(candidate.float() - reference.float())
    if denominator.item() == 0:
        return 0.0 if numerator.item() == 0 else float("inf")
    return float((numerator / denominator).item())


def bitwise_float32_equal(left: Any, right: Any) -> bool:
    import torch

    left_cpu = left.detach().to(device="cpu", dtype=torch.float32).contiguous()
    right_cpu = right.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return bool(torch.equal(left_cpu.view(torch.int32), right_cpu.view(torch.int32)))


def lens_comparison(reference: Any, candidate: Any) -> dict[str, Any]:
    import torch

    if reference.source_layers != SOURCE_LAYERS or candidate.source_layers != SOURCE_LAYERS:
        raise ValueError("lens source-layer mismatch")
    per_layer = {
        str(layer): relative_frobenius(
            reference.jacobians[layer], candidate.jacobians[layer]
        )
        for layer in SOURCE_LAYERS
    }
    exact = {
        str(layer): bitwise_float32_equal(
            reference.jacobians[layer], candidate.jacobians[layer]
        )
        for layer in SOURCE_LAYERS
    }
    return {
        "relative_frobenius_by_layer": per_layer,
        "max_relative_frobenius": max(per_layer.values()),
        "bitwise_equal_by_layer": exact,
        "all_layers_bitwise_equal": all(exact.values()),
    }


def validate_lens_matrices(lens: Any, expected_prompts: int) -> dict[str, Any]:
    import torch

    layers = {}
    for layer in SOURCE_LAYERS:
        matrix = lens.jacobians[layer]
        layers[str(layer)] = {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "finite": bool(torch.isfinite(matrix).all().item()),
        }
    checks = {
        "source_layers_exact": lens.source_layers == SOURCE_LAYERS,
        "d_model_1536": lens.d_model == 1536,
        "n_prompts_exact": lens.n_prompts == expected_prompts,
        "all_shape_1536x1536": all(
            value["shape"] == [1536, 1536] for value in layers.values()
        ),
        "all_float32_in_memory": all(
            value["dtype"] == "torch.float32" for value in layers.values()
        ),
        "all_finite": all(value["finite"] for value in layers.values()),
    }
    return {"layers": layers, "checks": checks, "pass": all(checks.values())}


def verify_manifest_token_ids(lens_model: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        actual = lens_model.encode(row["prompt"], max_length=MAX_SEQ_LEN)[0].cpu().tolist()
        if actual != row["fit_token_ids_128"]:
            raise ValueError(
                f"effective token IDs differ for selected_order={row['selected_order']}"
            )


def capture_source_activations(lens_model: Any, prompt: str) -> tuple[Any, Any]:
    import torch
    from jlens.hooks import ActivationRecorder

    input_ids = lens_model.encode(prompt, max_length=MAX_SEQ_LEN)
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=SOURCE_LAYERS) as recorder:
        lens_model.forward(input_ids)
        activations = torch.stack(
            [recorder.activations[layer][0].detach() for layer in SOURCE_LAYERS], dim=0
        )
    return input_ids, activations


def gpu_jacobian_stack(lens: Any, device: str) -> Any:
    import torch

    return torch.stack(
        [lens.jacobians[layer].to(device=device, dtype=torch.float32) for layer in SOURCE_LAYERS],
        dim=0,
    )


def residual_top25(lens_model: Any, residuals: Any) -> tuple[np.ndarray, int]:
    """Read [layer,position,d] residuals in one unembedding batch."""
    shape = residuals.shape[:-1]
    import torch

    full_logits = lens_model.unembed(residuals.reshape(-1, residuals.shape[-1]))
    logits = scoring.tokenizer_vocabulary_logits(
        full_logits, head_vocab_size=151936, tokenizer_vocab_size=151665
    )
    ids, ties = scoring.deterministic_topk_ids(logits, k=25)
    result = ids.reshape(*shape, 25).to(dtype=torch.int32).cpu().numpy()
    del full_logits, logits, ids
    return result, ties


def transport_top25(
    lens_model: Any, residuals: Any, jacobian_stack: Any
) -> tuple[np.ndarray, int]:
    import torch

    transported = torch.bmm(residuals.float(), jacobian_stack.transpose(1, 2))
    result, ties = residual_top25(lens_model, transported)
    del transported
    return result, ties


def heldout_readouts(
    lens_model: Any,
    rows: list[dict[str, Any]],
    j_a: Any,
    j_b: Any,
    *,
    chunk_positions: int = 32,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    all_a = []
    all_b = []
    ties = {"fit_a": 0, "fit_b": 0}
    for row_index, row in enumerate(rows):
        input_ids, activations = capture_source_activations(lens_model, row["prompt"])
        if input_ids.shape[1] != MAX_SEQ_LEN:
            raise ValueError(f"held-out row {row_index} did not truncate to 128 tokens")
        positions = list(range(SKIP_FIRST, input_ids.shape[1] - 1))
        row_a = []
        row_b = []
        for start in range(0, len(positions), chunk_positions):
            selected = positions[start : start + chunk_positions]
            residuals = activations[:, selected, :]
            a, n_ties = transport_top25(lens_model, residuals, j_a)
            ties["fit_a"] += n_ties
            b, n_ties = transport_top25(lens_model, residuals, j_b)
            ties["fit_b"] += n_ties
            row_a.append(a)
            row_b.append(b)
        all_a.append(np.concatenate(row_a, axis=1))
        all_b.append(np.concatenate(row_b, axis=1))
        del input_ids, activations
        logging.info("held-out readout %d/%d", row_index + 1, len(rows))
    return np.stack(all_a), np.stack(all_b), ties


def external_readout_item(
    lens_model: Any,
    prompt: str,
    j_a: Any,
    j_b: Any,
    j_merged: Any,
) -> tuple[dict[str, np.ndarray], dict[str, int], list[int]]:
    input_ids, activations = capture_source_activations(lens_model, prompt)
    position = input_ids.shape[1] - 1
    residuals = activations[:, position : position + 1, :]
    output: dict[str, np.ndarray] = {}
    ties: dict[str, int] = {}
    for name, stack in (("fit_a", j_a), ("fit_b", j_b), ("merged", j_merged)):
        ids, n_ties = transport_top25(lens_model, residuals, stack)
        output[name] = ids[:, 0, :]
        ties[name] = n_ties
    ids, n_ties = residual_top25(lens_model, residuals)
    output["logit"] = ids[:, 0, :]
    ties["logit"] = n_ties
    return output, ties, input_ids[0].cpu().tolist()


def run_external_evaluations(
    lens_model: Any,
    eligibility: dict[str, Any],
    j_a: Any,
    j_b: Any,
    j_merged: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reports: dict[str, Any] = {}
    stored: dict[str, np.ndarray] = {}
    permutation_rng = np.random.Generator(np.random.PCG64(20260812))
    total_ties = {name: 0 for name in ("fit_a", "fit_b", "merged", "logit")}
    for slug in eligibility["evaluation_order"]:
        eligible_items = [
            item
            for item in eligibility["evaluations"][slug]["items"]
            if item["item_eligible"]
        ]
        by_lens: dict[str, list[np.ndarray]] = {
            name: [] for name in ("fit_a", "fit_b", "merged", "logit")
        }
        labels: list[list[int]] = []
        for item_index, item in enumerate(eligible_items):
            outputs, ties, actual_ids = external_readout_item(
                lens_model, item["prompt"], j_a, j_b, j_merged
            )
            if actual_ids != item["prompt_token_ids"]:
                raise ValueError(f"external prompt IDs differ: {slug}:{item['name']}")
            if item["readout_position"] != len(actual_ids) - 1:
                raise ValueError(f"external readout position differs: {slug}:{item['name']}")
            for name, values in outputs.items():
                by_lens[name].append(values)
                total_ties[name] += ties[name]
            labels.append(
                [
                    int(label["scored_token_ids"][0])
                    for label in item["labels"]
                    if label["eligible"]
                ]
            )
            if (item_index + 1) % 10 == 0 or item_index + 1 == len(eligible_items):
                logging.info("external %s readout %d/%d", slug, item_index + 1, len(eligible_items))
        arrays = {name: np.stack(values) for name, values in by_lens.items()}
        all_null, l17_null = scoring.external_permutation_null(
            arrays["merged"], labels, rng=permutation_rng, n_perm=1_000
        )
        report = scoring.external_report(arrays, labels, all_null, l17_null)
        report["eligible_item_names"] = [item["name"] for item in eligible_items]
        reports[slug] = report
        key = slug.replace("-", "_")
        for name, values in arrays.items():
            stored[f"{key}__top25__{name}"] = values.astype(np.int32)
        stored[f"{key}__null__all_layers"] = all_null
        stored[f"{key}__null__l17"] = l17_null
    reports["top25_boundary_tie_rows"] = total_ties
    return reports, stored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorised", action="store_true")
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jlens-checkout", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--run-uuid", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.authorised:
        raise SystemExit("REFUSAL: Phase-1 execution requires --authorised")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root = args.root.resolve()
    output = args.output_root.resolve()
    scratch = args.scratch_root.resolve() / args.run_uuid
    if output.exists() or scratch.exists():
        raise FileExistsError("output or run-scoped scratch path already exists")
    output.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.FileHandler(output / "phase1_runner.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    environment_sections = []
    for command in (
        [sys.executable, "-m", "pip", "freeze"],
        ["nvidia-smi"],
        ["uname", "-a"],
    ):
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        environment_sections.append(
            f"$ {' '.join(command)}\nexit={completed.returncode}\n"
            f"{completed.stdout}{completed.stderr}\n"
        )
    atomic_text(output / "RUN_ENVIRONMENT.txt", "\n".join(environment_sections))
    report_path = output / "phase1_report.json"
    report: dict[str, Any] = {
        "schema_version": "rom-jspace-r1-phase1-report-v1",
        "execution_status": "running",
        "run_uuid": args.run_uuid,
        "authorised_flag": True,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phases": {},
        "exceptions": [],
    }
    atomic_json(report_path, report)
    started = time.perf_counter()
    try:
        execution = json.loads(args.execution_manifest.read_text())
        for relative, expected in execution["fixed_inputs_sha256"].items():
            require_hash(root / relative, expected, relative)
        if execution["run_uuid"] != args.run_uuid:
            raise ValueError("run UUID differs from execution manifest")
        checkout = args.jlens_checkout.resolve()
        if git_head(checkout) != JLENS_COMMIT:
            raise ValueError("J-lens commit mismatch")
        dirty_checkout = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=checkout,
        ).decode().strip()
        if dirty_checkout:
            raise ValueError(f"J-lens tracked checkout is dirty: {dirty_checkout}")
        for relative, expected in execution["jlens_files_sha256"].items():
            require_hash(checkout / relative, expected, f"J-lens/{relative}")
        corpus = json.loads((root / execution["corpus_manifest"]).read_text())
        eligibility = json.loads((root / execution["eligibility_manifest"]).read_text())
        if [row["split"] for row in corpus["rows"]].count("fit_a") != 50:
            raise ValueError("fit A is not 50 rows")
        if [row["split"] for row in corpus["rows"]].count("fit_b") != 50:
            raise ValueError("fit B is not 50 rows")
        if [row["split"] for row in corpus["rows"]].count("heldout") != 20:
            raise ValueError("heldout is not 20 rows")
        report["fixed_inputs_sha256"] = execution["fixed_inputs_sha256"]
        report["source_git_commit"] = execution["source_git_commit"]
        report["execution_manifest_sha256"] = sha256_file(args.execution_manifest)
        report["pod_id"] = os.environ.get("RUNPOD_POD_ID")
        report["phases"]["preflight"] = {"status": "passed"}
        atomic_json(report_path, report)

        sys.path.insert(0, str(checkout))
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        import jlens
        from jlens import JacobianLens, fit, from_hf

        imported_jlens = Path(jlens.__file__).resolve()
        if checkout not in imported_jlens.parents:
            raise ValueError(
                f"imported jlens from {imported_jlens}, not pinned checkout {checkout}"
            )

        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("sealed Phase 1 requires CUDA")
        torch.manual_seed(20260810)
        torch.cuda.manual_seed_all(20260810)
        torch.use_deterministic_algorithms(True)
        snapshot = Path(
            snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(args.cache_dir),
                local_files_only=True,
            )
        )
        actual_model_files = {
            str(path.relative_to(snapshot))
            for path in snapshot.rglob("*")
            if path.is_file()
        }
        if actual_model_files != set(execution["model_files_sha256"]):
            raise ValueError(
                "model snapshot file inventory mismatch: "
                f"missing={sorted(set(execution['model_files_sha256'])-actual_model_files)} "
                f"extra={sorted(actual_model_files-set(execution['model_files_sha256']))}"
            )
        for relative, expected in execution["model_files_sha256"].items():
            require_hash(snapshot / relative, expected, f"model/{relative}")
        for name, expected in corpus["tokenizer_files_sha256"].items():
            require_hash(snapshot / name, expected, f"tokenizer/{name}")
        common = {
            "revision": MODEL_REVISION,
            "cache_dir": str(args.cache_dir),
            "local_files_only": True,
        }
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, **common)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            **common,
        ).to(args.device)
        lens_model = from_hf(model, tokenizer, compile=False, force_bos=True)
        if lens_model.d_model != 1536 or lens_model.n_layers != 28:
            raise ValueError("model architecture mismatch")
        tokenizer_id_values = list(tokenizer.get_vocab().values())
        tokenizer_ids = set(tokenizer_id_values)
        if (
            len(tokenizer) != 151665
            or len(tokenizer_id_values) != len(tokenizer_ids)
            or tokenizer_ids != set(range(151665))
        ):
            raise ValueError("tokenizer IDs are not the registered contiguous 0..151664")
        if (
            model.config.vocab_size != 151936
            or model.lm_head.weight.shape[0] != 151936
        ):
            raise ValueError("model output head is not the registered padded size 151936")
        verify_manifest_token_ids(lens_model, corpus["rows"])
        properties = torch.cuda.get_device_properties(torch.device(args.device))
        observed_versions = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        }
        if observed_versions != execution["software_versions"]:
            raise ValueError(
                f"software versions differ: {observed_versions} != "
                f"{execution['software_versions']}"
            )
        report["environment"] = {
            **observed_versions,
            "device": args.device,
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "model_dtype": str(next(model.parameters()).dtype),
            "model_snapshot": str(snapshot),
            "raw_head_domain": int(model.lm_head.weight.shape[0]),
            "primary_ranking_domain": len(tokenizer),
            "valid_token_id_min": min(tokenizer_ids),
            "valid_token_id_max": max(tokenizer_ids),
            "stability_null_domain": len(tokenizer),
            "excluded_head_padding_rows": int(
                model.lm_head.weight.shape[0] - len(tokenizer)
            ),
            "jlens_commit": JLENS_COMMIT,
        }
        report["phases"]["model_and_token_identity"] = {"status": "passed"}
        atomic_json(report_path, report)

        rows_a = [row for row in corpus["rows"] if row["split"] == "fit_a"]
        rows_b = [row for row in corpus["rows"] if row["split"] == "fit_b"]
        heldout_rows = [row for row in corpus["rows"] if row["split"] == "heldout"]
        fit_started = time.perf_counter()
        lens_a = fit(
            lens_model,
            [row["prompt"] for row in rows_a],
            source_layers=SOURCE_LAYERS,
            target_layer=TARGET_LAYER,
            dim_batch=DIM_BATCH,
            max_seq_len=MAX_SEQ_LEN,
            skip_first=SKIP_FIRST,
            checkpoint_path=str(scratch / "fit_a_checkpoint.pt"),
            checkpoint_every=CHECKPOINT_EVERY,
            resume=False,
        )
        if lens_a.n_prompts != 50:
            raise ValueError(f"fit A completed {lens_a.n_prompts}, not 50 prompts")
        report["phases"]["fit_a"] = {
            "status": "passed",
            "n_prompts": lens_a.n_prompts,
            "wall_seconds": time.perf_counter() - fit_started,
        }
        atomic_json(report_path, report)
        fit_b_started = time.perf_counter()
        lens_b = fit(
            lens_model,
            [row["prompt"] for row in rows_b],
            source_layers=SOURCE_LAYERS,
            target_layer=TARGET_LAYER,
            dim_batch=DIM_BATCH,
            max_seq_len=MAX_SEQ_LEN,
            skip_first=SKIP_FIRST,
            checkpoint_path=str(scratch / "fit_b_checkpoint.pt"),
            checkpoint_every=CHECKPOINT_EVERY,
            resume=False,
        )
        if lens_b.n_prompts != 50:
            raise ValueError(f"fit B completed {lens_b.n_prompts}, not 50 prompts")
        report["phases"]["fit_b"] = {
            "status": "passed",
            "n_prompts": lens_b.n_prompts,
            "wall_seconds": time.perf_counter() - fit_b_started,
        }
        lens_merged = JacobianLens.merge([lens_a, lens_b])
        report["phases"]["fit_total_wall_seconds"] = time.perf_counter() - fit_started
        atomic_json(report_path, report)

        matrix_reports = {
            "fit_a": validate_lens_matrices(lens_a, 50),
            "fit_b": validate_lens_matrices(lens_b, 50),
            "merged": validate_lens_matrices(lens_merged, 100),
        }
        expected_merged = JacobianLens(
            {
                layer: (lens_a.jacobians[layer] + lens_b.jacobians[layer]) / 2
                for layer in SOURCE_LAYERS
            },
            n_prompts=100,
            d_model=1536,
        )
        merge_identity = lens_comparison(expected_merged, lens_merged)
        lenses_dir = output / "lenses"
        serialization: dict[str, Any] = {}
        for name, lens in (("fit_a", lens_a), ("fit_b", lens_b), ("merged", lens_merged)):
            loaded32, file32 = atomic_lens_save(
                lens, lenses_dir / f"{name}.fp32.pt", torch.float32
            )
            loaded16, file16 = atomic_lens_save(
                lens, lenses_dir / f"{name}.fp16.pt", torch.float16
            )
            serialization[name] = {
                "fp32": {**lens_comparison(lens, loaded32), **file32},
                "fp16": {**lens_comparison(lens, loaded16), **file16},
            }
        numerical_checks = {
            "all_matrix_checks": all(value["pass"] for value in matrix_reports.values()),
            "merge_max_relative_frobenius_at_most_1e-6": merge_identity[
                "max_relative_frobenius"
            ] <= 1e-6,
            "fp32_all_bitwise_equal": all(
                value["fp32"]["all_layers_bitwise_equal"]
                for value in serialization.values()
            ),
            "fp32_max_relative_frobenius_at_most_1e-6": all(
                value["fp32"]["max_relative_frobenius"] <= 1e-6
                for value in serialization.values()
            ),
            "fp16_max_relative_frobenius_at_most_1e-3": all(
                value["fp16"]["max_relative_frobenius"] <= 1e-3
                for value in serialization.values()
            ),
        }
        numerical = {
            "matrix_reports": matrix_reports,
            "merge_identity": merge_identity,
            "serialization": serialization,
            "checks": numerical_checks,
            "pass": all(numerical_checks.values()),
        }
        atomic_json(output / "numerical_validation.json", numerical)
        if not numerical["pass"]:
            raise RuntimeError("numerical/serialization validity gate failed")
        report["phases"]["numerical_validation"] = {"status": "passed"}
        atomic_json(report_path, report)

        j_a = gpu_jacobian_stack(lens_a, args.device)
        j_b = gpu_jacobian_stack(lens_b, args.device)
        j_merged = gpu_jacobian_stack(lens_merged, args.device)
        heldout_started = time.perf_counter()
        top_a, top_b, heldout_ties = heldout_readouts(
            lens_model, heldout_rows, j_a, j_b
        )
        scan_null, layer_null = scoring.stability_permutation_null(
            top_a,
            top_b,
            vocab_size=len(tokenizer),
            seed=20260811,
            n_perm=1_000,
            device=args.device,
        )
        stability = scoring.stability_report(top_a, top_b, scan_null, layer_null)
        stability["top25_boundary_tie_rows"] = heldout_ties
        stability["wall_seconds"] = time.perf_counter() - heldout_started
        atomic_npz(
            output / "heldout_stability_arrays.npz",
            fit_a_top25=top_a.astype(np.int32),
            fit_b_top25=top_b.astype(np.int32),
            scan_null=scan_null,
            layer_null=layer_null,
        )
        atomic_json(output / "heldout_stability.json", stability)
        report["phases"]["heldout_stability"] = {
            "status": "complete",
            "scientific_gate_pass": stability["gate"]["pass"],
            "wall_seconds": stability["wall_seconds"],
        }
        atomic_json(report_path, report)

        external_started = time.perf_counter()
        external, external_arrays = run_external_evaluations(
            lens_model, eligibility, j_a, j_b, j_merged
        )
        qualifying = [
            slug
            for slug in eligibility["evaluation_order"]
            if external[slug]["qualifying_success"]
        ]
        external_gate = {
            "qualifying_evaluations": qualifying,
            "n_qualifying": len(qualifying),
            "required": 2,
            "pass": len(qualifying) >= 2,
        }
        external["gate"] = external_gate
        external["wall_seconds"] = time.perf_counter() - external_started
        atomic_npz(output / "external_readout_arrays.npz", **external_arrays)
        atomic_json(output / "external_positive_controls.json", external)
        report["phases"]["external_positive_controls"] = {
            "status": "complete",
            "scientific_gate_pass": external_gate["pass"],
            "wall_seconds": external["wall_seconds"],
        }

        scientific_gate = {
            "numerical_serialization": numerical["pass"],
            "heldout_stability": stability["gate"]["pass"],
            "external_positive_controls": external_gate["pass"],
        }
        scientific_gate["pass"] = all(scientific_gate.values())
        report["scientific_gate"] = scientific_gate
        report["licensed_consequence"] = (
            "Phase_2_decomposition_may_be_considered_but_requires_separate_authorisation"
            if scientific_gate["pass"]
            else "stop_before_Phase_2_the_fitted_lens_did_not_pass_the_prespecified_validity_gate"
        )
        report["execution_status"] = "valid_complete"
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["total_wall_seconds"] = time.perf_counter() - started
        report["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        report["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        atomic_json(report_path, report)
        return 0
    except Exception as exc:
        report["execution_status"] = "error"
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["total_wall_seconds"] = time.perf_counter() - started
        report["exceptions"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
