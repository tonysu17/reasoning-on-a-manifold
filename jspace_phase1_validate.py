#!/usr/bin/env python3
"""Independently validate a completed R1 J-lens Phase-1 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

import jspace_phase1_scoring as scoring


SOURCE_LAYERS = list(range(27))
VocabSize = 151665
LENS_NAMES = ("fit_a", "fit_b", "merged")
EVAL_SLUGS = (
    "lens-eval-association",
    "lens-eval-typo",
    "lens-eval-multihop",
)
REQUIRED_RUN_FILES = (
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
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def rel_frobenius(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.float()
    cand = candidate.float()
    denominator = torch.linalg.vector_norm(ref)
    numerator = torch.linalg.vector_norm(cand - ref)
    if denominator.item() == 0:
        return 0.0 if numerator.item() == 0 else float("inf")
    return float((numerator / denominator).item())


def load_raw_lens(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if set(payload) != {"J", "n_prompts", "source_layers", "d_model"}:
        raise ValueError(f"unexpected lens keys in {path}: {sorted(payload)}")
    if payload["source_layers"] != SOURCE_LAYERS:
        raise ValueError(f"source layers differ in {path}")
    if payload["d_model"] != 1536:
        raise ValueError(f"d_model differs in {path}")
    if set(payload["J"]) != set(SOURCE_LAYERS):
        raise ValueError(f"J layer keys differ in {path}")
    for layer, matrix in payload["J"].items():
        if list(matrix.shape) != [1536, 1536]:
            raise ValueError(f"shape differs in {path}:L{layer}")
        if not bool(torch.isfinite(matrix).all().item()):
            raise ValueError(f"nonfinite matrix in {path}:L{layer}")
    return payload


def validate_top25(values: np.ndarray, expected_shape: tuple[int, ...], label: str) -> None:
    if values.shape != expected_shape:
        raise ValueError(f"{label} shape {values.shape} != {expected_shape}")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{label} is not integer")
    if values.min() < 0 or values.max() >= VocabSize:
        raise ValueError(f"{label} token ID outside vocabulary")
    if np.any(np.diff(np.sort(values, axis=-1), axis=-1) == 0):
        raise ValueError(f"{label} contains duplicate IDs in a top-25 set")


def same_float(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def assert_json_close(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, bool) or isinstance(right, bool) or left is None or right is None:
        if left != right:
            raise ValueError(f"reported value differs at {path}: {left!r} != {right!r}")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not same_float(left, right):
            raise ValueError(f"reported number differs at {path}: {left!r} != {right!r}")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f"reported keys differ at {path}")
        for key in left:
            assert_json_close(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(f"reported list length differs at {path}")
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            assert_json_close(left_value, right_value, f"{path}[{index}]")
        return
    if left != right:
        raise ValueError(f"reported value differs at {path}: {left!r} != {right!r}")


def validate(
    root: Path,
    run_root: Path,
    execution_manifest: Path,
    *,
    recompute_stability_null: bool = False,
    stability_device: str = "cuda",
    require_remote_attestation: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    try:
        execution = json.loads(execution_manifest.read_text())
        for relative, expected in execution["fixed_inputs_sha256"].items():
            path = root / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"fixed input missing/hash mismatch: {relative}")
        checks["fixed_inputs_match_execution_manifest"] = True
        for relative in REQUIRED_RUN_FILES:
            if not (run_root / relative).is_file():
                raise FileNotFoundError(f"missing run artifact: {relative}")
        checks["required_run_files_present"] = True

        report = json.loads((run_root / "phase1_report.json").read_text())
        if report.get("execution_status") != "valid_complete":
            raise ValueError(f"execution status is {report.get('execution_status')!r}")
        if report.get("run_uuid") != execution["run_uuid"]:
            raise ValueError("run UUID mismatch")
        if report.get("source_git_commit") != execution["source_git_commit"]:
            raise ValueError("source git commit mismatch")
        if report.get("pod_id") != execution["pod_id"]:
            raise ValueError("Pod ID mismatch in report")
        checks["report_identity_and_complete_status"] = True
        environment = report.get("environment", {})
        expected_vocab_domains = {
            "raw_head_domain": 151936,
            "primary_ranking_domain": VocabSize,
            "valid_token_id_min": 0,
            "valid_token_id_max": VocabSize - 1,
            "stability_null_domain": VocabSize,
            "excluded_head_padding_rows": 271,
        }
        if {
            key: environment.get(key) for key in expected_vocab_domains
        } != expected_vocab_domains:
            raise ValueError("reported model-head/tokenizer rank domains differ")
        checks["registered_vocabulary_domains_match"] = True

        lenses: dict[str, dict[str, dict[str, Any]]] = {}
        for name in LENS_NAMES:
            lenses[name] = {
                "fp32": load_raw_lens(run_root / f"lenses/{name}.fp32.pt"),
                "fp16": load_raw_lens(run_root / f"lenses/{name}.fp16.pt"),
            }
            expected_n = 100 if name == "merged" else 50
            if lenses[name]["fp32"]["n_prompts"] != expected_n:
                raise ValueError(f"{name} FP32 n_prompts mismatch")
            if lenses[name]["fp16"]["n_prompts"] != expected_n:
                raise ValueError(f"{name} FP16 n_prompts mismatch")
            if any(
                lenses[name]["fp32"]["J"][layer].dtype != torch.float32
                for layer in SOURCE_LAYERS
            ):
                raise ValueError(f"{name} FP32 file contains non-FP32 matrix")
            if any(
                lenses[name]["fp16"]["J"][layer].dtype != torch.float16
                for layer in SOURCE_LAYERS
            ):
                raise ValueError(f"{name} FP16 file contains non-FP16 matrix")
        checks["lens_inventory_shapes_dtypes_finite_counts"] = True

        merge_errors = []
        fp16_errors: dict[str, list[float]] = {name: [] for name in LENS_NAMES}
        for layer in SOURCE_LAYERS:
            expected = (
                lenses["fit_a"]["fp32"]["J"][layer]
                + lenses["fit_b"]["fp32"]["J"][layer]
            ) / 2
            merge_errors.append(
                rel_frobenius(expected, lenses["merged"]["fp32"]["J"][layer])
            )
            for name in LENS_NAMES:
                fp16_errors[name].append(
                    rel_frobenius(
                        lenses[name]["fp32"]["J"][layer],
                        lenses[name]["fp16"]["J"][layer],
                    )
                )
        if max(merge_errors) > 1e-6:
            raise ValueError(f"merged FP32 identity exceeds 1e-6: {max(merge_errors)}")
        if max(max(values) for values in fp16_errors.values()) > 1e-3:
            raise ValueError("FP16 relative Frobenius error exceeds 1e-3")
        numerical_report = json.loads((run_root / "numerical_validation.json").read_text())
        if numerical_report.get("pass") is not True:
            raise ValueError("runner numerical report did not pass")
        for name in LENS_NAMES:
            for precision in ("fp32", "fp16"):
                lens_path = run_root / f"lenses/{name}.{precision}.pt"
                recorded_file = numerical_report["serialization"][name][precision]
                if recorded_file.get("sha256") != sha256_file(lens_path):
                    raise ValueError(f"runner lens hash differs: {name}.{precision}")
                if recorded_file.get("size_bytes") != lens_path.stat().st_size:
                    raise ValueError(f"runner lens size differs: {name}.{precision}")
            if not same_float(
                numerical_report["serialization"][name]["fp16"][
                    "max_relative_frobenius"
                ],
                max(fp16_errors[name]),
                tolerance=1e-7,
            ):
                raise ValueError(f"runner FP16 error differs: {name}")
        if not same_float(
            max(merge_errors),
            numerical_report["merge_identity"]["max_relative_frobenius"],
            tolerance=1e-7,
        ):
            raise ValueError("recomputed merge error differs from runner report")
        checks["merge_and_fp16_recomputed"] = True
        details["merge_max_relative_frobenius"] = max(merge_errors)
        details["fp16_max_relative_frobenius"] = {
            name: max(values) for name, values in fp16_errors.items()
        }

        with np.load(run_root / "heldout_stability_arrays.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != {
                "fit_a_top25",
                "fit_b_top25",
                "scan_null",
                "layer_null",
            }:
                raise ValueError("unexpected held-out NPZ keys")
            top_a = arrays["fit_a_top25"]
            top_b = arrays["fit_b_top25"]
            scan_null = arrays["scan_null"]
            layer_null = arrays["layer_null"]
        validate_top25(top_a, (20, 27, 111, 25), "heldout fit A")
        validate_top25(top_b, (20, 27, 111, 25), "heldout fit B")
        if recompute_stability_null:
            regenerated_scan, regenerated_layers = scoring.stability_permutation_null(
                top_a,
                top_b,
                vocab_size=VocabSize,
                seed=20260811,
                n_perm=1_000,
                device=stability_device,
            )
            if not np.array_equal(regenerated_scan, scan_null):
                raise ValueError("regenerated stability scan null differs")
            if not np.array_equal(regenerated_layers, layer_null):
                raise ValueError("regenerated stability layer null differs")
            checks["stability_null_regenerated_from_seed"] = True
        else:
            details["stability_null_regenerated_from_seed"] = False
        stability_recomputed = scoring.stability_report(
            top_a, top_b, scan_null, layer_null
        )
        stability_report = json.loads((run_root / "heldout_stability.json").read_text())
        stability_protocol_keys = {
            "aggregation",
            "row_layer_median",
            "layer_median",
            "selected_band",
            "scan_null_p95_higher",
            "scan_empirical_upper_p",
            "absolute_floor",
            "bootstrap_10000",
            "pointwise_null_p95_higher",
            "gate",
        }
        if not stability_protocol_keys <= set(stability_report):
            raise ValueError("held-out report lacks protocol-defined fields")
        assert_json_close(
            {key: stability_recomputed[key] for key in stability_protocol_keys},
            {key: stability_report[key] for key in stability_protocol_keys},
            "heldout_stability",
        )
        checks["heldout_stability_recomputed"] = True

        eligibility = json.loads((root / execution["eligibility_manifest"]).read_text())
        with np.load(run_root / "external_readout_arrays.npz", allow_pickle=False) as arrays:
            external_arrays = {name: arrays[name] for name in arrays.files}
        expected_keys = set()
        external_recomputed: dict[str, Any] = {}
        external_rng = np.random.Generator(np.random.PCG64(20260812))
        for slug in EVAL_SLUGS:
            key = slug.replace("-", "_")
            eligible_items = [
                item
                for item in eligibility["evaluations"][slug]["items"]
                if item["item_eligible"]
            ]
            n_items = len(eligible_items)
            labels = [
                [
                    int(label["scored_token_ids"][0])
                    for label in item["labels"]
                    if label["eligible"]
                ]
                for item in eligible_items
            ]
            by_lens = {}
            for name in ("fit_a", "fit_b", "merged", "logit"):
                array_key = f"{key}__top25__{name}"
                expected_keys.add(array_key)
                values = external_arrays[array_key]
                validate_top25(values, (n_items, 27, 25), f"{slug} {name}")
                by_lens[name] = values
            all_key = f"{key}__null__all_layers"
            l17_key = f"{key}__null__l17"
            expected_keys.update((all_key, l17_key))
            regenerated_all, regenerated_l17 = scoring.external_permutation_null(
                by_lens["merged"], labels, rng=external_rng, n_perm=1_000
            )
            if not np.array_equal(regenerated_all, external_arrays[all_key]):
                raise ValueError(f"regenerated external all-layer null differs: {slug}")
            if not np.array_equal(regenerated_l17, external_arrays[l17_key]):
                raise ValueError(f"regenerated external L17 null differs: {slug}")
            external_recomputed[slug] = scoring.external_report(
                by_lens,
                labels,
                external_arrays[all_key],
                external_arrays[l17_key],
            )
        if set(external_arrays) != expected_keys:
            raise ValueError("unexpected external NPZ keys")
        external_report = json.loads(
            (run_root / "external_positive_controls.json").read_text()
        )
        qualifying = []
        for slug in EVAL_SLUGS:
            recomputed = external_recomputed[slug]
            observed = external_report[slug]
            protocol_keys = {
                "n_eligible_items",
                "n_eligible_labels",
                "scores",
                "permutation",
                "qualifying_checks",
                "qualifying_success",
            }
            if not protocol_keys <= set(observed):
                raise ValueError(f"external report lacks protocol fields: {slug}")
            assert_json_close(
                {key: recomputed[key] for key in protocol_keys},
                {key: observed[key] for key in protocol_keys},
                f"external.{slug}",
            )
            expected_names = [item["name"] for item in eligible_items]
            if observed.get("eligible_item_names") != expected_names:
                raise ValueError(f"external eligible item names differ: {slug}")
            if recomputed["qualifying_success"]:
                qualifying.append(slug)
        external_gate = {
            "qualifying_evaluations": qualifying,
            "n_qualifying": len(qualifying),
            "required": 2,
            "pass": len(qualifying) >= 2,
        }
        if external_gate != external_report["gate"]:
            raise ValueError("recomputed external gate differs")
        checks["external_positive_controls_recomputed"] = True
        checks["external_nulls_regenerated_in_fixed_rng_order"] = True

        expected_scientific = {
            "numerical_serialization": True,
            "heldout_stability": stability_recomputed["gate"]["pass"],
            "external_positive_controls": external_gate["pass"],
        }
        expected_scientific["pass"] = all(expected_scientific.values())
        if report["scientific_gate"] != expected_scientific:
            raise ValueError("overall scientific gate differs from recomputation")
        expected_consequence = (
            "Phase_2_decomposition_may_be_considered_but_requires_separate_authorisation"
            if expected_scientific["pass"]
            else "stop_before_Phase_2_the_fitted_lens_did_not_pass_the_prespecified_validity_gate"
        )
        if report.get("licensed_consequence") != expected_consequence:
            raise ValueError("licensed consequence differs from recomputed gate")
        checks["overall_scientific_gate_recomputed"] = True
        details["scientific_gate"] = expected_scientific
        details["artifact_sha256"] = {
            relative: sha256_file(run_root / relative) for relative in REQUIRED_RUN_FILES
        }
        if require_remote_attestation:
            attestation_path = run_root / "INDEPENDENT_VALIDATION.json"
            if not attestation_path.is_file():
                raise FileNotFoundError("remote independent validation attestation missing")
            attestation = json.loads(attestation_path.read_text())
            if attestation.get("ok") is not True:
                raise ValueError("remote independent validation attestation is not ok")
            if (
                attestation.get("execution_manifest_sha256")
                != sha256_file(execution_manifest)
            ):
                raise ValueError("remote attestation execution-manifest hash differs")
            if (
                attestation.get("checks", {}).get(
                    "stability_null_regenerated_from_seed"
                )
                is not True
            ):
                raise ValueError("remote attestation did not regenerate stability null")
            if (
                attestation.get("checks", {}).get(
                    "external_nulls_regenerated_in_fixed_rng_order"
                )
                is not True
            ):
                raise ValueError("remote attestation did not regenerate external nulls")
            if (
                attestation.get("checks", {}).get(
                    "overall_scientific_gate_recomputed"
                )
                is not True
            ):
                raise ValueError("remote attestation did not recompute overall gate")
            attested_hashes = attestation.get("details", {}).get("artifact_sha256")
            if attested_hashes != details["artifact_sha256"]:
                raise ValueError("remote attestation artifact hashes differ locally")
            checks["remote_cuda_null_attestation_verified"] = True
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    return {
        "schema_version": "rom-jspace-r1-phase1-independent-validation-v1",
        "ok": not errors and all(checks.values()),
        "run_root": str(run_root),
        "execution_manifest": str(execution_manifest),
        "execution_manifest_sha256": (
            sha256_file(execution_manifest) if execution_manifest.is_file() else None
        ),
        "checks": checks,
        "details": details,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--write-report", type=Path)
    parser.add_argument("--recompute-stability-null", action="store_true")
    parser.add_argument("--stability-device", default="cuda")
    parser.add_argument("--require-remote-attestation", action="store_true")
    args = parser.parse_args()
    report = validate(
        args.root.resolve(),
        args.run_root.resolve(),
        args.execution_manifest.resolve(),
        recompute_stability_null=args.recompute_stability_null,
        stability_device=args.stability_device,
        require_remote_attestation=args.require_remote_attestation,
    )
    if args.write_report is not None:
        atomic_json_new(args.write_report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
