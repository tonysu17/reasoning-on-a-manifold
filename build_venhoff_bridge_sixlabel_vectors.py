#!/usr/bin/env python3
"""Rebuild four held-out six-label directions at Venhoff's Qwen-1.5B layers.

This is deliberately a no-model, no-API bridge builder.  It reconstructs only
the unit single directions needed by the fixed-offset steering experiment:

    unit(mean(target rows) - mean(all other five labels' rows))

After unit normalisation this has the same direction as
``mean(target) - mean(overall)``.  The four target-label matrices come from the
current R1-1.5B extraction (``clip_window_to_sentence_end=false``), whereas the
initializing and deduction matrices are available only in the archived
six-label extraction (``clip_window_to_sentence_end=true``).  The output is
therefore explicitly a mixed-vintage bridge asset, not a clean uniform
re-extraction and not an exact reconstruction of Venhoff et al.'s vectors.

The exact 50 task ids recorded by E1 are removed using each activation source's
matching row_index.json.  No task split is recomputed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / "data/activations/R1-1.5B"
ARCHIVE = ROOT / "data/activations/_volume_R1-1.5B_6label_archive"
EVAL_IDS = ROOT / "results/eval/R1-1.5B__E1/eval_task_ids.json"
OUT = ROOT / "results/steering_vectors/R1-1.5B__venhoff_bridge_sixlabel"
REFERENCE = ROOT / "results/steering_vectors/R1-1.5B__E1_pooled"

# Preserve the six-label order used by build_e1_pooled_vectors.py so the
# floating-point reduction order also matches that reference where layers agree.
ALL_LABELS = [
    "initializing",
    "deduction",
    "adding-knowledge",
    "example-testing",
    "uncertainty-estimation",
    "backtracking",
]
TARGET_LAYERS = {
    "backtracking": 17,
    "uncertainty-estimation": 18,
    "example-testing": 15,
    "adding-knowledge": 18,
}
ARCHIVED_LABELS = {"initializing", "deduction"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _source(label: str) -> Path:
    return ARCHIVE if label in ARCHIVED_LABELS else CURRENT


def _atomic_save_npy(array: np.ndarray, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
    tmp.replace(path)


def _atomic_save_json(data: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def main() -> None:
    required = [
        CURRENT / "metadata.json",
        CURRENT / "row_index.json",
        ARCHIVE / "metadata.json",
        ARCHIVE / "row_index.json",
        EVAL_IDS,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing bridge artefacts in {OUT}"
        )
    OUT.mkdir(parents=True, exist_ok=True)

    current_meta = _load_json(CURRENT / "metadata.json")
    archive_meta = _load_json(ARCHIVE / "metadata.json")
    current_index = _load_json(CURRENT / "row_index.json")
    archive_index = _load_json(ARCHIVE / "row_index.json")
    eval_manifest = _load_json(EVAL_IDS)
    excluded_ids = set(eval_manifest["task_ids"])
    if len(excluded_ids) != 50:
        raise ValueError(
            f"Expected exactly 50 E1 hold-out ids, found {len(excluded_ids)}"
        )

    if current_meta.get("clip_window_to_sentence_end") is not False:
        raise ValueError("Current target source no longer has the expected unclipped contract")
    if archive_meta.get("clip_window_to_sentence_end") is not True:
        raise ValueError("Archived inert-label source no longer has the expected clipped contract")

    row_indices = {"current": current_index, "archive": archive_index}
    source_metadata = {"current": current_meta, "archive": archive_meta}
    input_paths = set(required)
    vectors: dict[str, np.ndarray] = {}
    behaviour_meta: dict[str, dict] = {}

    by_layer: dict[int, list[str]] = {}
    for behaviour, layer in TARGET_LAYERS.items():
        by_layer.setdefault(layer, []).append(behaviour)

    for layer in sorted(by_layer):
        matrices: dict[str, np.ndarray] = {}
        counts: dict[str, dict[str, int]] = {}
        hidden_dim = None

        for label in ALL_LABELS:
            source_key = "archive" if label in ARCHIVED_LABELS else "current"
            source_root = _source(label)
            array_path = source_root / f"{label}_layer{layer}.npy"
            if not array_path.is_file():
                raise FileNotFoundError(f"Missing activation matrix: {array_path}")
            input_paths.add(array_path)

            array = np.load(array_path).astype(np.float32)
            rows = row_indices[source_key].get("rows", {}).get(label)
            if rows is None:
                raise ValueError(
                    f"{source_root / 'row_index.json'} has no rows for {label}"
                )
            if len(rows) != array.shape[0]:
                raise ValueError(
                    f"Row-index mismatch for {label} at L{layer}: "
                    f"matrix={array.shape[0]}, index={len(rows)}"
                )
            if array.ndim != 2:
                raise ValueError(f"Expected 2-D matrix at {array_path}, got {array.shape}")
            if hidden_dim is None:
                hidden_dim = int(array.shape[1])
            elif int(array.shape[1]) != hidden_dim:
                raise ValueError(
                    f"Hidden-width mismatch at L{layer}: {label} has {array.shape[1]}, "
                    f"expected {hidden_dim}"
                )

            chain_ids = np.asarray([row["chain_id"] for row in rows], dtype=object)
            keep = ~np.isin(chain_ids, list(excluded_ids))
            matrices[label] = array[keep]
            counts[label] = {
                "raw": int(array.shape[0]),
                "excluded": int((~keep).sum()),
                "retained": int(keep.sum()),
            }

        for behaviour in by_layer[layer]:
            on = matrices[behaviour]
            off = np.concatenate(
                [matrices[label] for label in ALL_LABELS if label != behaviour],
                axis=0,
            )
            difference = on.mean(axis=0) - off.mean(axis=0)
            raw_norm = float(np.linalg.norm(difference))
            if not np.isfinite(raw_norm) or raw_norm < 1e-10:
                raise ValueError(
                    f"{behaviour} at L{layer} has invalid mean-difference norm {raw_norm}"
                )
            vector = (difference / raw_norm).astype(np.float32)
            vectors[behaviour] = vector
            behaviour_meta[behaviour] = {
                "layer": layer,
                "n_on": int(on.shape[0]),
                "n_off": int(off.shape[0]),
                "n_excluded": int(counts[behaviour]["excluded"]),
                "counts_by_label": counts,
                "raw_difference_norm": raw_norm,
                "vector_norm": float(np.linalg.norm(vector)),
                # These fields keep the narrow artefact legible to existing
                # single-vector consumers without claiming any PCA construction.
                "auto_k": None,
                "k_values": [],
            }

        del matrices

    for behaviour, vector in vectors.items():
        path = OUT / f"{behaviour}_single.npy"
        _atomic_save_npy(vector, path)
        behaviour_meta[behaviour]["vector_file"] = path.relative_to(ROOT).as_posix()
        behaviour_meta[behaviour]["vector_sha256"] = _sha256(path)

    verification: dict[str, dict] = {}
    for behaviour in ("backtracking", "example-testing"):
        reference_path = REFERENCE / f"{behaviour}_single.npy"
        if not reference_path.is_file():
            continue
        reference = np.load(reference_path).astype(np.float32)
        cosine = float(
            np.dot(vectors[behaviour], reference)
            / (np.linalg.norm(vectors[behaviour]) * np.linalg.norm(reference))
        )
        verification[behaviour] = {
            "reference": reference_path.relative_to(ROOT).as_posix(),
            "reference_sha256": _sha256(reference_path),
            "same_layer": True,
            "cosine": cosine,
            "one_minus_cosine": float(1.0 - cosine),
        }

    relative_hashes = {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in sorted(input_paths)
    }
    metadata = {
        "_provenance": {
            "builder": Path(__file__).name,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "layers": TARGET_LAYERS,
            "vector_definition": (
                "unit(mean(target rows) - mean(concatenated rows from the other "
                "five labels)); direction-equivalent after unit normalization to "
                "mean(target) - frequency-weighted mean(overall six labels)"
            ),
            "normalization": "L2 unit norm; no Venhoff magnitude scaling applied here",
            "holdout": {
                "manifest": EVAL_IDS.relative_to(ROOT).as_posix(),
                "manifest_sha256": _sha256(EVAL_IDS),
                "n_task_ids": len(excluded_ids),
                "task_ids": eval_manifest["task_ids"],
                "rule_as_recorded": eval_manifest.get("rule"),
                "implementation": (
                    "exact manifest ids removed through each source's matching "
                    "row_index.json; split not recomputed"
                ),
            },
            "source_contract": {
                "status": "mixed clip-window vintages",
                "warning": (
                    "Bridge/sensitivity asset only. This is not a clean uniform "
                    "six-label re-extraction and carries no claim that the target "
                    "and inert-label rows share one extraction window convention."
                ),
                "target_labels": {
                    "labels": [label for label in ALL_LABELS if label not in ARCHIVED_LABELS],
                    "root": CURRENT.relative_to(ROOT).as_posix(),
                    "clip_window_to_sentence_end": source_metadata["current"].get(
                        "clip_window_to_sentence_end"
                    ),
                    "row_index": (CURRENT / "row_index.json").relative_to(ROOT).as_posix(),
                },
                "archived_inert_labels": {
                    "labels": sorted(ARCHIVED_LABELS),
                    "root": ARCHIVE.relative_to(ROOT).as_posix(),
                    "clip_window_to_sentence_end": source_metadata["archive"].get(
                        "clip_window_to_sentence_end"
                    ),
                    "row_index": (ARCHIVE / "row_index.json").relative_to(ROOT).as_posix(),
                },
                "shared_fields_checked": {
                    "pooling": [
                        source_metadata["current"].get("pooling"),
                        source_metadata["archive"].get("pooling"),
                    ],
                    "sentence_matching": [
                        source_metadata["current"].get("sentence_matching"),
                        source_metadata["archive"].get("sentence_matching"),
                    ],
                    "n_preceding": [
                        source_metadata["current"].get("n_preceding"),
                        source_metadata["archive"].get("n_preceding"),
                    ],
                    "n_execution": [
                        source_metadata["current"].get("n_execution"),
                        source_metadata["archive"].get("n_execution"),
                    ],
                },
            },
            "input_sha256": relative_hashes,
            "reference_verification_not_used_in_construction": verification,
        },
        **behaviour_meta,
    }
    _atomic_save_json(metadata, OUT / "metadata.json")

    print(f"Saved four mixed-vintage six-label unit directions -> {OUT}")
    for behaviour in TARGET_LAYERS:
        info = behaviour_meta[behaviour]
        suffix = ""
        if behaviour in verification:
            suffix = f", cos(E1_pooled)={verification[behaviour]['cosine']:.10f}"
        print(
            f"  {behaviour:24s} L{info['layer']:>2d}: "
            f"n_on={info['n_on']:>5d}, n_off={info['n_off']:>5d}, "
            f"norm={info['vector_norm']:.8f}{suffix}"
        )


if __name__ == "__main__":
    main()
