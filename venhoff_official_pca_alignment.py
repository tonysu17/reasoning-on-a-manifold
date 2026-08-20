#!/usr/bin/env python3
"""Compare released Venhoff directions with the thesis PCA subspaces.

This is a no-model, no-API post-hoc sensitivity.  It derives the exact
code-defined directions from Venhoff et al.'s released Qwen-1.5B mean tensor,
then measures how much of each direction lies in PCA bases fitted to the
thesis activation corpus.

The analysis deliberately separates two quantities:

* PCA spectra/dimensionality describe an activation cloud and cannot change
  when only a steering vector is replaced.
* vector-to-subspace alignment can change.  For unit direction ``u`` and an
  orthonormal top-k basis ``U_k`` this script reports

      retained_norm R_k = ||U_k u||,
      captured_squared_norm E_k = R_k**2,
      subspace_angle = arccos(R_k).

The historical ``steering_geometry.json`` field named ``retained_energy`` is
``R_k`` (and is identical to the cosine with the normalised projection), not
the squared-energy fraction ``E_k``.  Both are written here with unambiguous
names.

The headline comparison holds a PCA cloud fixed and contrasts the
thesis/hybrid direction with the exact released Venhoff direction at
Venhoff's layers L17/L18/L15/L18.  It records both (i) the holdout-excluded
E1-training cloud used to reconstruct the historical steering subspace and
(ii) a genuinely out-of-sample cloud containing only the 50 held-out E1 task
IDs.  Same-E1-layer comparisons and pooled/ON-only sensitivities are recorded.
Because the six-label pooled thesis cloud mixes current target-label
activations with archived initializing/deduction activations, every such basis
remains a mixed-vintage sensitivity rather than a clean new primary result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / "data/activations/R1-1.5B"
ARCHIVE = ROOT / "data/activations/_volume_R1-1.5B_6label_archive"
EVAL_IDS = ROOT / "results/eval/R1-1.5B__E1/eval_task_ids.json"
E1 = ROOT / "results/steering_vectors/R1-1.5B__E1_pooled"
E1_GEOMETRY = ROOT / "results/eval/R1-1.5B__E1/steering_geometry.json"
HYBRID = ROOT / "results/steering_vectors/R1-1.5B__venhoff_bridge_hybrid"
ON_ONLY_PCA = ROOT / "results/pca/R1-1.5B"

EXPECTED_OFFICIAL_SHA256 = (
    "bbf7bcac3758df3236d485c41dcf3033a53fd6333a4abd555290e8ef71419412"
)
EXPECTED_OFFICIAL_REPO_COMMIT = "93259bc3410c99293351df41141cd16b4110422a"

LABELS = [
    "initializing",
    "deduction",
    "adding-knowledge",
    "example-testing",
    "uncertainty-estimation",
    "backtracking",
]
TARGETS = [
    "backtracking",
    "uncertainty-estimation",
    "example-testing",
    "adding-knowledge",
]
ARCHIVED_LABELS = {"initializing", "deduction"}
PRIMARY_LAYERS = {
    "backtracking": 17,
    "uncertainty-estimation": 15,
    "example-testing": 15,
    "adding-knowledge": 17,
}
VENHOFF_LAYERS = {
    "backtracking": 17,
    "uncertainty-estimation": 18,
    "example-testing": 15,
    "adding-knowledge": 18,
}
K_VALUES = (1, 3, 5, 10)
MAX_COMPONENTS = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def git_output(cwd: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"Cannot normalise vector with norm={norm}")
    return vector / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))


def direction_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(cosine(a, b), -1.0, 1.0))))


def alignment(direction: np.ndarray, components: np.ndarray, k_values: list[int]) -> dict:
    """Return unambiguously named vector-to-subspace alignment statistics."""
    u = unit(direction)
    U = np.asarray(components, dtype=np.float64)
    if U.ndim != 2 or U.shape[1] != u.shape[0]:
        raise ValueError(f"Basis/vector mismatch: U={U.shape}, u={u.shape}")
    gram = U @ U.T
    if not np.allclose(gram, np.eye(U.shape[0]), atol=2e-4, rtol=2e-4):
        raise ValueError("PCA component rows are not sufficiently orthonormal")

    squared_loadings = np.square(U @ u)
    cumulative = np.cumsum(squared_loadings)
    cells: dict[str, dict[str, float]] = {}
    for k in k_values:
        if not 1 <= k <= U.shape[0]:
            raise ValueError(f"Invalid k={k} for {U.shape[0]} components")
        captured = float(np.clip(cumulative[k - 1], 0.0, 1.0))
        retained = float(math.sqrt(captured))
        cells[str(k)] = {
            "retained_norm": retained,
            "captured_squared_norm": captured,
            "subspace_angle_degrees": float(
                math.degrees(math.acos(np.clip(retained, 0.0, 1.0)))
            ),
        }

    threshold_k: dict[str, int | str] = {}
    for threshold in (0.5, 0.7, 0.9):
        hits = np.flatnonzero(cumulative >= threshold)
        threshold_k[f"captured_{int(threshold * 100)}pct"] = (
            int(hits[0] + 1) if hits.size else f">{U.shape[0]}"
        )
    return {
        "by_k": cells,
        "smallest_k_for_captured_squared_norm": threshold_k,
        "component_squared_loadings": squared_loadings.tolist(),
    }


def _source_for(label: str) -> Path:
    return ARCHIVE if label in ARCHIVED_LABELS else CURRENT


def load_selected_rows(
    label: str,
    layer: int,
    row_indices: dict[str, dict],
    heldout_ids: set[str],
    split: str,
) -> tuple[np.ndarray, dict[str, int], Path]:
    source_key = "archive" if label in ARCHIVED_LABELS else "current"
    source = _source_for(label)
    path = source / f"{label}_layer{layer}.npy"
    array = np.load(path, allow_pickle=False)
    rows = row_indices[source_key].get("rows", {}).get(label)
    if rows is None or len(rows) != array.shape[0]:
        raise ValueError(
            f"Row provenance mismatch for {label}@L{layer}: "
            f"matrix={array.shape[0]}, rows={None if rows is None else len(rows)}"
        )
    chain_ids = np.asarray([row["chain_id"] for row in rows], dtype=object)
    in_holdout = np.isin(chain_ids, list(heldout_ids))
    if split == "holdout_excluded_e1_training":
        keep = ~in_holdout
    elif split == "heldout_50_task":
        keep = in_holdout
    else:
        raise ValueError(f"Unknown row split: {split}")
    selected = np.asarray(array[keep], dtype=np.float32)
    return selected, {
        "raw": int(array.shape[0]),
        "heldout_rows": int(in_holdout.sum()),
        "training_rows": int((~in_holdout).sum()),
        "selected_rows": int(keep.sum()),
    }, path


def covariance_pca(
    matrix: np.ndarray,
    max_components: int = MAX_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fit PCA by a stable float64 covariance eigendecomposition."""
    X = np.asarray(matrix)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D activation matrix, got {X.shape}")
    n, d = X.shape
    if n < 2 or d < 1:
        raise ValueError(f"Insufficient PCA matrix shape: {X.shape}")
    n_components = min(max_components, n - 1, d)

    # Convert before centring and before the Gram product.  Computing X.T @ X
    # on float32 inputs and casting afterwards does not make the accumulation
    # float64 and is needlessly vulnerable to cancellation.
    centered = np.asarray(X, dtype=np.float64)
    mean = centered.mean(axis=0, dtype=np.float64)
    centered -= mean
    covariance = (centered.T @ centered) / (n - 1)
    covariance = (covariance + covariance.T) / 2.0
    raw_eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(raw_eigenvalues)[::-1]
    raw_eigenvalues = raw_eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    numerical_negative_count = int((raw_eigenvalues < 0).sum())
    negative_tolerance = 1e-10 * max(float(raw_eigenvalues[0]), 1.0)
    material_negative_count = int(
        (raw_eigenvalues < -negative_tolerance).sum()
    )
    minimum_raw_eigenvalue = float(raw_eigenvalues.min())
    eigenvalues = np.maximum(raw_eigenvalues, 0.0)
    components = eigenvectors[:, :n_components].T
    total_variance = float(eigenvalues.sum())
    explained = eigenvalues[:n_components] / total_variance
    cumulative = np.cumsum(explained)
    auto_hits = np.flatnonzero(cumulative >= 0.70)
    auto_k: int | str = (
        int(auto_hits[0] + 1) if auto_hits.size else f">{n_components}"
    )
    diagnostics = {
        "n_rows": int(n),
        "hidden_dim": int(d),
        "total_variance": total_variance,
        "cumulative_variance_at_k": {
            str(k): float(cumulative[k - 1])
            for k in K_VALUES
            if k <= n_components
        },
        "auto_k_70pct_variance": auto_k,
        "minimum_raw_eigenvalue": minimum_raw_eigenvalue,
        "negative_eigenvalue_count_before_clipping": numerical_negative_count,
        "material_negative_eigenvalue_count": material_negative_count,
        "material_negative_tolerance": negative_tolerance,
        "numeric_method": (
            "mean-centred float64 covariance eigendecomposition; covariance "
            "formed as X_centered.T @ X_centered / (n-1)"
        ),
    }
    return components, eigenvalues[:n_components], mean, diagnostics


def pooled_pca(
    layer: int,
    row_indices: dict[str, dict],
    heldout_ids: set[str],
    split: str,
    max_components: int = MAX_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """PCA of one explicit row split of the mixed-vintage six-label cloud."""
    matrices: list[np.ndarray] = []
    counts: dict[str, dict[str, int]] = {}
    paths: dict[str, str] = {}
    for label in LABELS:
        matrix, count, path = load_selected_rows(
            label, layer, row_indices, heldout_ids, split
        )
        matrices.append(matrix)
        counts[label] = count
        paths[label] = path.relative_to(ROOT).as_posix()

    X = np.concatenate(matrices, axis=0)
    components, eigenvalues, mean, diagnostics = covariance_pca(
        X, max_components=max_components
    )
    metadata = {
        "layer": layer,
        "row_split": split,
        "counts_by_label": counts,
        "source_paths": paths,
        "method": (
            "mean-centred row-pooled six-label covariance eigendecomposition; "
            "target labels current/unclipped, initializing+deduction archived/clipped"
        ),
        **diagnostics,
    }
    return components, eigenvalues, mean, metadata


def on_only_pca(
    behaviour: str,
    layer: int,
    row_indices: dict[str, dict],
    heldout_ids: set[str],
    split: str,
    max_components: int = MAX_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """PCA of the selected target-behaviour rows only."""
    matrix, counts, path = load_selected_rows(
        behaviour, layer, row_indices, heldout_ids, split
    )
    components, eigenvalues, mean, diagnostics = covariance_pca(
        matrix, max_components=max_components
    )
    metadata = {
        "behaviour": behaviour,
        "layer": layer,
        "row_split": split,
        "counts": counts,
        "source_path": path.relative_to(ROOT).as_posix(),
        "method": "mean-centred target-label-only covariance eigendecomposition",
        **diagnostics,
    }
    return components, eigenvalues, mean, metadata


def official_direction(means: dict, behaviour: str, layer: int) -> tuple[np.ndarray, dict]:
    target = means[behaviour]["mean"][layer].detach().cpu().numpy()
    overall = means["overall"]["mean"][layer].detach().cpu().numpy()
    difference = np.asarray(target - overall, dtype=np.float32)
    raw_norm = float(np.linalg.norm(difference))
    overall_norm = float(np.linalg.norm(overall))
    return unit(difference), {
        "definition": "unit(released_mean[target][layer] - released_mean[overall][layer])",
        "raw_difference_norm": raw_norm,
        "released_code_scale_norm": overall_norm,
        "scale_definition": (
            "The released loader rescales each difference vector to the L2 norm "
            "of released_mean[overall] at this layer; this is code-derived, not "
            "a claim that the numeric norm appears in the paper."
        ),
        "released_target_count": int(means[behaviour]["count"]),
        "released_overall_count": int(means["overall"]["count"]),
    }


def compare_directions(
    current: np.ndarray,
    official: np.ndarray,
    components: np.ndarray,
    k_values: list[int],
) -> dict:
    cos = cosine(current, official)
    current_alignment = alignment(current, components, k_values)
    official_alignment = alignment(official, components, k_values)
    deltas = {}
    for k in k_values:
        key = str(k)
        c = current_alignment["by_k"][key]
        v = official_alignment["by_k"][key]
        deltas[key] = {
            "official_minus_current_retained_norm": (
                v["retained_norm"] - c["retained_norm"]
            ),
            "official_minus_current_captured_squared_norm": (
                v["captured_squared_norm"] - c["captured_squared_norm"]
            ),
        }
    return {
        "direction_cosine_current_vs_official": cos,
        "direction_angle_degrees": direction_angle_degrees(current, official),
        "current_direction": current_alignment,
        "official_venhoff_direction": official_alignment,
        "official_minus_current": deltas,
    }


def save_basis_triplet(
    out: Path,
    stem: str,
    components: np.ndarray,
    eigenvalues: np.ndarray,
    mean: np.ndarray,
    output_hashes: dict[str, str],
) -> dict[str, str]:
    """Save one reproducible float64 PCA basis/eigenspectrum/mean triplet."""
    paths = {
        "component_file": out / f"{stem}_components.npy",
        "eigenvalue_file": out / f"{stem}_eigenvalues.npy",
        "mean_file": out / f"{stem}_mean.npy",
    }
    np.save(paths["component_file"], np.asarray(components, dtype=np.float64))
    np.save(paths["eigenvalue_file"], np.asarray(eigenvalues, dtype=np.float64))
    np.save(paths["mean_file"], np.asarray(mean, dtype=np.float64))
    saved: dict[str, str] = {}
    for key, path in paths.items():
        digest = sha256(path)
        output_hashes[path.name] = digest
        saved[key] = path.name
        saved[key.replace("_file", "_sha256")] = digest
    return saved


def comparison_cell(
    current: np.ndarray,
    official: np.ndarray,
    components: np.ndarray,
    k_values: list[int],
    *,
    layer: int,
    current_path: Path,
    official_meta: dict,
    pca_basis: str,
) -> dict:
    cell = compare_directions(current, official, components, k_values)
    cell.update({
        "layer": layer,
        "current_direction_source": current_path.relative_to(ROOT).as_posix(),
        "current_direction_sha256": sha256(current_path),
        "official_direction": official_meta,
        "pca_basis": pca_basis,
    })
    return cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-repo",
        type=Path,
        default=Path("/Users/tonysu/steering-thinking-llms"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            ROOT
            / "results/pca/R1-1.5B__venhoff_official_direction_alignment_v2"
        ),
    )
    args = parser.parse_args()
    official_repo = args.official_repo.resolve()
    official_asset = (
        official_repo
        / "train-steering-vectors/results/vars/"
        "mean_vectors_deepseek-r1-distill-qwen-1.5b.pt"
    )
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {out}")

    required = [
        CURRENT / "metadata.json",
        CURRENT / "row_index.json",
        ARCHIVE / "metadata.json",
        ARCHIVE / "row_index.json",
        EVAL_IDS,
        E1 / "metadata.json",
        E1_GEOMETRY,
        HYBRID / "metadata.json",
        official_asset,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    official_sha = sha256(official_asset)
    if official_sha != EXPECTED_OFFICIAL_SHA256:
        raise ValueError(
            f"Official mean-vector SHA changed: expected {EXPECTED_OFFICIAL_SHA256}, "
            f"got {official_sha}"
        )
    official_commit = git_output(official_repo, "rev-parse", "HEAD")
    if official_commit != EXPECTED_OFFICIAL_REPO_COMMIT:
        raise ValueError(
            f"Official repository commit changed: expected "
            f"{EXPECTED_OFFICIAL_REPO_COMMIT}, got {official_commit}"
        )

    # Imported lazily so pure helper tests require only NumPy.
    import torch

    means = torch.load(official_asset, map_location="cpu", weights_only=True)
    current_metadata = load_json(CURRENT / "metadata.json")
    archive_metadata = load_json(ARCHIVE / "metadata.json")
    if current_metadata.get("clip_window_to_sentence_end") is not False:
        raise ValueError("Unexpected current target-label extraction contract")
    if archive_metadata.get("clip_window_to_sentence_end") is not True:
        raise ValueError("Unexpected archived inert-label extraction contract")

    eval_manifest = load_json(EVAL_IDS)
    heldout_ids = set(eval_manifest["task_ids"])
    if len(heldout_ids) != 50:
        raise ValueError(f"Expected 50 held-out task ids, found {len(heldout_ids)}")
    e1_metadata = load_json(E1 / "metadata.json")
    e1_geometry = load_json(E1_GEOMETRY)
    hybrid_metadata = load_json(HYBRID / "metadata.json")
    row_indices = {
        "current": load_json(CURRENT / "row_index.json"),
        "archive": load_json(ARCHIVE / "row_index.json"),
    }

    out.mkdir(parents=True)
    training_pooled_bases: dict[int, np.ndarray] = {}
    heldout_pooled_bases: dict[int, np.ndarray] = {}
    training_pooled_basis_metadata: dict[str, dict] = {}
    heldout_pooled_basis_metadata: dict[str, dict] = {}
    heldout_on_bases: dict[tuple[str, int], np.ndarray] = {}
    heldout_on_basis_metadata: dict[str, dict] = {}
    output_hashes: dict[str, str] = {}
    input_paths = set(required)

    for layer in sorted(set(PRIMARY_LAYERS.values()) | set(VENHOFF_LAYERS.values())):
        for split, bases, metadata_by_layer, stem_prefix in (
            (
                "holdout_excluded_e1_training",
                training_pooled_bases,
                training_pooled_basis_metadata,
                "training_pooled",
            ),
            (
                "heldout_50_task",
                heldout_pooled_bases,
                heldout_pooled_basis_metadata,
                "heldout_pooled",
            ),
        ):
            components, eigenvalues, mean, metadata = pooled_pca(
                layer, row_indices, heldout_ids, split
            )
            bases[layer] = components
            for path_text in metadata["source_paths"].values():
                input_paths.add(ROOT / path_text)
            metadata.update(
                save_basis_triplet(
                    out,
                    f"{stem_prefix}_layer{layer}",
                    components,
                    eigenvalues,
                    mean,
                    output_hashes,
                )
            )
            metadata_by_layer[str(layer)] = metadata

    # The independent ON-only sensitivity uses only rows from the 50 held-out
    # task IDs.  Neither thesis direction was fitted on these rows.
    for behaviour in TARGETS:
        for layer in sorted({PRIMARY_LAYERS[behaviour], VENHOFF_LAYERS[behaviour]}):
            components, eigenvalues, mean, metadata = on_only_pca(
                behaviour,
                layer,
                row_indices,
                heldout_ids,
                "heldout_50_task",
            )
            heldout_on_bases[(behaviour, layer)] = components
            input_paths.add(ROOT / metadata["source_path"])
            stem = f"heldout_on_{behaviour}_layer{layer}"
            metadata.update(
                save_basis_triplet(
                    out, stem, components, eigenvalues, mean, output_hashes
                )
            )
            heldout_on_basis_metadata[f"{behaviour}|L{layer}"] = metadata

    training_protocol_layer: dict[str, dict] = {}
    training_primary_layer: dict[str, dict] = {}
    heldout_protocol_layer: dict[str, dict] = {}
    heldout_primary_layer: dict[str, dict] = {}
    validation: dict[str, dict] = {}
    historical_on_only_sensitivity: dict[str, dict] = {}
    heldout_on_protocol_layer: dict[str, dict] = {}
    heldout_on_primary_layer: dict[str, dict] = {}

    def k_values_for(metadata: dict) -> list[int]:
        values = list(K_VALUES)
        auto = metadata["auto_k_70pct_variance"]
        if isinstance(auto, int) and auto not in values:
            values.append(auto)
        return values

    for behaviour in TARGETS:
        # Protocol-layer comparisons use the hybrid bridge direction: BT/EX are
        # exact E1 vectors; uncertainty/adding-knowledge are mixed-vintage L18
        # reconstructions.  Propagate that distinction into every result cell.
        v_layer = VENHOFF_LAYERS[behaviour]
        current_path = HYBRID / f"{behaviour}_single.npy"
        input_paths.add(current_path)
        current = unit(np.load(current_path, allow_pickle=False))
        official, official_meta = official_direction(means, behaviour, v_layer)
        source_info = hybrid_metadata[behaviour]
        training_protocol_layer[behaviour] = comparison_cell(
            current,
            official,
            training_pooled_bases[v_layer],
            k_values_for(training_pooled_basis_metadata[str(v_layer)]),
            layer=v_layer,
            current_path=current_path,
            official_meta=official_meta,
            pca_basis=(
                "holdout-excluded E1-training six-label row-pooled "
                "mixed-vintage cloud"
            ),
        )
        heldout_protocol_layer[behaviour] = comparison_cell(
            current,
            official,
            heldout_pooled_bases[v_layer],
            k_values_for(heldout_pooled_basis_metadata[str(v_layer)]),
            layer=v_layer,
            current_path=current_path,
            official_meta=official_meta,
            pca_basis=(
                "independent 50-task heldout six-label row-pooled mixed-vintage cloud"
            ),
        )
        for cell in (
            training_protocol_layer[behaviour],
            heldout_protocol_layer[behaviour],
        ):
            cell["current_direction_source_kind"] = source_info["source_kind"]
            cell["current_direction_interpretation"] = source_info["interpretation"]

        # Same primary E1 layer: holds layer/PCA cloud fixed while replacing only
        # the direction source, subject to different source-corpus construction.
        p_layer = PRIMARY_LAYERS[behaviour]
        e1_path = E1 / f"{behaviour}_single.npy"
        input_paths.add(e1_path)
        e1_direction = unit(np.load(e1_path, allow_pickle=False))
        official_at_primary, official_primary_meta = official_direction(
            means, behaviour, p_layer
        )
        training_primary_layer[behaviour] = comparison_cell(
            e1_direction,
            official_at_primary,
            training_pooled_bases[p_layer],
            k_values_for(training_pooled_basis_metadata[str(p_layer)]),
            layer=p_layer,
            current_path=e1_path,
            official_meta=official_primary_meta,
            pca_basis=(
                "holdout-excluded E1-training six-label row-pooled "
                "mixed-vintage cloud"
            ),
        )
        heldout_primary_layer[behaviour] = comparison_cell(
            e1_direction,
            official_at_primary,
            heldout_pooled_bases[p_layer],
            k_values_for(heldout_pooled_basis_metadata[str(p_layer)]),
            layer=p_layer,
            current_path=e1_path,
            official_meta=official_primary_meta,
            pca_basis=(
                "independent 50-task heldout six-label row-pooled mixed-vintage cloud"
            ),
        )
        for cell in (
            training_primary_layer[behaviour],
            heldout_primary_layer[behaviour],
        ):
            cell["current_direction_source_kind"] = (
                "saved_E1_pooled_single_direction"
            )
            cell["current_direction_provenance_status"] = (
                "unresolved provenance: E1 metadata records git_commit null"
            )

        # Validate the reconstructed basis against the stored E1 projected
        # vectors.  Near-unity, not bit identity, is expected because the E1
        # build provenance does not uniquely establish the old source files.
        validation[behaviour] = {
            "auto_k_70pct_variance": {
                "recorded_E1": int(e1_metadata[behaviour]["auto_k"]),
                "reconstructed_training_cloud": (
                    training_pooled_basis_metadata[str(p_layer)][
                        "auto_k_70pct_variance"
                    ]
                ),
                "independent_heldout_cloud": (
                    heldout_pooled_basis_metadata[str(p_layer)][
                        "auto_k_70pct_variance"
                    ]
                ),
            },
            "by_k": {},
        }
        for k in (3, 5):
            saved_path = E1 / f"{behaviour}_manifold_k{k}.npy"
            input_paths.add(saved_path)
            saved = unit(np.load(saved_path, allow_pickle=False))
            U = training_pooled_bases[p_layer][:k]
            projected = unit(U.T @ (U @ e1_direction))
            signed_projection_cosine = cosine(projected, saved)
            reconstructed_retained = float(np.linalg.norm(U @ e1_direction))
            recorded_retained = float(
                e1_geometry[behaviour][str(k)]["retained_energy"]
            )
            validation[behaviour]["by_k"][str(k)] = {
                "signed_cosine_reconstructed_projection_vs_saved": (
                    signed_projection_cosine
                ),
                "absolute_cosine_reconstructed_projection_vs_saved": abs(
                    signed_projection_cosine
                ),
                "reconstructed_retained_norm": reconstructed_retained,
                "recorded_E1_retained_norm": recorded_retained,
                "reconstructed_minus_recorded_retained_norm": (
                    reconstructed_retained - recorded_retained
                ),
                "retained_norm_derived_from_saved_vector": abs(
                    cosine(e1_direction, saved)
                ),
                "saved_file": saved_path.relative_to(ROOT).as_posix(),
                "saved_sha256": sha256(saved_path),
            }

        # Sensitivity: the already-saved ON-only behaviour PCA basis.  Unlike
        # the headline pooled basis, this contains no between-class mean term.
        on_path = ON_ONLY_PCA / f"{behaviour}_components_layer{v_layer}.npy"
        input_paths.add(on_path)
        on_provenance_path = ON_ONLY_PCA / f"provenance_layer{v_layer}.json"
        on_provenance = None
        if on_provenance_path.is_file():
            input_paths.add(on_provenance_path)
            on_provenance = load_json(on_provenance_path)
        on_components = np.load(on_path, allow_pickle=False)
        on_cell = compare_directions(
            current, official, on_components, [1, 3, 5, 10, 50, 100]
        )
        on_cell.update({
            "layer": v_layer,
            "basis_file": on_path.relative_to(ROOT).as_posix(),
            "basis_sha256": sha256(on_path),
            "basis_definition": (
                "ON-label-only PCA; historical saved all-row basis including "
                "the 50 later-designated E1 holdout task IDs"
            ),
            "basis_provenance": on_provenance,
        })
        historical_on_only_sensitivity[behaviour] = on_cell

        heldout_on_protocol_layer[behaviour] = comparison_cell(
            current,
            official,
            heldout_on_bases[(behaviour, v_layer)],
            [1, 3, 5, 10, 50, 100],
            layer=v_layer,
            current_path=current_path,
            official_meta=official_meta,
            pca_basis=(
                "independent 50-task heldout target-label-only current/unclipped cloud"
            ),
        )
        heldout_on_primary_layer[behaviour] = comparison_cell(
            e1_direction,
            official_at_primary,
            heldout_on_bases[(behaviour, p_layer)],
            [1, 3, 5, 10, 50, 100],
            layer=p_layer,
            current_path=e1_path,
            official_meta=official_primary_meta,
            pca_basis=(
                "independent 50-task heldout target-label-only current/unclipped cloud"
            ),
        )

    result = {
        "schema_version": "venhoff-official-direction-pca-alignment-v2",
        "status": {
            "empirical": "post-hoc exploratory latent-space sensitivity",
            "confirmatory": False,
            "claim_boundary": (
                "Measures how exact released Venhoff directions align with thesis-corpus "
                "PCA bases. The holdout-excluded E1-training basis is partly in-sample for "
                "the thesis direction and its pooled covariance contains a between-class "
                "mean term; the 50-task heldout bases are the less circular sensitivity. "
                "This does not measure Venhoff-corpus PCA geometry, does not estimate a "
                "vector's dimensionality, and does not change the thesis activation-cloud "
                "PCA spectra or dimensionality results."
            ),
            "uncertainty_boundary": (
                "These are deterministic alignment summaries without resampling intervals. "
                "The released Venhoff asset contains means, not row activations, so the "
                "Venhoff direction cannot be independently bootstrapped from that asset."
            ),
        },
        "definitions": {
            "retained_norm": "R_k = ||U_k u|| for unit u",
            "captured_squared_norm": "E_k = ||U_k u||^2 = R_k^2",
            "subspace_angle_degrees": "acos(R_k) in degrees",
            "historical_naming_note": (
                "The existing steering_geometry retained_energy field equals R_k, "
                "not E_k; it is also identical to cosine(u, normalised(P_k u))."
            ),
        },
        "holdout_excluded_training_pooled_protocol_layer_comparison": (
            training_protocol_layer
        ),
        "holdout_excluded_training_pooled_primary_layer_comparison": (
            training_primary_layer
        ),
        "heldout_50_task_pooled_protocol_layer_comparison": heldout_protocol_layer,
        "heldout_50_task_pooled_primary_layer_comparison": heldout_primary_layer,
        "historical_all_row_on_only_basis_sensitivity": (
            historical_on_only_sensitivity
        ),
        "heldout_50_task_on_only_protocol_layer_comparison": (
            heldout_on_protocol_layer
        ),
        "heldout_50_task_on_only_primary_layer_comparison": (
            heldout_on_primary_layer
        ),
        "holdout_excluded_training_pooled_basis_metadata": (
            training_pooled_basis_metadata
        ),
        "heldout_50_task_pooled_basis_metadata": heldout_pooled_basis_metadata,
        "heldout_50_task_on_only_basis_metadata": heldout_on_basis_metadata,
        "validation_against_saved_e1_projection": validation,
        "provenance": {
            "builder": Path(__file__).name,
            "builder_sha256": sha256(Path(__file__)),
            "analysis_repo_git_commit": git_output(ROOT, "rev-parse", "HEAD"),
            "analysis_repo_git_dirty": bool(git_output(ROOT, "status", "--porcelain")),
            "official_repository": str(official_repo),
            "official_repository_commit": official_commit,
            "official_repository_dirty": bool(
                git_output(official_repo, "status", "--porcelain")
            ),
            "official_mean_vector_file": str(official_asset),
            "official_mean_vector_sha256": official_sha,
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "holdout_manifest": EVAL_IDS.relative_to(ROOT).as_posix(),
            "holdout_manifest_sha256": sha256(EVAL_IDS),
            "n_held_out_tasks": len(heldout_ids),
            "row_split_contract": {
                "holdout_excluded_e1_training": (
                    "all activation rows whose chain_id is not in the exact E1 "
                    "50-task manifest"
                ),
                "heldout_50_task": (
                    "only activation rows whose chain_id is in the exact E1 "
                    "50-task manifest"
                ),
            },
            "source_contract": {
                "status": "mixed clip-window vintages",
                "target_labels": {
                    "root": CURRENT.relative_to(ROOT).as_posix(),
                    "clip_window_to_sentence_end": False,
                },
                "initializing_and_deduction": {
                    "root": ARCHIVE.relative_to(ROOT).as_posix(),
                    "clip_window_to_sentence_end": True,
                },
                "e1_execution_lineage": (
                    "unresolved provenance: source E1 metadata records git_commit null"
                ),
            },
            "input_sha256": {
                path.relative_to(ROOT).as_posix()
                if path.is_relative_to(ROOT)
                else str(path): sha256(path)
                for path in sorted(input_paths)
            },
            "output_sha256": output_hashes,
        },
    }
    report_path = out / "venhoff_official_pca_alignment.json"
    with report_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    output_hashes[report_path.name] = sha256(report_path)

    print(f"Saved -> {report_path}")
    for title, cells in (
        ("Independent heldout pooled", heldout_protocol_layer),
        ("Independent heldout ON-only", heldout_on_protocol_layer),
    ):
        print(f"\n{title} protocol-layer comparison (captured squared norm E_k):")
        print(
            f"{'behaviour':24s} {'cos(dir)':>9s} {'current E3':>11s} "
            f"{'official E3':>12s} {'current E5':>11s} {'official E5':>12s}"
        )
        for behaviour in TARGETS:
            cell = cells[behaviour]
            c3 = cell["current_direction"]["by_k"]["3"]["captured_squared_norm"]
            v3 = cell["official_venhoff_direction"]["by_k"]["3"][
                "captured_squared_norm"
            ]
            c5 = cell["current_direction"]["by_k"]["5"]["captured_squared_norm"]
            v5 = cell["official_venhoff_direction"]["by_k"]["5"][
                "captured_squared_norm"
            ]
            print(
                f"{behaviour:24s} "
                f"{cell['direction_cosine_current_vs_official']:9.3f} "
                f"{c3:11.3f} {v3:12.3f} {c5:11.3f} {v5:12.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
