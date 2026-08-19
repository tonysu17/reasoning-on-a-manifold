#!/usr/bin/env python3
"""Assemble the four-direction hybrid asset for the Venhoff bridge.

This builder does not recompute any direction.  It copies the exact current
``E1_pooled`` single-vector files for backtracking (L17) and example-testing
(L15), so those two bridge arms keep the directions used by the thesis E1
experiment.  Only the two directions absent at Venhoff's layers are taken from
the separately reconstructed six-label bridge asset: uncertainty-estimation
(L18) and adding-knowledge (L18).

The distinction is load-bearing.  The first pair isolates operator/write-dose
changes relative to E1, subject to E1's unresolved execution/input lineage.
The L18 pair is a mixed-clip-window-vintage reconstruction and does not provide
the same isolation.  The generated metadata records that boundary per
behaviour and hashes every source file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
E1 = ROOT / "results/steering_vectors/R1-1.5B__E1_pooled"
RECONSTRUCTED = (
    ROOT / "results/steering_vectors/R1-1.5B__venhoff_bridge_sixlabel"
)
OUT = ROOT / "results/steering_vectors/R1-1.5B__venhoff_bridge_hybrid"

TARGET_LAYERS = {
    "backtracking": 17,
    "uncertainty-estimation": 18,
    "example-testing": 15,
    "adding-knowledge": 18,
}
SOURCE_ROOTS = {
    "backtracking": E1,
    "uncertainty-estimation": RECONSTRUCTED,
    "example-testing": E1,
    "adding-knowledge": RECONSTRUCTED,
}
SOURCE_KINDS = {
    "backtracking": "exact_current_E1_pooled_single_direction",
    "uncertainty-estimation": "reconstructed_exact_layer_six_label_direction",
    "example-testing": "exact_current_E1_pooled_single_direction",
    "adding-knowledge": "reconstructed_exact_layer_six_label_direction",
}

# Pin the intended inputs rather than silently composing a different bridge if
# an upstream artefact is replaced in place.
EXPECTED_VECTOR_SHA256 = {
    "backtracking": "10b12bdd2e90afafee4198e13156d30e585cde41f29993dca2ca91d7c221b2ca",
    "uncertainty-estimation": "1462474e842167ed00b3cf89ab537f5dc7066273dc9503d320bf186986b371c6",
    "example-testing": "e943cd417ff650df5eea05366fae33fe7bdf355de9213310770138168a43a3bb",
    "adding-knowledge": "b57a6301a392994008615fc55d4e39a899e4864e76520c25fba5cbbfdd800a5b",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _write_json(data: dict, path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing hybrid asset: {OUT}")

    metadata_paths = {
        "current_E1_pooled": E1 / "metadata.json",
        "reconstructed_exact_layer": RECONSTRUCTED / "metadata.json",
    }
    required = list(metadata_paths.values()) + [
        root / f"{behaviour}_single.npy"
        for behaviour, root in SOURCE_ROOTS.items()
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))

    source_metadata = {
        key: _load_json(path) for key, path in metadata_paths.items()
    }
    e1_provenance = source_metadata["current_E1_pooled"].get("_provenance", {})
    reconstructed_provenance = source_metadata["reconstructed_exact_layer"].get(
        "_provenance", {}
    )
    if e1_provenance.get("git_commit") is not None:
        raise ValueError(
            "The current E1 source no longer has the expected unresolved "
            "git_commit: null provenance; review this bridge contract before rebuilding"
        )
    if (
        reconstructed_provenance.get("source_contract", {}).get("status")
        != "mixed clip-window vintages"
    ):
        raise ValueError(
            "The reconstructed source no longer has the audited mixed-vintage contract"
        )

    behaviour_metadata: dict[str, dict] = {}
    source_hashes: dict[str, str] = {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in metadata_paths.values()
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=OUT.parent, prefix=f".{OUT.name}."
    ) as tmp_name:
        tmp_dir = Path(tmp_name)
        for behaviour, expected_layer in TARGET_LAYERS.items():
            source_root = SOURCE_ROOTS[behaviour]
            source_key = (
                "current_E1_pooled"
                if source_root == E1
                else "reconstructed_exact_layer"
            )
            source_path = source_root / f"{behaviour}_single.npy"
            source_record = source_metadata[source_key].get(behaviour, {})
            actual_layer = int(source_record.get("layer", -1))
            if actual_layer != expected_layer:
                raise ValueError(
                    f"{behaviour}: source layer is {actual_layer}, expected "
                    f"Venhoff layer {expected_layer}"
                )

            source_sha = _sha256(source_path)
            if source_sha != EXPECTED_VECTOR_SHA256[behaviour]:
                raise ValueError(
                    f"{behaviour}: source SHA-256 changed; expected "
                    f"{EXPECTED_VECTOR_SHA256[behaviour]}, got {source_sha}"
                )
            source_hashes[source_path.relative_to(ROOT).as_posix()] = source_sha

            source_array = np.load(source_path, allow_pickle=False)
            if source_array.shape != (1536,) or source_array.dtype != np.float32:
                raise ValueError(
                    f"{behaviour}: expected float32[1536], got "
                    f"{source_array.dtype}{source_array.shape}"
                )
            vector_norm = float(np.linalg.norm(source_array))
            if not np.isfinite(vector_norm) or not np.isclose(
                vector_norm, 1.0, rtol=1e-6, atol=1e-6
            ):
                raise ValueError(
                    f"{behaviour}: expected a finite unit vector, norm={vector_norm}"
                )

            destination = tmp_dir / f"{behaviour}_single.npy"
            shutil.copyfile(source_path, destination)
            destination_sha = _sha256(destination)
            destination_array = np.load(destination, allow_pickle=False)
            if destination_sha != source_sha or not np.array_equal(
                destination_array, source_array
            ):
                raise RuntimeError(
                    f"{behaviour}: destination is not an exact byte/value copy"
                )

            interpretation = (
                "Exact current thesis E1 direction; changing the operator and "
                "write dose can be isolated while holding this direction fixed."
                if source_key == "current_E1_pooled"
                else
                "Exact-layer six-label reconstruction from mixed clip-window "
                "vintages; this arm does not isolate only operator/write dose "
                "relative to the thesis E1 direction."
            )
            behaviour_metadata[behaviour] = {
                "layer": expected_layer,
                "source_kind": SOURCE_KINDS[behaviour],
                "source_file": source_path.relative_to(ROOT).as_posix(),
                "source_sha256": source_sha,
                "source_metadata": metadata_paths[source_key]
                .relative_to(ROOT)
                .as_posix(),
                "source_metadata_sha256": source_hashes[
                    metadata_paths[source_key].relative_to(ROOT).as_posix()
                ],
                "vector_file": (OUT / destination.name)
                .relative_to(ROOT)
                .as_posix(),
                "vector_sha256": destination_sha,
                "byte_identical_to_source": True,
                "array_equal_to_source": True,
                "vector_norm": vector_norm,
                "n_on": source_record.get("n_on"),
                "n_off": source_record.get("n_off"),
                "n_excluded": source_record.get("n_excluded"),
                "interpretation": interpretation,
            }

        source_contracts = {
            "current_E1_pooled": {
                "behaviours": ["backtracking", "example-testing"],
                "metadata": metadata_paths["current_E1_pooled"]
                .relative_to(ROOT)
                .as_posix(),
                "metadata_sha256": source_hashes[
                    metadata_paths["current_E1_pooled"].relative_to(ROOT).as_posix()
                ],
                "source_git_commit": e1_provenance.get("git_commit"),
                "source_git_dirty": e1_provenance.get("git_dirty"),
                "provenance_status": (
                    "unresolved provenance — E1 metadata records git_commit: null "
                    "and does not hash or uniquely establish its activation inputs"
                ),
                "copy_contract": (
                    "exact .npy bytes and float32 values; no direction recomputation"
                ),
            },
            "reconstructed_exact_layer": {
                "behaviours": ["uncertainty-estimation", "adding-knowledge"],
                "metadata": metadata_paths["reconstructed_exact_layer"]
                .relative_to(ROOT)
                .as_posix(),
                "metadata_sha256": source_hashes[
                    metadata_paths["reconstructed_exact_layer"]
                    .relative_to(ROOT)
                    .as_posix()
                ],
                "provenance_status": "input files and hashes recorded by source asset",
                "source_contract_status": reconstructed_provenance[
                    "source_contract"
                ]["status"],
                "limitation": reconstructed_provenance["source_contract"][
                    "warning"
                ],
                "copy_contract": (
                    "exact .npy bytes and float32 values from the reconstructed asset; "
                    "no further direction recomputation"
                ),
            },
        }
        metadata = {
            "_provenance": {
                "builder": Path(__file__).name,
                "builder_sha256": _sha256(Path(__file__)),
                "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "layers": TARGET_LAYERS,
                "construction": (
                    "byte-for-byte composition of existing unit-vector artefacts; "
                    "no directions recomputed"
                ),
                "provenance_status": (
                    "hybrid: exact hashed composition, with unresolved source "
                    "provenance retained for the E1 backtracking/example-testing leg"
                ),
                "claim_boundary": (
                    "Backtracking and example-testing hold the current E1 directions "
                    "fixed while changing operator/write dose. Uncertainty-estimation "
                    "and adding-knowledge use reconstructed L18 directions and are a "
                    "mixed-vintage sensitivity, not the same isolation."
                ),
                "source_contracts": source_contracts,
                "input_sha256": source_hashes,
            },
            **behaviour_metadata,
        }
        _write_json(metadata, tmp_dir / "metadata.json")
        tmp_dir.replace(OUT)

    print(f"Saved four-direction hybrid bridge asset -> {OUT}")
    for behaviour in TARGET_LAYERS:
        info = behaviour_metadata[behaviour]
        print(
            f"  {behaviour:24s} L{info['layer']:>2d}: "
            f"{info['source_kind']}, sha256={info['vector_sha256']}"
        )


if __name__ == "__main__":
    main()
