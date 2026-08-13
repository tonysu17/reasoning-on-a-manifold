#!/usr/bin/env python3
"""Create a hash-bound, post-hoc/non-gating J-lens failure map."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_UUID = "jspace-p1-20260810T224338Z-996077031021"
VALIDATION_ATTEMPT_UUID = "jspace-p1v-20260810T231952Z-6ad3de6f3754"
STATUS = "exploratory_posthoc_non_gating"
L17 = 17
K = 25
N_LAYERS = 27
EXPECTED_LICENSED_CONSEQUENCE = (
    "stop_before_Phase_2_the_fitted_lens_did_not_pass_the_prespecified_validity_gate"
)
EXPECTED_SOURCE_HASHES = {
    "ARTIFACT_MANIFEST.json": (
        "4664395bc660b510634d02382dbe6e01caa0a41d86801887547d0a6513782977"
    ),
    "DONE.json": "9a3362a84ab067bef7cea279ee2aacb241d46c703f14ad17df6494322d5ad326",
    "external_positive_controls.json": (
        "f86e83dbaae81cbe748e81d6684f63cd6c106c0911820fc9a768c27156de7ed2"
    ),
    "external_readout_arrays.npz": (
        "02ee47f173b034db744758ac915460d1bd7ed5c347bd4022ca97dd95760c6289"
    ),
    "phase1_report.json": (
        "004145b129f289972e740ad23e20084eb0a9109ef0d89383645f3fc269fc35d9"
    ),
    (
        "validation_attempts/"
        f"{VALIDATION_ATTEMPT_UUID}/INDEPENDENT_VALIDATION.json"
    ): "032fdd87f5d66083a854e9e16e782f8089021dc1eed9bee704cef2cbadd72fa7",
}
EXPECTED_ELIGIBILITY_HASH = (
    "a193ce15ff18d1852a870703dba104763a01b06740c99ec8fd955e3a15927f7c"
)
SUITES = (
    "lens-eval-association",
    "lens-eval-typo",
    "lens-eval-multihop",
)
LENSES = ("fit_a", "fit_b", "merged", "logit")
SUITE_LABELS = {
    "lens-eval-association": "Association",
    "lens-eval-typo": "Typo",
    "lens-eval-multihop": "Multihop",
}
LENS_LABELS = {
    "fit_a": "Fit A",
    "fit_b": "Fit B",
    "merged": "Merged J-lens",
    "logit": "Logit lens",
}
COLORS = {
    "fit_a": "#56B4E9",
    "fit_b": "#009E73",
    "merged": "#D55E00",
    "logit": "#6B7280",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def npz_key(slug: str, lens: str) -> str:
    return f"{slug.replace('-', '_')}__top25__{lens}"


def expected_npz_keys() -> set[str]:
    return {
        npz_key(slug, lens) for slug in SUITES for lens in LENSES
    } | {
        f"{slug.replace('-', '_')}__null__all_layers" for slug in SUITES
    } | {f"{slug.replace('-', '_')}__null__l17" for slug in SUITES}


def require_exact_npz_inventory(files: Iterable[str]) -> None:
    if set(files) != expected_npz_keys():
        raise ValueError("external NPZ key inventory mismatch")


def eligible_items(
    eligibility: dict[str, Any], slug: str
) -> list[dict[str, Any]]:
    evaluations = eligibility.get("evaluations", {})
    if slug not in evaluations:
        raise ValueError(f"missing eligibility suite {slug}")
    items = [
        item
        for item in evaluations[slug].get("items", [])
        if item.get("item_eligible") is True
    ]
    if not items:
        raise ValueError(f"no eligible items for {slug}")
    return items


def eligible_labels(item: dict[str, Any]) -> list[dict[str, Any]]:
    labels = [label for label in item.get("labels", []) if label.get("eligible") is True]
    if not labels:
        raise ValueError(f"eligible item {item.get('name')} has no eligible labels")
    for label in labels:
        token_ids = label.get("scored_token_ids")
        if not isinstance(token_ids, list) or len(token_ids) != 1:
            raise ValueError("post-hoc diagnostic requires one scored token per label")
    return labels


def pass_at_25(
    top25: np.ndarray,
    label_token_lists: list[list[int]],
    *,
    layers: Iterable[int],
) -> float:
    values = np.asarray(top25)
    chosen_layers = list(layers)
    if values.ndim != 3 or values.shape[0] != len(label_token_lists):
        raise ValueError("expected [item,layer,25] with one label list per item")
    if values.shape[1:] != (N_LAYERS, K):
        raise ValueError(f"expected layer/top-k shape {(N_LAYERS, K)}")
    if not chosen_layers or min(chosen_layers) < 0 or max(chosen_layers) >= N_LAYERS:
        raise ValueError("invalid layer selection")
    item_fractions = []
    for index, labels in enumerate(label_token_lists):
        seen = set(int(token) for token in values[index, chosen_layers].reshape(-1))
        item_fractions.append(np.mean([int(label) in seen for label in labels]))
    return float(np.mean(item_fractions))


def items_with_any_hit(
    top25: np.ndarray,
    label_token_lists: list[list[int]],
    *,
    layers: Iterable[int],
) -> int:
    chosen_layers = list(layers)
    count = 0
    for index, labels in enumerate(label_token_lists):
        seen = set(int(token) for token in top25[index, chosen_layers].reshape(-1))
        count += int(any(int(label) in seen for label in labels))
    return count


def censored_rank(row: np.ndarray, token_id: int) -> int:
    positions = np.flatnonzero(np.asarray(row) == int(token_id))
    if len(positions) == 0:
        return K + 1
    return int(positions[0]) + 1


def lowest_peak_layer(scores: list[float]) -> int:
    maximum = max(scores)
    return next(index for index, value in enumerate(scores) if value == maximum)


def validate_sources(
    bundle_root: Path, eligibility_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        require_hash(bundle_root / relative, expected)
    require_hash(eligibility_path, EXPECTED_ELIGIBILITY_HASH)

    artifact_manifest = load_json(bundle_root / "ARTIFACT_MANIFEST.json")
    done = load_json(bundle_root / "DONE.json")
    report = load_json(bundle_root / "phase1_report.json")
    external = load_json(bundle_root / "external_positive_controls.json")
    validation = load_json(
        bundle_root
        / "validation_attempts"
        / VALIDATION_ATTEMPT_UUID
        / "INDEPENDENT_VALIDATION.json"
    )
    if artifact_manifest.get("run_uuid") != RUN_UUID:
        raise ValueError("artifact manifest run UUID mismatch")
    if done.get("run_uuid") != RUN_UUID or done.get("state") != "DONE":
        raise ValueError("DONE identity/state mismatch")
    if done.get("integrity_valid_complete") is not True:
        raise ValueError("bundle is not integrity-valid complete")
    if done.get("scientific_gate_pass") is not False:
        raise ValueError("diagnostic is licensed only for the sealed negative gate")
    if validation.get("ok") is not True or not all(validation.get("checks", {}).values()):
        raise ValueError("authoritative independent validation is not fully true")
    if report.get("licensed_consequence") != EXPECTED_LICENSED_CONSEQUENCE:
        raise ValueError("licensed consequence mismatch")
    if report.get("scientific_gate", {}).get("pass") is not False:
        raise ValueError("scientific gate mismatch")
    if external.get("gate") != {
        "n_qualifying": 1,
        "pass": False,
        "qualifying_evaluations": ["lens-eval-typo"],
        "required": 2,
    }:
        raise ValueError("external gate differs from sealed outcome")

    registered_files = artifact_manifest.get("files", {})
    for relative in (
        "external_positive_controls.json",
        "external_readout_arrays.npz",
        "phase1_report.json",
        (
            "validation_attempts/"
            f"{VALIDATION_ATTEMPT_UUID}/INDEPENDENT_VALIDATION.json"
        ),
    ):
        if registered_files.get(relative, {}).get("sha256") != EXPECTED_SOURCE_HASHES[relative]:
            raise ValueError(f"artifact manifest does not register expected {relative}")
    eligibility = load_json(eligibility_path)
    return eligibility, external, report


def build_diagnostic(
    bundle_root: Path,
    eligibility_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    eligibility, external, report = validate_sources(bundle_root, eligibility_path)
    arrays = np.load(bundle_root / "external_readout_arrays.npz", allow_pickle=False)
    require_exact_npz_inventory(arrays.files)

    layer_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    suite_summaries: dict[str, Any] = {}
    for slug in SUITES:
        items = eligible_items(eligibility, slug)
        item_names = [str(item["name"]) for item in items]
        if item_names != external[slug]["eligible_item_names"]:
            raise ValueError(f"eligible item order mismatch for {slug}")
        label_records = [eligible_labels(item) for item in items]
        label_token_lists = [
            [int(label["scored_token_ids"][0]) for label in labels]
            for labels in label_records
        ]
        n_labels = sum(len(labels) for labels in label_token_lists)
        lens_arrays: dict[str, np.ndarray] = {}
        lens_summaries: dict[str, Any] = {}
        for lens in LENSES:
            values = np.asarray(arrays[npz_key(slug, lens)], dtype=np.int64)
            if values.shape != (len(items), N_LAYERS, K):
                raise ValueError(f"unexpected {slug}/{lens} shape {values.shape}")
            if values.min() < 0 or values.max() >= 151_665:
                raise ValueError(f"token ID outside sealed domain for {slug}/{lens}")
            lens_arrays[lens] = values
            per_layer = [
                pass_at_25(values, label_token_lists, layers=[layer])
                for layer in range(N_LAYERS)
            ]
            cumulative = [
                pass_at_25(values, label_token_lists, layers=range(layer + 1))
                for layer in range(N_LAYERS)
            ]
            all_layers = pass_at_25(values, label_token_lists, layers=range(N_LAYERS))
            l17 = pass_at_25(values, label_token_lists, layers=[L17])
            registered_scores = external[slug]["scores"][lens]
            if not math.isclose(all_layers, registered_scores["all_layers"], abs_tol=1e-15):
                raise ValueError(f"all-layer endpoint mismatch for {slug}/{lens}")
            if not math.isclose(l17, registered_scores["l17"], abs_tol=1e-15):
                raise ValueError(f"L17 endpoint mismatch for {slug}/{lens}")
            peak = lowest_peak_layer(per_layer)
            lens_summaries[lens] = {
                "all_layers_union_pass_at25": all_layers,
                "l17_pass_at25": l17,
                "peak_layer": peak,
                "peak_layer_pass_at25": per_layer[peak],
                "n_items_with_any_hit_all_layers": items_with_any_hit(
                    values, label_token_lists, layers=range(N_LAYERS)
                ),
                "n_items_with_any_hit_l17": items_with_any_hit(
                    values, label_token_lists, layers=[L17]
                ),
            }
            for layer in range(N_LAYERS):
                layer_rows.append(
                    {
                        "suite": slug,
                        "lens": lens,
                        "layer": layer,
                        "pass_at25": per_layer[layer],
                        "cumulative_union_pass_at25": cumulative[layer],
                        "n_items_with_any_hit_at_layer": items_with_any_hit(
                            values, label_token_lists, layers=[layer]
                        ),
                        "n_items": len(items),
                        "n_labels": n_labels,
                    }
                )
            for item_index, (item, labels) in enumerate(zip(items, label_records)):
                for label in labels:
                    token_id = int(label["scored_token_ids"][0])
                    for layer in range(N_LAYERS):
                        rank = censored_rank(values[item_index, layer], token_id)
                        rank_rows.append(
                            {
                                "suite": slug,
                                "lens": lens,
                                "item_index": int(item["item_index"]),
                                "item_name": str(item["name"]),
                                "label_index": int(label["label_index"]),
                                "label": str(label["label"]),
                                "scored_form": str(label["scored_form"]),
                                "token_id": token_id,
                                "layer": layer,
                                "censored_rank": rank,
                                "within_top25": rank <= K,
                            }
                        )

        merged_scores = [
            row["pass_at25"]
            for row in layer_rows
            if row["suite"] == slug and row["lens"] == "merged"
        ]
        logit_scores = [
            row["pass_at25"]
            for row in layer_rows
            if row["suite"] == slug and row["lens"] == "logit"
        ]
        merged_hit_ranks = [
            row["censored_rank"]
            for row in rank_rows
            if row["suite"] == slug
            and row["lens"] == "merged"
            and row["within_top25"]
        ]
        merged_l17_hit_ranks = [
            row["censored_rank"]
            for row in rank_rows
            if row["suite"] == slug
            and row["lens"] == "merged"
            and row["layer"] == L17
            and row["within_top25"]
        ]
        suite_summaries[slug] = {
            "n_items": len(items),
            "n_labels": n_labels,
            "registered_gate": external[slug]["qualifying_success"],
            "registered_permutation": external[slug]["permutation"],
            "registered_checks": external[slug]["qualifying_checks"],
            "lenses": lens_summaries,
            "merged_minus_logit_by_layer": [
                merged - logit for merged, logit in zip(merged_scores, logit_scores)
            ],
            "merged_hit_rank_summary": (
                {
                    "n_label_layer_hits": len(merged_hit_ranks),
                    "median": float(np.median(merged_hit_ranks)),
                    "minimum": min(merged_hit_ranks),
                    "maximum": max(merged_hit_ranks),
                }
                if merged_hit_ranks
                else None
            ),
            "merged_l17_hit_rank_summary": (
                {
                    "n_label_hits": len(merged_l17_hit_ranks),
                    "median": float(np.median(merged_l17_hit_ranks)),
                    "minimum": min(merged_l17_hit_ranks),
                    "maximum": max(merged_l17_hit_ranks),
                }
                if merged_l17_hit_ranks
                else None
            ),
        }

    diagnostic = {
        "schema_version": "rom-jspace-r1-phase1-posthoc-failure-map-v1",
        "status": STATUS,
        "run_uuid": RUN_UUID,
        "validation_attempt_uuid": VALIDATION_ATTEMPT_UUID,
        "source_integrity_valid_complete": True,
        "scientific_gate_pass": False,
        "licensed_consequence": EXPECTED_LICENSED_CONSEQUENCE,
        "registered_external_gate": external["gate"],
        "method": {
            "rank_domain": "top25_only_rank_26_means_greater_than_25",
            "per_layer_estimand": (
                "mean_over_items_of_within_item_fraction_of_eligible_labels_in_layer_top25"
            ),
            "cumulative_estimand": (
                "same_item_fraction_after_unioning_top25_sets_from_layer_0_through_layer_l"
            ),
            "registered_all_layers_estimand": (
                "same_item_fraction_after_unioning_all_27_layer_top25_sets_not_a_layer_average"
            ),
            "interpretation_boundary": (
                "posthoc_descriptive_only_does_not_reclassify_phase1_or_license_phase2"
            ),
        },
        "source_sha256": {
            **EXPECTED_SOURCE_HASHES,
            "eligibility_manifest": EXPECTED_ELIGIBILITY_HASH,
        },
        "suite_summaries": suite_summaries,
        "registered_scientific_gate": report["scientific_gate"],
    }
    return diagnostic, layer_rows, rank_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_layerwise(layer_rows: list[dict[str, Any]], path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharex=True, sharey=True)
    for axis, slug in zip(axes, SUITES):
        for lens in LENSES:
            rows = [
                row for row in layer_rows if row["suite"] == slug and row["lens"] == lens
            ]
            rows.sort(key=lambda row: row["layer"])
            width = 2.5 if lens == "merged" else 1.4
            style = "--" if lens == "logit" else "-"
            alpha = 1.0 if lens in {"merged", "logit"} else 0.75
            axis.plot(
                [row["layer"] for row in rows],
                [100 * row["pass_at25"] for row in rows],
                label=LENS_LABELS[lens],
                color=COLORS[lens],
                linewidth=width,
                linestyle=style,
                alpha=alpha,
            )
        merged = [
            row for row in layer_rows if row["suite"] == slug and row["lens"] == "merged"
        ]
        merged.sort(key=lambda row: row["layer"])
        scores = [row["pass_at25"] for row in merged]
        peak = lowest_peak_layer(scores)
        axis.scatter(
            [peak],
            [100 * scores[peak]],
            color=COLORS["merged"],
            s=28,
            zorder=5,
        )
        axis.annotate(
            f"peak L{peak}",
            (peak, 100 * scores[peak]),
            xytext=(4, 7),
            textcoords="offset points",
            fontsize=8,
            color=COLORS["merged"],
        )
        axis.axvline(L17, color="#111827", linewidth=1.1, linestyle=":")
        axis.set_title(SUITE_LABELS[slug], fontsize=12, fontweight="bold")
        axis.set_xlim(0, N_LAYERS - 1)
        axis.set_ylim(0, 100)
        axis.set_xticks([0, 5, 10, 15, 17, 20, 25])
        axis.set_xlabel("Source layer (zero-based)", fontsize=10)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel(
        "Mean within-item eligible-label hit fraction (%)", fontsize=10
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "External-control pass@25 varied by suite and source layer",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.995,
        0.015,
        "Post-hoc, non-gating diagnostic; dotted line = preregistered L17",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        metadata={"Software": "matplotlib", "Creation Time": "2026-08-11"},
    )
    plt.close(fig)


def count_fraction(summary: dict[str, Any], field: str) -> str:
    lens = summary["lenses"]["merged"]
    return f"{lens[field]}/{summary['n_items']}"


def write_report(path: Path, diagnostic: dict[str, Any]) -> None:
    suites = diagnostic["suite_summaries"]
    association = suites["lens-eval-association"]
    typo = suites["lens-eval-typo"]
    multihop = suites["lens-eval-multihop"]
    text = f"""# R1 J-lens Phase-1 post-hoc failure map

**Status:** exploratory post-hoc diagnostic; non-gating.  
**Source run:** `{RUN_UUID}`; independently validated under `{VALIDATION_ATTEMPT_UUID}`.  
**Registered outcome remains:** the fitted lens did not pass the prespecified validity gate. Phase 2 remains prospective/unrun.

## Result

The lens was numerically valid and split-half stable, but descriptive external-control pass@25 varied by suite and source layer. The registered inferential endpoints remain the all-layer union and L17; the external gate required two of three suites, and only typo qualified.

| Suite | Merged any-layer union (items with >=1 hit) | Merged L17 (items with >=1 hit) | Post-hoc descriptive peak (merged) | Registered disposition |
|---|---:|---:|---:|---|
| Association | {association['lenses']['merged']['all_layers_union_pass_at25']:.4f} ({count_fraction(association, 'n_items_with_any_hit_all_layers')}) | {association['lenses']['merged']['l17_pass_at25']:.4f} ({count_fraction(association, 'n_items_with_any_hit_l17')}) | L{association['lenses']['merged']['peak_layer']}: {association['lenses']['merged']['peak_layer_pass_at25']:.4f} | non-qualifying |
| Typo | {typo['lenses']['merged']['all_layers_union_pass_at25']:.4f} ({count_fraction(typo, 'n_items_with_any_hit_all_layers')}) | {typo['lenses']['merged']['l17_pass_at25']:.4f} ({count_fraction(typo, 'n_items_with_any_hit_l17')}) | L{typo['lenses']['merged']['peak_layer']}: {typo['lenses']['merged']['peak_layer_pass_at25']:.4f} | qualifying |
| Multihop | {multihop['lenses']['merged']['all_layers_union_pass_at25']:.4f} ({count_fraction(multihop, 'n_items_with_any_hit_all_layers')}) | {multihop['lenses']['merged']['l17_pass_at25']:.4f} ({count_fraction(multihop, 'n_items_with_any_hit_l17')}) | L{multihop['lenses']['merged']['peak_layer']}: {multihop['lenses']['merged']['peak_layer_pass_at25']:.4f} | non-qualifying |

The scores are means over items of the within-item fraction of eligible labels found in the relevant top-25 set; the parenthetic counts are the distinct number of items with at least one hit. “Any-layer union” is the registered item-level union over all 27 source layers, not an average of layer scores. Multihop passed the registered all-layer permutation criterion but failed the L17 criterion. Descriptively, association is sparse across the layer profile, while typo is strong through the middle layers and remains substantial at L17.

![Layerwise pass@25](jspace_phase1_layerwise_pass25.png)

## Interpretation boundary

This diagnostic characterizes why the sealed gate did not pass. It does not replace the registered endpoint, select a new layer, change K, alter target eligibility, rescue the lens, or license J-space decomposition or causal intervention. Moving a future study from L17 to L24–26 would be a new hypothesis requiring a new preregistration and fresh confirmatory controls.

## Files

- `failure_map.json`: hash-bound structured summary and source identities.
- `layerwise_pass25.csv`: per-suite, per-lens layer profiles and cumulative unions.
- `target_rank_censored.csv`: eligible-label ranks, where 26 means “not in top 25.”
- `jspace_phase1_layerwise_pass25.png`: aligned three-panel visualization.
- `DIAGNOSTIC_MANIFEST.json`: output and source hashes.
"""
    path.write_text(text)


def write_manifest(output_dir: Path) -> None:
    generated = {}
    for name in (
        "failure_map.json",
        "layerwise_pass25.csv",
        "target_rank_censored.csv",
        "jspace_phase1_layerwise_pass25.png",
        "REPORT.md",
    ):
        path = output_dir / name
        generated[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema_version": "rom-jspace-r1-phase1-posthoc-diagnostic-manifest-v1",
        "status": STATUS,
        "run_uuid": RUN_UUID,
        "validation_attempt_uuid": VALIDATION_ATTEMPT_UUID,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_sha256": {
            **EXPECTED_SOURCE_HASHES,
            "eligibility_manifest": EXPECTED_ELIGIBILITY_HASH,
        },
        "generated": generated,
        "scientific_gate_pass": False,
        "phase2_licensed": False,
    }
    (output_dir / "DIAGNOSTIC_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--eligibility-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    eligibility_path = args.eligibility_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostic, layer_rows, rank_rows = build_diagnostic(bundle_root, eligibility_path)
    (output_dir / "failure_map.json").write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
    )
    write_csv(
        output_dir / "layerwise_pass25.csv",
        layer_rows,
        [
            "suite",
            "lens",
            "layer",
            "pass_at25",
            "cumulative_union_pass_at25",
            "n_items_with_any_hit_at_layer",
            "n_items",
            "n_labels",
        ],
    )
    write_csv(
        output_dir / "target_rank_censored.csv",
        rank_rows,
        [
            "suite",
            "lens",
            "item_index",
            "item_name",
            "label_index",
            "label",
            "scored_form",
            "token_id",
            "layer",
            "censored_rank",
            "within_top25",
        ],
    )
    plot_layerwise(layer_rows, output_dir / "jspace_phase1_layerwise_pass25.png")
    write_report(output_dir / "REPORT.md", diagnostic)
    write_manifest(output_dir)
    print(
        json.dumps(
            {
                "status": STATUS,
                "output_dir": str(output_dir),
                "scientific_gate_pass": False,
                "phase2_licensed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
