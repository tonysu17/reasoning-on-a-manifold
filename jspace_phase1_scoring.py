#!/usr/bin/env python3
"""Pure scoring utilities for the sealed R1 J-lens Phase-1 validity gates."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


K = 25
L17 = 17


def higher_quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="higher"))


def empirical_upper_p(null: np.ndarray, observed: float) -> float:
    values = np.asarray(null, dtype=np.float64)
    return float((1 + np.count_nonzero(values >= observed)) / (len(values) + 1))


def deterministic_topk_ids(logits: Any, k: int = K) -> tuple[Any, int]:
    """Return IDs ordered by descending logit then ascending token ID.

    The fast path uses ``topk`` when the kth boundary is strict.  If any row
    ties at the boundary, a stable full sort implements the registered token-ID
    tie break.  The second return value is the number of tied rows.
    """
    import torch

    if logits.ndim < 1 or logits.shape[-1] <= k:
        raise ValueError(f"need vocab dimension > {k}, got {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("readout logits contain NaN or infinity")
    values, ids = torch.topk(logits, k=k + 1, dim=-1, largest=True, sorted=True)
    tied = values[..., k - 1] == values[..., k]
    n_tied = int(tied.sum().item())
    ids = ids[..., :k]
    if n_tied:
        # Resolve only tied rows.  ``nonzero`` traverses token IDs in ascending
        # order, so the boundary subset is deterministic without sorting the
        # full vocabulary for every non-tied row in the batch.
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_ids = ids.reshape(-1, k)
        tied_flat = tied.reshape(-1)
        flat_values = values[..., :k].reshape(-1, k)
        for row_index in tied_flat.nonzero(as_tuple=True)[0].tolist():
            threshold = flat_values[row_index, k - 1]
            strict = flat_ids[row_index][flat_values[row_index] > threshold]
            need = k - strict.numel()
            boundary = (flat_logits[row_index] == threshold).nonzero(as_tuple=True)[0][
                :need
            ]
            flat_ids[row_index] = torch.cat((strict, boundary))
        ids = flat_ids.reshape(*ids.shape)
    # Top-k does not promise a token-ID order for non-boundary ties.  Sort
    # lexicographically for storage; set-based scoring is unchanged.
    chosen_values = torch.gather(logits, -1, ids)
    order_id = torch.argsort(ids, dim=-1, stable=True)
    ids_by_id = torch.gather(ids, -1, order_id)
    values_by_id = torch.gather(chosen_values, -1, order_id)
    order_value = torch.argsort(values_by_id, dim=-1, descending=True, stable=True)
    ids = torch.gather(ids_by_id, -1, order_value)
    return ids, n_tied


def tokenizer_vocabulary_logits(
    logits: Any, *, head_vocab_size: int, tokenizer_vocab_size: int
) -> Any:
    """Restrict a padded model head to the registered tokenizer-ID domain.

    Qwen-family output heads may contain padding rows that are not token IDs.
    Validate the complete head, including those rows, before returning the
    contiguous tokenizer prefix used by the registered vocabulary null.
    """
    import torch

    if logits.ndim < 1 or logits.shape[-1] != head_vocab_size:
        observed = logits.shape[-1] if logits.ndim else None
        raise ValueError(f"readout head size {observed} != {head_vocab_size}")
    if tokenizer_vocab_size <= 0 or tokenizer_vocab_size > head_vocab_size:
        raise ValueError("invalid tokenizer/head vocabulary dimensions")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("full readout head contains NaN or infinity")
    return logits[..., :tokenizer_vocab_size]


def _intersection_counts(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.shape[-1] != K:
        raise ValueError(f"top-k arrays must have equal [...,{K}] shape")
    if np.any(np.diff(np.sort(left, axis=-1), axis=-1) == 0):
        raise ValueError("left top-k rows contain duplicate token IDs")
    if np.any(np.diff(np.sort(right, axis=-1), axis=-1) == 0):
        raise ValueError("right top-k rows contain duplicate token IDs")
    return (left[..., :, None] == right[..., None, :]).sum(axis=(-1, -2))


def jaccard_topk(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    intersection = _intersection_counts(left, right).astype(np.float64)
    return intersection / (2 * K - intersection)


def row_layer_and_layer_medians(
    top25_a: np.ndarray, top25_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate [row,layer,position,25] top-k arrays in registered order."""
    if top25_a.ndim != 4 or top25_a.shape[1:] != top25_b.shape[1:]:
        raise ValueError("expected matching [row,layer,position,25] arrays")
    per_position = jaccard_topk(top25_a, top25_b)
    row_layer = np.median(per_position, axis=2)
    by_layer = np.median(row_layer, axis=0)
    return per_position, row_layer, by_layer


def all_candidate_bands(n_layers: int, anchor: int = L17, min_length: int = 4) -> list[tuple[int, int]]:
    if not 0 <= anchor < n_layers:
        raise ValueError("anchor out of range")
    return [
        (start, end)
        for start in range(anchor + 1)
        for end in range(anchor, n_layers)
        if end - start + 1 >= min_length
    ]


def band_scan(layer_values: np.ndarray, anchor: int = L17, min_length: int = 4) -> dict[str, Any]:
    values = np.asarray(layer_values, dtype=np.float64)
    candidates = []
    for start, end in all_candidate_bands(len(values), anchor, min_length):
        candidates.append(
            {
                "start": start,
                "end": end,
                "length": end - start + 1,
                "score": float(np.min(values[start : end + 1])),
            }
        )
    # Max score; then shortest length; then lowest start.
    winner = sorted(candidates, key=lambda row: (-row["score"], row["length"], row["start"]))[0]
    return winner


def bootstrap_layer_medians(
    row_layer: np.ndarray, *, seed: int = 20260813, n_boot: int = 10_000
) -> dict[str, list[float]]:
    values = np.asarray(row_layer, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, values.shape[0], size=(n_boot, values.shape[0]))
    boot = np.median(values[indices, :], axis=1)
    return {
        "lower_0.025": np.quantile(boot, 0.025, axis=0).tolist(),
        "upper_0.975": np.quantile(boot, 0.975, axis=0).tolist(),
    }


def _torch_midpoint_median(values: Any, dim: int) -> Any:
    ordered = values.sort(dim=dim).values
    n = ordered.shape[dim]
    if n % 2:
        return ordered.select(dim, n // 2)
    return (ordered.select(dim, n // 2 - 1) + ordered.select(dim, n // 2)) / 2


def stability_permutation_null(
    top25_a: np.ndarray,
    top25_b: np.ndarray,
    *,
    vocab_size: int,
    seed: int = 20260811,
    n_perm: int = 1_000,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply each global vocabulary bijection to B only and repeat full scan.

    Returns ``(scan_scores[n_perm], layer_medians[n_perm,n_layers])``.
    """
    import torch

    a = np.asarray(top25_a, dtype=np.int64)
    b = np.asarray(top25_b, dtype=np.int64)
    if a.shape != b.shape or a.ndim != 4 or a.shape[-1] != K:
        raise ValueError("expected matching [row,layer,position,25] arrays")
    if a.min() < 0 or b.min() < 0 or a.max() >= vocab_size or b.max() >= vocab_size:
        raise ValueError("token ID outside registered vocabulary")
    a_t = torch.from_numpy(a).to(device)
    b_t = torch.from_numpy(b).to(device)
    rng = np.random.Generator(np.random.PCG64(seed))
    layer_null = np.empty((n_perm, a.shape[1]), dtype=np.float64)
    scan_null = np.empty(n_perm, dtype=np.float64)
    with torch.no_grad():
        for index in range(n_perm):
            permutation = torch.from_numpy(
                rng.permutation(vocab_size).astype(np.int64, copy=False)
            ).to(device)
            mapped_b = permutation[b_t]
            intersection = (
                a_t[..., :, None] == mapped_b[..., None, :]
            ).sum(dim=(-1, -2)).double()
            per_position = intersection / (2 * K - intersection)
            row_layer = _torch_midpoint_median(per_position, dim=2)
            by_layer = _torch_midpoint_median(row_layer, dim=0)
            layer_values = by_layer.cpu().numpy().astype(np.float64, copy=False)
            layer_null[index] = layer_values
            scan_null[index] = band_scan(layer_values)["score"]
            del permutation, mapped_b, intersection, per_position, row_layer, by_layer
    return scan_null, layer_null


def stability_report(
    top25_a: np.ndarray,
    top25_b: np.ndarray,
    scan_null: np.ndarray,
    layer_null: np.ndarray,
    *,
    absolute_floor: float = 0.10,
) -> dict[str, Any]:
    if np.asarray(top25_a).shape != (20, 27, 111, K):
        raise ValueError("production held-out A shape must be [20,27,111,25]")
    if np.asarray(top25_b).shape != (20, 27, 111, K):
        raise ValueError("production held-out B shape must be [20,27,111,25]")
    if np.asarray(scan_null).shape != (1_000,) or np.asarray(layer_null).shape != (
        1_000,
        27,
    ):
        raise ValueError("stability null must be exactly [1000] and [1000,27]")
    if not np.isfinite(scan_null).all() or not np.isfinite(layer_null).all():
        raise ValueError("stability null contains NaN or infinity")
    _, row_layer, by_layer = row_layer_and_layer_medians(top25_a, top25_b)
    winner = band_scan(by_layer)
    q95 = higher_quantile(scan_null, 0.95)
    p = empirical_upper_p(scan_null, winner["score"])
    return {
        "aggregation": "median_positions_then_median_rows",
        "row_layer_median": row_layer.tolist(),
        "layer_median": by_layer.tolist(),
        "selected_band": winner,
        "scan_null_p95_higher": q95,
        "scan_empirical_upper_p": p,
        "absolute_floor": absolute_floor,
        "bootstrap_10000": {
            "interval_quantile_method": "numpy_default_linear",
            **bootstrap_layer_medians(row_layer),
        },
        "pointwise_null_p95_higher": [
            higher_quantile(layer_null[:, layer], 0.95)
            for layer in range(layer_null.shape[1])
        ],
        "gate": {
            "above_scan_null_p95": bool(winner["score"] > q95),
            "empirical_p_at_most_0.05": bool(p <= 0.05),
            "at_least_absolute_floor": bool(winner["score"] >= absolute_floor),
            "pass": bool(winner["score"] > q95 and p <= 0.05 and winner["score"] >= absolute_floor),
        },
    }


def pass_at_25(top25: np.ndarray, label_lists: list[list[int]], *, layers: Iterable[int]) -> float:
    values = np.asarray(top25)
    if values.ndim != 3 or values.shape[0] != len(label_lists) or values.shape[2] != K:
        raise ValueError("expected [item,layer,25] and one label list per item")
    selected = values[:, list(layers), :]
    item_fractions = []
    for item_index, labels in enumerate(label_lists):
        if not labels:
            raise ValueError("eligible item has no eligible labels")
        seen = set(int(value) for value in selected[item_index].reshape(-1))
        item_fractions.append(np.mean([int(label) in seen for label in labels]))
    return float(np.mean(item_fractions))


def external_permutation_null(
    merged_top25: np.ndarray,
    label_lists: list[list[int]],
    *,
    rng: np.random.Generator,
    n_perm: int = 1_000,
    l17: int = L17,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(merged_top25)
    n_items = values.shape[0]
    if n_items != len(label_lists):
        raise ValueError("label list count mismatch")
    hit_all = np.empty((n_items, n_items), dtype=np.float64)
    hit_l17 = np.empty((n_items, n_items), dtype=np.float64)
    seen_all = [set(int(value) for value in values[i].reshape(-1)) for i in range(n_items)]
    seen_l17 = [set(int(value) for value in values[i, l17].reshape(-1)) for i in range(n_items)]
    for prompt_index in range(n_items):
        for source_index, labels in enumerate(label_lists):
            hit_all[prompt_index, source_index] = np.mean(
                [int(label) in seen_all[prompt_index] for label in labels]
            )
            hit_l17[prompt_index, source_index] = np.mean(
                [int(label) in seen_l17[prompt_index] for label in labels]
            )
    all_null = np.empty(n_perm, dtype=np.float64)
    l17_null = np.empty(n_perm, dtype=np.float64)
    prompt_indices = np.arange(n_items)
    for index in range(n_perm):
        assignment = rng.permutation(n_items)
        all_null[index] = float(np.mean(hit_all[prompt_indices, assignment]))
        l17_null[index] = float(np.mean(hit_l17[prompt_indices, assignment]))
    return all_null, l17_null


def external_report(
    top25_by_lens: dict[str, np.ndarray],
    label_lists: list[list[int]],
    all_null: np.ndarray,
    l17_null: np.ndarray,
) -> dict[str, Any]:
    expected = {"fit_a", "fit_b", "merged", "logit"}
    if set(top25_by_lens) != expected:
        raise ValueError(f"lens keys must be {sorted(expected)}")
    if np.asarray(all_null).shape != (1_000,) or np.asarray(l17_null).shape != (1_000,):
        raise ValueError("external null arrays must each contain exactly 1,000 draws")
    if not np.isfinite(all_null).all() or not np.isfinite(l17_null).all():
        raise ValueError("external null contains NaN or infinity")
    scores: dict[str, dict[str, float]] = {}
    n_layers = top25_by_lens["merged"].shape[1]
    for name, values in top25_by_lens.items():
        scores[name] = {
            "all_layers": pass_at_25(values, label_lists, layers=range(n_layers)),
            "l17": pass_at_25(values, label_lists, layers=[L17]),
        }
    all_q95 = higher_quantile(all_null, 0.95)
    l17_q95 = higher_quantile(l17_null, 0.95)
    all_p = empirical_upper_p(all_null, scores["merged"]["all_layers"])
    l17_p = empirical_upper_p(l17_null, scores["merged"]["l17"])
    checks = {
        "merged_all_above_p95": scores["merged"]["all_layers"] > all_q95,
        "merged_all_p_at_most_0.05": all_p <= 0.05,
        "merged_l17_above_p95": scores["merged"]["l17"] > l17_q95,
        "merged_l17_p_at_most_0.05": l17_p <= 0.05,
        "fit_ab_all_difference_at_most_0.10": abs(
            scores["fit_a"]["all_layers"] - scores["fit_b"]["all_layers"]
        ) <= 0.10,
        "fit_ab_l17_difference_at_most_0.10": abs(
            scores["fit_a"]["l17"] - scores["fit_b"]["l17"]
        ) <= 0.10,
    }
    return {
        "n_eligible_items": len(label_lists),
        "n_eligible_labels": sum(len(labels) for labels in label_lists),
        "scores": scores,
        "permutation": {
            "n": len(all_null),
            "all_layers_p95_higher": all_q95,
            "all_layers_empirical_upper_p": all_p,
            "l17_p95_higher": l17_q95,
            "l17_empirical_upper_p": l17_p,
        },
        "qualifying_checks": checks,
        "qualifying_success": bool(all(checks.values())),
    }
