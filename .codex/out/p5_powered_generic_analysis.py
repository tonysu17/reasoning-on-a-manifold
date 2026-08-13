#!/usr/bin/env python3
"""Prospective P5 task-paired primary behavioural analysis.

The module contains only deterministic statistical functions and validation.
It does not read current pilot outcomes, call a model, or perform network/pod
work.  Sentence/checkpoint rows remain nested; tasks are the resampling unit.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from typing import Any


SCHEMA_VERSION = "p5-powered-generic-primary-analysis-1"
CONTROL_ROLE = "p5_owned_fullft_control_s42"
SAFETY_ROLE = "p5_owned_fullft_safety_s42"
ENDPOINTS = (
    "backtracking",
    "uncertainty-estimation",
    "example-testing",
    "adding-knowledge",
)
BOOTSTRAP_DRAWS = 10_000
SIGN_FLIP_DRAWS = 100_000
SEED = 20260808


class AnalysisError(RuntimeError):
    pass


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise AnalysisError("invalid percentile input")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def validate_and_index(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    digests: dict[tuple[str, str], str] = {}
    for record in records:
        task_id = record.get("task_id")
        role = record.get("checkpoint_role")
        category = record.get("category")
        if not all(isinstance(value, str) and value for value in (task_id, role, category)):
            raise AnalysisError("task_id, checkpoint_role, and category are required")
        if role not in {CONTROL_ROLE, SAFETY_ROLE}:
            raise AnalysisError("primary analysis received a non-primary checkpoint role")
        endpoints = record.get("endpoints")
        if not isinstance(endpoints, dict) or set(endpoints) != set(ENDPOINTS):
            raise AnalysisError("endpoint keys differ from the frozen primary family")
        for value in endpoints.values():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1
            ):
                raise AnalysisError("endpoint values must be fractions in [0,1] or null")
        for field in ("generation_row_sha256", "prefix_row_sha256", "annotation_row_sha256"):
            if not isinstance(record.get(field), str) or len(record[field]) != 64:
                raise AnalysisError(f"missing or invalid {field}")
        key = (task_id, role)
        digest = sha256_json(record)
        if key in digests:
            if digests[key] != digest:
                raise AnalysisError("conflicting duplicate task-role record")
            continue
        digests[key] = digest
        indexed[task_id][role] = record

    for task_id, roles in indexed.items():
        categories = {record["category"] for record in roles.values()}
        if len(categories) != 1:
            raise AnalysisError(f"category mismatch across roles for {task_id}")
    return dict(indexed)


def endpoint_pairs(
    indexed: dict[str, dict[str, dict[str, Any]]], endpoint: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if endpoint not in ENDPOINTS:
        raise AnalysisError("unknown endpoint")
    pairs = []
    missing = Counter()
    for task_id in sorted(indexed):
        roles = indexed[task_id]
        control = roles.get(CONTROL_ROLE)
        safety = roles.get(SAFETY_ROLE)
        if control is None:
            missing["missing_control_row"] += 1
        if safety is None:
            missing["missing_safety_row"] += 1
        if control is None or safety is None:
            continue
        control_value = control["endpoints"][endpoint]
        safety_value = safety["endpoints"][endpoint]
        if control_value is None:
            missing["missing_control_endpoint"] += 1
        if safety_value is None:
            missing["missing_safety_endpoint"] += 1
        if control_value is None or safety_value is None:
            continue
        pairs.append(
            {
                "task_id": task_id,
                "category": control["category"],
                "control": float(control_value),
                "safety": float(safety_value),
                "difference": float(safety_value) - float(control_value),
            }
        )
    return pairs, dict(sorted(missing.items()))


def stratified_bootstrap_interval(
    pairs: list[dict[str, Any]], *, draws: int = BOOTSTRAP_DRAWS, seed: int = SEED
) -> list[float]:
    if draws < 1 or not pairs:
        raise AnalysisError("bootstrap requires pairs and positive draws")
    by_category: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        by_category[pair["category"]].append(pair["difference"])
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = []
        for category in sorted(by_category):
            values = by_category[category]
            sample.extend(rng.choice(values) for _ in values)
        means.append(sum(sample) / len(sample))
    return [percentile(means, 0.025), percentile(means, 0.975)]


def sign_flip_p_value(
    differences: list[float], *, draws: int = SIGN_FLIP_DRAWS, seed: int = SEED
) -> float:
    if draws < 1 or not differences:
        raise AnalysisError("sign-flip test requires differences and positive draws")
    observed = abs(sum(differences) / len(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        simulated = abs(
            sum(value if rng.getrandbits(1) else -value for value in differences)
            / len(differences)
        )
        if simulated >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (draws + 1)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    if set(p_values) != set(ENDPOINTS):
        raise AnalysisError("Holm family must contain exactly four primary endpoints")
    if any(isinstance(value, bool) or not 0 <= value <= 1 for value in p_values.values()):
        raise AnalysisError("invalid p-value")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def analyze_primary(
    records: list[dict[str, Any]],
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    sign_flip_draws: int = SIGN_FLIP_DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    indexed = validate_and_index(records)
    results = {}
    raw_p = {}
    for endpoint in ENDPOINTS:
        pairs, missing = endpoint_pairs(indexed, endpoint)
        if not pairs:
            raise AnalysisError(f"no complete pairs for {endpoint}")
        differences = [pair["difference"] for pair in pairs]
        raw_p[endpoint] = sign_flip_p_value(differences, draws=sign_flip_draws, seed=seed)
        results[endpoint] = {
            "estimand": "mean task-level safety minus control sentence-fraction difference",
            "unit": "task/prompt",
            "n_planned_tasks_present": len(indexed),
            "n_complete_pairs": len(pairs),
            "complete_pairs_by_category": dict(sorted(Counter(pair["category"] for pair in pairs).items())),
            "control_task_mean": sum(pair["control"] for pair in pairs) / len(pairs),
            "safety_task_mean": sum(pair["safety"] for pair in pairs) / len(pairs),
            "paired_mean_difference": sum(differences) / len(differences),
            "bootstrap_95_interval": stratified_bootstrap_interval(
                pairs, draws=bootstrap_draws, seed=seed
            ),
            "raw_sign_flip_p": raw_p[endpoint],
            "missingness": missing,
        }
    adjusted = holm_adjust(raw_p)
    for endpoint in ENDPOINTS:
        results[endpoint]["holm_adjusted_p"] = adjusted[endpoint]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "computed_primary_checkpoint_contrast",
        "independent_unit": "task/prompt",
        "primary_contrast": f"{SAFETY_ROLE} minus {CONTROL_ROLE}",
        "bootstrap_draws": bootstrap_draws,
        "sign_flip_draws": sign_flip_draws,
        "seed": seed,
        "multiple_testing": "Holm across four primary endpoints",
        "training_seed_boundary": "one owned training seed; checkpoint-bounded, not recipe-level replication",
        "endpoints": results,
    }
