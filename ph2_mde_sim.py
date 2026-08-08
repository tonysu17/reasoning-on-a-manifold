#!/usr/bin/env python3
"""Simulation MDE for the Phase-2 transported-frame intervention battery.

This is a sizing model, not a hypothesis test and not a result regeneration. It reads the
existing E8 backtracking single-direction arm and its authoritative energy-matched floor,
separates empirical task heterogeneity from a binomial sentence-label noise floor, and
simulates two model batteries under either independent or shared task sampling.

The legacy Phase-0 first cut is also reproduced under its exact extra-conservative
assumptions as a calibration check. See the generated Markdown's README section for the
limits of the variance decomposition and the n>49 extrapolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "results/eval/R1-1.5B__E1"
DEFAULT_JSON = ROOT / ".codex/out/ph2_mde_sim.json"
DEFAULT_MD = ROOT / ".codex/out/ph2_mde_sim.md"

BEHAVIOUR = "backtracking"
ARM = "single_direction"
TASK_COUNTS = (50, 100, 150, 200, 300, 400)
ATTENUATIONS = tuple(round(x / 10, 1) for x in range(1, 11))
ESTIMANDS = ("sentence_fraction", "per_1k_tokens")
DESIGNS = ("unpaired_across_models", "task_paired_across_models")
ALPHA = 0.05
TARGET_POWER = 0.80
SEED = 20260802
DEFAULT_N_SIM = 5_000
LEGACY_B = 10_000
Z_SUM = 1.6449 + 0.8416


@dataclass(frozen=True)
class Observation:
    count: int
    n_sentences: int
    n_tokens: int


@dataclass(frozen=True)
class TaskProfile:
    task_id: str
    arm_value: float
    floor_value: float
    delta: float
    arm_noise_variance: float
    floor_noise_variance: float

    @property
    def delta_noise_variance(self) -> float:
        return self.arm_noise_variance + self.floor_noise_variance


def _key(row: dict) -> tuple[str, str, str, float]:
    return (str(row["task_id"]), str(row["behaviour"]), str(row["method"]),
            float(row["alpha"]))


def _base_task(row: dict) -> str:
    value = row.get("base_task_id")
    return str(value) if value else str(row.get("task_id", "")).split("#")[0]


def _count_target(annotations: Iterable[dict]) -> int:
    return sum(1 for annotation in annotations if annotation.get("label") == BEHAVIOUR)


def load_observations(method: str, alpha: float, annotated: list[dict],
                      steered: list[dict]) -> dict[str, list[Observation]]:
    """Load valid annotated records, retaining every sample/subspace replicate per task."""
    valid_keys = {_key(row) for row in steered}
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for row in annotated:
        if row.get("behaviour") != BEHAVIOUR or row.get("method") != method:
            continue
        if abs(float(row.get("alpha", np.nan)) - alpha) > 1e-9:
            continue
        if _key(row) not in valid_keys:
            continue
        annotations = row.get("annotations") or []
        if not annotations:
            continue
        n_tokens = int(row.get("n_tokens") or 0)
        if n_tokens <= 0:
            continue
        grouped[_base_task(row)].append(Observation(
            count=_count_target(annotations),
            n_sentences=len(annotations),
            n_tokens=n_tokens,
        ))
    return dict(grouped)


def _observation_value_and_variance(obs: Observation, estimand: str) -> tuple[float, float]:
    """Observed value and an independent-sentence binomial sampling-variance floor."""
    p = obs.count / obs.n_sentences
    if estimand == "sentence_fraction":
        value = p
        variance = p * (1.0 - p) / obs.n_sentences
    elif estimand == "per_1k_tokens":
        scale = 1000.0 / obs.n_tokens
        value = scale * obs.count
        variance = scale * scale * obs.n_sentences * p * (1.0 - p)
    else:  # pragma: no cover - guarded by the fixed ESTIMANDS set
        raise ValueError(f"unknown estimand: {estimand}")
    return float(value), float(variance)


def _pooled_value_and_variance(observations: list[Observation],
                               estimand: str) -> tuple[float, float]:
    """Pool replicates as E8 does: equal-weight mean within task before resampling."""
    pairs = [_observation_value_and_variance(obs, estimand) for obs in observations]
    values = np.asarray([pair[0] for pair in pairs], dtype=float)
    variances = np.asarray([pair[1] for pair in pairs], dtype=float)
    r = len(pairs)
    return float(values.mean()), float(variances.sum() / (r * r))


def build_profiles(arm_rows: dict[str, list[Observation]],
                   floor_rows: dict[str, list[Observation]],
                   estimand: str) -> list[TaskProfile]:
    profiles = []
    for task_id in sorted(set(arm_rows) & set(floor_rows)):
        arm_value, arm_var = _pooled_value_and_variance(arm_rows[task_id], estimand)
        floor_value, floor_var = _pooled_value_and_variance(floor_rows[task_id], estimand)
        profiles.append(TaskProfile(
            task_id=task_id,
            arm_value=arm_value,
            floor_value=floor_value,
            delta=floor_value - arm_value,
            arm_noise_variance=arm_var,
            floor_noise_variance=floor_var,
        ))
    if len(profiles) < 2:
        raise ValueError(f"only {len(profiles)} paired task profiles for {estimand}")
    return profiles


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return (SEED + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def variance_components(profiles: list[TaskProfile]) -> dict:
    deltas = np.asarray([profile.delta for profile in profiles], dtype=float)
    noise = np.asarray([profile.delta_noise_variance for profile in profiles], dtype=float)
    observed_variance = float(deltas.var(ddof=1))
    mean_noise_variance = float(noise.mean())
    shared_variance = max(observed_variance - mean_noise_variance, 0.0)
    if observed_variance > 0 and shared_variance > 0:
        task_effect = (deltas - deltas.mean()) * np.sqrt(shared_variance / observed_variance)
    else:
        task_effect = np.zeros_like(deltas)
    return {
        "mean_delta": float(deltas.mean()),
        "observed_task_delta_variance": observed_variance,
        "mean_arm_noise_variance": mean_noise_variance,
        "estimated_shared_task_variance": shared_variance,
        "shared_fraction_of_observed_variance": (
            float(shared_variance / observed_variance) if observed_variance > 0 else 0.0),
        "task_effect": task_effect,
        "noise_variance_by_task": noise,
    }


def simulate_mean_difference(profiles: list[TaskProfile], components: dict, n_tasks: int,
                             attenuation: float, design: str, n_sim: int,
                             stream: str) -> np.ndarray:
    """Simulate base Delta_floor minus attenuated-target Delta_floor.

    The empirical task component is shared between models in the paired design and sampled
    independently in the unpaired design. Arm noise is independent across models in both.
    """
    rng = np.random.default_rng(_stable_seed(stream, design, n_tasks, attenuation, n_sim))
    n_profiles = len(profiles)
    base_idx = rng.integers(0, n_profiles, size=(n_sim, n_tasks))
    if design == "task_paired_across_models":
        target_idx = base_idx
    elif design == "unpaired_across_models":
        target_idx = rng.integers(0, n_profiles, size=(n_sim, n_tasks))
    else:  # pragma: no cover - guarded by DESIGNS
        raise ValueError(f"unknown design: {design}")

    task_effect = components["task_effect"]
    noise_sd = np.sqrt(components["noise_variance_by_task"])
    mean_delta = components["mean_delta"]

    base = (mean_delta + task_effect[base_idx]
            + rng.normal(size=(n_sim, n_tasks)) * noise_sd[base_idx])
    target = ((1.0 - attenuation) * (mean_delta + task_effect[target_idx])
              + rng.normal(size=(n_sim, n_tasks)) * noise_sd[target_idx])
    return base.mean(axis=1) - target.mean(axis=1)


def _monotone_power(raw: list[float]) -> list[float]:
    """Monte Carlo jitter must not make greater attenuation appear less detectable."""
    return np.maximum.accumulate(np.asarray(raw, dtype=float)).tolist()


def interpolate_f_star(attenuations: tuple[float, ...], power: list[float],
                       target: float = TARGET_POWER) -> float | None:
    if power[-1] < target:
        return None
    for i, value in enumerate(power):
        if value < target:
            continue
        if i == 0:
            return float(attenuations[0])
        x0, x1 = attenuations[i - 1], attenuations[i]
        y0, y1 = power[i - 1], value
        if y1 <= y0:
            return float(x1)
        return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    return None


def legacy_first_cut(profiles: list[TaskProfile]) -> dict:
    """Exact reproduction of ph0's current single-direction assumptions."""
    floor = np.asarray([profile.floor_value for profile in profiles], dtype=float)
    arm = np.asarray([profile.arm_value for profile in profiles], dtype=float)
    n = len(profiles)
    rng = np.random.default_rng(SEED)
    i = rng.integers(0, n, size=(LEGACY_B, n))
    j = rng.integers(0, n, size=(LEGACY_B, n))
    paired = (floor[i] - arm[i]).mean(axis=1)
    arm_floor_unpaired = floor[i].mean(axis=1) - arm[j].mean(axis=1)
    se_paired = float(paired.std(ddof=1))
    se_unpaired = float(arm_floor_unpaired.std(ddof=1))
    se_used = max(se_paired, se_unpaired)
    delta = float((floor - arm).mean())
    f_star = float(Z_SUM * np.sqrt(2.0) * se_used / abs(delta))
    target = 1.2934415841793263
    return {
        "n_tasks": n,
        "B": LEGACY_B,
        "delta": delta,
        "se_paired_arm_floor": se_paired,
        "se_unpaired_arm_floor": se_unpaired,
        "se_used": se_used,
        "f_star": f_star,
        "acceptance_target": target,
        "relative_error": abs(f_star - target) / target,
        "within_10_percent": bool(abs(f_star - target) / target <= 0.10),
        "assumptions": [
            "arm/floor task pairing discarded when choosing the conservative single-battery SE",
            "two model batteries independent and equal variance (additional sqrt(2))",
            "normal critical-value approximation",
        ],
    }


def run_simulation(n_sim: int) -> dict:
    annotated = json.loads((EVAL / "annotated_steered.json").read_text())
    steered = json.loads((EVAL / "steering_results.json").read_text())
    report = json.loads((EVAL / "delta_floor_report.json").read_text())
    cell = report["cells"][f"{BEHAVIOUR}|{ARM}"]
    alpha = float(cell["alpha"])
    floor = str(cell["floor"])

    arm_rows = load_observations(ARM, alpha, annotated, steered)
    floor_rows = load_observations(floor, alpha, annotated, steered)
    profiles_by_estimand = {
        estimand: build_profiles(arm_rows, floor_rows, estimand) for estimand in ESTIMANDS
    }

    fraction_profiles = profiles_by_estimand["sentence_fraction"]
    fraction_delta = float(np.mean([profile.delta for profile in fraction_profiles]))
    if not np.isclose(fraction_delta, float(cell["delta_floor"]), rtol=0, atol=1e-12):
        raise RuntimeError(
            f"authoritative E8 estimand mismatch: {fraction_delta} vs {cell['delta_floor']}")
    legacy = legacy_first_cut(fraction_profiles)
    if not legacy["within_10_percent"]:
        raise RuntimeError("legacy first-cut acceptance check failed")

    estimand_results = {}
    for estimand, profiles in profiles_by_estimand.items():
        components = variance_components(profiles)
        serializable_components = {
            key: value for key, value in components.items()
            if key not in {"task_effect", "noise_variance_by_task"}
        }
        designs = {}
        for design in DESIGNS:
            curve = []
            for n_tasks in TASK_COUNTS:
                null = simulate_mean_difference(
                    profiles, components, n_tasks, 0.0, design, n_sim,
                    stream=f"{estimand}:null")
                critical = float(np.quantile(null, 1.0 - ALPHA))
                raw_power = []
                for attenuation in ATTENUATIONS:
                    alternative = simulate_mean_difference(
                        profiles, components, n_tasks, attenuation, design, n_sim,
                        stream=f"{estimand}:alternative")
                    raw_power.append(float(np.mean(alternative > critical)))
                power = _monotone_power(raw_power)
                curve.append({
                    "n_tasks": n_tasks,
                    "null_critical_value": critical,
                    "attenuation": list(ATTENUATIONS),
                    "power_raw": raw_power,
                    "power_monotone": power,
                    "f_star_interpolated": interpolate_f_star(ATTENUATIONS, power),
                })
            eligible = [row["n_tasks"] for row in curve
                        if row["f_star_interpolated"] is not None
                        and row["f_star_interpolated"] <= 0.5]
            designs[design] = {
                "curve": curve,
                "n_star_for_f_star_le_0_5_on_tested_grid": min(eligible) if eligible else None,
                "n_star_definition": (
                    "smallest tested task count with interpolated f* <= 0.5; "
                    "not an interpolation in n"),
            }
        estimand_results[estimand] = {
            "n_empirical_tasks": len(profiles),
            "empirical_delta": components["mean_delta"],
            "variance_components": serializable_components,
            "designs": designs,
        }

    return {
        "script": "ph2_mde_sim.py",
        "date": "2026-08-02",
        "status": "simulation sizing model; not inferential evidence",
        "seed": SEED,
        "n_sim": n_sim,
        "alpha_one_sided": ALPHA,
        "target_power": TARGET_POWER,
        "task_counts": list(TASK_COUNTS),
        "attenuations": list(ATTENUATIONS),
        "source": {
            "annotated": "results/eval/R1-1.5B__E1/annotated_steered.json",
            "steered": "results/eval/R1-1.5B__E1/steering_results.json",
            "report": "results/eval/R1-1.5B__E1/delta_floor_report.json",
            "behaviour": BEHAVIOUR,
            "arm": ARM,
            "floor": floor,
            "alpha": alpha,
        },
        "legacy_first_cut_reproduction": legacy,
        "estimands": estimand_results,
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# Phase-2 transport MDE simulation",
        "",
        "**Status:** sizing model only; not inferential evidence and not a Phase-2 result.",
        "",
        f"Seed {result['seed']}; {result['n_sim']:,} Monte Carlo trials per cell; "
        f"one-sided alpha={result['alpha_one_sided']:.2f}; target power="
        f"{result['target_power']:.2f}.",
        "",
        "## Calibration acceptance",
        "",
    ]
    legacy = result["legacy_first_cut_reproduction"]
    lines += [
        f"The exact Phase-0 `single_direction` assumptions reproduce **f*="
        f"{legacy['f_star']:.3f}** at n={legacy['n_tasks']} "
        f"(target {legacy['acceptance_target']:.3f}; relative error "
        f"{legacy['relative_error']:.2%}; "
        f"{'PASS' if legacy['within_10_percent'] else 'FAIL'} against the ±10% check).",
        "This calibration is retained as a legacy reference, not used as the paired design's "
        "variance model.",
        "",
        "## MDE curves",
        "",
    ]
    for estimand, erow in result["estimands"].items():
        vc = erow["variance_components"]
        lines += [
            f"### {estimand}",
            "",
            f"Empirical E8 Delta_floor: {erow['empirical_delta']:+.6f} over "
            f"{erow['n_empirical_tasks']} paired tasks. Estimated shared-task fraction of "
            f"observed delta variance: {vc['shared_fraction_of_observed_variance']:.1%}.",
            "",
            "| design | f*(50) | f*(100) | f*(150) | f*(200) | f*(300) | f*(400) | "
            "first tested n with f* <= 0.5 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for design, drow in erow["designs"].items():
            fs = [row["f_star_interpolated"] for row in drow["curve"]]
            display = [">1.0" if value is None else f"{value:.3f}" for value in fs]
            nstar = drow["n_star_for_f_star_le_0_5_on_tested_grid"]
            lines.append("| " + " | ".join([
                design, *display, "not reached" if nstar is None else str(nstar)
            ]) + " |")
        lines.append("")

    lines += [
        "## README — what this simulation does and does not establish",
        "",
        "- The independent unit is the task. The E8 arm and its energy-matched floor are "
        "pooled within task before any resampling. The primary source cell is "
        "`backtracking|single_direction` at sealed alpha=1.0.",
        "- `unpaired_across_models` samples different empirical task profiles for the base "
        "and target batteries. `task_paired_across_models` samples the same task profile for "
        "both, while drawing independent arm noise. This keeps cross-model task pairing "
        "separate from within-model arm/floor pairing.",
        "- The hierarchical variance split is model-based. Observed per-task Delta_floor "
        "variance is decomposed into a binomial sentence-label noise floor and a residual "
        "shared task component. Sentence labels within a chain are not guaranteed independent; "
        "the binomial component may therefore understate arm noise and make the paired design "
        "optimistic.",
        "- The target battery is simulated by multiplying both the mean effect and its "
        "task-specific component by `(1 - attenuation)`. Independent model-specific arm noise "
        "is not attenuated. Other transport failures can have different variance structures.",
        "- Power uses the 95th percentile of a separately simulated null distribution of the "
        "mean base-minus-target effect. `f*` is linearly interpolated only between the sealed "
        "0.1 attenuation grid points after applying a monotone power envelope to Monte Carlo "
        "jitter.",
        "- Counts above the 49 observed E8 tasks are bootstrap extrapolations from the same "
        "empirical task distribution. They do not demonstrate that a newly collected 300-task "
        "battery has the same difficulty mix, effect variance, annotation quality, or support.",
        "- The per-1k estimand counts target-labelled sentences per 1,000 generated tokens. It "
        "is a variance-reduced candidate only if Phase 2 seals it before outcome inspection; "
        "it is not the E8 headline estimand.",
        "- The simulator sizes attenuation of the existing grounded single-direction effect. "
        "It does not rescue the archived manifold first cut, whose floor was misassigned. "
        "Target-model sham/random-orthogonal gates, matched count/energy floors, multiplicity, "
        "and injection-recovery still belong in the sealed Phase-2 protocol.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sim", type=int, default=DEFAULT_N_SIM)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    if args.n_sim < 500:
        raise SystemExit("--n-sim must be >= 500 for a stable sizing curve")

    result = run_simulation(args.n_sim)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    args.out_md.write_text(render_markdown(result))
    print(json.dumps({
        "legacy_f_star": result["legacy_first_cut_reproduction"]["f_star"],
        "legacy_acceptance": result["legacy_first_cut_reproduction"]["within_10_percent"],
        "outputs": [str(args.out_json), str(args.out_md)],
    }, indent=2))


if __name__ == "__main__":
    main()
