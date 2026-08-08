#!/usr/bin/env python3
"""ph0 — transport-MDE first cut (Phase 0 item 5/6 of the unified plan).

Pre-registered in results/prereg/PHASE0_TRANSPORT_FREEZE_2026-08-02.md §1.4 (written first).

Sizes the Phase 2 frame-transport battery: task-cluster bootstrap of the E8 backtracking
delta_floor (single_direction and manifold_k5 arms, alpha*=1.0, n~50 tasks) -> SE ->
conservative detectable attenuation fraction under an unpaired two-battery comparison:
    f* = (z_.95 + z_.80) * sqrt(2) * SE / delta_hat
Estimand recovery is gated: the recomputed delta_floor must match the reported value
within 20% relative, else only an SE-ratio fallback is reported. CPU, deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
EVAL = ROOT / "results/eval/R1-1.5B__E1"
OUT_JSON = ROOT / "results/safety_posttrain/ph0_mde_firstcut.json"
OUT_MD = ROOT / "results/safety_posttrain/PH0_MDE_FIRSTCUT.md"

BEH = "backtracking"
ARMS = ["single_direction", "manifold_k5"]
FLOOR = "energy_matched_random"
B = 10_000
SEED = 20260802
Z = 1.6449 + 0.8416          # alpha=.05 one-sided, power .80
GATE_REL = 0.20


def _bt_counts(row: dict) -> tuple[int, int]:
    """(#backtracking sentences, #sentences) from the annotation payload, defensively."""
    ann = row.get("annotations")
    n_bt = n_all = 0
    if isinstance(ann, list):
        for a in ann:
            if not isinstance(a, dict):
                continue
            n_all += 1
            labs = a.get("labels", a.get("label", a.get("behaviours", [])))
            if isinstance(labs, str):
                labs = [labs]
            if any(BEH in str(l) for l in (labs or [])):
                n_bt += 1
    elif isinstance(ann, dict):
        for _, labs in ann.items():
            n_all += 1
            if isinstance(labs, str):
                labs = [labs]
            if any(BEH in str(l) for l in (labs or [])):
                n_bt += 1
    return n_bt, n_all


def _rates(rows: list[dict], method: str, behaviour: str) -> dict[str, dict]:
    out = {}
    for r in rows:
        if r["method"] != method or r["behaviour"] != behaviour:
            continue
        n_bt, n_all = _bt_counts(r)
        if n_all == 0:
            continue
        out[r["base_task_id"]] = {
            "frac": n_bt / n_all,
            "per1k": 1000.0 * n_bt / max(1, r.get("n_tokens") or 1),
        }
    return out


def main() -> None:
    rows = json.loads((EVAL / "annotated_steered.json").read_text())
    rep = json.loads((EVAL / "delta_floor_report.json").read_text())
    rng = np.random.default_rng(SEED)

    floor = _rates(rows, FLOOR, BEH)
    results = {}
    for arm in ARMS:
        cell = rep["cells"][f"{BEH}|{arm}"]
        reported = float(cell["delta_floor"])
        arm_rates = _rates(rows, arm, BEH)
        common = sorted(set(arm_rates) & set(floor))

        picked, best = None, None
        for variant in ("frac", "per1k"):
            a = np.array([arm_rates[t][variant] for t in common])
            f = np.array([floor[t][variant] for t in common])
            # E8's delta_floor is suppression-oriented at alpha*: floor minus arm
            d = float(f.mean() - a.mean())
            rel = abs(d - reported) / abs(reported)
            if best is None or rel < best["rel_err"]:
                best = {"variant": variant, "delta_recomputed": d, "rel_err": rel}
            if rel <= GATE_REL and picked is None:
                picked = {"variant": variant, "delta_recomputed": d, "rel_err": rel}
        chosen = picked or best
        gate_pass = chosen["rel_err"] <= GATE_REL

        a = np.array([arm_rates[t][chosen["variant"]] for t in common])
        f = np.array([floor[t][chosen["variant"]] for t in common])
        n = len(common)
        idx = rng.integers(0, n, size=(B, n))
        boot_paired = (f[idx] - a[idx]).mean(axis=1)
        idx2 = rng.integers(0, n, size=(B, n))
        boot_unpaired = f[idx].mean(axis=1) - a[idx2].mean(axis=1)
        se = float(max(boot_paired.std(ddof=1), boot_unpaired.std(ddof=1)))  # conservative

        delta_for_f = reported if gate_pass else chosen["delta_recomputed"]
        f_star = float(Z * np.sqrt(2) * se / abs(delta_for_f))
        n_for_half = int(np.ceil(n * (f_star / 0.5) ** 2))
        results[arm] = {
            "reported_delta_floor": reported,
            "estimand_recovery": chosen, "estimand_gate_pass": bool(gate_pass),
            "n_tasks_paired": n,
            "se_paired": float(boot_paired.std(ddof=1)),
            "se_unpaired": float(boot_unpaired.std(ddof=1)),
            "se_used": se,
            "f_star_detectable_attenuation": f_star,
            "n_tasks_for_f_star_0.5": n_for_half,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = {"script": "ph0_mde_firstcut.py", "date": "2026-08-02", "seed": SEED, "B": B,
           "prereg": "results/prereg/PHASE0_TRANSPORT_FREEZE_2026-08-02.md §1.4",
           "formula": "f* = (z.95+z.80)*sqrt(2)*SE/delta; unpaired-across-models, conservative",
           "results": results}
    OUT_JSON.write_text(json.dumps(out, indent=1))

    lines = ["# PH0 — transport-MDE first cut (sizing only, not inference)", "",
             f"Seed {SEED}, B={B}. f* = smallest attenuation of the E8 delta_floor a two-battery",
             "comparison could detect (alpha .05 one-sided, power .80, no pairing benefit assumed).", ""]
    for arm, r in results.items():
        lines += [f"## {BEH} | {arm}",
                  f"- reported delta_floor {r['reported_delta_floor']:+.4f}; estimand recovery: "
                  f"{r['estimand_recovery']['variant']} rel-err {r['estimand_recovery']['rel_err']:.1%}"
                  f" ({'PASS' if r['estimand_gate_pass'] else 'FAIL -> SE-ratio fallback only'})",
                  f"- SE (paired/unpaired bootstrap): {r['se_paired']:.4f} / {r['se_unpaired']:.4f}",
                  f"- **f\\* = {r['f_star_detectable_attenuation']:.2f}** at n={r['n_tasks_paired']}"
                  f" → tasks needed for f\\*=0.5: **{r['n_tasks_for_f_star_0.5']}**", ""]
    lines += ["Consequence rule (§1.4): Phase 2 prereg must size its battery so f* ≤ 0.5, or drop",
              "the 'retained' verdict in favour of 'not disabled'."]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
