#!/usr/bin/env python3
"""PT-B1 executor — safety-versus-control behavioural cell.

Sealed authority: results/prereg/PTB1_SAFETY_CONTROL_BEHAVIOUR_PREREG_2026-08-20.md
(+ Amendment 1). Stage graph:

    identity_reference   local   $0   build the adapter identity-gate reference
                                      bundle + verification-chain file to push
    (pod stages)         pod          ptb1_pod_arm.py per arm: train -> gate ->
                                      generate (launched via runpod_ptb1.sh)
    annotate             local   SPENDS  six PT-B1 shards through the A4/A5R10
                                      pipeline (guard manifest + --authorised)
    analyse              local   $0   sealed endpoints: Holm primary family,
                                      missingness bounds, secondaries, report

Every spend-capable stage refuses to run without --authorised. Nothing here
regenerates or re-annotates the Phase-2 base vanilla baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "ptb1"
PREREG = ROOT / "results/prereg/PTB1_SAFETY_CONTROL_BEHAVIOUR_PREREG_2026-08-20.md"
AMENDMENT_1 = ROOT / "results/prereg/PTB1_AMENDMENT_1_2026-08-20.md"

PTB1_SEED = 20260820
N_VERIF_CHAINS = 150
GATE_LAYERS = (12, 16)
G1_PASS = 0.98
G1_AMENDED = 0.95
G2_MAX = 0.50
MIN_ROW_COVERAGE = 0.95

BASE_ACTS = ROOT / "data/activations/R1-1.5B"
ANNOTATED_CORPUS = ROOT / "data/annotated_R1-1.5B.json"
BEHAVIOURS = ("backtracking", "uncertainty-estimation", "example-testing",
              "adding-knowledge")

#: arm name -> (training data file, seed, stored activation dir)
ARMS = {
    "safety1000-s42": ("data/safety_star1_sft.json", 42,
                       "data/activations/R1-1.5B-lora-safety1000"),
    "safety1000-s43": ("data/safety_star1_sft.json", 43,
                       "data/activations/R1-1.5B-lora-safety1000-s43"),
    "safety1000-s44": ("data/safety_star1_sft.json", 44,
                       "data/activations/R1-1.5B-lora-safety1000-s44"),
    "control1000-s42": ("data/control_generic_sft.json", 42,
                        "data/activations/R1-1.5B-lora-control1000"),
    "control1000-s43": ("data/control_generic_sft.json", 43,
                        "data/activations/R1-1.5B-lora-control1000-s43"),
    "control1000-s44": ("data/control_generic_sft.json", 44,
                        "data/activations/R1-1.5B-lora-control1000-s44"),
}
SAFETY_ARMS = tuple(a for a in ARMS if a.startswith("safety"))
CONTROL_ARMS = tuple(a for a in ARMS if a.startswith("control"))


def opposite_arm(arm: str) -> str:
    """Same-seed opposite-recipe arm (for the G2 polarity check)."""
    if arm.startswith("safety1000"):
        return arm.replace("safety1000", "control1000")
    return arm.replace("control1000", "safety1000")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    tmp.replace(path)


def write_provenance(stage: str, payload: dict, input_paths=()) -> None:
    """House schema (rom-result-provenance-v1), PT-B1-scoped directory."""
    import subprocess
    from datetime import datetime, timezone
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, capture_output=True, text=True).stdout.strip())
    if not head:
        # Pod payloads are pushed without .git; the launcher records the
        # source commit in a file instead.
        marker = ROOT / "PTB1_SOURCE_COMMIT.txt"
        head = (marker.read_text().strip() if marker.exists()
                else "unavailable (no git checkout, no source-commit marker)")
        dirty = None
    doc = {
        "schema_version": "rom-result-provenance-v1",
        "stage": f"ptb1-{stage}",
        "prereg": {"path": str(PREREG.relative_to(ROOT)), "sha256": sha256(PREREG)},
        "amendments": [{"path": str(AMENDMENT_1.relative_to(ROOT)),
                        "sha256": sha256(AMENDMENT_1)}],
        "git_commit": head, "git_dirty": dirty,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": PTB1_SEED,
        "inputs_sha256": {str(Path(p).relative_to(ROOT) if Path(p).is_absolute()
                              else p): sha256(ROOT / p) for p in input_paths},
        **payload,
    }
    _atomic_json(doc, OUT / "provenance" / f"{stage}.json")


# ── row-index alignment helpers (shared with the pod-side gate) ──────────────

def load_row_keys(act_dir: Path) -> dict:
    """behaviour -> [(chain_id, annotation_index), ...] in stored row order."""
    ri = json.loads((act_dir / "row_index.json").read_text())
    return {b: [(r["chain_id"], r["annotation_index"]) for r in rows]
            for b, rows in ri["rows"].items() if b in BEHAVIOURS}


def key_positions(keys_by_behaviour: dict) -> dict:
    """(behaviour, chain_id, annotation_index) -> stored row position."""
    return {(b, c, a): i
            for b, keys in keys_by_behaviour.items()
            for i, (c, a) in enumerate(keys)}


def gather_rows(act_dir: Path, layer: int, wanted: list) -> "object":
    """Stack rows for ``wanted`` [(behaviour, chain, ann_idx)] preserving order."""
    import numpy as np
    pos = key_positions(load_row_keys(act_dir))
    mats = {b: np.load(act_dir / f"{b}_layer{layer}.npy", mmap_mode="r")
            for b in BEHAVIOURS}
    out = np.empty((len(wanted), next(iter(mats.values())).shape[1]),
                   dtype=np.float32)
    for i, key in enumerate(wanted):
        out[i] = mats[key[0]][pos[key]]
    return out


def gate_verdict(cos_same: dict, cos_opp: dict, coverage: float) -> dict:
    """Sealed G1/G2 bands. cos_same/cos_opp: layer -> cosine."""
    if coverage < MIN_ROW_COVERAGE:
        return {"verdict": "STOP", "reason": f"row coverage {coverage:.3f} < "
                                             f"{MIN_ROW_COVERAGE}"}
    if any(c > G2_MAX for c in cos_opp.values()):
        return {"verdict": "STOP", "reason": f"G2 polarity failed: {cos_opp}"}
    if all(c >= G1_PASS for c in cos_same.values()):
        return {"verdict": "PASS", "reason": None}
    if all(c >= G1_AMENDED for c in cos_same.values()):
        return {"verdict": "AMENDED", "reason": "recipe-replicate "
                                                f"(cos in [0.95,0.98)): {cos_same}"}
    return {"verdict": "STOP", "reason": f"G1 identity failed: {cos_same}"}


# ── stage: identity_reference (local, $0) ────────────────────────────────────

def stage_identity_reference(authorised: bool) -> None:
    """Build the reference bundle the pod-side gate compares against:
    a seeded 150-chain verification subset, the row-matched base + per-arm
    stored activation rows (float16 NPZ), and the filtered annotated-chain
    file the pod extracts with. Deterministic; binds every consumed hash."""
    import numpy as np

    gate_dir = OUT / "identity_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)

    # Chain universe: present in the base index AND every stored arm index.
    base_keys = load_row_keys(BASE_ACTS)
    arm_keys = {a: load_row_keys(ROOT / d) for a, (_f, _s, d) in ARMS.items()}
    chains_of = lambda kb: {c for keys in kb.values() for c, _ in keys}
    universe = chains_of(base_keys)
    for kb in arm_keys.values():
        universe &= chains_of(kb)
    sample = sorted(random.Random(PTB1_SEED).sample(sorted(universe),
                                                    N_VERIF_CHAINS))
    sample_set = set(sample)

    # Verification rows: keys in the sample present in base AND all arms.
    base_pos = key_positions(base_keys)
    wanted = []
    for b in BEHAVIOURS:
        keys = [(b, c, a) for c, a in base_keys.get(b, []) if c in sample_set]
        for kb in arm_keys.values():
            present = set(kb.get(b, []))
            keys = [k for k in keys if (k[1], k[2]) in present]
        wanted.extend(sorted(keys))
    if not wanted:
        raise SystemExit("no verification rows survived the intersection")

    _atomic_json(wanted, gate_dir / "reference_keys.json")
    bundle = {}
    for layer in GATE_LAYERS:
        bundle[f"base_l{layer}"] = gather_rows(BASE_ACTS, layer,
                                               wanted).astype(np.float16)
        for arm, (_f, _s, d) in ARMS.items():
            bundle[f"arm_{arm}_l{layer}"] = gather_rows(
                ROOT / d, layer, wanted).astype(np.float16)
    np.savez_compressed(gate_dir / "reference.npz", **bundle)

    # Filtered annotated chains for the pod-side extraction.
    corpus = json.loads(ANNOTATED_CORPUS.read_text())
    verification_chains = [r for r in corpus if r["task_id"] in sample_set]
    _atomic_json(verification_chains, gate_dir / "verification_chains.json")

    # Sanity echo: stored same-seed vs cross-recipe direction separation.
    sanity = {}
    for layer in GATE_LAYERS:
        base_m = bundle[f"base_l{layer}"].astype(np.float32)
        dirs = {arm: (bundle[f"arm_{arm}_l{layer}"].astype(np.float32)
                      - base_m).mean(axis=0) for arm in ARMS}
        cos = lambda u, v: float(np.dot(u, v)
                                 / (np.linalg.norm(u) * np.linalg.norm(v)))
        sanity[str(layer)] = {
            "within_safety_min": min(cos(dirs[a], dirs[b])
                                     for a in SAFETY_ARMS for b in SAFETY_ARMS
                                     if a < b),
            "within_control_min": min(cos(dirs[a], dirs[b])
                                      for a in CONTROL_ARMS for b in CONTROL_ARMS
                                      if a < b),
            "cross_recipe_max": max(cos(dirs[a], dirs[opposite_arm(a)])
                                    for a in SAFETY_ARMS),
        }

    inputs = [str(ANNOTATED_CORPUS.relative_to(ROOT))]
    for d in [BASE_ACTS] + [ROOT / d for _a, (_f, _s, d) in ARMS.items()]:
        inputs.append(str((d / "row_index.json").relative_to(ROOT)))
        for b in BEHAVIOURS:
            for layer in GATE_LAYERS:
                inputs.append(str((d / f"{b}_layer{layer}.npy").relative_to(ROOT)))
    meta = {
        "n_verification_chains": len(sample),
        "chains": sample,
        "n_rows": len(wanted),
        "rows_per_behaviour": {b: sum(1 for k in wanted if k[0] == b)
                               for b in BEHAVIOURS},
        "gate": {"layers": list(GATE_LAYERS), "g1_pass": G1_PASS,
                 "g1_amended": G1_AMENDED, "g2_max": G2_MAX,
                 "min_row_coverage": MIN_ROW_COVERAGE},
        "stored_direction_sanity": sanity,
        "outputs": {"reference.npz": sha256(gate_dir / "reference.npz"),
                    "reference_keys.json":
                        sha256(gate_dir / "reference_keys.json"),
                    "verification_chains.json":
                        sha256(gate_dir / "verification_chains.json")},
    }
    _atomic_json(meta, gate_dir / "reference_meta.json")
    write_provenance("identity_reference", {"authorised": authorised,
                                            "counts": meta["rows_per_behaviour"],
                                            "sanity": sanity},
                     input_paths=inputs)
    print(json.dumps({"chains": len(sample), "rows": len(wanted),
                      "sanity": sanity}, indent=1))


# ── stage: annotate (local, SPENDS) ──────────────────────────────────────────

ANNOTATION_MAX_TOKENS = 2400   # mirrors ph2_executor.py


def stage_annotate(authorised: bool) -> None:
    """Six PT-B1 arm shards through the exact Phase-2 pipeline (A3 Sonnet-only,
    A4 window, A5R10 coverage + bounded attempts). Guard manifest is bound via
    PTB1_ANNOTATION_GUARD_MANIFEST(+_SHA256); fail-closed on every ceiling."""
    if not authorised:
        raise SystemExit("annotate SPENDS — refuse without --authorised")
    if not (os.environ.get("CLAUDE_PROXY_URL") and os.environ.get("CLAUDE_PROXY_KEY")):
        raise SystemExit("CLAUDE_PROXY_URL/KEY not set — annotation is proxy-gated.")
    guard_manifest_raw = os.environ.get("PTB1_ANNOTATION_GUARD_MANIFEST")
    guard_manifest_sha = os.environ.get("PTB1_ANNOTATION_GUARD_MANIFEST_SHA256")
    if not (guard_manifest_raw and guard_manifest_sha):
        raise SystemExit("PTB1_ANNOTATION_GUARD_MANIFEST(+_SHA256) required; "
                         "refusing all annotation API calls.")
    from src.annotation import (ANNOTATION_MODEL, annotate_chains,
                                annotation_initial_request_count,
                                max_prompt_chars)
    from src.annotation_budget import (AnnotationAttemptGuard,
                                       AnnotationAttemptLimitError)
    from src.annotation_coverage import (COVERAGE_RULE_VERSION,
                                         region_source_text)
    from src.ph2_stages import (ANNOTATION_DEDUP_KEYS,
                                ANNOTATION_INCLUDE_POST_THINK,
                                ANNOTATION_WINDOW_TOKENS, annotation_window,
                                proxy_chunk_budget_ok)
    budget = proxy_chunk_budget_ok()
    if not budget["ok"]:
        raise SystemExit(f"29-s chunk budget violated: {budget}")

    ann_dir = OUT / "annotation"
    ann_dir.mkdir(parents=True, exist_ok=True)
    shards = [(OUT / "battery" / f"{arm}.json", ann_dir / f"{arm}.json")
              for arm in ARMS]
    guard_manifest = Path(guard_manifest_raw)
    if not guard_manifest.is_absolute():
        guard_manifest = ROOT / guard_manifest
    bound_paths = ("ptb1_executor.py", "src/annotation.py",
                   "src/annotation_budget.py", "src/annotation_coverage.py",
                   "src/annotation_quarantine.py", "src/ph2_stages.py",
                   "src/delta_floor.py",
                   "results/prereg/PTB1_SAFETY_CONTROL_BEHAVIOUR_PREREG_2026-08-20.md",
                   "results/prereg/PTB1_AMENDMENT_1_2026-08-20.md",
                   "results/prereg/phase2_task_manifest.json")
    try:
        attempt_guard = AnnotationAttemptGuard.from_manifest(
            guard_manifest, guard_manifest_sha,
            ann_dir / "api_attempt_journal.jsonl", root=ROOT,
            expected_model=ANNOTATION_MODEL,
            expected_annotation_window_tokens=ANNOTATION_WINDOW_TOKENS,
            expected_bound_paths=bound_paths,
            expected_max_output_tokens=ANNOTATION_MAX_TOKENS,
            expected_max_prompt_chars=max_prompt_chars(),
            expected_coverage_rule_version=COVERAGE_RULE_VERSION)
    except AnnotationAttemptLimitError as exc:
        raise SystemExit(f"annotation attempt guard refused execution: {exc}")

    prepared, observed_paths, observed_initial = [], [], 0
    for src_path, dst_path in shards:
        if not src_path.exists():
            raise SystemExit(f"missing battery shard {src_path} — run the pod "
                             f"stages first (no partial annotation campaigns)")
        observed_paths.append(str(src_path.relative_to(ROOT)))
        rows, n_windowed = [], 0
        for r in json.loads(src_path.read_text()):
            win, est, truncated = annotation_window(r["chain"])
            n_windowed += int(truncated)
            rows.append({**r, "chain": win, "chain_full": r["chain"],
                         "annotated_tokens": est,
                         "annotation_window_tokens": ANNOTATION_WINDOW_TOKENS,
                         "annotation_window_truncated": truncated})
            observed_initial += annotation_initial_request_count(
                region_source_text(win,
                                   include_post_think=ANNOTATION_INCLUDE_POST_THINK))
        prepared.append((src_path, dst_path, rows, n_windowed))
    if tuple(sorted(observed_paths)) != attempt_guard.policy.source_paths:
        raise SystemExit("annotation attempt guard source-set mismatch; "
                         "refusing all API calls")
    try:
        attempt_guard.assert_planned_initial_requests(observed_initial)
    except AnnotationAttemptLimitError as exc:
        raise SystemExit(f"annotation attempt guard refused execution: {exc}")

    status: dict = {}
    for src_path, dst_path, rows, n_windowed in prepared:
        annotated = annotate_chains(
            rows, save_path=dst_path, dedup_keys=ANNOTATION_DEDUP_KEYS,
            model=ANNOTATION_MODEL, max_tokens=ANNOTATION_MAX_TOKENS,
            shrink_on_retry=True, max_retries=3, attempt_guard=attempt_guard,
            coverage_validation=True,
            include_post_think=ANNOTATION_INCLUDE_POST_THINK)
        status[dst_path.name] = {
            "n_rows": len(annotated), "n_windowed": n_windowed,
            "n_unresolved": sum(1 for r in annotated
                                if r.get("annotation_complete") is False
                                or not r.get("annotations")),
            "n_coverage_incomplete": sum(
                1 for r in annotated
                if r.get("annotations")
                and not r.get("annotation_coverage_complete", False)),
            "coverage_rule_version": COVERAGE_RULE_VERSION,
        }
    _atomic_json(status, ann_dir / "annotation_status.json")
    write_provenance("annotate", {
        "authorised": authorised, "counts": status,
        "spend": attempt_guard.summary(),
        "annotator": {"provider": "lab AWS proxy", "model": ANNOTATION_MODEL,
                      "caveat": "builder-annotator (Sonnet) — carried into "
                                "every PT-B1 verdict sentence"}},
        input_paths=[str(p.relative_to(ROOT)) for p, _ in shards])
    print(json.dumps(status, indent=1))


# ── stage: analyse (local, $0) ───────────────────────────────────────────────

def class_cell(cells_by_arm: dict, arms: tuple) -> dict:
    """Merge per-arm task cells into one arm-class cell. With exactly one row
    per (arm, task), pooling rows across seeds equals the sealed
    mean-over-resolved-seed-rows estimand."""
    merged: dict = {}
    for arm in arms:
        for task, entry in cells_by_arm[arm].items():
            slot = merged.setdefault(task, {"values": [], "n_missing": 0})
            slot["values"].extend(entry["values"])
            slot["n_missing"] += entry["n_missing"]
    return merged


def stage_analyse(authorised: bool) -> None:
    import numpy as np
    from src.delta_floor import empirical_two_sided_p, paired_bootstrap_mean
    from src.ph2_stages import chain_stats
    from src.steered_inference import default_max_new_tokens
    from src.steering_analysis import holm_bonferroni
    from ph2_missingness_bounds import bound_contrast, vanilla_cell

    ann_dir = OUT / "annotation"
    bat_dir = OUT / "battery"
    ann = {arm: json.loads((ann_dir / f"{arm}.json").read_text())
           for arm in ARMS}
    gen = {arm: json.loads((bat_dir / f"{arm}.json").read_text())
           for arm in ARMS}
    cap = default_max_new_tokens()

    analysis: dict = {
        "prereg": str(PREREG.relative_to(ROOT)),
        "estimand": "paired per-task safety-minus-control difference in "
                    "windowed six-label behaviour fraction; task = cluster; "
                    "arm-class value = mean over resolved seed rows",
        "caveats": ["builder-annotator (Sonnet)",
                    "control-corpus asymmetry (control adapter trained on own "
                    "corpus chains; tasks A1-disjoint; format transfer "
                    "direction unknown)",
                    "recipe-level, three seeds per class, one 1.5B "
                    "response-distilled model, this battery only",
                    "no refusal/compliance/benchmark endpoint; boxed "
                    "correctness not computable (no gold answers)"],
    }

    # Primary family: four behaviours, Holm.
    primary: dict = {}
    pvals: dict = {}
    for beh in BEHAVIOURS:
        cells = {arm: vanilla_cell(ann[arm], beh) for arm in ARMS}
        s_cell = class_cell(cells, SAFETY_ARMS)
        c_cell = class_cell(cells, CONTROL_ARMS)
        res = bound_contrast(s_cell, c_cell)
        diffs = []
        for t in sorted(set(s_cell) & set(c_cell)):
            sv = [v for v in s_cell[t]["values"]]
            cv = [v for v in c_cell[t]["values"]]
            if sv and cv:
                diffs.append(float(np.mean(sv)) - float(np.mean(cv)))
        bres, boot = paired_bootstrap_mean(diffs, n_resamples=10_000,
                                           seed=PTB1_SEED,
                                           return_distribution=True)
        p = empirical_two_sided_p(boot, bres.estimate)
        primary[beh] = {
            "complete_case": {"diff_mean": bres.estimate,
                              "n_pairs": len(diffs),
                              "ci_low": bres.ci_low, "ci_high": bres.ci_high,
                              "raw_p": p},
            "missingness": {"manski": res["manski"],
                            "tipping_point": res["tipping_point"],
                            "behaviour_dense_scenarios":
                                res["behaviour_dense_scenarios"],
                            "label": res["robustness"]["label"]},
            "arm_class_stats": {"safety": res["arm_minuend"],
                                "control": res["arm_subtrahend"]},
        }
        pvals[beh] = p
    holm = holm_bonferroni({b: p for b, p in pvals.items() if p is not None})
    for beh in BEHAVIOURS:
        cell = primary[beh]
        adj = holm.get(beh)
        cell["holm_adjusted_p"] = adj
        fragile = cell["missingness"]["label"] == "missingness-fragile"
        sig = adj is not None and adj < 0.05
        cell["verdict"] = (
            "resolved nonzero (Holm) — cite complete-case + unresolved counts "
            "+ robustness label" if (sig and not fragile)
            else "Holm-significant but missingness-fragile — report as NOT "
                 "robust" if (sig and fragile)
            else "point estimate only")
    analysis["primary"] = primary

    # Secondary: bt_per_1k (windowed numerator and denominator).
    def bt_rate_cell(rows):
        cell: dict = {}
        for r in rows:
            if r["method"] != "vanilla":
                continue
            slot = cell.setdefault(r["task_id"], {"values": [], "n_missing": 0})
            anns = r.get("annotations")
            complete = (r.get("annotation_complete") is not False and anns
                        and r.get("annotation_coverage_complete", False))
            denom = r.get("annotated_region_tokens") or r.get("annotated_tokens")
            if complete and denom:
                n_bt = sum(1 for a in anns if a.get("label") == "backtracking")
                slot["values"].append(1000.0 * n_bt / denom)
            else:
                slot["n_missing"] += 1
        return cell
    s_rate = class_cell({a: bt_rate_cell(ann[a]) for a in SAFETY_ARMS},
                        SAFETY_ARMS)
    c_rate = class_cell({a: bt_rate_cell(ann[a]) for a in CONTROL_ARMS},
                        CONTROL_ARMS)
    rate_diffs = [float(np.mean(s_rate[t]["values"]))
                  - float(np.mean(c_rate[t]["values"]))
                  for t in sorted(set(s_rate) & set(c_rate))
                  if s_rate[t]["values"] and c_rate[t]["values"]]
    if len(rate_diffs) >= 2:
        rt = paired_bootstrap_mean(rate_diffs, n_resamples=10_000,
                                   seed=PTB1_SEED)
        analysis["secondary_bt_per_1k"] = {
            "diff_mean": rt.estimate, "ci_low": rt.ci_low,
            "ci_high": rt.ci_high, "n_pairs": len(rate_diffs),
            "descriptive_only": True}

    # Secondary: per-seed 3x3 sign grid (descriptive).
    grid: dict = {}
    for beh in BEHAVIOURS:
        cells = {arm: vanilla_cell(ann[arm], beh) for arm in ARMS}
        g = {}
        for sa in SAFETY_ARMS:
            for ca in CONTROL_ARMS:
                diffs = [float(np.mean(cells[sa][t]["values"]))
                         - float(np.mean(cells[ca][t]["values"]))
                         for t in sorted(set(cells[sa]) & set(cells[ca]))
                         if cells[sa][t]["values"] and cells[ca][t]["values"]]
                g[f"{sa}|{ca}"] = float(np.mean(diffs)) if diffs else None
        vals = [v for v in g.values() if v is not None]
        grid[beh] = {"pairs": g,
                     "n_negative": sum(1 for v in vals if v < 0),
                     "n_positive": sum(1 for v in vals if v > 0)}
    analysis["secondary_per_seed_grid"] = grid

    # Full-chain endpoints (annotation-free, paired by task).
    def chain_cell(rows):
        cell: dict = {}
        for r in rows:
            if r["method"] != "vanilla":
                continue
            st = chain_stats(r["chain"], r["n_tokens"], cap, None)
            cell.setdefault(r["task_id"], []).append({
                "length": float(r["n_tokens"]),
                "looped": float(st["looped"]),
                "truncated": float(st["truncated"]),
                "boxed_emitted": float("\\boxed" in r["chain"])})
        return cell
    s_chain = {}
    for a in SAFETY_ARMS:
        for t, v in chain_cell(gen[a]).items():
            s_chain.setdefault(t, []).extend(v)
    c_chain = {}
    for a in CONTROL_ARMS:
        for t, v in chain_cell(gen[a]).items():
            c_chain.setdefault(t, []).extend(v)
    full_chain: dict = {}
    for ep in ("length", "looped", "truncated", "boxed_emitted"):
        diffs = [float(np.mean([x[ep] for x in s_chain[t]]))
                 - float(np.mean([x[ep] for x in c_chain[t]]))
                 for t in sorted(set(s_chain) & set(c_chain))]
        bt = paired_bootstrap_mean(diffs, n_resamples=10_000, seed=PTB1_SEED)
        full_chain[ep] = {"diff_mean": bt.estimate, "ci_low": bt.ci_low,
                          "ci_high": bt.ci_high, "n_pairs": len(diffs)}
    analysis["full_chain"] = full_chain

    # Sealed damage-context rule.
    trip = (abs(full_chain["length"]["diff_mean"]) > 500
            or abs(full_chain["looped"]["diff_mean"]) > 0.10
            or abs(full_chain["truncated"]["diff_mean"]) > 0.10)
    analysis["damage_context_rule"] = {
        "tripped": bool(trip),
        "rule": "every prevalence citation must carry the length/loop shift "
                "adjacent and co-report bt_per_1k" if trip
                else "not tripped"}

    # Base reference levels (reused Phase-2 artefacts; never regenerated).
    base_ann_path = ROOT / "results/ph2/annotation/base.json"
    if base_ann_path.exists():
        base_ann = [r for r in json.loads(base_ann_path.read_text())
                    if r["method"] == "vanilla"]
        levels = {}
        for beh in BEHAVIOURS:
            cell = vanilla_cell(base_ann, beh)
            vals = [float(np.mean(e["values"])) for e in cell.values()
                    if e["values"]]
            levels[beh] = {"mean": float(np.mean(vals)) if vals else None,
                           "n_tasks": len(vals)}
        analysis["base_reference_levels"] = levels

    _atomic_json(analysis, OUT / "analysis" / "ptb1_analysis.json")

    lines = ["# PT-B1 — safety-versus-control behavioural cell", "",
             f"Sealed authority: `{PREREG.relative_to(ROOT)}` (+ Amendment 1).",
             "Caveats (travel with every number): " + "; ".join(analysis["caveats"]) + ".",
             "", "## Primary family (safety − control, Holm over 4)", "",
             "| Behaviour | Δ complete-case (n) | 95% BCa | raw p | Holm | "
             "Missingness | Verdict |", "|---|---|---|---|---|---|---|"]
    for beh in BEHAVIOURS:
        c = primary[beh]["complete_case"]
        h = primary[beh].get("holm_adjusted_p")
        lines.append(
            f"| {beh} | {c['diff_mean']:+.4f} ({c['n_pairs']}) | "
            f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {c['raw_p']:.4f} | "
            f"{'—' if h is None else f'{h:.4f}'} | "
            f"{primary[beh]['missingness']['label']} | "
            f"{primary[beh]['verdict']} |")
    lines += ["", "## Full-chain endpoints (annotation-free)", "",
              "| Endpoint | Δ (safety − control) | 95% BCa | n |",
              "|---|---|---|---|"]
    for ep, v in full_chain.items():
        lines.append(f"| {ep} | {v['diff_mean']:+.2f} | "
                     f"[{v['ci_low']:+.2f}, {v['ci_high']:+.2f}] | "
                     f"{v['n_pairs']} |")
    lines += ["", f"Damage-context rule: {analysis['damage_context_rule']['rule']}.",
              "", "Per-seed sign grids, bt_per_1k, base reference levels, and "
              "full missingness bounds: `ptb1_analysis.json`.", ""]
    (OUT / "analysis" / "PTB1_REPORT.md").write_text("\n".join(lines))

    write_provenance("analyse", {
        "authorised": authorised,
        "bootstrap": {"B": 10_000, "seed": PTB1_SEED, "kind": "paired BCa"},
        "multiplicity": "Holm over the four behaviours",
        "missingness_contract":
            "results/prereg/PH2_MISSINGNESS_BOUNDING_SPEC_2026-08-19.md"},
        input_paths=[f"results/ptb1/annotation/{a}.json" for a in ARMS]
        + [f"results/ptb1/battery/{a}.json" for a in ARMS])
    print(json.dumps({b: primary[b]["verdict"] for b in BEHAVIOURS}, indent=1))


STAGES = {
    "identity_reference": stage_identity_reference,
    "annotate": stage_annotate,
    "analyse": stage_analyse,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--authorised", action="store_true",
                    help="required for spend-capable stages")
    args = ap.parse_args()
    STAGES[args.stage](args.authorised)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
