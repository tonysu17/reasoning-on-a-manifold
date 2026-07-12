#!/usr/bin/env python
"""E10.1 P2 — swap vs projective ablation at matched on-target displacement.

The sealed collapse-account prediction (E10_DAS_PREREG.md P2): a data-bounded
coordinate CLAMP (the generation-time analogue of the interchange swap: set the
direction's coordinate to a class-mean value the model actually exhibits, no
dose knob) produces LESS repetition-collapse than the projective operator
(eq steer-apply, alpha-scaled) at matched realized coordinate displacement.

Behaviour: backtracking (the grounded causal frame from E10.1 + the known
bidirectional collapse handle from E9.1b: ablate 24% / vanilla 36% / amplify
58-64% collapse). Two directions, each applied at its OWN site:
  - DAS-learned (E10.1, trained at hs[17])          -> SteeredModel(layer=16)
  - E1 diff-of-means bt17 (built at block-17 output) -> SteeredModel(layer=17)
Arms per direction (greedy, cap 8192, 50 held-out tasks):
  add alpha in {0.5, 1.0} | clamp_on (c = class-mean source coordinate) |
  sub alpha = 1.0         | clamp_off (c = class-mean base coordinate)
plus one shared vanilla (alpha=0 through the identical generation path).

Endpoints (annotation-free): collapse rate (4-gram repetition > 0.8), mean
tokens, boxed rate; matching variable = mean realized |Delta(r^T h)| recorded
by the hook. Primary comparison: clamp_on collapse vs the add-curve collapse
interpolated to the clamp's realized displacement (paired bootstrap over
tasks); secondary: exact McNemar clamp_on vs add(1.0).

Stages: clamps -> generate -> analyse -> all. GPU ~4-5 h (550 chains).
Output: results/eval/R1-1.5B__E10_P2/
"""

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))
from src.steered_inference import SteeredModel          # noqa: E402
from src.task_gen import stratified_eval_split          # noqa: E402
from src.evaluation import repetition_rate              # noqa: E402

OUT = ROOT / "results" / "eval" / "R1-1.5B__E10_P2"
DAS_DIR = ROOT / "results" / "das" / "R1-1.5B" / "main"
DM_VEC = ROOT / "results" / "steering_vectors" / "R1-1.5B__E1_pooled" / "backtracking_single.npy"

# direction name -> (vector path, hs index for clamp calibration, SteeredModel layer)
DIRECTIONS = {
    "das": (DAS_DIR / "dir_learned_L17.npy", 17, 16),
    "dm": (DM_VEC, 18, 17),
}
COLLAPSE_THRESH = 0.8


def log(msg):
    print(f"[e10.p2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_das_module():
    spec = importlib.util.spec_from_file_location("das", ROOT / "20_das_backtracking.py")
    das = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(das)
    return das


def stage_clamps(cfg):
    """Class-mean clamp targets c_on / c_off per direction at its own site."""
    das = load_das_module()
    device, dtype = das.pick_device()
    tok, model = das.load_model(device, dtype)
    pairs = json.load(open(DAS_DIR / "pairs.json"))[:cfg.n_pairs_clamp]
    targets = {}
    for name, (vec_path, hs_idx, _) in DIRECTIONS.items():
        v = np.load(vec_path)
        d = torch.tensor(v / (np.linalg.norm(v) + 1e-8), dtype=torch.float32, device=device)
        batches = das.make_batches(pairs, tok, model, hs_idx, device, cfg.bs, 1)
        on, off = [], []
        for b in batches:
            on += [float(x) for x in (b["hs"] @ d)]
            off += [float(x) for x in (b["hb"] @ d)]
        targets[name] = {"c_on": float(np.mean(on)), "c_off": float(np.mean(off)),
                         "sd_on": float(np.std(on)), "sd_off": float(np.std(off)),
                         "hs_index": hs_idx, "n": len(on)}
        log(f"{name}: c_on={targets[name]['c_on']:+.3f} (sd {targets[name]['sd_on']:.3f}) "
            f"c_off={targets[name]['c_off']:+.3f} (sd {targets[name]['sd_off']:.3f})")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(targets, open(OUT / "clamp_targets.json", "w"), indent=2)
    del model
    return targets


def build_arms(targets):
    """arm name -> (direction, mode, alpha, clamp_value)."""
    arms = {"vanilla": (None, "subtract", 0.0, 0.0)}
    for name in DIRECTIONS:
        t = targets[name]
        arms[f"{name}_add_a0.5"] = (name, "add", 0.5, 0.0)
        arms[f"{name}_add_a1.0"] = (name, "add", 1.0, 0.0)
        arms[f"{name}_clamp_on"] = (name, "clamp", 0.0, t["c_on"])
        arms[f"{name}_sub_a1.0"] = (name, "subtract", 1.0, 0.0)
        arms[f"{name}_clamp_off"] = (name, "clamp", 0.0, t["c_off"])
    return arms


def stage_generate(cfg):
    targets = json.load(open(OUT / "clamp_targets.json"))
    das = load_das_module()
    device, dtype = das.pick_device()
    tok, model = das.load_model(device, dtype)

    tasks_all = json.load(open(ROOT / "data" / "tasks_final.json"))
    eval_tasks, rule = stratified_eval_split(tasks_all, n_test=cfg.n_tasks)
    eval_tasks = eval_tasks[:cfg.n_tasks]
    instructions = [t["prompt"] for t in eval_tasks]     # tasks_final.json schema: id/prompt
    task_ids = [t["id"] for t in eval_tasks]
    log(f"{len(eval_tasks)} held-out tasks ({rule})")

    vecs = {}
    for name, (vec_path, _, _) in DIRECTIONS.items():
        v = np.load(vec_path)
        vecs[name] = v / (np.linalg.norm(v) + 1e-8)

    arms = build_arms(targets)
    results_path = OUT / "steering_results.json"
    records = json.load(open(results_path)) if results_path.exists() else []
    done = {(r["method"], r["task_id"]) for r in records}

    for arm_name, (dname, mode, alpha, cval) in arms.items():
        todo_idx = [i for i, tid in enumerate(task_ids) if (arm_name, tid) not in done]
        if not todo_idx:
            log(f"{arm_name}: already complete")
            continue
        vec = vecs[dname] if dname else vecs["dm"]  # vanilla: any vector, alpha=0
        layer = DIRECTIONS[dname][2] if dname else DIRECTIONS["dm"][2]
        sm = SteeredModel(model, tok, vec, layer, alpha=alpha, mode=mode,
                          clamp_value=cval)
        log(f"=== {arm_name} (layer={layer} mode={mode} alpha={alpha} "
            f"clamp={cval:+.3f}) — {len(todo_idx)} tasks ===")
        for lo in range(0, len(todo_idx), cfg.batch):
            chunk = todo_idx[lo:lo + cfg.batch]
            outs = sm.generate_batch([instructions[i] for i in chunk],
                                     max_new_tokens=cfg.max_new_tokens)
            disp = sm.mean_abs_displacement()
            for i, rec in zip(chunk, outs):
                rec.update({"task_id": task_ids[i], "method": arm_name,
                            "direction": dname or "none", "clamp_value": cval,
                            "mean_abs_displacement": disp})
                records.append(rec)
            json.dump(records, open(results_path, "w"))
            log(f"  {arm_name}: {min(lo + cfg.batch, len(todo_idx))}/{len(todo_idx)} "
                f"(disp {disp if disp is None else round(disp, 3)})")
    log(f"generation complete: {len(records)} records -> {results_path}")


def stage_analyse(cfg):
    records = json.load(open(OUT / "steering_results.json"))
    by_arm = {}
    for r in records:
        by_arm.setdefault(r["method"], {})[r["task_id"]] = r

    def stats(rs):
        reps = {tid: repetition_rate(r.get("chain", "")) for tid, r in rs.items()}
        coll = {tid: rep > COLLAPSE_THRESH for tid, rep in reps.items()}
        return {
            "n": len(rs),
            "collapse_rate": float(np.mean(list(coll.values()))),
            "mean_rep": float(np.mean(list(reps.values()))),
            "mean_tokens": float(np.mean([r.get("n_tokens", 0) for r in rs.values()])),
            "boxed_rate": float(np.mean([("\\boxed" in r.get("chain", "")) for r in rs.values()])),
            "mean_displacement": float(np.mean([r.get("mean_abs_displacement") or 0.0
                                                for r in rs.values()])),
            "_collapse_by_task": coll,
        }

    table = {arm: stats(rs) for arm, rs in by_arm.items()}

    analysis = {"experiment": "E10.1 P2 swap vs ablation", "date": time.strftime("%Y-%m-%d"),
                "collapse_thresh": COLLAPSE_THRESH,
                "arms": {a: {k: v for k, v in s.items() if not k.startswith("_")}
                         for a, s in table.items()}, "primary": {}}

    rng = np.random.default_rng(0)
    for dname in DIRECTIONS:
        van, a05, a10, cl = (table.get("vanilla"), table.get(f"{dname}_add_a0.5"),
                             table.get(f"{dname}_add_a1.0"), table.get(f"{dname}_clamp_on"))
        if not all([van, a05, a10, cl]):
            continue
        # interpolate the add-curve collapse at the clamp's realized displacement
        xs = [0.0, a05["mean_displacement"], a10["mean_displacement"]]
        ys = [van["collapse_rate"], a05["collapse_rate"], a10["collapse_rate"]]
        xc = cl["mean_displacement"]
        pred = float(np.interp(xc, xs, ys))
        obs = cl["collapse_rate"]
        # paired bootstrap over tasks for obs - pred
        tids = sorted(set(van["_collapse_by_task"]) & set(a05["_collapse_by_task"])
                      & set(a10["_collapse_by_task"]) & set(cl["_collapse_by_task"]))
        diffs = []
        for _ in range(10000):
            samp = rng.choice(tids, size=len(tids), replace=True)
            v = np.mean([van["_collapse_by_task"][t] for t in samp])
            y05 = np.mean([a05["_collapse_by_task"][t] for t in samp])
            y10 = np.mean([a10["_collapse_by_task"][t] for t in samp])
            c = np.mean([cl["_collapse_by_task"][t] for t in samp])
            diffs.append(c - float(np.interp(xc, xs, [v, y05, y10])))
        # exact two-sided sign test clamp_on vs add_a1.0 on discordant tasks
        b = sum(1 for t in tids if cl["_collapse_by_task"][t] and not a10["_collapse_by_task"][t])
        c_ = sum(1 for t in tids if not cl["_collapse_by_task"][t] and a10["_collapse_by_task"][t])
        n_disc = b + c_
        p_sign = (min(1.0, 2 * sum(math.comb(n_disc, i) for i in range(0, min(b, c_) + 1))
                      / 2 ** n_disc) if n_disc else 1.0)
        analysis["primary"][dname] = {
            "clamp_displacement": xc, "add_curve_x": xs, "add_curve_collapse": ys,
            "predicted_collapse_at_clamp_disp": pred, "observed_clamp_collapse": obs,
            "diff_obs_minus_pred": obs - pred,
            "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "mcnemar_clamp_vs_add1": {"clamp_only": b, "add_only": c_, "p_two_sided": p_sign},
            "P2_supported": bool(obs - pred < 0 and np.percentile(diffs, 97.5) < 0),
        }

    json.dump(analysis, open(OUT / "p2_analysis.json", "w"), indent=2)
    lines = ["# E10.1 P2 — swap vs ablation: REPORT", "",
             f"Date: {analysis['date']} · collapse = 4-gram repetition > {COLLAPSE_THRESH}", "",
             "| arm | n | collapse | mean rep | mean tokens | boxed | displacement |",
             "|---|---|---|---|---|---|---|"]
    for a, s in sorted(analysis["arms"].items()):
        lines.append(f"| {a} | {s['n']} | {s['collapse_rate']:.2f} | {s['mean_rep']:.3f} | "
                     f"{s['mean_tokens']:.0f} | {s['boxed_rate']:.2f} | "
                     f"{s['mean_displacement']:.3f} |")
    lines += ["", "## Primary (sealed P2): clamp collapse vs add-curve at matched displacement"]
    for d, p in analysis["primary"].items():
        lines.append(f"- **{d}**: observed clamp_on collapse {p['observed_clamp_collapse']:.2f} vs "
                     f"predicted {p['predicted_collapse_at_clamp_disp']:.2f} at matched "
                     f"displacement {p['clamp_displacement']:.3f}; diff "
                     f"{p['diff_obs_minus_pred']:+.2f} CI95 {p['diff_ci95']}; McNemar vs add(1.0) "
                     f"p={p['mcnemar_clamp_vs_add1']['p_two_sided']:.4f}; "
                     f"P2 supported: {p['P2_supported']}")
    (OUT / "P2_REPORT.md").write_text("\n".join(lines))
    log(f"analysis -> {OUT}/p2_analysis.json + P2_REPORT.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["clamps", "generate", "analyse", "all"], default="all")
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--bs", type=int, default=16)          # clamp-calibration batches
    ap.add_argument("--n-pairs-clamp", type=int, default=400)
    ap.add_argument("--smoke", action="store_true")
    cfg = ap.parse_args()
    if cfg.smoke:
        cfg.n_tasks, cfg.max_new_tokens, cfg.batch, cfg.n_pairs_clamp = 2, 64, 2, 8
    OUT.mkdir(parents=True, exist_ok=True)
    if cfg.stage in ("clamps", "all"):
        stage_clamps(cfg)
    if cfg.stage in ("generate", "all"):
        stage_generate(cfg)
    if cfg.stage in ("analyse", "all"):
        stage_analyse(cfg)


if __name__ == "__main__":
    main()
