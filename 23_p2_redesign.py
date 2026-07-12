#!/usr/bin/env python
"""E10.1 P2 (REDESIGN) — swap/clamp vs projective ablation, matched on DISPLACEMENT.

Why a redesign (see E10_DAS_PREREG.md Adjudication 2): the first P2 compared a
single full-clamp point (displacement ~20-24) against a projective α-sweep that
only reached ~7-16, so "matched displacement" flat-extrapolated off the curve.
FIX: dose BOTH intervention families so their realized-displacement ranges
OVERLAP, then compare the collapse-vs-displacement CURVES on common support.

  - projective amplify (Huang op):   h' = h + α(rᵀh)r,  α ∈ {0.5,1,2,3}
  - bounded clamp toward c_on:        h' = h + β(c_on − rᵀh)r,  β ∈ {0.5,1.0}
    (β<1 = dose-able partial clamp; c_on = class-mean source coordinate)

Both applied to BOTH directions: the grounded causal L17-DAS direction (E10.1
headline) and the correlational E1 diff-of-means (each at its own site). One
shared vanilla. Greedy, cap 8192, the 50 held-out tasks (collapse-prone regime,
matching E8/E9.1b). 1 + 2×(4+2) = 13 arms × 50 = 650 chains (~3-4 h, ~$3-4).

Matching variable (primary) = mean realized |Δ(rᵀh)| recorded by the hook — the
clean, annotation-free "how far did you move the coordinate". Secondary on-target
axis = lexical backtracking-cue rate per 1000 tokens (annotation-free proxy;
crude, reported as secondary only).

Endpoints (annotation-free): collapse = 4-gram repetition > 0.8; boxed rate;
tokens; cue rate.

SEALED prediction (Adjudication 3): on OVERLAPPING displacement support, the
bounded clamp collapses <= the projective shift at matched displacement (the
"on-distribution targeting is gentler" hypothesis). Equality => "collapse is a
pure function of displacement, intervention type is irrelevant" (also a clean,
publishable outcome). Per-direction; grounded DAS is the headline, dm secondary.

Stages: clamps -> generate -> analyse. Output: results/eval/R1-1.5B__E10_P2b/
"""

import argparse
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))
from src.steered_inference import SteeredModel          # noqa: E402
from src.task_gen import stratified_eval_split          # noqa: E402
from src.evaluation import repetition_rate              # noqa: E402

OUT = ROOT / "results" / "eval" / "R1-1.5B__E10_P2b"
DAS_DIR = ROOT / "results" / "das" / "R1-1.5B" / "main"
DM_VEC = ROOT / "results" / "steering_vectors" / "R1-1.5B__E1_pooled" / "backtracking_single.npy"

# direction -> (vector path, hs index for clamp calibration, SteeredModel layer)
DIRECTIONS = {
    "das": (DAS_DIR / "dir_learned_L17.npy", 17, 16),
    "dm": (DM_VEC, 18, 17),
}
ALPHAS = [0.5, 1.0, 2.0, 3.0]      # projective amplify doses (span the clamp displacement)
BETAS = [0.5, 1.0]                 # bounded-clamp doses toward c_on
COLLAPSE_THRESH = 0.8
CUE_RE = re.compile(r"\b(wait|actually|hmm+|hold on|let me reconsider|on second thought|"
                    r"but wait|no,|scratch that|never mind)\b", re.IGNORECASE)


def log(msg):
    print(f"[e10.p2b {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_das_module():
    spec = importlib.util.spec_from_file_location("das", ROOT / "20_das_backtracking.py")
    das = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(das)
    return das


def cue_rate_per_1k(text, n_tokens):
    if not n_tokens:
        return 0.0
    return 1000.0 * len(CUE_RE.findall(text)) / n_tokens


def stage_clamps(cfg):
    das = load_das_module()
    device, dtype = das.pick_device()
    tok, model = das.load_model(device, dtype)
    pairs = json.load(open(DAS_DIR / "pairs.json"))[:cfg.n_pairs_clamp]
    targets = {}
    for name, (vec_path, hs_idx, _) in DIRECTIONS.items():
        v = np.load(vec_path)
        d = das.torch.tensor(v / (np.linalg.norm(v) + 1e-8), dtype=das.torch.float32, device=device)
        batches = das.make_batches(pairs, tok, model, hs_idx, device, cfg.bs, 1)
        on = [float(x) for b in batches for x in (b["hs"] @ d)]
        targets[name] = {"c_on": float(np.mean(on)), "sd_on": float(np.std(on)),
                         "hs_index": hs_idx, "n": len(on)}
        log(f"{name}: c_on={targets[name]['c_on']:+.3f} (sd {targets[name]['sd_on']:.3f})")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(targets, open(OUT / "clamp_targets.json", "w"), indent=2)
    del model


def build_arms(targets):
    """arm name -> (direction, mode, alpha, clamp_value, clamp_gain)."""
    arms = {"vanilla": (None, "subtract", 0.0, 0.0, 1.0)}
    for name in DIRECTIONS:
        c_on = targets[name]["c_on"]
        for a in ALPHAS:
            arms[f"{name}_proj_a{a}"] = (name, "add", a, 0.0, 1.0)
        for b in BETAS:
            arms[f"{name}_clamp_b{b}"] = (name, "clamp", 0.0, c_on, b)
    return arms


def stage_generate(cfg):
    targets = json.load(open(OUT / "clamp_targets.json"))
    das = load_das_module()
    device, dtype = das.pick_device()
    tok, model = das.load_model(device, dtype)

    tasks_all = json.load(open(ROOT / "data" / "tasks_final.json"))
    eval_tasks, rule = stratified_eval_split(tasks_all, n_test=cfg.n_tasks)
    eval_tasks = eval_tasks[:cfg.n_tasks]
    instructions = [t["prompt"] for t in eval_tasks]
    task_ids = [t["id"] for t in eval_tasks]
    log(f"{len(eval_tasks)} held-out tasks ({rule})")

    vecs = {n: (lambda v: v / (np.linalg.norm(v) + 1e-8))(np.load(p))
            for n, (p, _, _) in DIRECTIONS.items()}
    arms = build_arms(targets)
    rp = OUT / "steering_results.json"
    records = json.load(open(rp)) if rp.exists() else []
    done = {(r["method"], r["task_id"]) for r in records}

    for arm, (dname, mode, alpha, cval, cgain) in arms.items():
        todo = [i for i, tid in enumerate(task_ids) if (arm, tid) not in done]
        if not todo:
            log(f"{arm}: complete"); continue
        vec = vecs[dname] if dname else vecs["dm"]
        layer = DIRECTIONS[dname][2] if dname else DIRECTIONS["dm"][2]
        sm = SteeredModel(model, tok, vec, layer, alpha=alpha, mode=mode,
                          clamp_value=cval, clamp_gain=cgain)
        log(f"=== {arm} (layer={layer} {mode} a={alpha} c={cval:+.2f} b={cgain}) — {len(todo)} tasks ===")
        for lo in range(0, len(todo), cfg.batch):
            chunk = todo[lo:lo + cfg.batch]
            outs = sm.generate_batch([instructions[i] for i in chunk],
                                     max_new_tokens=cfg.max_new_tokens)
            disp = sm.mean_abs_displacement()
            for i, rec in zip(chunk, outs):
                rec.update({"task_id": task_ids[i], "method": arm, "direction": dname or "none",
                            "mode": mode, "alpha": alpha, "clamp_gain": cgain,
                            "mean_abs_displacement": disp})
                records.append(rec)
            json.dump(records, open(rp, "w"))
            log(f"  {arm}: {min(lo + cfg.batch, len(todo))}/{len(todo)} (disp "
                f"{disp if disp is None else round(disp, 2)})")
    log(f"generation complete: {len(records)} records -> {rp}")


def _boot_diff(clamp_by_task, proj_x, proj_y_by_task, xc, n_boot=10000, seed=0):
    """Paired bootstrap of (clamp collapse − projective collapse interpolated to
    the clamp displacement xc). proj_y_by_task: list of (x, {tid:0/1}) for the
    projective dose points; xc must lie within [min,max] proj_x (in-range)."""
    rng = np.random.default_rng(seed)
    tids = sorted(set(clamp_by_task) & set.intersection(*[set(d) for _, d in proj_y_by_task]))
    diffs = []
    for _ in range(n_boot):
        samp = rng.choice(tids, size=len(tids), replace=True)
        c = np.mean([clamp_by_task[t] for t in samp])
        ys = [np.mean([d[t] for t in samp]) for _, d in proj_y_by_task]
        diffs.append(c - float(np.interp(xc, proj_x, ys)))
    return float(np.mean(diffs)), [float(np.percentile(diffs, 2.5)),
                                   float(np.percentile(diffs, 97.5))]


def stage_analyse(cfg):
    records = json.load(open(OUT / "steering_results.json"))
    by_arm = {}
    for r in records:
        by_arm.setdefault(r["method"], {})[r["task_id"]] = r

    def stats(rs):
        reps = {t: repetition_rate(r.get("chain", "")) for t, r in rs.items()}
        coll = {t: reps[t] > COLLAPSE_THRESH for t in reps}
        return {"n": len(rs), "collapse_rate": float(np.mean(list(coll.values()))),
                "mean_rep": float(np.mean(list(reps.values()))),
                "mean_tokens": float(np.mean([r.get("n_tokens", 0) for r in rs.values()])),
                "boxed_rate": float(np.mean([("\\boxed" in r.get("chain", "")) for r in rs.values()])),
                "cue_per_1k": float(np.mean([cue_rate_per_1k(r.get("chain", ""), r.get("n_tokens", 0))
                                             for r in rs.values()])),
                "mean_displacement": float(np.mean([r.get("mean_abs_displacement") or 0.0
                                                    for r in rs.values()])),
                "_collapse_by_task": coll}

    table = {a: stats(rs) for a, rs in by_arm.items()}
    analysis = {"experiment": "E10.1 P2 redesign (matched displacement)",
                "date": time.strftime("%Y-%m-%d"), "collapse_thresh": COLLAPSE_THRESH,
                "arms": {a: {k: v for k, v in s.items() if not k.startswith("_")}
                         for a, s in table.items()}, "primary": {}}

    for dname in DIRECTIONS:
        proj = [(table[f"{dname}_proj_a{a}"]["mean_displacement"],
                 f"{dname}_proj_a{a}") for a in ALPHAS if f"{dname}_proj_a{a}" in table]
        proj.sort()
        proj_x = [x for x, _ in proj]
        proj_y_by_task = [(x, table[a]["_collapse_by_task"]) for x, a in proj]
        for b in BETAS:
            cl_arm = f"{dname}_clamp_b{b}"
            if cl_arm not in table or len(proj_x) < 2:
                continue
            xc = table[cl_arm]["mean_displacement"]
            in_range = proj_x[0] <= xc <= proj_x[-1]
            mean_d, ci = _boot_diff(table[cl_arm]["_collapse_by_task"], proj_x,
                                    proj_y_by_task, xc)
            analysis["primary"][cl_arm] = {
                "clamp_displacement": xc, "proj_displacement_range": [proj_x[0], proj_x[-1]],
                "in_range": bool(in_range),
                "clamp_collapse": table[cl_arm]["collapse_rate"],
                "proj_collapse_interp_at_clamp_disp": float(np.interp(
                    xc, proj_x, [table[a]["collapse_rate"] for _, a in proj])),
                "diff_clamp_minus_proj": mean_d, "diff_ci95": ci,
                "P2_supported_clamp_gentler": bool(mean_d < 0 and ci[1] < 0),
                "type_irrelevant_equal": bool(ci[0] <= 0 <= ci[1]),
            }

    json.dump(analysis, open(OUT / "p2b_analysis.json", "w"), indent=2)
    lines = ["# E10.1 P2 REDESIGN — matched-displacement collapse curves", "",
             f"Date: {analysis['date']} · collapse = 4-gram repetition > {COLLAPSE_THRESH}", "",
             "| arm | n | collapse | displacement | cue/1k | boxed | tokens |",
             "|---|---|---|---|---|---|---|"]
    for a, s in sorted(analysis["arms"].items()):
        lines.append(f"| {a} | {s['n']} | {s['collapse_rate']:.2f} | {s['mean_displacement']:.2f} | "
                     f"{s['cue_per_1k']:.2f} | {s['boxed_rate']:.2f} | {s['mean_tokens']:.0f} |")
    lines += ["", "## Primary: clamp vs projective at MATCHED displacement (in-range now)"]
    for arm, p in analysis["primary"].items():
        verdict = ("clamp GENTLER" if p["P2_supported_clamp_gentler"]
                   else "type IRRELEVANT (equal)" if p["type_irrelevant_equal"]
                   else "clamp WORSE")
        lines.append(f"- **{arm}**: clamp collapse {p['clamp_collapse']:.2f} vs projective "
                     f"{p['proj_collapse_interp_at_clamp_disp']:.2f} at displacement "
                     f"{p['clamp_displacement']:.2f} (proj range {p['proj_displacement_range'][0]:.1f}"
                     f"–{p['proj_displacement_range'][1]:.1f}, in-range={p['in_range']}); "
                     f"diff {p['diff_clamp_minus_proj']:+.2f} CI95 {[round(x,2) for x in p['diff_ci95']]} "
                     f"⇒ **{verdict}**")
    (OUT / "P2b_REPORT.md").write_text("\n".join(lines))
    log(f"analysis -> {OUT}/p2b_analysis.json + P2b_REPORT.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["clamps", "generate", "analyse", "all"], default="all")
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--bs", type=int, default=16)
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
