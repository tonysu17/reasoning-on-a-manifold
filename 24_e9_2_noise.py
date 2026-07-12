#!/usr/bin/env python
"""E9.2 — structured state-entropy injection (collapse ladder rung three).

Does the GEOMETRY of injected state noise decide whether entropy rescues
diversity or damages reasoning? Four k-matched noise geometries at hs[17]
(block-16 output, the E10 grounded site), energy-matched:

  iso     — isotropic N(0, sigma^2 I) in the full 1536-D stream
  causal  — noise confined to the GROUNDED 2-D causal frame (E10.2 frame_k2)
  pca     — noise confined to the top-2 PCA frame of backtracking activations
            (correlational comparator, k-matched, eval-rows excluded;
            nearly disjoint from causal: principal-angle cos 0.39/0.04)
  random  — noise in a Haar-random 2-D frame (floor)

Greedy decoding throughout: all cross-sample diversity is attributable to the
state noise alone (3 noise seeds per task per cell). Energy calibrated at
runtime: target E||eps|| = frac x mean||h|| at the site, frac in {0.01, 0.03};
per-geometry sigma = target / sqrt(k).

Endpoints (annotation-free): collapse rate (4-gram rep > 0.8), BOXED RATE (the
completion guard P2b taught: loop-avoidance != preserved reasoning),
mean tokens, and cross-seed diversity = 1 - |shared 4-grams| / |union| over
the 3 seeds per task (plus distinct-4gram fraction of the union).

SEALED predictions: COLLAPSE_AND_ENTROPY.md §5 E9.2 amendment (2026-07-07).
Stages: calibrate -> generate -> analyse. ~1250 chains, one 4090 (~9-11 h).
Output: results/eval/R1-1.5B__E9_2/
"""

import argparse
import importlib.util
import json
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

OUT = ROOT / "results" / "eval" / "R1-1.5B__E9_2"
FRAMES = ROOT / "results" / "e9_2" / "frames.npz"
SITE_LAYER = 16          # SteeredModel layer=16 -> block-16 output == hs[17]
FRACS = [0.01, 0.03]
SEEDS = [0, 1, 2]
COLLAPSE_THRESH = 0.8


def log(msg):
    print(f"[e9.2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_das_module():
    spec = importlib.util.spec_from_file_location("das", ROOT / "20_das_backtracking.py")
    das = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(das)
    return das


class NoiseModel(SteeredModel):
    """State-noise injection through the SteeredModel generation machinery.

    Overrides the hook: h <- h + eps with eps = sigma * (U z) (frame arms,
    z ~ N(0, I_k) fresh per position) or eps = sigma * z (isotropic, U=None).
    Greedy decoding + per-call noise seed => diversity attributable to the
    state noise alone. Reuses generate_batch (left-padding, OOM fallback)."""

    def __init__(self, model, tokenizer, U, sigma, noise_seed=0):
        d = model.config.hidden_size
        dummy = np.zeros(d, dtype=np.float32); dummy[0] = 1.0
        super().__init__(model, tokenizer, dummy, SITE_LAYER, alpha=0.0, mode="noise")
        self.U = (torch.tensor(U, dtype=torch.float32).to(next(model.parameters()).device)
                  if U is not None else None)
        self.sigma = float(sigma)
        self.noise_seed = int(noise_seed)
        self._gen = None
        self._eps_sum, self._eps_count = 0.0, 0

    def _reseed(self):
        self._gen = torch.Generator().manual_seed(self.noise_seed)
        self._eps_sum, self._eps_count = 0.0, 0

    def generate_batch(self, instructions, max_new_tokens=None, temperature=0.0, seed=0):
        self.noise_seed = seed
        self._reseed()
        return super().generate_batch(instructions, max_new_tokens,
                                      temperature=temperature, seed=seed)

    def _hook_fn(self, module, input, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        b, s, d = hidden.shape
        if self.U is not None:
            z = torch.randn(b, s, self.U.shape[1], generator=self._gen)
            eps = self.sigma * (z.to(hidden.device) @ self.U.T)
        else:
            z = torch.randn(b, s, d, generator=self._gen)
            eps = self.sigma * z.to(hidden.device)
        self._eps_sum += float(eps.float().norm(dim=-1).sum())
        self._eps_count += b * s
        h = (hidden.float() + eps).to(hidden.dtype)
        return ((h,) + output[1:]) if is_tuple else h

    def mean_eps_norm(self):
        return self._eps_sum / self._eps_count if self._eps_count else None


def stage_calibrate(cfg, das, tok, model, device):
    """Mean ||h|| at hs[17] over a few corpus chains -> sigma per (geometry, frac)."""
    chains = json.load(open(ROOT / "data" / "chains_R1-1.5B.json"))[:5]
    norms = []
    with torch.no_grad():
        for c in chains:
            ids = tok(c["full_text"], return_tensors="pt", truncation=True,
                      max_length=1024).input_ids.to(device)
            hs = model(ids, output_hidden_states=True).hidden_states[SITE_LAYER + 1]
            norms.append(float(hs[0].float().norm(dim=-1).mean()))
    mean_h = float(np.mean(norms))
    frames = np.load(FRAMES)
    calib = {"mean_h_norm": mean_h, "site": f"hs[{SITE_LAYER+1}]", "sigmas": {}}
    for geom in ("iso", "causal", "pca", "random"):
        k = 1536 if geom == "iso" else int(frames[geom].shape[1]) if geom != "iso" else 1536
        for frac in FRACS:
            calib["sigmas"][f"{geom}_f{frac}"] = frac * mean_h / np.sqrt(k)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(calib, open(OUT / "calibration.json", "w"), indent=2)
    log(f"calibrated: mean||h||={mean_h:.1f} at hs[{SITE_LAYER+1}]; sigmas -> calibration.json")
    return calib


def stage_generate(cfg, das, tok, model, device):
    calib = json.load(open(OUT / "calibration.json"))
    frames = np.load(FRAMES)
    tasks_all = json.load(open(ROOT / "data" / "tasks_final.json"))
    eval_tasks, rule = stratified_eval_split(tasks_all, n_test=cfg.n_tasks)
    eval_tasks = eval_tasks[:cfg.n_tasks]
    instructions = [t["prompt"] for t in eval_tasks]
    task_ids = [t["id"] for t in eval_tasks]
    log(f"{len(eval_tasks)} held-out tasks ({rule})")

    geoms = cfg.geometries
    rp = OUT / "steering_results.json"
    records = json.load(open(rp)) if rp.exists() else []
    done = {(r["method"], r["task_id"], r.get("noise_seed", -1)) for r in records}

    # vanilla (greedy, deterministic, one per task; no hook)
    todo = [i for i, tid in enumerate(task_ids) if ("vanilla", tid, -1) not in done]
    if todo:
        sm = SteeredModel(model, tok, np.eye(1, model.config.hidden_size,
                          dtype=np.float32)[0], SITE_LAYER, alpha=0.0, mode="subtract")
        log(f"=== vanilla — {len(todo)} tasks ===")
        for lo in range(0, len(todo), cfg.batch):
            chunk = todo[lo:lo + cfg.batch]
            outs = sm.generate_batch([instructions[i] for i in chunk],
                                     max_new_tokens=cfg.max_new_tokens)
            for i, rec in zip(chunk, outs):
                rec.update({"task_id": task_ids[i], "method": "vanilla", "noise_seed": -1})
                records.append(rec)
            json.dump(records, open(rp, "w"))
            log(f"  vanilla: {min(lo+cfg.batch, len(todo))}/{len(todo)}")

    for geom in geoms:
        U = None if geom == "iso" else frames[geom]
        for frac in cfg.fracs:
            sigma = calib["sigmas"][f"{geom}_f{frac}"]
            arm = f"{geom}_f{frac}"
            nm = NoiseModel(model, tok, U, sigma)
            for seed in cfg.seeds:
                todo = [i for i, tid in enumerate(task_ids)
                        if (arm, tid, seed) not in done]
                if not todo:
                    continue
                log(f"=== {arm} seed{seed} (sigma={sigma:.4f}) — {len(todo)} tasks ===")
                for lo in range(0, len(todo), cfg.batch):
                    chunk = todo[lo:lo + cfg.batch]
                    outs = nm.generate_batch([instructions[i] for i in chunk],
                                             max_new_tokens=cfg.max_new_tokens, seed=seed)
                    eps = nm.mean_eps_norm()
                    for i, rec in zip(chunk, outs):
                        rec.update({"task_id": task_ids[i], "method": arm,
                                    "geometry": geom, "energy_frac": frac,
                                    "noise_seed": seed, "sigma": sigma,
                                    "mean_eps_norm": eps})
                        records.append(rec)
                    json.dump(records, open(rp, "w"))
                    log(f"  {arm} s{seed}: {min(lo+cfg.batch, len(todo))}/{len(todo)} "
                        f"(||eps|| {eps if eps is None else round(eps, 2)})")
    log(f"generation complete: {len(records)} records")


def _ngrams(text, n=4):
    w = text.split()
    return set(tuple(w[i:i+n]) for i in range(len(w) - n + 1))


def stage_analyse(cfg):
    records = json.load(open(OUT / "steering_results.json"))
    by_arm = {}
    for r in records:
        by_arm.setdefault(r["method"], {}).setdefault(r["task_id"], {})[
            r.get("noise_seed", -1)] = r

    def arm_stats(tasks):
        chains = [r for t in tasks.values() for r in t.values()]
        reps = [repetition_rate(r.get("chain", "")) for r in chains]
        coll = [x > COLLAPSE_THRESH for x in reps]
        div, dist = [], []
        for t, seeds in tasks.items():
            if len(seeds) < 2:
                continue
            gs = [_ngrams(r.get("chain", "")) for r in seeds.values()]
            union = set().union(*gs)
            inter = set.intersection(*gs) if all(gs) else set()
            if union:
                div.append(1.0 - len(inter) / len(union))
                dist.append(len(union) / max(1, sum(len(g) for g in gs)))
        return {"n_chains": len(chains),
                "collapse_rate": float(np.mean(coll)),
                "boxed_rate": float(np.mean([("\\boxed" in r.get("chain", "")) for r in chains])),
                "mean_tokens": float(np.mean([r.get("n_tokens", 0) for r in chains])),
                "cross_seed_diversity": float(np.mean(div)) if div else None,
                "distinct4_fraction": float(np.mean(dist)) if dist else None,
                "mean_eps_norm": float(np.mean([r.get("mean_eps_norm") or 0.0 for r in chains]))}

    table = {a: arm_stats(t) for a, t in by_arm.items()}
    analysis = {"experiment": "E9.2 structured state-entropy injection",
                "date": time.strftime("%Y-%m-%d"), "arms": table, "verdicts": {}}

    van = table.get("vanilla", {})
    for frac in cfg.fracs:
        iso = table.get(f"iso_f{frac}")
        cau = table.get(f"causal_f{frac}")
        if iso and cau and van:
            analysis["verdicts"][f"P1_f{frac}"] = {
                "causal_diversity": cau["cross_seed_diversity"],
                "iso_diversity": iso["cross_seed_diversity"],
                "causal_boxed_drop": van["boxed_rate"] - cau["boxed_rate"],
                "iso_boxed_drop": van["boxed_rate"] - iso["boxed_rate"],
                "P1_causal_dominates": bool(
                    cau["cross_seed_diversity"] is not None and
                    iso["cross_seed_diversity"] is not None and
                    cau["cross_seed_diversity"] >= iso["cross_seed_diversity"] and
                    (van["boxed_rate"] - cau["boxed_rate"]) <
                    (van["boxed_rate"] - iso["boxed_rate"]))}
    json.dump(analysis, open(OUT / "e9_2_analysis.json", "w"), indent=2)

    lines = ["# E9.2 structured state-entropy injection — REPORT", "",
             f"Date: {analysis['date']} · site hs[17] · greedy + noise seeds "
             f"{cfg.seeds} · collapse = rep4 > {COLLAPSE_THRESH}", "",
             "| arm | chains | collapse | boxed | tokens | x-seed diversity | distinct4 | ||eps|| |",
             "|---|---|---|---|---|---|---|---|"]
    for a in sorted(table):
        s = table[a]
        fmt = lambda v, sp='.2f': (format(v, sp) if isinstance(v, float) else 'n/a')
        lines.append(f"| {a} | {s['n_chains']} | {s['collapse_rate']:.2f} | {s['boxed_rate']:.2f} | "
                     f"{s['mean_tokens']:.0f} | {fmt(s['cross_seed_diversity'], '.3f')} | "
                     f"{fmt(s['distinct4_fraction'], '.3f')} | {s['mean_eps_norm']:.2f} |")
    lines += ["", "## Sealed P1 (causal Pareto-dominates iso at matched energy)"]
    for k, v in analysis["verdicts"].items():
        lines.append(f"- {k}: causal div {v['causal_diversity']} vs iso {v['iso_diversity']}; "
                     f"boxed drop {v['causal_boxed_drop']:.2f} vs {v['iso_boxed_drop']:.2f} "
                     f"=> P1 {'SUPPORTED' if v['P1_causal_dominates'] else 'not supported'}")
    (OUT / "REPORT.md").write_text("\n".join(lines))
    log(f"analysis -> {OUT}/e9_2_analysis.json + REPORT.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["calibrate", "generate", "analyse", "all"], default="all")
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--geometries", nargs="+",
                    default=["iso", "causal", "pca", "random"])
    ap.add_argument("--fracs", type=float, nargs="+", default=FRACS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--smoke", action="store_true")
    cfg = ap.parse_args()
    if cfg.smoke:
        cfg.n_tasks, cfg.max_new_tokens, cfg.batch = 2, 64, 2
        cfg.geometries, cfg.fracs, cfg.seeds = ["causal", "iso"], [0.03], [0, 1]
    OUT.mkdir(parents=True, exist_ok=True)

    if cfg.stage == "analyse":
        stage_analyse(cfg)
        return
    das = load_das_module()
    device, dtype = das.pick_device()
    tok, model = das.load_model(device, dtype)
    log(f"device={device} geoms={cfg.geometries} fracs={cfg.fracs} seeds={cfg.seeds}")
    if cfg.stage in ("calibrate", "all"):
        stage_calibrate(cfg, das, tok, model, device)
    if cfg.stage in ("generate", "all"):
        stage_generate(cfg, das, tok, model, device)
    if cfg.stage == "all":
        stage_analyse(cfg)


if __name__ == "__main__":
    main()
