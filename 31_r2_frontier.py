#!/usr/bin/env python3
"""
R2 — entropy–value frontier (creativity–entropy rung 2).

Pre-registration: R2_FRONTIER_PREREG.md (sealed 2026-07-12). Programme doc:
../creativity_entropy_extension.md §6 (R2).

Two entropy knobs traced on the value(=boxed completion) vs solution-diversity
plane for R1-Distill-1.5B, asking which buys diversity at less completion cost:

  Knob T (thermostat) : vanilla sampling, temperature swept {0.3, 0.6, 0.9, 1.2}
  Knob A (pump)       : backtracking single_direction steering, α swept
                        {0, 0.5, 1.0, 1.5} at fixed T=0.6

Knob A + the T=0.6 point of knob T are FULLY REUSED from E9.1 T06; only the
vanilla temperature sweep at T∈{0.3,0.9,1.2} is new generation (~1 GPU-h).

Stages:
  generate  (GPU)  vanilla r1 temperature sweep → r2 sweep rows (steering-schema)
                   → results/r2_frontier/sweep_gen.json
  analyse   (CPU)  ingest E9.1 T06 (+ optional sweep + R1 r1 arm), build both
                   frontiers, run P-R2.1 matched-diversity / P-R2.2 monotone /
                   P-R2.3 collapse → results/r2_frontier/{REPORT.md,report.json}

Value axis = annotation-free `boxed` rate (completion, NOT correctness — CF-P).
Runs analyse on existing data first (preliminary PUMP frontier before the
thermostat sweep exists); the sweep only ADDS knob-T points.

Usage:
  python3 31_r2_frontier.py --stage analyse                 # runs now, existing data
  python3 31_r2_frontier.py --stage generate --temps 0.3 0.9 1.2   # pod, later
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("r2_frontier")

EVAL_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
GEN_SEEDS = [0, 1, 2]
PUMP_T = 0.6                    # fixed temperature for the amplification knob


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="analyse", choices=["analyse", "generate"])
    ap.add_argument("--t06", default="results/eval/R1-1.5B__E9_1_T06/steering_results.json",
                    help="E9.1 T06 arms (knob A + vanilla T=0.6)")
    ap.add_argument("--sweep", default="results/r2_frontier/sweep_gen.json",
                    help="R2 vanilla temperature-sweep rows (added when present)")
    ap.add_argument("--chains", default="data/chains_R1-1.5B.json")
    ap.add_argument("--eval-ids",
                    default="results/eval/R1-1.5B__E9_1_greedy/eval_task_ids.json")
    ap.add_argument("--out", default="results/r2_frontier")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.3, 0.9, 1.2],
                    help="NEW vanilla temperatures to generate (T=0.6 already exists)")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


# ── shared metric helpers (definitions inherited verbatim from E9.1/R1) ────────

def _ngrams(text: str, n: int = 4) -> set:
    toks = text.split()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _boxed(chain: str) -> bool:
    return "\\boxed" in chain


def _boxed_answer(chain: str) -> str | None:
    """Content of the last brace-balanced \\boxed{...} (annotation-free answer)."""
    import re
    ms = list(re.finditer(r"\\boxed\{", chain))
    if not ms:
        return None
    i, depth, out = ms[-1].end(), 1, []
    for c in chain[i:]:
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
    return "".join(out).strip()


def _answer_diversity(sample_chains: list[str]) -> float | None:
    """Distinct \\boxed answers / n samples among a task's samples that produced
    an answer. None if <2 answered — the honest, completion-conditioned axis
    (4-gram Jaccard is saturated ~0.96 and non-discriminating; PREP FINDING
    2026-07-12, R2_FRONTIER_PREREG.md addendum)."""
    ans = [a for c in sample_chains if (a := _boxed_answer(c)) is not None]
    if len(ans) < 2:
        return None
    return len(set(ans)) / len(ans)


def _diversity(sample_chains: list[str]) -> float | None:
    """1 − mean pairwise 4-gram Jaccard across a task's samples (≥2)."""
    if len(sample_chains) < 2:
        return None
    grams = [_ngrams(c) for c in sample_chains]
    jac = [len(a & b) / max(len(a | b), 1) for a, b in combinations(grams, 2)]
    return 1.0 - float(np.mean(jac))


def _cell_metrics(rows: list[dict], rep_fn) -> dict:
    """Per-cell value/diversity/collapse. rows share one (knob, level) cell."""
    by_task: dict = {}
    for r in rows:
        by_task.setdefault(r["base_task_id"], []).append(r)
    divs, adivs, boxed, collapsed, ntok, answered = [], [], [], [], [], []
    for _t, rs in by_task.items():
        chains = [r["chain"] for r in rs]
        d = _diversity(chains)
        if d is not None:
            divs.append(d)
        ad = _answer_diversity(chains)
        if ad is not None:
            adivs.append(ad)
        answered.append(any(_boxed(c) for c in chains))
        for r in rs:
            boxed.append(_boxed(r["chain"]))
            collapsed.append(rep_fn(r["chain"]) > 0.8)
            ntok.append(r.get("n_tokens") or len(r["chain"].split()))
    # collapse-excluded boxed (P-R2.3 secondary endpoint)
    keep = [(_boxed(r["chain"])) for rs in by_task.values() for r in rs
            if rep_fn(r["chain"]) <= 0.8]
    return {
        "n_rows": sum(len(rs) for rs in by_task.values()),
        "n_tasks": len(by_task),
        "diversity": float(np.mean(divs)) if divs else None,       # 4-gram (SATURATED)
        "diversity_n": len(divs),
        "answer_diversity": float(np.mean(adivs)) if adivs else None,  # distinct-boxed
        "answer_diversity_n": len(adivs),                          # tasks with ≥2 answers
        "answered_frac": float(np.mean(answered)) if answered else None,
        "boxed": float(np.mean(boxed)) if boxed else None,
        "boxed_uncollapsed": float(np.mean(keep)) if keep else None,
        "collapse": float(np.mean(collapsed)) if collapsed else None,
        "mean_tokens": float(np.mean(ntok)) if ntok else None,
    }


# ── stage: analyse ────────────────────────────────────────────────────────────

def _load_rows(args) -> list[dict]:
    rows = json.loads(Path(args.t06).read_text())
    sweep_p = Path(args.sweep)
    n_sweep = 0
    if sweep_p.exists():
        sweep = json.loads(sweep_p.read_text())
        rows += sweep
        n_sweep = len(sweep)
    logger.info(f"analyse: {len(rows)} rows ({n_sweep} from R2 sweep)")
    return rows


def _frontier(cells: dict) -> list[dict]:
    """(diversity, boxed) points sorted by diversity, dropping diversity-less cells."""
    pts = [{"level": lvl, **m} for lvl, m in cells.items() if m["diversity"] is not None]
    return sorted(pts, key=lambda p: p["diversity"])


def _matched_diversity_gap(pump: list[dict], thermo: list[dict], n_grid: int = 25) -> dict:
    """P-R2.1: boxed(pump) − boxed(thermo) on a common diversity grid (overlap only)."""
    if len(pump) < 2 or len(thermo) < 2:
        return {"status": "insufficient points (need ≥2 per knob with diversity)"}
    lo = max(pump[0]["diversity"], thermo[0]["diversity"])
    hi = min(pump[-1]["diversity"], thermo[-1]["diversity"])
    if not (hi > lo):
        return {"status": "no diversity overlap between knobs — widen a grid (kill criterion)",
                "pump_range": [pump[0]["diversity"], pump[-1]["diversity"]],
                "thermo_range": [thermo[0]["diversity"], thermo[-1]["diversity"]]}
    grid = np.linspace(lo, hi, n_grid)

    def interp(pts, grid):
        xs = [p["diversity"] for p in pts]
        ys = [p["boxed"] for p in pts]
        return np.interp(grid, xs, ys)

    gaps = interp(pump, grid) - interp(thermo, grid)
    n_pos = int((gaps > 0).sum())
    # bootstrap CI on the mean gap over grid points
    rng = np.random.default_rng(0)
    boot = [float(np.mean(rng.choice(gaps, len(gaps), replace=True))) for _ in range(2000)]
    return {
        "overlap_diversity": [float(lo), float(hi)],
        "mean_gap_boxed_pump_minus_thermo": float(np.mean(gaps)),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "grid_points_pump_higher": f"{n_pos}/{n_grid}",
        "verdict": ("P-R2.1 SUPPORTED — pump retains more completion at matched diversity"
                    if np.percentile(boot, 2.5) > 0 else
                    "P-R2.1 REFUTED — thermostat retains more (pump is only an overthinking tax)"
                    if np.percentile(boot, 97.5) < 0 else
                    "P-R2.1 INCONCLUSIVE — gap CI straddles 0"),
    }


def _monotone(front: list[dict]) -> dict:
    from scipy.stats import spearmanr
    if len(front) < 3:
        return {"status": "too few points", "n": len(front)}
    rho, p = spearmanr([f["diversity"] for f in front], [f["boxed"] for f in front])
    return {"spearman_rho": round(float(rho), 3), "p": float(p), "n": len(front)}


def stage_analyse(args) -> None:
    from src.evaluation import repetition_rate
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args)

    # Knob A (pump): backtracking single_direction @ T=0.6, keyed by α;
    # α=0 anchor = vanilla @ T=0.6.
    pump_cells = {}
    van_t06 = [r for r in rows if r.get("method") == "vanilla"
               and abs(float(r.get("temperature", 0)) - PUMP_T) < 1e-6]
    if van_t06:
        pump_cells[0.0] = _cell_metrics(van_t06, repetition_rate)
    for a in sorted({r["alpha"] for r in rows
                     if r.get("behaviour") == "backtracking"
                     and r.get("method") == "single_direction"
                     and abs(float(r.get("temperature", 0)) - PUMP_T) < 1e-6}):
        cell = [r for r in rows if r.get("behaviour") == "backtracking"
                and r.get("method") == "single_direction"
                and r["alpha"] == a
                and abs(float(r.get("temperature", 0)) - PUMP_T) < 1e-6]
        pump_cells[float(a)] = _cell_metrics(cell, repetition_rate)

    # Knob T (thermostat): vanilla, keyed by temperature (T>0 for diversity).
    thermo_cells = {}
    for t in sorted({float(r.get("temperature", 0)) for r in rows
                     if r.get("method") == "vanilla" and float(r.get("temperature", 0)) > 0}):
        cell = [r for r in rows if r.get("method") == "vanilla"
                and abs(float(r.get("temperature", 0)) - t) < 1e-6]
        thermo_cells[t] = _cell_metrics(cell, repetition_rate)

    pump_front = _frontier(pump_cells)
    thermo_front = _frontier(thermo_cells)

    report = {
        "value_axis": "boxed completion rate (annotation-free; NOT correctness — CF-P)",
        "pump_cells": pump_cells, "thermo_cells": thermo_cells,
        "pump_frontier": pump_front, "thermo_frontier": thermo_front,
        "P_R2_1_matched_diversity": _matched_diversity_gap(pump_front, thermo_front),
        "P_R2_2_monotone": {"pump": _monotone(pump_front),
                            "thermostat": _monotone(thermo_front)},
        "thermostat_complete": len(thermo_front) >= 2,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    _write_md(out, report)
    logger.info(f"analyse: wrote {out}/report.json + REPORT.md "
                f"(pump {len(pump_front)} pts / thermostat {len(thermo_front)} pts)")


def _write_md(out: Path, r: dict) -> None:
    def _row(lvl, m):
        f = lambda x, p=3: "—" if x is None else format(x, f".{p}f")
        return (f"| {lvl} | {m['n_tasks']} | {f(m['diversity'])} | {f(m['answer_diversity'])} "
                f"({m['answer_diversity_n']}) | {f(m['answered_frac'],2)} | {f(m['boxed'])} "
                f"| {f(m['collapse'])} | {f(m['mean_tokens'],0)} |")
    hdr = ("| level | tasks | 4gram-div | ans-div (n≥2) | answered | boxed | collapse | mean_tok |\n"
           "|--:|--:|--:|--:|--:|--:|--:|--:|")
    md = ["# R2 — entropy–value frontier (creativity–entropy rung 2)\n",
          f"Value axis: {r['value_axis']}. Prereg: R2_FRONTIER_PREREG.md\n",
          "> **PREP FINDING 2026-07-12:** 4-gram diversity is SATURATED (~0.96, range 0.87–0.99, "
          "0% of tasks <0.9) — surface token variation, not solution diversity, no dynamic range. "
          "Answer-level diversity has range but is SPARSE (only ~30–46% of the 50 general-reasoning "
          "eval tasks produce any \\boxed answer). ⇒ the value-vs-diversity FRONTIER is not "
          "measurable on this task set; R2's clean result is the completion/length effect below. "
          "True diversity frontier deferred to R3's dedicated multi-solution task family.\n",
          "## Knob A — PUMP (backtracking amplification α @ T=0.6)\n", hdr]
    for lvl, m in sorted(r["pump_cells"].items()):
        md.append(_row(lvl, m))
    md += ["\n## Knob T — THERMOSTAT (vanilla temperature)\n", hdr]
    for lvl, m in sorted(r["thermo_cells"].items()):
        md.append(_row(lvl, m))
    p1 = r["P_R2_1_matched_diversity"]
    md += ["\n## P-R2.1 — pump vs thermostat at matched diversity\n"]
    if "verdict" in p1:
        md.append(f"Overlap diversity {p1['overlap_diversity']}; mean boxed gap "
                  f"(pump−thermo) **{p1['mean_gap_boxed_pump_minus_thermo']:+.3f}** "
                  f"CI95 {p1['ci95']}; pump higher at {p1['grid_points_pump_higher']} grid pts.\n"
                  f"\n**{p1['verdict']}**\n")
    else:
        md.append(f"_{p1['status']}_"
                  + (f" (pump {p1.get('pump_range')} vs thermo {p1.get('thermo_range')})"
                     if 'pump_range' in p1 else "") + "\n")
    p2 = r["P_R2_2_monotone"]
    md += [f"\n## P-R2.2 — value falls as diversity rises (both knobs)\n",
           f"- pump: {p2['pump']}\n- thermostat: {p2['thermostat']}\n"]
    if not r["thermostat_complete"]:
        md.append("\n> ⚠️ PRELIMINARY: thermostat frontier has <2 diversity points — the "
                  "temperature sweep (T∈{0.3,0.9,1.2}) is not yet generated. Pump frontier "
                  "stands on existing data; P-R2.1 awaits the sweep.\n")
    (out / "REPORT.md").write_text("\n".join(md) + "\n")


# ── stage: generate (pod; vanilla temperature sweep) ──────────────────────────

def stage_generate(args) -> None:
    """Vanilla r1 at each NEW temperature × 3 samples × eval tasks. Reuses R1's
    proven batched HF generation path (imported at runtime)."""
    import importlib.util
    import torch
    from src.chain_gen import load_model, format_prompt, _seed_torch

    spec = importlib.util.spec_from_file_location("r1c", "30_r1_compression.py")
    r1c = importlib.util.module_from_spec(spec); spec.loader.exec_module(r1c)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "sweep_gen.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["base_task_id"], r["temperature"], r["sample"]) for r in rows}

    eval_ids = json.loads(Path(args.eval_ids).read_text())["task_ids"]
    chains = {r["task_id"]: r for r in json.loads(Path(args.chains).read_text())}
    tasks = [(tid, chains[tid]["instruction"]) for tid in eval_ids if tid in chains]
    if args.smoke:
        tasks, args.temps = tasks[:2], args.temps[:1]

    todo = [(tid, ins, T, s) for T in args.temps for s in GEN_SEEDS
            for (tid, ins) in tasks if (tid, T, s) not in done]
    if args.limit:
        todo = todo[:args.limit]
    logger.info(f"generate: {len(todo)} vanilla chains (temps {args.temps}, {len(done)} done)")
    if not todo:
        return

    model, tokenizer = load_model(EVAL_MODEL, dtype="float16")
    model.eval()
    from tqdm import tqdm
    for i in tqdm(range(0, len(todo), args.batch), desc="r2-sweep"):
        batch = todo[i:i + args.batch]
        try:
            prompts = [format_prompt(tokenizer, ins) for (_, ins, _, _) in batch]
            T = batch[0][2]
            gen = r1c._hf_generate_batch(model, tokenizer, prompts,
                                         max_new=args.max_new_tokens, temperature=T,
                                         seed=1000 * (batch[0][3] + 1) + i)
            for (tid, _, t, s), g in zip(batch, gen):
                rows.append({"behaviour": "shared", "method": "vanilla", "alpha": 0.0,
                             "base_task_id": tid, "task_id": tid, "sample": s,
                             "temperature": t, **g})
        except Exception as e:
            logger.warning(f"r2-sweep FAILED batch@{i}: {e}")
        finally:
            r1c.r0._clear_accel_cache()
        path.write_text(json.dumps(rows, indent=1))
    logger.info(f"generate: done ({len(rows)} rows)")


def main() -> None:
    args = parse_args()
    if args.stage == "generate":
        stage_generate(args)
    else:
        stage_analyse(args)


if __name__ == "__main__":
    main()
