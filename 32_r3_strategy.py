#!/usr/bin/env python3
"""
R3 — strategy-entropy pilot gate (creativity–entropy rung 3, prep).

Pre-registration: R3_PILOT_PREREG.md (sealed 2026-07-12 before any generation).
Motivated by the R2 prep finding (R2_FRONTIER_PREREG.md addendum): 4-gram
diversity is saturated on long sampled chains, so solution diversity must be
measured at the STRATEGY level on a dedicated multi-solution, answer-checkable
task family. This pilot buys three facts for $0 (local MPS) before any pod or
annotation spend: the model answers the family (G-R3.1), the value axis has
headroom (G-R3.2), the lexical classifier covers it (G-R3.3), and sampled
solutions actually vary in strategy (G-R3.4 — THE gate).

Stages:
  tasks     (CPU)  8 parametric templates × 2 instances = 16 tasks, INTEGER
                   golds computed by construction, per-template declared
                   strategy space → data/r3_tasks_pilot.json
  generate  (GPU)  16 tasks × 3 samples @ T=0.6 (seeds 0/1/2, E9.1 batching
                   contract), max_new 3072, batch 4, checkpointed
                   → results/r3_strategy/pilot_gen.json
  gate      (CPU)  G-R3.1–G-R3.4 verdicts + per-template table
                   → results/r3_strategy/{PILOT_GATE.md,pilot_gate.json}

Value axis here is TRUE CORRECTNESS (computable golds) — the eval-50
completion-only caveat does not apply. The lexical strategy classifier is a
RANGE-FINDER only (CF-T): no scientific claim rests on which strategy a chain
used; the full R3 upgrades to judged labels under the R2.2 multi-annotator
protocol iff G-R3.4 shows there is variation to label.

Usage:
  python3 32_r3_strategy.py --stage tasks
  python3 32_r3_strategy.py --stage generate            # local MPS pilot
  python3 32_r3_strategy.py --stage gate
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from math import comb
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("r3_strategy")

EVAL_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
GEN_SEEDS = [0, 1, 2]
SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="gate", choices=["tasks", "generate", "gate"])
    ap.add_argument("--tasks-file", default="data/r3_tasks_pilot.json")
    ap.add_argument("--out", default="results/r3_strategy")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


# ── strategy classifier (fixed pre-pilot; RANGE-FINDER only, CF-T) ─────────────

FAMILIES: dict[str, list[str]] = {
    "recursion":      ["recurrence", "recursion", "recursive", "f(n-1)", "a_{n-1}",
                       "dynamic programming", "previous term"],
    "pattern":        ["pattern", "small cases", "first few", "cycle", "repeats",
                       "fibonacci", "known sequence"],
    "casework":       ["case 1", "case 2", "casework", "enumerat", "list all",
                       "exactly one", "exactly two"],
    "formula":        ["formula", "binomial", "choose", "\\binom", "closed form",
                       "combination"],
    "telescoping":    ["telescop"],
    "induction":      ["induction"],
    "substitution":   ["substitut"],
    "elimination":    ["eliminat", "subtract the", "add the two equations",
                       "multiply the first"],
    "matrix":         ["matrix", "determinant", "cramer"],
    "modular":        ["mod 4", "modulo", "congruen", "mod 10"],
    "complement":     ["complement", "no six", "no sixes", "none of the",
                       "total number of sequences minus"],
    "vieta":          ["vieta"],
    "explicit_roots": ["quadratic formula", "discriminant"],
    "identity":       ["(r+s)", "expand", "p^2 - 2q", "square of the sum"],
}

SPACES: dict[str, list[str]] = {           # declared per-template strategy spaces
    "T1": ["recursion", "pattern", "casework"],
    "T2": ["formula", "telescoping", "induction", "pattern"],
    "T3": ["formula", "recursion", "casework"],
    "T4": ["substitution", "elimination", "matrix"],
    "T5": ["formula", "induction", "pattern"],
    "T6": ["pattern", "modular"],
    "T7": ["complement", "casework"],
    "T8": ["vieta", "identity", "explicit_roots"],
}


def primary_strategy(chain: str, template: str) -> str:
    """Most-hit keyword family within the template's declared space; ties break
    by declared order; zero hits → 'unclassified'."""
    text = chain.lower()
    best, best_hits = "unclassified", 0
    for fam in SPACES[template]:
        hits = sum(text.count(kw) for kw in FAMILIES[fam])
        if hits > best_hits:
            best, best_hits = fam, hits
    return best


# ── answer extraction (verbatim from 31_r2_frontier.py) ───────────────────────

def _boxed_answer(chain: str) -> str | None:
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


def normalise(ans: str | None) -> str | None:
    """Integer normalisation: strip $, commas, \\, spaces, trailing dot."""
    if ans is None:
        return None
    a = ans.replace("$", "").replace("\\,", "").replace(",", "").replace(" ", "")
    a = a.rstrip(".")
    return a if re.fullmatch(r"-?\d+", a) else None


# ── stage: tasks (parametric, golds computed by construction) ─────────────────

def _fib(n: int) -> int:
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b


def build_tasks() -> list[dict]:
    T = []

    def add(tpl, prompt, gold):
        T.append({"task_id": f"{tpl}_{sum(t['template'] == tpl for t in T)}",
                  "template": tpl, "prompt": prompt + SUFFIX, "gold": str(gold),
                  "strategy_space": SPACES[tpl]})

    for n in (10, 12):                                   # T1 — Fib(n+1)
        add("T1", f"In how many ways can a 2×{n} board be completely tiled by "
                  f"1×2 dominoes?", _fib(n + 1))
    for n in (15, 20):                                   # T2 — n(n+1)(n+2)/3
        add("T2", f"Evaluate the sum 1·2 + 2·3 + 3·4 + … + {n}·{n + 1}.",
            n * (n + 1) * (n + 2) // 3)
    for m, n in ((6, 5), (7, 6)):                        # T3 — C(m+n, m)
        add("T3", f"How many paths are there from the bottom-left corner to the "
                  f"top-right corner of a {m}×{n} grid of unit squares, moving "
                  f"only right or up along the grid lines?", comb(m + n, m))
    for (a, b, c, d, e, f, ans) in ((3, 5, 41, 2, -1, 10, 28),   # sol (7,4)
                                    (4, 3, 51, 1, 2, 24, 54)):   # sol (6,9)
        add("T4", f"Let x and y be real numbers with {a}x + {b}y = {c} and "
                  f"{d}x {'−' if e < 0 else '+'} {abs(e)}y = {f}. Find the value "
                  f"of xy.", ans)
    for n in (20, 24):                                   # T5 — n(n+1)(2n+1)/6
        add("T5", f"Evaluate 1² + 2² + 3² + … + {n}².", n * (n + 1) * (2 * n + 1) // 6)
    for a, b in ((7, 2026), (3, 2025)):                  # T6 — last digit of a^b
        add("T6", f"What is the last digit of {a}^{b}?", pow(a, b, 10))
    for n in (3, 4):                                     # T7 — 6^n − 5^n
        add("T7", f"A fair six-sided die is rolled {n} times, and the sequence of "
                  f"results is recorded. How many distinct sequences contain at "
                  f"least one 6?", 6 ** n - 5 ** n)
    for p, q in ((7, 5), (9, 14)):                       # T8 — p² − 2q
        add("T8", f"Let r and s be the roots of x² − {p}x + {q} = 0. "
                  f"Find r² + s².", p * p - 2 * q)
    return T


def stage_tasks(args) -> None:
    tasks = build_tasks()
    Path(args.tasks_file).write_text(json.dumps(tasks, indent=1, ensure_ascii=False))
    logger.info(f"tasks: wrote {len(tasks)} tasks "
                f"({len(set(t['template'] for t in tasks))} templates) → {args.tasks_file}")


# ── stage: generate (local MPS pilot) ─────────────────────────────────────────

def stage_generate(args) -> None:
    import importlib.util
    from src.chain_gen import load_model, format_prompt
    from tqdm import tqdm

    spec = importlib.util.spec_from_file_location("r1c", "30_r1_compression.py")
    r1c = importlib.util.module_from_spec(spec); spec.loader.exec_module(r1c)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "pilot_gen.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["task_id"], r["sample"]) for r in rows}

    tasks = json.loads(Path(args.tasks_file).read_text())
    if args.smoke:
        tasks = tasks[:2]
    todo = [(t, s) for s in GEN_SEEDS for t in tasks if (t["task_id"], s) not in done]
    if args.limit:
        todo = todo[:args.limit]
    logger.info(f"generate: {len(todo)} chains to run ({len(done)} done)")
    if not todo:
        return

    model, tokenizer = load_model(EVAL_MODEL, dtype="float16")
    model.eval()
    for i in tqdm(range(0, len(todo), args.batch), desc="r3-pilot"):
        batch = todo[i:i + args.batch]
        try:
            prompts = [format_prompt(tokenizer, t["prompt"]) for t, _ in batch]
            s = batch[0][1]                              # seed-homogeneous batches
            gen = r1c._hf_generate_batch(model, tokenizer, prompts,
                                         max_new=args.max_new_tokens,
                                         temperature=0.6, seed=1000 * (s + 1) + i)
            for (t, s_), g in zip(batch, gen):
                rows.append({"task_id": t["task_id"], "template": t["template"],
                             "gold": t["gold"], "sample": s_, "temperature": 0.6,
                             **g})
        except Exception as e:
            logger.warning(f"generate FAILED batch@{i}: {e}")
        finally:
            r1c.r0._clear_accel_cache()
        path.write_text(json.dumps(rows, indent=1))
    logger.info(f"generate: done ({len(rows)} rows)")


# ── stage: gate ───────────────────────────────────────────────────────────────

def stage_gate(args) -> None:
    out = Path(args.out)
    rows = json.loads((out / "pilot_gen.json").read_text())
    tasks = {t["task_id"]: t for t in json.loads(Path(args.tasks_file).read_text())}
    for r in rows:
        r["answer"] = normalise(_boxed_answer(r["chain"]))
        r["correct"] = r["answer"] is not None and r["answer"] == r["gold"]
        r["strategy"] = primary_strategy(r["chain"], r["template"])
        r["cap_hit"] = r["n_tokens"] >= args.max_new_tokens

    parse_rate = float(np.mean([r["answer"] is not None for r in rows]))
    accuracy = float(np.mean([r["correct"] for r in rows]))
    answered = [r for r in rows if r["answer"] is not None]
    coverage = (float(np.mean([r["strategy"] != "unclassified" for r in answered]))
                if answered else 0.0)

    by_task: dict[str, list] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    varied = [tid for tid, rs in by_task.items()
              if len({r["strategy"] for r in rs} - {"unclassified"}) >= 2]
    variation = len(varied) / len(by_task) if by_task else 0.0

    gates = {
        "G_R3_1_answerable": {"parse_rate": round(parse_rate, 3), "bar": 0.70,
                              "pass": parse_rate >= 0.70},
        "G_R3_2_headroom": {"accuracy": round(accuracy, 3), "bar": [0.20, 0.95],
                            "pass": 0.20 <= accuracy <= 0.95},
        "G_R3_3_coverage": {"coverage_answered": round(coverage, 3), "bar": 0.60,
                            "pass": coverage >= 0.60},
        "G_R3_4_variation": {"tasks_with_2plus_strategies": f"{len(varied)}/{len(by_task)}",
                             "fraction": round(variation, 3), "bar": 0.25,
                             "pass": variation >= 0.25},
    }
    verdict = ("ALL GATES PASS — full R3 as designed"
               if all(g["pass"] for g in gates.values())
               else "GATE FAILURE — redesign per prereg options before full R3")

    per_tpl = {}
    for tpl in sorted({r["template"] for r in rows}):
        rs = [r for r in rows if r["template"] == tpl]
        per_tpl[tpl] = {
            "n": len(rs),
            "parse": round(float(np.mean([r["answer"] is not None for r in rs])), 2),
            "acc": round(float(np.mean([r["correct"] for r in rs])), 2),
            "cap_hit": round(float(np.mean([r["cap_hit"] for r in rs])), 2),
            "mean_tok": int(np.mean([r["n_tokens"] for r in rs])),
            "strategies": sorted({r["strategy"] for r in rs}),
        }

    report = {"n_rows": len(rows), "n_tasks": len(by_task), "gates": gates,
              "verdict": verdict, "per_template": per_tpl,
              "varied_tasks": varied}
    (out / "pilot_gate.json").write_text(json.dumps(report, indent=1))

    md = ["# R3 pilot gate — strategy-entropy substrate check\n",
          f"{len(rows)} chains / {len(by_task)} tasks. Prereg: R3_PILOT_PREREG.md\n",
          f"## VERDICT: **{verdict}**\n",
          "| gate | value | bar | pass |", "|---|--:|--:|:--|"]
    for k, g in gates.items():
        val = [v for v in g.values() if not isinstance(v, bool)][:2]
        md.append(f"| {k} | {val[0]} | {val[1]} | {'✅' if g['pass'] else '❌'} |")
    md += ["\n## Per template\n",
           "| tpl | n | parse | acc | cap-hit | mean tok | strategies seen |",
           "|---|--:|--:|--:|--:|--:|---|"]
    for tpl, m in per_tpl.items():
        md.append(f"| {tpl} | {m['n']} | {m['parse']} | {m['acc']} | {m['cap_hit']} "
                  f"| {m['mean_tok']} | {', '.join(m['strategies'])} |")
    (out / "PILOT_GATE.md").write_text("\n".join(md) + "\n")
    logger.info(f"gate: {verdict} → {out}/PILOT_GATE.md")


def main() -> None:
    args = parse_args()
    {"tasks": stage_tasks, "generate": stage_generate, "gate": stage_gate}[args.stage](args)


if __name__ == "__main__":
    main()
