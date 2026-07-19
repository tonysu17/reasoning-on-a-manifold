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

Pilot stages (EXECUTED 2026-07-13; ALL 4 GATES PASSED — results/r3_strategy/PILOT_GATE.md):
  tasks     (CPU)  8 parametric templates × 2 instances = 16 tasks, INTEGER
                   golds computed by construction, per-template declared
                   strategy space → data/r3_tasks_pilot.json
  generate  (GPU)  16 tasks × 3 samples @ T=0.6 (seeds 0/1/2, E9.1 batching
                   contract), max_new 6144 (Amendment 1), batch 4, checkpointed
                   → results/r3_strategy/pilot_gen.json
  gate      (CPU)  G-R3.1–G-R3.4 verdicts + per-template table
                   → results/r3_strategy/{PILOT_GATE.md,pilot_gate.json}

FULL R3 stages (this file, post-gate; prereg §"What the full R3 adds if the
gates pass" — pod-scale task set, more samples, the value×strategy-diversity
plane with the reframed P-R2.1 pump-vs-thermostat comparison):
  full-tasks     (CPU)  8 templates × 8 instances (4 easy + 4 hard; parameters
                        DISJOINT from the 16 pilot instances) = 64 tasks
                        → data/r3_tasks_full.json
  full-generate  (GPU)  64 tasks × k=8 samples (seeds 0–7) × 7 cells:
                        thermostat = vanilla T ∈ {0.3, 0.6, 0.9, 1.2};
                        pump = backtracking single_direction (E1-pooled vector,
                        layer from vector metadata = 17), mode=subtract,
                        α ∈ {0.5, 1.0, 1.5} @ T=0.6 (the R2 knob-A grid, as
                        EXECUTED in E9.1 T06 — R2_FRONTIER_PREREG.md).
                        Shared anchor: vanilla T=0.6 doubles as pump α=0.
                        = 3584 chains, max_new 6144, resume-safe (skips
                        already-generated (task, cell, sample) rows), atomic
                        checkpoint after every batch, (cell, seed)-homogeneous
                        batches (E9.1 batch-seeding contract), progress line
                        every 10 batches (= 10 tasks' worth at --batch 8, k=8)
                        → results/r3_strategy/full_gen.json
  full-analyse   (CPU)  strategy-entropy analysis: per-(cell, task) strategy
                        distributions + Shannon entropy (bits, labelled chains,
                        ≥2 required) via the FIXED pilot classifier; per-cell
                        value(=TRUE correctness) × strategy-entropy plane;
                        P-R2.1/2/3 (reframed onto the working diversity axis);
                        difficulty strata (CF-V); strategy×correctness
                        cross-tab (CF-U) → results/r3_strategy/{FULL_REPORT.md,
                        full_report.json}

Value axis is TRUE CORRECTNESS (computable golds) — the eval-50
completion-only caveat does not apply. The lexical strategy classifier is a
RANGE-FINDER only (CF-T): no scientific claim rests on which strategy a chain
used; FAMILIES/SPACES are frozen exactly as sealed pre-pilot.

AMENDMENT 2 (2026-07-13, design of the full run, before any full-run
generation; precedent: Amendment 1 / E10 Amendment 1 — instrument scope fixed
pre-analysis). The pilot verdict (RESULTS_LEDGER §B5) imposes three
requirements the sealed prereg text does not spell out; they are REGISTERED
here (and in R3_PILOT_PREREG.md Amendment 2) as part of the full design:
  1. DIFFICULTY STRATIFIED UPWARD (CF-V): pilot accuracy 0.896 = near ceiling
     (every parsed answer correct; headroom was T1/T2 truncation only). The
     full set therefore adds a HARD stratum: 4 easy + 4 hard instances per
     template, hard = larger parameters where arithmetic/strategy choice bites.
  2. k ≥ 5 samples/task/cell: G-R3.4 passed EXACTLY at the 0.25 bar at k=3 —
     too thin for an entropy axis. Full run uses k=8 (seeds 0–7).
  3. TEMPERATURE ARM: strategy entropy must be readable against decoding
     entropy → vanilla T sweep {0.3, 0.6, 0.9, 1.2} (= the R2 thermostat grid).
  4. Cap lesson: max_new ≥ 6144 everywhere (Amendment 1's uniform cap kept —
     3072 was truncation-killed by the model's own overthinking).

PREREG-SILENT CHOICES for the full run (minimal, documented here):
  * 64 tasks — inside the prereg's "~50–80"; 8 instances/template keeps every
    declared strategy space unchanged (classifier untouched). Full-set
    parameters are disjoint from the pilot's 16 instances.
  * Cells = the R2 prereg's shelved P-R2.1 sweep transplanted verbatim
    (thermostat T {0.3,0.6,0.9,1.2}; pump α {0.5,1.0,1.5} @ T=0.6): R2 folded
    into R3 explicitly so the frontier could run "with a working diversity
    axis"; inheriting its grid adds no new knobs. Pump mode defaults to
    "subtract" because that is what E9.1 T06 (= R2 knob A, reused by the R2
    analysis) actually ran (T06 predates --steer-mode; SteeredModel default =
    subtract). --steer-mode add exposes the E9.1b amplify sign as a NON-default
    exploratory arm only.
  * Strategy entropy = Shannon entropy in BITS over primary-strategy labels of
    a task's k chains, unclassified EXCLUDED (coverage reported separately;
    an unclassified-included sensitivity is also reported); None if <2
    labelled chains. Cell-level x = mean over tasks with defined entropy;
    y = accuracy over all rows (unparsed counts incorrect, pilot convention).
  * DECLARED FOLLOW-ONS, deliberately NOT in this harness: LLM-judge strategy
    labels under the R2.2 multi-annotator κ protocol (vs this lexical proxy),
    and the R3(ii) excursion signature at within-chain strategy switches (R0
    E-2 instruments + matched-position controls). Both consume full_gen.json.

Usage:
  python3 32_r3_strategy.py --stage tasks|generate|gate        # pilot (done)
  python3 32_r3_strategy.py --stage full-tasks
  python3 32_r3_strategy.py --stage full-generate --device auto --batch 8
  python3 32_r3_strategy.py --stage full-analyse
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("r3_strategy")

EVAL_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
GEN_SEEDS = [0, 1, 2]                                  # pilot seeds
SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."

# ── full-run constants (prereg-silent choices documented in the docstring) ────
FULL_K = 8                                             # samples/cell (seeds 0..7)
FULL_TEMPS = [0.3, 0.6, 0.9, 1.2]                      # thermostat knob (R2 grid)
FULL_ALPHAS = [0.5, 1.0, 1.5]                          # pump knob (R2 grid)
PUMP_T = 0.6                                           # pump runs at fixed T=0.6
PUMP_BEHAVIOUR = "backtracking"
PROGRESS_EVERY_BATCHES = 10


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="gate",
                    choices=["tasks", "generate", "gate",
                             "full-tasks", "full-generate", "full-analyse"])
    ap.add_argument("--tasks-file", default="data/r3_tasks_pilot.json")
    ap.add_argument("--full-tasks-file", default="data/r3_tasks_full.json")
    ap.add_argument("--out", default="results/r3_strategy")
    # Amendment 1 (2026-07-12): uniform cap 6144 (3072 truncation-killed batch 1).
    ap.add_argument("--max-new-tokens", type=int, default=6144)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    # full-run knobs
    ap.add_argument("--k", type=int, default=FULL_K,
                    help="samples per (task, cell) in full-generate (seeds 0..k-1)")
    ap.add_argument("--temps", type=float, nargs="+", default=FULL_TEMPS,
                    help="thermostat (vanilla) temperatures")
    ap.add_argument("--alphas", type=float, nargs="+", default=FULL_ALPHAS,
                    help="pump (bt single_direction) alpha levels @ T=0.6")
    ap.add_argument("--steer-mode", default="subtract", choices=["subtract", "add"],
                    help="pump sign; subtract = R2 knob A as executed (E9.1 T06)")
    ap.add_argument("--vectors-dir",
                    default="results/steering_vectors/R1-1.5B__E1_pooled",
                    help="E1-pooled steering vectors (pump arm)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                    help="device for full-generate (auto → cuda>mps>cpu)")
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
                             "max_new": args.max_new_tokens, **g})
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
        r["cap_hit"] = r["n_tokens"] >= r.get("max_new", args.max_new_tokens)

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


# ═══════════════════════ FULL R3 (post-gate) ═════════════════════════════════
# Prereg §"What the full R3 adds if the gates pass" + R2_FRONTIER_PREREG.md's
# knob definitions (R2 folded into R3). All prereg-silent choices are listed in
# the module docstring. Classifier (FAMILIES/SPACES) is FROZEN as sealed.

# ── stage: full-tasks (64 tasks, 4 easy + 4 hard per template, disjoint) ──────

def _sgn(x: int) -> str:
    """'+'/'−' joiner using the pilot's U+2212 minus."""
    return "+" if x >= 0 else "−"


def _lin_eq(a: int, b: int, c: int) -> str:
    """'ax ± |b|y = c' with the pilot's sign formatting (a > 0 by construction)."""
    rhs = str(c) if c >= 0 else f"−{abs(c)}"
    return f"{a}x {_sgn(b)} {abs(b)}y = {rhs}"


def _quad_eq(p: int, q: int) -> str:
    """'x² − px + q' rendered with resolved signs (coefficient of x is −p)."""
    xterm = f"{_sgn(-p)} {abs(p)}x" if p != 0 else ""
    cterm = f"{_sgn(q)} {abs(q)}"
    return f"x² {xterm} {cterm}".replace("  ", " ")


#: Full-set instance parameters: 4 easy + 4 hard per template, all DISJOINT
#: from the pilot's 16 instances (pilot: T1 n∈{10,12}; T2 n∈{15,20};
#: T3 (6,5),(7,6); T4 two fixed systems; T5 n∈{20,24}; T6 (7,2026),(3,2025);
#: T7 n∈{3,4}; T8 (7,5),(9,14)). Golds computed by construction below.
FULL_PARAMS: dict[str, dict[str, list]] = {
    "T1": {"easy": [8, 9, 11, 13],            "hard": [14, 16, 18, 20]},
    "T2": {"easy": [10, 12, 18, 24],          "hard": [30, 36, 45, 60]},
    "T3": {"easy": [(4, 4), (5, 5), (6, 4), (7, 5)],
           "hard": [(8, 7), (9, 8), (10, 9), (10, 10)]},
    # T4 entries: (a, b, d, e, x, y) — c = ax+by and f = dx+ey are derived, so
    # the gold xy is correct by construction; all determinants are nonzero.
    "T4": {"easy": [(2, 3, 1, -1, 5, 3), (5, 2, 3, 1, 4, 7),
                    (3, 4, 2, -3, 6, 2), (7, 2, 2, 5, 3, 8)],
           "hard": [(6, -5, 4, 3, 7, -2), (9, 4, 3, -7, -3, 8),
                    (11, 7, 5, -6, 8, -4), (8, 13, 12, -5, 9, 6)]},
    "T5": {"easy": [15, 18, 22, 26],          "hard": [35, 40, 55, 75]},
    "T6": {"easy": [(2, 2029), (8, 2030), (4, 2027), (9, 2026)],
           "hard": [(17, 2026), (23, 1999), (12, 2045), (38, 2027)]},
    "T7": {"easy": [2, 5, 6, 7],              "hard": [8, 9, 10, 12]},
    "T8": {"easy": [(8, 3), (11, 10), (12, 20), (5, 2)],
           "hard": [(13, -22), (-9, 14), (23, 56), (31, -47)]},
}


def build_full_tasks() -> list[dict]:
    """64 tasks (8 templates × 8 instances), golds computable by construction,
    per-template declared strategy spaces UNCHANGED from the sealed pilot."""
    T: list[dict] = []

    def add(tpl: str, difficulty: str, prompt: str, gold: int, params) -> None:
        T.append({"task_id": f"{tpl}_f{sum(t['template'] == tpl for t in T)}",
                  "template": tpl, "difficulty": difficulty,
                  "prompt": prompt + SUFFIX, "gold": str(gold),
                  "params": params, "strategy_space": SPACES[tpl]})

    for diff in ("easy", "hard"):
        for n in FULL_PARAMS["T1"][diff]:                    # T1 — Fib(n+1)
            add("T1", diff, f"In how many ways can a 2×{n} board be completely "
                            f"tiled by 1×2 dominoes?", _fib(n + 1), {"n": n})
        for n in FULL_PARAMS["T2"][diff]:                    # T2 — n(n+1)(n+2)/3
            add("T2", diff, f"Evaluate the sum 1·2 + 2·3 + 3·4 + … + {n}·{n + 1}.",
                n * (n + 1) * (n + 2) // 3, {"n": n})
        for m, n in FULL_PARAMS["T3"][diff]:                 # T3 — C(m+n, m)
            add("T3", diff, f"How many paths are there from the bottom-left corner "
                            f"to the top-right corner of a {m}×{n} grid of unit "
                            f"squares, moving only right or up along the grid "
                            f"lines?", comb(m + n, m), {"m": m, "n": n})
        for a, b, d, e, x, y in FULL_PARAMS["T4"][diff]:     # T4 — xy at solution
            c, f = a * x + b * y, d * x + e * y
            add("T4", diff, f"Let x and y be real numbers with {_lin_eq(a, b, c)} "
                            f"and {_lin_eq(d, e, f)}. Find the value of xy.",
                x * y, {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f,
                        "x": x, "y": y})
        for n in FULL_PARAMS["T5"][diff]:                    # T5 — n(n+1)(2n+1)/6
            add("T5", diff, f"Evaluate 1² + 2² + 3² + … + {n}².",
                n * (n + 1) * (2 * n + 1) // 6, {"n": n})
        for a, b in FULL_PARAMS["T6"][diff]:                 # T6 — last digit a^b
            add("T6", diff, f"What is the last digit of {a}^{b}?",
                pow(a, b, 10), {"a": a, "b": b})
        for n in FULL_PARAMS["T7"][diff]:                    # T7 — 6^n − 5^n
            add("T7", diff, f"A fair six-sided die is rolled {n} times, and the "
                            f"sequence of results is recorded. How many distinct "
                            f"sequences contain at least one 6?",
                6 ** n - 5 ** n, {"n": n})
        for p, q in FULL_PARAMS["T8"][diff]:                 # T8 — p² − 2q
            add("T8", diff, f"Let r and s be the roots of {_quad_eq(p, q)} = 0. "
                            f"Find r² + s².", p * p - 2 * q, {"p": p, "q": q})
    return T


def stage_full_tasks(args) -> None:
    tasks = build_full_tasks()
    pilot_prompts = {t["prompt"] for t in build_tasks()}
    overlap = [t["task_id"] for t in tasks if t["prompt"] in pilot_prompts]
    if overlap:                                       # guard, should be impossible
        raise ValueError(f"full tasks overlap pilot instances: {overlap}")
    Path(args.full_tasks_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.full_tasks_file).write_text(json.dumps(tasks, indent=1,
                                                     ensure_ascii=False))
    n_e = sum(t["difficulty"] == "easy" for t in tasks)
    logger.info(f"full-tasks: wrote {len(tasks)} tasks "
                f"({len(set(t['template'] for t in tasks))} templates, "
                f"{n_e} easy / {len(tasks) - n_e} hard) → {args.full_tasks_file}")


# ── stage: full-generate (pod; thermostat + pump cells, resume-safe) ──────────

def build_cells(temps: list[float], alphas: list[float], steer_mode: str) -> list[dict]:
    """The 7 default cells: vanilla at each T (thermostat; T=0.6 doubles as the
    pump's α=0 anchor) + bt single_direction at each α @ T=0.6 (pump)."""
    cells = [{"cell": f"vanilla_T{t:g}", "method": "vanilla", "behaviour": "shared",
              "mode": None, "layer": None, "temperature": float(t), "alpha": 0.0}
             for t in temps]
    cells += [{"cell": f"pump_{steer_mode}_a{a:g}", "method": "single_direction",
               "behaviour": PUMP_BEHAVIOUR, "mode": steer_mode, "layer": None,
               "temperature": PUMP_T, "alpha": float(a)} for a in alphas]
    return cells


def _build_full_batches(tasks: list[dict], cells: list[dict], seeds: list[int],
                        done: set, batch_size: int) -> list[tuple]:
    """(cell, seed, [tasks]) batches. Every batch is (cell, seed)-HOMOGENEOUS —
    the E9.1 batch-seeding contract — and stays so under resume holes (pending
    tasks are regrouped per (cell, seed), never across)."""
    batches = []
    for cell in cells:
        for s in seeds:
            pend = [t for t in tasks if (t["task_id"], cell["cell"], s) not in done]
            for j in range(0, len(pend), batch_size):
                batches.append((cell, s, pend[j:j + batch_size]))
    return batches


def _atomic_write_json(path: Path, obj) -> None:
    """Checkpoint write that survives a deadman kill mid-write (tmp + rename)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _load_pump_vector(vectors_dir: str) -> tuple[np.ndarray, int]:
    """E1-pooled bt single_direction vector + its layer (17 per metadata)."""
    vd = Path(vectors_dir)
    vec_path = vd / f"{PUMP_BEHAVIOUR}_single.npy"
    if not vec_path.exists():
        raise FileNotFoundError(
            f"pump vector missing: {vec_path} — sync results/steering_vectors "
            f"before full-generate (or drop the pump cells via --alphas)")
    vec = np.load(vec_path)
    layer = 17
    meta = vd / "metadata.json"
    if meta.exists():
        layer = int(json.loads(meta.read_text())
                    .get("_provenance", {}).get("layers", {})
                    .get(PUMP_BEHAVIOUR, layer))
    return vec, layer


def stage_full_generate(args) -> None:
    import importlib.util
    import time
    from src.chain_gen import load_model, format_prompt
    from src.steered_inference import SteeredModel
    from tqdm import tqdm

    spec = importlib.util.spec_from_file_location("r1c", "30_r1_compression.py")
    r1c = importlib.util.module_from_spec(spec); spec.loader.exec_module(r1c)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "full_gen.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["task_id"], r["cell"], r["sample"]) for r in rows}

    tasks = json.loads(Path(args.full_tasks_file).read_text())
    cells = build_cells(args.temps, args.alphas, args.steer_mode)
    seeds = list(range(args.k))
    if args.smoke:                                   # tiny end-to-end: 2 tasks ×
        tasks = tasks[:2]                            # (anchor + 1 pump cell) × 2 seeds
        anchor = f"vanilla_T{PUMP_T:g}"
        pumps = [c for c in cells if c["method"] != "vanilla"]
        cells = [c for c in cells if c["cell"] == anchor] + pumps[:1]
        seeds = seeds[:2]

    batches = _build_full_batches(tasks, cells, seeds, done, args.batch)
    if args.limit:
        kept, n = [], 0
        for c, s, b in batches:
            if n >= args.limit:
                break
            b = b[:args.limit - n]; n += len(b); kept.append((c, s, b))
        batches = kept
    total = sum(len(b) for _, _, b in batches)
    logger.info(f"full-generate: {total} chains to run in {len(batches)} batches "
                f"({len(done)} done; k={len(seeds)}, "
                f"cells={[c['cell'] for c in cells]})")
    if not batches:
        return

    device_map = "auto" if args.device == "auto" else {"": args.device}
    model, tokenizer = load_model(EVAL_MODEL, dtype="float16", device_map=device_map)
    model.eval()

    steered: dict[str, SteeredModel] = {}
    pump_cells = [c for c in cells if c["method"] != "vanilla"]
    if pump_cells:
        vec, layer = _load_pump_vector(args.vectors_dir)
        for c in pump_cells:
            c["layer"] = layer
            steered[c["cell"]] = SteeredModel(model, tokenizer, vec, layer,
                                              alpha=c["alpha"], mode=c["mode"])
        logger.info(f"full-generate: pump = {PUMP_BEHAVIOUR} single_direction "
                    f"@L{layer} mode={args.steer_mode} (vector {args.vectors_dir})")

    t0, n_run, tasks_seen = time.time(), 0, set()
    for bi, (cell, s, batch) in enumerate(tqdm(batches, desc="r3-full")):
        try:
            seed = 1000 * (s + 1) + bi               # per-(batch, seed) contract
            if cell["method"] == "vanilla":
                prompts = [format_prompt(tokenizer, t["prompt"]) for t in batch]
                gen = r1c._hf_generate_batch(model, tokenizer, prompts,
                                             max_new=args.max_new_tokens,
                                             temperature=cell["temperature"],
                                             seed=seed)
            else:                                    # pump: hooked batched generate
                gen = steered[cell["cell"]].generate_batch(
                    [t["prompt"] for t in batch],
                    max_new_tokens=args.max_new_tokens,
                    temperature=cell["temperature"], seed=seed)
            for t, g in zip(batch, gen):
                row = {"task_id": t["task_id"], "template": t["template"],
                       "difficulty": t["difficulty"], "gold": t["gold"],
                       "cell": cell["cell"], "method": cell["method"],
                       "behaviour": cell["behaviour"], "mode": cell["mode"],
                       "layer": cell["layer"], "alpha": cell["alpha"],
                       "temperature": cell["temperature"], "sample": s,
                       "seed": seed, "max_new": args.max_new_tokens,
                       "chain": g["chain"], "n_tokens": g["n_tokens"]}
                if cell["method"] != "vanilla":      # diagnostic only
                    row["mean_abs_proj"] = g.get("mean_abs_proj")
                rows.append(row)
                n_run += 1; tasks_seen.add(t["task_id"])
        except Exception as e:                       # fail-soft; resume re-runs it
            logger.warning(f"full-generate FAILED batch@{bi} "
                           f"[{cell['cell']} s{s}]: {e}")
        finally:
            r1c.r0._clear_accel_cache()
        _atomic_write_json(path, rows)
        if (bi + 1) % PROGRESS_EVERY_BATCHES == 0 or bi == len(batches) - 1:
            el = time.time() - t0
            eta = (total - n_run) * el / n_run / 3600 if n_run else float("inf")
            logger.info(f"progress: {n_run}/{total} chains "
                        f"({len(tasks_seen)} tasks touched) | "
                        f"{el / 3600:.2f}h elapsed | ETA {eta:.2f}h | "
                        f"at {cell['cell']} s{s}")
    logger.info(f"full-generate: done ({len(rows)} rows in {path})")


# ── stage: full-analyse (strategy-entropy measures + the frontier) ────────────

def strategy_entropy(labels: list[str], include_unclassified: bool = False):
    """Shannon entropy (BITS) of a task's primary-strategy distribution.
    Unclassified chains are excluded unless include_unclassified; None if <2
    (labels can't define a distribution)."""
    labs = [l for l in labels if include_unclassified or l != "unclassified"]
    if len(labs) < 2:
        return None
    p = np.array(list(Counter(labs).values()), dtype=float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def _ngrams(text: str, n: int = 4) -> set:
    toks = text.split()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _fourgram_diversity(chains: list[str]):
    """1 − mean pairwise 4-gram Jaccard (verbatim from 31_r2_frontier.py; kept
    to DOCUMENT the saturated token-level axis next to the working one)."""
    if len(chains) < 2:
        return None
    grams = [_ngrams(c) for c in chains]
    jac = [len(a & b) / max(len(a | b), 1) for a, b in combinations(grams, 2)]
    return 1.0 - float(np.mean(jac))


def _answer_div(chains: list[str]):
    """Distinct normalised \\boxed answers / n answered (≥2), else None."""
    ans = [a for c in chains if (a := normalise(_boxed_answer(c))) is not None]
    if len(ans) < 2:
        return None
    return len(set(ans)) / len(ans)


def _annotate_full(rows: list[dict], repetition_rate) -> None:
    for r in rows:
        r["answer"] = normalise(_boxed_answer(r["chain"]))
        r["correct"] = r["answer"] is not None and r["answer"] == r["gold"]
        r["strategy"] = primary_strategy(r["chain"], r["template"])
        r["cap_hit"] = r["n_tokens"] >= r.get("max_new", 6144)
        r["collapsed"] = repetition_rate(r["chain"]) > 0.8


def _mean_or_none(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _full_cell_metrics(cell_rows: list[dict]) -> tuple[dict, dict]:
    """(summary, per_task) for one cell of pre-annotated rows."""
    by_task: dict[str, list] = {}
    for r in cell_rows:
        by_task.setdefault(r["task_id"], []).append(r)
    per_task = {}
    for tid, rs in sorted(by_task.items()):
        labels = [r["strategy"] for r in rs]
        space = SPACES[rs[0]["template"]]
        H = strategy_entropy(labels)
        per_task[tid] = {
            "template": rs[0]["template"], "difficulty": rs[0]["difficulty"],
            "n": len(rs), "labels": dict(Counter(labels)),
            "strategy_entropy": H,
            "strategy_entropy_norm": (H / float(np.log2(len(space)))
                                      if H is not None and len(space) > 1 else None),
            "strategy_entropy_incl_unclassified":
                strategy_entropy(labels, include_unclassified=True),
            "n_distinct_labelled": len(set(labels) - {"unclassified"}),
            "accuracy": float(np.mean([r["correct"] for r in rs])),
            "answer_diversity": _answer_div([r["chain"] for r in rs]),
            "fourgram_diversity": _fourgram_diversity([r["chain"] for r in rs]),
        }
    answered = [r for r in cell_rows if r["answer"] is not None]
    uncol = [r for r in cell_rows if not r["collapsed"]]
    summary = {
        "n_rows": len(cell_rows), "n_tasks": len(by_task),
        "parse_rate": float(np.mean([r["answer"] is not None for r in cell_rows])),
        "accuracy": float(np.mean([r["correct"] for r in cell_rows])),
        "accuracy_uncollapsed": (float(np.mean([r["correct"] for r in uncol]))
                                 if uncol else None),
        "coverage_answered": (float(np.mean([r["strategy"] != "unclassified"
                                             for r in answered]))
                              if answered else None),
        "strategy_entropy": _mean_or_none([m["strategy_entropy"]
                                           for m in per_task.values()]),
        "strategy_entropy_n": sum(m["strategy_entropy"] is not None
                                  for m in per_task.values()),
        "strategy_entropy_norm": _mean_or_none([m["strategy_entropy_norm"]
                                                for m in per_task.values()]),
        "strategy_entropy_incl_unclassified":
            _mean_or_none([m["strategy_entropy_incl_unclassified"]
                           for m in per_task.values()]),
        "multi_strategy_frac": float(np.mean([m["n_distinct_labelled"] >= 2
                                              for m in per_task.values()])),
        "answer_diversity": _mean_or_none([m["answer_diversity"]
                                           for m in per_task.values()]),
        "fourgram_diversity": _mean_or_none([m["fourgram_diversity"]
                                             for m in per_task.values()]),
        "collapse": float(np.mean([r["collapsed"] for r in cell_rows])),
        "cap_hit": float(np.mean([r["cap_hit"] for r in cell_rows])),
        "mean_tokens": float(np.mean([r["n_tokens"] for r in cell_rows])),
    }
    return summary, per_task


def _frontier_points(cells: dict[str, dict], keys: list[tuple],
                     y: str = "accuracy") -> list[dict]:
    """[(level, cell_name)] → frontier points sorted by strategy entropy,
    dropping cells whose entropy (or y) is undefined."""
    pts = []
    for level, name in keys:
        m = cells.get(name)
        if m and m["strategy_entropy"] is not None and m.get(y) is not None:
            pts.append({"level": level, "cell": name,
                        "strategy_entropy": m["strategy_entropy"], "y": m[y]})
    return sorted(pts, key=lambda p: p["strategy_entropy"])


def _matched_entropy_gap(pump: list[dict], thermo: list[dict],
                         n_grid: int = 25) -> dict:
    """P-R2.1 reframed: y(pump) − y(thermo) interpolated onto a common
    strategy-entropy grid over the overlapping range (logic verbatim from
    31_r2_frontier.py, axes swapped onto the working substrate)."""
    if len(pump) < 2 or len(thermo) < 2:
        return {"status": "insufficient points (need ≥2 per knob with entropy)"}
    lo = max(pump[0]["strategy_entropy"], thermo[0]["strategy_entropy"])
    hi = min(pump[-1]["strategy_entropy"], thermo[-1]["strategy_entropy"])
    if not (hi > lo):
        return {"status": "no strategy-entropy overlap between knobs — widen a "
                          "grid before claiming anything (R2 kill criterion)",
                "pump_range": [pump[0]["strategy_entropy"],
                               pump[-1]["strategy_entropy"]],
                "thermo_range": [thermo[0]["strategy_entropy"],
                                 thermo[-1]["strategy_entropy"]]}
    grid = np.linspace(lo, hi, n_grid)

    def interp(pts):
        return np.interp(grid, [p["strategy_entropy"] for p in pts],
                         [p["y"] for p in pts])

    gaps = interp(pump) - interp(thermo)
    rng = np.random.default_rng(0)
    boot = [float(np.mean(rng.choice(gaps, len(gaps), replace=True)))
            for _ in range(2000)]
    lo_ci, hi_ci = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    return {
        "overlap_entropy": [float(lo), float(hi)],
        "mean_gap_pump_minus_thermo": float(np.mean(gaps)),
        "ci95": [float(lo_ci), float(hi_ci)],
        "grid_points_pump_higher": f"{int((gaps > 0).sum())}/{n_grid}",
        "verdict": ("P-R2.1 (reframed) SUPPORTED — pump retains more value at "
                    "matched strategy entropy" if lo_ci > 0 else
                    "P-R2.1 (reframed) REFUTED — thermostat retains more value "
                    "at matched strategy entropy" if hi_ci < 0 else
                    "P-R2.1 (reframed) INCONCLUSIVE — gap CI straddles 0"),
    }


def _monotone_pts(pts: list[dict]) -> dict:
    from scipy.stats import spearmanr
    if len(pts) < 3:
        return {"status": "too few points", "n": len(pts)}
    rho, p = spearmanr([q["strategy_entropy"] for q in pts],
                       [q["y"] for q in pts])
    return {"spearman_rho": round(float(rho), 3), "p": float(p), "n": len(pts)}


def stage_full_analyse(args) -> None:
    from src.evaluation import repetition_rate

    out = Path(args.out)
    rows = json.loads((out / "full_gen.json").read_text())
    _annotate_full(rows, repetition_rate)
    anchor = f"vanilla_T{PUMP_T:g}"

    by_cell: dict[str, list] = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    cells, per_task = {}, {}
    for name, rs in sorted(by_cell.items()):
        cells[name], per_task[name] = _full_cell_metrics(rs)

    # value × strategy-entropy plane (both knobs share the T=0.6/α=0 anchor)
    pump_keys = ([(0.0, anchor)] if anchor in cells else []) + \
        sorted((by_cell[n][0]["alpha"], n) for n in cells if n.startswith("pump_"))
    thermo_keys = sorted((by_cell[n][0]["temperature"], n)
                         for n in cells if n.startswith("vanilla_"))
    pump_front = _frontier_points(cells, pump_keys)
    thermo_front = _frontier_points(cells, thermo_keys)
    # P-R2.3 (reframed) secondary endpoint: same gap, collapse-excluded value
    pump_front_u = _frontier_points(cells, pump_keys, y="accuracy_uncollapsed")
    thermo_front_u = _frontier_points(cells, thermo_keys, y="accuracy_uncollapsed")

    strata: dict[str, dict] = {}
    for name, rs in sorted(by_cell.items()):
        strata[name] = {}
        for diff in ("easy", "hard"):
            sub = [r for r in rs if r["difficulty"] == diff]
            if not sub:
                continue
            ents = [m["strategy_entropy"] for tid, m in per_task[name].items()
                    if m["difficulty"] == diff]
            strata[name][diff] = {
                "n": len(sub),
                "parse_rate": float(np.mean([r["answer"] is not None for r in sub])),
                "accuracy": float(np.mean([r["correct"] for r in sub])),
                "strategy_entropy": _mean_or_none(ents),
                "cap_hit": float(np.mean([r["cap_hit"] for r in sub])),
            }

    xtab: dict[str, dict] = {}                       # CF-U visibility (pooled)
    for r in rows:
        cell_x = xtab.setdefault(r["template"], {})
        st = cell_x.setdefault(r["strategy"], {"n": 0, "n_correct": 0})
        st["n"] += 1
        st["n_correct"] += int(r["correct"])
    for tpl in xtab:
        for st in xtab[tpl].values():
            st["accuracy"] = round(st["n_correct"] / st["n"], 3)

    anchor_m = cells.get(anchor)
    report = {
        "prereg": "R3_PILOT_PREREG.md §'What the full R3 adds' + "
                  "R2_FRONTIER_PREREG.md knob definitions (R2 folded into R3)",
        "definitions": {
            "value": "TRUE correctness (computable golds; unparsed = incorrect)",
            "strategy_entropy": "Shannon entropy (bits) of primary-strategy "
                                "labels over a task's samples, unclassified "
                                "excluded, None if <2 labelled; cell value = "
                                "mean over tasks with defined entropy",
            "classifier": "FROZEN pilot lexical classifier — RANGE-FINDER only "
                          "(CF-T); judged labels are a declared follow-on",
            "collapse": "house 4-gram repetition_rate > 0.8",
        },
        "n_rows": len(rows), "n_tasks": len({r["task_id"] for r in rows}),
        "cells": cells,
        "substrate_diagnostics_anchor": (None if anchor_m is None else {
            "cell": anchor,
            "parse_rate": anchor_m["parse_rate"],
            "accuracy": anchor_m["accuracy"],
            "coverage_answered": anchor_m["coverage_answered"],
            "multi_strategy_frac": anchor_m["multi_strategy_frac"],
            "note": "pilot-gate analogues at full scale (informational — the "
                    "gates themselves were the pilot's, already passed)",
        }),
        "pump_frontier": pump_front, "thermo_frontier": thermo_front,
        "P_R2_1_reframed": _matched_entropy_gap(pump_front, thermo_front),
        "P_R2_2_reframed": {"pump": _monotone_pts(pump_front),
                            "thermostat": _monotone_pts(thermo_front)},
        "P_R2_3_reframed": {
            "collapse_by_pump_level": {n: cells[n]["collapse"]
                                       for _, n in pump_keys if n in cells},
            "collapse_by_thermo_level": {n: cells[n]["collapse"]
                                         for _, n in thermo_keys if n in cells},
            "matched_gap_uncollapsed":
                _matched_entropy_gap(pump_front_u, thermo_front_u),
        },
        "difficulty_strata": strata,
        "strategy_correctness_xtab": xtab,
        "per_task": per_task,
    }
    (out / "full_report.json").write_text(json.dumps(report, indent=1))
    _write_full_md(out, report)
    logger.info(f"full-analyse: wrote {out}/full_report.json + FULL_REPORT.md "
                f"({len(rows)} rows, {len(cells)} cells; pump "
                f"{len(pump_front)} pts / thermostat {len(thermo_front)} pts)")


def _write_full_md(out: Path, r: dict) -> None:
    def f(x, p=3):
        return "—" if x is None else format(x, f".{p}f")

    md = ["# R3 FULL — strategy entropy on the multi-solution family\n",
          f"{r['n_rows']} chains / {r['n_tasks']} tasks / {len(r['cells'])} cells. "
          f"Prereg: {r['prereg']}\n",
          f"Value = {r['definitions']['value']}. "
          f"Strategy entropy = {r['definitions']['strategy_entropy']}. "
          f"Classifier: {r['definitions']['classifier']}.\n"]

    sd = r.get("substrate_diagnostics_anchor")
    if sd:
        md += ["## Substrate diagnostics at the anchor cell "
               "(pilot-gate analogues, informational)\n",
               f"- parse {f(sd['parse_rate'])} | accuracy {f(sd['accuracy'])} | "
               f"coverage {f(sd['coverage_answered'])} | multi-strategy tasks "
               f"{f(sd['multi_strategy_frac'])}\n"]

    md += ["## Cells\n",
           "| cell | rows | acc | acc(uncol) | parse | strat-H (bits) | H-norm "
           "| multi-strat | ans-div | 4gram-div | collapse | cap-hit | mean tok |",
           "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name, m in r["cells"].items():
        md.append(f"| {name} | {m['n_rows']} | {f(m['accuracy'])} "
                  f"| {f(m['accuracy_uncollapsed'])} | {f(m['parse_rate'],2)} "
                  f"| {f(m['strategy_entropy'])} ({m['strategy_entropy_n']}) "
                  f"| {f(m['strategy_entropy_norm'])} "
                  f"| {f(m['multi_strategy_frac'],2)} | {f(m['answer_diversity'])} "
                  f"| {f(m['fourgram_diversity'])} | {f(m['collapse'],2)} "
                  f"| {f(m['cap_hit'],2)} | {f(m['mean_tokens'],0)} |")

    p1 = r["P_R2_1_reframed"]
    md += ["\n## Value × strategy-entropy plane\n",
           f"- pump frontier: {[(q['level'], round(q['strategy_entropy'], 3), round(q['y'], 3)) for q in r['pump_frontier']]}",
           f"- thermostat frontier: {[(q['level'], round(q['strategy_entropy'], 3), round(q['y'], 3)) for q in r['thermo_frontier']]}",
           "\n### P-R2.1 (reframed) — pump vs thermostat at matched strategy entropy\n"]
    if "verdict" in p1:
        md.append(f"Overlap H {p1['overlap_entropy']}; mean accuracy gap "
                  f"(pump−thermo) **{p1['mean_gap_pump_minus_thermo']:+.3f}** "
                  f"CI95 {p1['ci95']}; pump higher at "
                  f"{p1['grid_points_pump_higher']} grid pts.\n\n**{p1['verdict']}**\n")
    else:
        md.append(f"_{p1['status']}_\n")
    p2 = r["P_R2_2_reframed"]
    md += ["\n### P-R2.2 (reframed) — value falls as strategy entropy rises\n",
           f"- pump: {p2['pump']}\n- thermostat: {p2['thermostat']}\n"]
    p3 = r["P_R2_3_reframed"]
    md += ["\n### P-R2.3 (reframed) — collapse asymmetry (secondary)\n",
           f"- collapse by pump level: { {k: round(v, 3) for k, v in p3['collapse_by_pump_level'].items()} }",
           f"- collapse by thermo level: { {k: round(v, 3) for k, v in p3['collapse_by_thermo_level'].items()} }",
           f"- collapse-excluded matched gap: "
           f"{p3['matched_gap_uncollapsed'].get('verdict', p3['matched_gap_uncollapsed'].get('status'))}\n"]

    md += ["\n## Difficulty strata (CF-V)\n",
           "| cell | stratum | n | parse | acc | strat-H | cap-hit |",
           "|---|---|--:|--:|--:|--:|--:|"]
    for name, ds in r["difficulty_strata"].items():
        for diff, m in ds.items():
            md.append(f"| {name} | {diff} | {m['n']} | {f(m['parse_rate'],2)} "
                      f"| {f(m['accuracy'])} | {f(m['strategy_entropy'])} "
                      f"| {f(m['cap_hit'],2)} |")

    md += ["\n## Strategy × correctness (CF-U, pooled over cells)\n",
           "| tpl | strategy | n | acc |", "|---|---|--:|--:|"]
    for tpl, sts in sorted(r["strategy_correctness_xtab"].items()):
        for st, m in sorted(sts.items()):
            md.append(f"| {tpl} | {st} | {m['n']} | {m['accuracy']} |")

    md += ["\n## Declared follow-ons NOT in this run\n",
           "- LLM-judge strategy labels under the R2.2 multi-annotator κ "
           "protocol, compared against this lexical proxy (CF-T upgrade).",
           "- R3(ii): excursion signature at within-chain strategy switches "
           "(R0 E-2 instruments, matched-position controls) on these chains.\n"]
    (out / "FULL_REPORT.md").write_text("\n".join(md) + "\n")


def main() -> None:
    args = parse_args()
    {"tasks": stage_tasks, "generate": stage_generate, "gate": stage_gate,
     "full-tasks": stage_full_tasks, "full-generate": stage_full_generate,
     "full-analyse": stage_full_analyse}[args.stage](args)


if __name__ == "__main__":
    main()
