#!/usr/bin/env python3
"""
R1 — Strata-differential compression across post-training (creativity–entropy
rung 1 + RL.a). Pre-registration: R1_COMPRESSION_PREREG.md (sealed 2026-07-12).

Arms (the post-training ladder at 1.5B, all public):
  r1         deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B   (reference; matched-text
             metrics REUSED from R0 ent shards + E9.0 state shards — never recomputed)
  deepscaler agentica-org/DeepScaleR-1.5B-Preview        (GRPO-RLVR on top of r1;
             PRIMARY contrast, matched-ids tier)
  star1      UCSC-VLAA/STAR1-R1-Distill-1.5B             (safety-SFT control, matched-ids)
  qwenmath   Qwen/Qwen2.5-Math-1.5B                      (base anchor, matched-TEXT tier —
             tokenizer gate failed; own grid, summary comparisons only, MATH-primary)

Stages (checkpointed; --limit bounds items/process for the MPS restart loop):
  gate       (CPU)  tokenizer-compatibility gate per arm → gate_tokenizer.json
  extract    (GPU)  teacher-forced E-1 + E-2@L17 on the R0 200-chain sample (same
                    full_texts for every arm; E9.0 grid; matched-ids arms assert
                    center-grid equality with the E9.0 state shard)
  generate   (GPU)  own-generation arm: 50 E9.1 eval tasks × (greedy + 3×T0.6),
                    max_new 8192; per-(batch,seed) reproducibility contract
  score-gen  (GPU)  each arm re-scores its OWN generations (on-policy E-1 + PR@17;
                    the CF-M cross-check)
  analyse    (CPU)  P-R1.1–P-R1.6 verdicts + kills → report.json + REPORT.md

Annotation-free behaviour proxies (documented here, sealed): backtracking-cue
rate = case-insensitive matches of BT_CUE_RE per 1k whitespace tokens; boxed =
"\\boxed" present; collapse = house 4-gram repetition_rate > 0.8.

Usage:
  python3 30_r1_compression.py --stage gate
  python3 30_r1_compression.py --stage all --models deepscaler star1 qwenmath
  python3 30_r1_compression.py --stage extract --models star1 --limit 20
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
from itertools import combinations
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("r1_compression")

r0 = importlib.import_module("29_r0_entropy_ladder")   # entropy/window/cache helpers

MODEL_IDS = {
    "r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepscaler": "agentica-org/DeepScaleR-1.5B-Preview",
    "star1": "UCSC-VLAA/STAR1-R1-Distill-1.5B",
    "qwenmath": "Qwen/Qwen2.5-Math-1.5B",
}
MATCHED_IDS_ARMS = ("deepscaler", "star1")             # gate-verified byte-identical to r1
GEN_SEEDS = (0, 1, 2)                                  # T06 samples; greedy is sample=-1
BT_CUE_RE = re.compile(
    r"\b(wait|alternatively|hmm|let me (?:reconsider|re-?check|double-?check)"
    r"|on second thought)\b", re.IGNORECASE)
QWENMATH_SYSTEM = ("Please reason step by step, and put your final answer "
                   "within \\boxed{}.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "gate", "extract", "generate", "score-gen", "analyse"])
    ap.add_argument("--models", nargs="+", default=["deepscaler", "star1", "qwenmath"],
                    choices=list(MODEL_IDS), help="arms to process (r1 = reuse-only "
                    "for extract; include r1 for generate/score-gen)")
    ap.add_argument("--chains", default="data/chains_R1-1.5B.json")
    ap.add_argument("--r0-out", default="results/r0_entropy_ladder/R1-1.5B")
    ap.add_argument("--state-shards", default="results/loop_geometry/R1-1.5B/shards")
    ap.add_argument("--eval-ids", default="results/eval/R1-1.5B__E9_1_greedy/eval_task_ids.json")
    ap.add_argument("--out", default="results/r1_compression")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--max-seq-tokens", type=int, default=10240)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--lm-head-chunk", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="2 extract chains + 2 gen tasks (greedy only), first arm only")
    return ap.parse_args()


# ── stage: gate ───────────────────────────────────────────────────────────────

def stage_gate(args) -> None:
    from transformers import AutoTokenizer
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chains = json.loads(Path(args.chains).read_text())[:20]
    ref = AutoTokenizer.from_pretrained(MODEL_IDS["r1"])
    res = {}
    for arm, mid in MODEL_IDS.items():
        if arm == "r1":
            continue
        t = AutoTokenizer.from_pretrained(mid)
        res[arm] = {
            "vocab_identical": t.get_vocab() == ref.get_vocab(),
            "chain_body_ids_identical": all(
                t(r["chain"], add_special_tokens=False).input_ids ==
                ref(r["chain"], add_special_tokens=False).input_ids for r in chains),
            "tier": "matched-ids" if arm in MATCHED_IDS_ARMS else "matched-text",
        }
    (out / "gate_tokenizer.json").write_text(json.dumps(res, indent=1))
    logger.info(f"gate: {json.dumps(res)}")
    for arm in MATCHED_IDS_ARMS:
        if arm in res and not res[arm]["vocab_identical"]:
            raise SystemExit(f"gate FAILED: {arm} declared matched-ids but vocab differs")


# ── stage: extract (teacher-forced matched-text ladder metrics) ───────────────

def stage_extract(args, arm: str) -> None:
    import torch
    from src.chain_gen import load_model
    from src.hooks import ActivationCache
    from src.loop_geometry import windowed_state_metrics

    if arm == "r1":
        logger.info("extract: r1 is reuse-only (R0 ent shards + E9.0 state shards)")
        return
    out = Path(args.out) / arm / "ent_shards"
    out.mkdir(parents=True, exist_ok=True)
    sample = json.loads((Path(args.r0_out) / "sample.json").read_text())

    todo = [t for t in sample["loop"] + sample["clean"]
            if not (out / f"{t}.npz").exists()]
    if args.smoke:
        todo = todo[:2]
    if args.limit:
        todo = todo[:args.limit]
    logger.info(f"extract[{arm}]: {len(todo)} chains to run")
    if not todo:
        return

    chains = {r["task_id"]: r for r in json.loads(Path(args.chains).read_text())}
    model, tokenizer = load_model(MODEL_IDS[arm], dtype="float16")
    if args.fp32:
        model = model.float()
    model.eval()
    device = next(model.parameters()).device
    L = args.layer

    from tqdm import tqdm
    n_fail = 0
    for tid in tqdm(todo, desc=f"extract[{arm}]"):
        try:
            with np.load(Path(args.state_shards) / f"{tid}.npz") as z:
                ref_centers = z["centers"].copy()
                is_loop = int(z["is_loop"])
                onset = int(z["onset_tok_gen"])

            rec = chains[tid]
            prompt = rec["prompt"]
            full_text = rec.get("full_text") or (prompt + rec["chain"])
            enc, gen_tok = r0._tokenize(tokenizer, full_text, len(prompt),
                                        args.max_seq_tokens, device)
            if gen_tok.size < r0.WINDOW:
                raise ValueError(f"only {gen_tok.size} generated tokens")
            gen_lo = int(gen_tok[0])

            with ActivationCache(model, layers=[L]) as cache:
                ent_next = r0.entropy_series(model, enc, args.lm_head_chunk)
                H = cache[L][0].float().numpy()
            ent_gen = r0.gen_entropy(ent_next, gen_lo)
            if np.isnan(ent_gen).all():
                raise ValueError("all-NaN entropy (numerical blowup — retry with --fp32)")
            centers, ent_win = r0.windowed_mean(ent_gen)
            m = windowed_state_metrics(H[gen_lo:], window=r0.WINDOW, stride=r0.STRIDE)
            if arm in MATCHED_IDS_ARMS and not np.array_equal(centers, ref_centers):
                raise ValueError("grid mismatch vs E9.0 shard on a matched-ids arm")

            np.savez_compressed(
                out / f"{tid}.npz",
                ent_win=ent_win, centers=centers,
                pr=m["pr"], unif=m["unif"],
                ent_mean=np.float64(np.nanmean(ent_gen)),
                n_gen_tokens=np.int64(gen_tok.size),
                is_loop=np.int64(is_loop), onset_tok_gen=np.int64(onset),
            )
        except Exception as e:                          # fail-soft, counted
            n_fail += 1
            logger.warning(f"extract[{arm}] FAILED {tid}: {e}")
        finally:
            r0._clear_accel_cache()
    logger.info(f"extract[{arm}]: done ({n_fail} failures)")


# ── stage: generate (own-generation arm) ─────────────────────────────────────

def _build_prompt(arm: str, tokenizer, instruction: str) -> str:
    if arm == "qwenmath":                              # CF-N: its own template, declared
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": QWENMATH_SYSTEM},
             {"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True)
    from src.chain_gen import format_prompt            # deepseek family (r1 lineage)
    return format_prompt(tokenizer, instruction)


def _hf_generate_batch(model, tokenizer, prompts: list[str], *, max_new: int,
                       temperature: float, seed: int) -> list[dict]:
    """Left-padded batched generation. Reproducibility contract mirrors E9.1:
    per (batch composition, seed), not per sequence."""
    import torch
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    add_special_tokens=False).to(model.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0 if temperature > 0 else None, top_k=0 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id)
    rows = []
    plen = enc["input_ids"].shape[1]
    for j in range(len(prompts)):
        new_ids = out[j][plen:]
        new_ids = new_ids[new_ids != tokenizer.pad_token_id]
        rows.append({"chain": tokenizer.decode(new_ids, skip_special_tokens=True),
                     "n_tokens": int(new_ids.shape[0])})
    return rows


def stage_generate(args, arm: str) -> None:
    from src.chain_gen import load_model
    from tqdm import tqdm

    out = Path(args.out) / arm
    out.mkdir(parents=True, exist_ok=True)
    path = out / "gen.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["task"], r["sample"]) for r in rows}

    eval_ids = json.loads(Path(args.eval_ids).read_text())["task_ids"]
    chains = {r["task_id"]: r for r in json.loads(Path(args.chains).read_text())}
    tasks = [(tid, chains[tid]["instruction"], chains[tid].get("category", "?"))
             for tid in eval_ids if tid in chains]
    decodes = [(-1, 0.0, 0)] + [(s, 0.6, s) for s in GEN_SEEDS]   # (sample, T, seed)
    if args.smoke:
        tasks, decodes = tasks[:2], decodes[:1]

    todo = [(t, d) for d in decodes for t in tasks if (t[0], d[0]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    logger.info(f"generate[{arm}]: {len(todo)} chains to run ({len(done)} done)")
    if not todo:
        return

    model, tokenizer = load_model(MODEL_IDS[arm], dtype="float16")
    model.eval()
    for i in tqdm(range(0, len(todo), args.batch), desc=f"generate[{arm}]"):
        batch = todo[i:i + args.batch]
        try:
            prompts = [_build_prompt(arm, tokenizer, ins) for (_, ins, _), _ in batch]
            sample_idx = batch[0][1][0]                # batches are decode-homogeneous
            gen = _hf_generate_batch(model, tokenizer, prompts,
                                     max_new=args.max_new_tokens,
                                     temperature=batch[0][1][1],
                                     seed=1000 * (sample_idx + 1) + i)
            for ((tid, _, cat), (s, T, _)), g in zip(batch, gen):
                rows.append({"arm": arm, "task": tid, "category": cat, "sample": s,
                             "temperature": T, **g})
        except Exception as e:
            logger.warning(f"generate[{arm}] FAILED batch@{i}: {e}")
        finally:
            r0._clear_accel_cache()
        path.write_text(json.dumps(rows, indent=1))
    logger.info(f"generate[{arm}]: done ({len(rows)} rows)")


# ── stage: score-gen (on-policy re-scoring of own generations) ────────────────

def stage_score_gen(args, arm: str) -> None:
    from src.chain_gen import load_model
    from src.hooks import ActivationCache
    from src.loop_geometry import windowed_state_metrics
    from tqdm import tqdm

    path = Path(args.out) / arm / "gen.json"
    if not path.exists():
        logger.warning(f"score-gen[{arm}]: no gen.json — skipped")
        return
    rows = json.loads(path.read_text())
    chains = {r["task_id"]: r for r in json.loads(Path(args.chains).read_text())}
    todo = [r for r in rows if "ent_mean" not in r and r["n_tokens"] >= r0.WINDOW]
    if args.limit:
        todo = todo[:args.limit]
    logger.info(f"score-gen[{arm}]: {len(todo)} chains to score")
    if not todo:
        return

    model, tokenizer = load_model(MODEL_IDS[arm], dtype="float16")
    if args.fp32:
        model = model.float()
    model.eval()
    device = next(model.parameters()).device
    L = args.layer
    n_fail = 0
    for i, r in enumerate(tqdm(todo, desc=f"score-gen[{arm}]")):
        try:
            prompt = _build_prompt(arm, tokenizer, chains[r["task"]]["instruction"])
            enc, gen_tok = r0._tokenize(tokenizer, prompt + r["chain"], len(prompt),
                                        args.max_seq_tokens, device)
            if gen_tok.size < r0.WINDOW:
                raise ValueError("too short after retokenization")
            gen_lo = int(gen_tok[0])
            with ActivationCache(model, layers=[L]) as cache:
                ent_next = r0.entropy_series(model, enc, args.lm_head_chunk)
                H = cache[L][0].float().numpy()
            ent_gen = r0.gen_entropy(ent_next, gen_lo)
            if np.isnan(ent_gen).all():
                raise ValueError("all-NaN entropy (retry with --fp32)")
            m = windowed_state_metrics(H[gen_lo:], window=r0.WINDOW, stride=r0.STRIDE)
            r["ent_mean"] = float(np.nanmean(ent_gen))
            r["pr_mean"] = float(np.nanmean(m["pr"]))
            r["unif_mean"] = float(np.nanmean(m["unif"]))
        except Exception as e:
            n_fail += 1
            logger.warning(f"score-gen[{arm}] FAILED {r['task']}#{r['sample']}: {e}")
        finally:
            r0._clear_accel_cache()
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            path.write_text(json.dumps(rows, indent=1))
    logger.info(f"score-gen[{arm}]: done ({n_fail} failures)")


# ── stage: analyse ────────────────────────────────────────────────────────────

def _rep_summaries(args, arm: str) -> dict[str, dict]:
    """Per-chain matched-text summaries {tid: {ent, pr, unif, is_loop}} for one arm."""
    res = {}
    if arm == "r1":                                    # reuse R0 + E9.0 shards
        sample = json.loads((Path(args.r0_out) / "sample.json").read_text())
        for tid in sample["loop"] + sample["clean"]:
            ep = Path(args.r0_out) / "ent_shards" / f"{tid}.npz"
            sp = Path(args.state_shards) / f"{tid}.npz"
            if not ep.exists():
                continue
            with np.load(ep) as z:
                ent = float(np.nanmean(z["ent_win"]))
                is_loop = int(z["is_loop"])
            with np.load(sp) as z:
                pr = float(np.nanmean(z[f"pr_{args.layer}"]))
                unif = float(np.nanmean(z[f"unif_{args.layer}"]))
            res[tid] = {"ent": ent, "pr": pr, "unif": unif, "is_loop": is_loop}
        return res
    for p in sorted((Path(args.out) / arm / "ent_shards").glob("*.npz")):
        with np.load(p) as z:
            res[p.stem] = {"ent": float(np.nanmean(z["ent_win"])),
                           "pr": float(np.nanmean(z["pr"])),
                           "unif": float(np.nanmean(z["unif"])),
                           "is_loop": int(z["is_loop"])}
    return res


def stage_analyse(args) -> None:
    from scipy.stats import wilcoxon, mannwhitneyu, spearmanr
    from src.evaluation import repetition_rate

    out = Path(args.out)
    report: dict = {"layer": args.layer}

    # ── R1-rep: matched-text paired deltas vs r1 ─────────────────────────────
    ref = _rep_summaries(args, "r1")
    rep = {}
    for arm in ("deepscaler", "star1", "qwenmath"):
        s = _rep_summaries(args, arm)
        common = sorted(set(s) & set(ref))
        row = {"n": len(common)}
        for key in ("ent", "pr", "unif"):
            d = [s[t][key] - ref[t][key] for t in common
                 if np.isfinite(s[t][key]) and np.isfinite(ref[t][key])]
            if len(d) >= 10:
                row[key] = {"median_delta": float(np.median(d)),
                            "mean_delta": float(np.mean(d)),
                            "wilcoxon_p": float(wilcoxon(d).pvalue), "n": len(d)}
        rep[arm] = row
    report["rep_paired_vs_r1"] = rep

    if "pr" in rep.get("deepscaler", {}):
        d = rep["deepscaler"]["pr"]
        report["P_R1_1_rlvr_compression"] = {
            **d, "verdict": ("CONFIRMED (deepscaler PR < r1 on matched text)"
                             if d["median_delta"] < 0 and d["wilcoxon_p"] < 0.05
                             else "NOT confirmed")}
    if "pr" in rep.get("qwenmath", {}):
        d = rep["qwenmath"]["pr"]
        report["P_R1_2_base_vs_distill"] = {
            **d, "verdict": ("SFT-ENTROPY-SEEKING (r1 rank >= base)" if d["median_delta"] <= 0
                             else "INHERITED-COMPRESSION direction (base rank > r1... "
                                  "i.e. r1 < base)"),
            "note": "matched-TEXT tier: direction only (qwenmath grid differs)"}
    if "pr" in rep.get("star1", {}):
        d = rep["star1"]["pr"]
        report["P_R1_6_safety_control"] = {
            **d, "verdict": ("PASS (star1 ~ r1)" if d["wilcoxon_p"] >= 0.05 or
                             abs(d["median_delta"]) < 0.5 else "star1 differs from r1")}

    # ── R1-beh: own-generation metrics ───────────────────────────────────────
    beh = {}
    for arm in MODEL_IDS:
        path = out / arm / "gen.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        for r in rows:
            r["rep4"] = repetition_rate(r["chain"])
            words = max(len(r["chain"].split()), 1)
            r["bt_per_1k"] = 1000 * len(BT_CUE_RE.findall(r["chain"])) / words
            r["boxed"] = "\\boxed" in r["chain"]
        t06 = [r for r in rows if r["sample"] >= 0]
        by_task: dict[str, list] = {}
        for r in t06:
            by_task.setdefault(r["task"], []).append(r)
        div = []
        for task, rs in sorted(by_task.items()):
            if len(rs) < 2:
                continue
            grams = [r0._ngrams(r["chain"]) for r in rs]
            jac = [len(a & b) / max(len(a | b), 1) for a, b in combinations(grams, 2)]
            div.append({"task": task, "category": rs[0]["category"],
                        "diversity": 1.0 - float(np.mean(jac))})
        beh[arm] = {
            "n_rows": len(rows), "n_tasks_t06": len(div),
            "diversity_mean": float(np.mean([d["diversity"] for d in div])) if div else None,
            "diversity_math": (float(np.mean([d["diversity"] for d in div
                                              if d["category"].startswith("math")]))
                               if any(d["category"].startswith("math") for d in div) else None),
            "len_mean": float(np.mean([r["n_tokens"] for r in rows])),
            "bt_per_1k_mean": float(np.mean([r["bt_per_1k"] for r in rows])),
            "boxed_rate": float(np.mean([r["boxed"] for r in rows])),
            "collapse_rate": float(np.mean([r["rep4"] > 0.8 for r in rows])),
            "ent_mean": (float(np.mean([r["ent_mean"] for r in rows if "ent_mean" in r]))
                         if any("ent_mean" in r for r in rows) else None),
            "pr_mean": (float(np.mean([r["pr_mean"] for r in rows if "pr_mean" in r]))
                        if any("pr_mean" in r for r in rows) else None),
            "_diversity_per_task": div,
        }
    report["beh"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in beh.items()}

    # P-R1.3 ordering test (pairwise MW on per-task diversity, MATH-primary for base)
    def _div(arm, math_only=False):
        d = beh.get(arm, {}).get("_diversity_per_task", [])
        return [x["diversity"] for x in d
                if not math_only or x["category"].startswith("math")]
    order = {}
    for a, b, math_only in (("qwenmath", "r1", True), ("r1", "deepscaler", False)):
        da, db = _div(a, math_only), _div(b, math_only)
        if len(da) >= 5 and len(db) >= 5:
            mw = mannwhitneyu(da, db, alternative="greater")
            order[f"{a}>{b}"] = {"mean_a": float(np.mean(da)), "mean_b": float(np.mean(db)),
                                 "mw_p_one_sided": float(mw.pvalue),
                                 "math_only": math_only}
    report["P_R1_3_diversity_ordering"] = order

    (out / "report.json").write_text(json.dumps(report, indent=1))
    _write_md(out, report)
    logger.info(f"analyse: wrote {out}/report.json + REPORT.md")


def _write_md(out: Path, r: dict) -> None:
    md = ["# R1 — strata-differential compression (creativity–entropy rung 1 + RL.a)\n",
          "Prereg: R1_COMPRESSION_PREREG.md. Primary contrast r1↔deepscaler (matched-ids).\n",
          "## R1-rep: paired matched-text deltas vs r1 (per-chain, R0 sample)\n",
          "| arm | n | Δent median (p) | ΔPR median (p) | Δunif median (p) |",
          "|---|--:|--:|--:|--:|"]
    for arm, row in r.get("rep_paired_vs_r1", {}).items():
        cells = []
        for key in ("ent", "pr", "unif"):
            v = row.get(key)
            cells.append(f"{v['median_delta']:+.3f} ({v['wilcoxon_p']:.2g})" if v else "—")
        md.append(f"| {arm} | {row.get('n', 0)} | " + " | ".join(cells) + " |")
    for k in ("P_R1_1_rlvr_compression", "P_R1_2_base_vs_distill", "P_R1_6_safety_control"):
        if k in r:
            md.append(f"\n**{k}:** {r[k]['verdict']}")
    md += ["\n## R1-beh: own-generation ladder (greedy + 3×T0.6, 50 tasks)\n",
           "| arm | diversity (T06) | math-only | len | bt/1k | boxed | collapse | on-policy ent | on-policy PR |",
           "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for arm, b in r.get("beh", {}).items():
        md.append(f"| {arm} | {b['diversity_mean']} | {b['diversity_math']} | "
                  f"{b['len_mean']:.0f} | {b['bt_per_1k_mean']:.2f} | {b['boxed_rate']:.2f} | "
                  f"{b['collapse_rate']:.2f} | {b['ent_mean']} | {b['pr_mean']} |")
    if "P_R1_3_diversity_ordering" in r:
        md.append("\n**P-R1.3 orderings:** " + json.dumps(r["P_R1_3_diversity_ordering"]))
    (out / "REPORT.md").write_text("\n".join(md) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    models = args.models[:1] if args.smoke else args.models
    if args.stage in ("all", "gate"):
        stage_gate(args)
    for arm in models:
        if args.stage in ("all", "extract"):
            stage_extract(args, arm)
        if args.stage in ("all", "generate"):
            stage_generate(args, arm)
        if args.stage in ("all", "score-gen"):
            stage_score_gen(args, arm)
    if args.stage in ("all", "analyse") and not args.smoke:
        stage_analyse(args)


if __name__ == "__main__":
    main()
