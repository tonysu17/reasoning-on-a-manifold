#!/usr/bin/env python3
"""
R0 — Entropy-ladder bookkeeping (creativity–entropy programme, rung 0).

Pre-registration: R0_ENTROPY_LADDER_PREREG.md (sealed 2026-07-11 before any
entropy computation). Programme doc: ../creativity_entropy_extension.md §6.

Four checkpointed stages:

  sample       (CPU)  Stratified subsample of the E9.0 loop-geometry shards:
                      100 loop (onset>0) + 100 clean, rng seed 0.
                      → <out>/sample.json
  extract-base (GPU)  One teacher-forced pass per sampled chain; token-level
                      next-token predictive entropy (nats, float32) over the
                      GENERATED region, windowed on the exact E9.0 grid
                      (window 128 / stride 64); center-grid equality with the
                      state shard asserted per chain (mismatch ⇒ excluded).
                      → <out>/ent_shards/<task_id>.npz
  extract-t06  (GPU)  E9.1 T=0.6 vanilla arm (50 tasks × 3 samples): same
                      entropy pass + fresh L17 windowed PR/uniformity
                      (ActivationCache), prompts reconstructed from the corpus
                      chains file via base_task_id.
                      → <out>/t06_ent.json  (incrementally checkpointed)
  analyse      (CPU)  R0.a cross-level Spearman (primary regions: clean = all
                      generated windows, loop = pre-onset windows only);
                      R0.b loop entropy profile with the MANDATORY
                      matched-relative-position clean control (E9.0b lesson);
                      R0.c per-task E-1/E-2 vs cross-sample diversity
                      (1 − mean pairwise 4-gram Jaccard, e9_1_analysis def).
                      → <out>/report.json + <out>/REPORT.md

E-1 is computed by re-scoring the model's own stored text: exact for the
greedy base corpus; for T06 samples it measures local uncertainty along the
sampled path (caveat carried into every table that uses it).

Usage:
  python3 29_r0_entropy_ladder.py --stage all
  python3 29_r0_entropy_ladder.py --stage analyse       # after shards exist
  python3 29_r0_entropy_ladder.py --stage all --smoke   # 3 base + 2 t06 tasks
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("r0_entropy")

WINDOW, STRIDE = 128, 64            # the E9.0 grid — do not change (grid equality asserted)
PRE, BASE_LO, BASE_HI = 512, 128, 640   # E9.0b pre-onset / early-baseline design


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "sample", "extract-base", "extract-t06", "analyse"])
    ap.add_argument("--chains", default="data/chains_R1-1.5B.json")
    ap.add_argument("--state-shards", default="results/loop_geometry/R1-1.5B/shards")
    ap.add_argument("--t06-results", default="results/eval/R1-1.5B__E9_1_T06/steering_results.json")
    ap.add_argument("--out", default="results/r0_entropy_ladder/R1-1.5B")
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--layer", type=int, default=17, help="primary E-2 layer")
    ap.add_argument("--max-seq-tokens", type=int, default=10240)
    ap.add_argument("--lm-head-chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap items per extract invocation (0 = all); used by the "
                         "r0_runner.sh restart loop to bound MPS allocator growth")
    ap.add_argument("--fp32", action="store_true",
                    help="run the forward pass in float32 — retry path for chains "
                         "whose bf16 pass produced all-NaN entropy (3/350 observed)")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


# ── shared helpers ────────────────────────────────────────────────────────────

def _ngrams(text: str, n: int = 4) -> set:
    """4-gram set — verbatim from e9_1_analysis.py so diversity is the same measure."""
    toks = text.split()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def windowed_mean(x: np.ndarray, window: int = WINDOW, stride: int = STRIDE):
    """Mean over sliding windows; centers mirror windowed_state_metrics exactly."""
    T = len(x)
    centers, vals = [], []
    for start in range(0, max(T - window + 1, 0), stride):
        centers.append(start + window // 2)
        vals.append(float(np.nanmean(x[start:start + window])))
    return np.asarray(centers, dtype=np.int64), np.asarray(vals, dtype=np.float64)


def _tokenize(tokenizer, full_text: str, gen_start_char: int, max_seq: int, device):
    enc = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt",
                    add_special_tokens=False, truncation=True, max_length=max_seq)
    offsets = enc.pop("offset_mapping")[0].tolist()
    starts = np.asarray([s for s, _ in offsets], dtype=np.int64)
    ends = np.asarray([e for _, e in offsets], dtype=np.int64)
    gen_tok = np.nonzero((starts >= gen_start_char) & (ends > starts))[0]
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc, gen_tok


def entropy_series(model, enc, chunk: int) -> np.ndarray:
    """Next-token predictive entropy (nats) at every position, chunked LM head.

    Returns ent_next of length T where ent_next[i] = H(softmax(logits_i)) — the
    distribution that PREDICTS token i+1. The full (T, vocab) logits tensor is
    never materialised (2.3 GB at 8k tokens): the base transformer runs once,
    then lm_head is applied in position chunks with float32 softmax. Each chunk's
    device tensors are dropped eagerly — on MPS the caching allocator otherwise
    accumulates the freed blocks across variable-length chains until even KB-scale
    allocations OOM (observed 2026-07-11: latency 67→670 s/chain, then a 280-chain
    OOM cascade; the E9.0 KV-fragmentation lesson, one level down).
    """
    import torch
    with torch.no_grad():
        out = model.model(**enc, use_cache=False)
        h = out.last_hidden_state                      # (1, T, hidden), post final norm
        ents = []
        for i in range(0, h.shape[1], chunk):
            logits = model.lm_head(h[:, i:i + chunk]).float()
            logp = torch.log_softmax(logits, dim=-1)
            ents.append((-(logp.exp() * logp).sum(-1))[0].cpu())
            del logits, logp
        del h, out
        return torch.cat(ents).numpy()


def _clear_accel_cache() -> None:
    """Release the caching allocator's pool (CUDA and MPS) between chains."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def gen_entropy(ent_next: np.ndarray, gen_lo: int) -> np.ndarray:
    """Entropy aligned to generated rows gen_lo..T-1: row j ← ent_next[gen_lo+j-1]
    (the distribution that generated that token). Same row space as the state
    windows in the E9.0 shards, so the window grids coincide."""
    T = len(ent_next)
    return ent_next[gen_lo - 1:T - 1].copy()


# ── stage: sample ─────────────────────────────────────────────────────────────

def stage_sample(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sample.json"
    if path.exists():
        sample = json.loads(path.read_text())
        logger.info(f"sample: reusing {path} "
                    f"({len(sample['loop'])} loop / {len(sample['clean'])} clean)")
        return sample

    loops, cleans = [], []
    for p in sorted(Path(args.state_shards).glob("*.npz")):
        with np.load(p) as z:
            if int(z["is_loop"]) and int(z["onset_tok_gen"]) > 0:
                loops.append(p.stem)
            elif not int(z["is_loop"]):
                cleans.append(p.stem)
    rng = np.random.default_rng(args.seed)
    sample = {
        "seed": args.seed,
        "loop": sorted(rng.choice(loops, min(args.n_per_class, len(loops)),
                                  replace=False).tolist()),
        "clean": sorted(rng.choice(cleans, min(args.n_per_class, len(cleans)),
                                   replace=False).tolist()),
    }
    path.write_text(json.dumps(sample, indent=1))
    logger.info(f"sample: {len(sample['loop'])} loop / {len(sample['clean'])} clean "
                f"(from {len(loops)}/{len(cleans)} eligible)")
    return sample


# ── stage: extract-base ───────────────────────────────────────────────────────

def stage_extract_base(args, sample: dict) -> None:
    import torch
    from src.chain_gen import load_model
    from tqdm import tqdm

    out = Path(args.out)
    ent_dir = out / "ent_shards"
    ent_dir.mkdir(parents=True, exist_ok=True)

    todo = [t for t in sample["loop"] + sample["clean"]
            if not (ent_dir / f"{t}.npz").exists()]
    if args.smoke:
        todo = [t for t in sample["loop"][:2] + sample["clean"][:1]
                if not (ent_dir / f"{t}.npz").exists()]
    if args.limit:
        todo = todo[:args.limit]
    logger.info(f"extract-base: {len(todo)} chains to run")
    if not todo:
        return

    chains = {r["task_id"]: r for r in json.loads(Path(args.chains).read_text())}
    model, tokenizer = load_model(args.model_id, dtype="float16")
    if args.fp32:
        model = model.float()          # NaN-retry path: full-precision forward
    model.eval()
    device = next(model.parameters()).device

    n_fail = 0
    for tid in tqdm(todo, desc="extract-base"):
        try:
            with np.load(Path(args.state_shards) / f"{tid}.npz") as z:
                shard_centers = z["centers"].copy()
                shard_n_gen = int(z["n_gen_tokens"])
                is_loop = int(z["is_loop"])
                onset = int(z["onset_tok_gen"])

            rec = chains[tid]
            prompt = rec["prompt"]
            full_text = rec.get("full_text") or (prompt + rec["chain"])
            enc, gen_tok = _tokenize(tokenizer, full_text, len(prompt),
                                     args.max_seq_tokens, device)
            if int(gen_tok.size) != shard_n_gen:
                raise ValueError(f"gen-token mismatch: {gen_tok.size} vs shard {shard_n_gen}")
            gen_lo = int(gen_tok[0])

            ent_next = entropy_series(model, enc, args.lm_head_chunk)
            ent_gen = gen_entropy(ent_next, gen_lo)
            if np.isnan(ent_gen).all():
                raise ValueError("all-NaN entropy (numerical blowup — retry with --fp32)")
            centers, ent_win = windowed_mean(ent_gen)
            if centers.shape != shard_centers.shape or not np.array_equal(centers, shard_centers):
                raise ValueError("window-center grid mismatch vs state shard")

            np.savez_compressed(
                ent_dir / f"{tid}.npz",
                ent_win=ent_win, centers=centers,
                ent_tok=ent_gen.astype(np.float16),
                ent_mean=np.float64(np.nanmean(ent_gen)),
                n_gen_tokens=np.int64(gen_tok.size),
                is_loop=np.int64(is_loop), onset_tok_gen=np.int64(onset),
            )
        except Exception as e:                          # fail-soft per chain (counted)
            n_fail += 1
            logger.warning(f"extract-base FAILED {tid}: {e}")
        finally:
            _clear_accel_cache()
    logger.info(f"extract-base: done ({n_fail} failures)")


# ── stage: extract-t06 ────────────────────────────────────────────────────────

def stage_extract_t06(args) -> None:
    import torch
    from src.chain_gen import load_model
    from src.hooks import ActivationCache
    from src.loop_geometry import windowed_state_metrics
    from tqdm import tqdm

    out = Path(args.out)
    path = out / "t06_ent.json"
    done = {r["task_id"] for r in json.loads(path.read_text())} if path.exists() else set()

    rows = [r for r in json.loads(Path(args.t06_results).read_text())
            if r["method"] == "vanilla"]
    if args.smoke:
        base_ids = sorted({r["base_task_id"] for r in rows})[:2]
        rows = [r for r in rows if r["base_task_id"] in base_ids]
    todo = [r for r in rows if r["task_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    logger.info(f"extract-t06: {len(todo)} vanilla samples to run ({len(done)} done)")
    if not todo:
        return

    chains = {r["task_id"]: r for r in json.loads(Path(args.chains).read_text())}
    model, tokenizer = load_model(args.model_id, dtype="float16")
    if args.fp32:
        model = model.float()          # NaN-retry path: full-precision forward
    model.eval()
    device = next(model.parameters()).device
    L = args.layer

    records = json.loads(path.read_text()) if path.exists() else []
    n_fail = 0
    for i, r in enumerate(tqdm(todo, desc="extract-t06")):
        try:
            prompt = chains[r["base_task_id"]]["prompt"]
            full_text = prompt + r["chain"]
            enc, gen_tok = _tokenize(tokenizer, full_text, len(prompt),
                                     args.max_seq_tokens, device)
            if gen_tok.size < WINDOW:
                raise ValueError(f"only {gen_tok.size} generated tokens")
            gen_lo = int(gen_tok[0])

            with ActivationCache(model, layers=[L]) as cache:
                ent_next = entropy_series(model, enc, args.lm_head_chunk)
                H = cache[L][0].float().numpy()
            ent_gen = gen_entropy(ent_next, gen_lo)
            if np.isnan(ent_gen).all():
                raise ValueError("all-NaN entropy (numerical blowup — retry with --fp32)")
            m = windowed_state_metrics(H[gen_lo:], window=WINDOW, stride=STRIDE)

            records.append({
                "task_id": r["task_id"], "base_task_id": r["base_task_id"],
                "sample": r.get("sample"), "n_gen_tokens": int(gen_tok.size),
                "mean_ent": float(np.nanmean(ent_gen)),
                "mean_pr": float(np.nanmean(m["pr"])),
                "mean_unif": float(np.nanmean(m["unif"])),
            })
        except Exception as e:
            n_fail += 1
            logger.warning(f"extract-t06 FAILED {r['task_id']}: {e}")
        finally:
            _clear_accel_cache()
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):    # incremental checkpoint
            path.write_text(json.dumps(records, indent=1))
    logger.info(f"extract-t06: done ({n_fail} failures)")


# ── stage: analyse ────────────────────────────────────────────────────────────

def _load_pair(args, tid: str) -> dict | None:
    """Ent shard + matching state-shard series for one chain."""
    ep = Path(args.out) / "ent_shards" / f"{tid}.npz"
    if not ep.exists():
        return None
    with np.load(ep) as z:
        d = {k: z[k].copy() for k in z.files}
    with np.load(Path(args.state_shards) / f"{tid}.npz") as z:
        for L in (15, 16, 17):
            d[f"pr_{L}"] = z[f"pr_{L}"].copy()
            d[f"unif_{L}"] = z[f"unif_{L}"].copy()
    d["task_id"] = tid
    return d


def stage_analyse(args, sample: dict) -> None:
    from scipy.stats import spearmanr, mannwhitneyu, wilcoxon

    out = Path(args.out)
    L = args.layer
    pairs = [p for tid in sample["loop"] + sample["clean"]
             if (p := _load_pair(args, tid)) is not None]
    n_nan_ent = sum(1 for p in pairs if bool(np.all(np.isnan(p["ent_win"]))))
    pairs = [p for p in pairs if not bool(np.all(np.isnan(p["ent_win"])))]
    n_loop = sum(int(p["is_loop"]) for p in pairs)
    logger.info(f"analyse: {len(pairs)} chains with ent+state shards "
                f"({n_loop} loop; {n_nan_ent} all-NaN ent excluded)")
    report: dict = {"n_chains": len(pairs), "n_loop": n_loop,
                    "n_excluded": len(sample["loop"] + sample["clean"]) - len(pairs),
                    "n_all_nan_ent_excluded": n_nan_ent,
                    "layer": L}

    # ── R0.a cross-level correlations (primary region per class) ─────────────
    def _summ(p) -> dict | None:
        c = p["centers"]
        mask = (c < p["onset_tok_gen"]) if p["is_loop"] else np.ones_like(c, bool)
        if mask.sum() < 3:
            return None
        row = {"ent": float(np.nanmean(p["ent_win"][mask])),
               "n_gen": int(p["n_gen_tokens"]), "is_loop": int(p["is_loop"])}
        for l in (15, 16, 17):
            row[f"pr{l}"] = float(np.nanmean(p[f"pr_{l}"][mask]))
            row[f"unif{l}"] = float(np.nanmean(p[f"unif_{l}"][mask]))
        if not all(np.isfinite(v) for k, v in row.items() if k not in ("is_loop",)):
            return None                                  # all-NaN ent chain (counted upstream)
        return row

    summ = [s for p in pairs if (s := _summ(p)) is not None]
    report["r0a_n_summarised"] = len(summ)

    def _corr(rows, keys) -> dict:
        res = {}
        for a, b in combinations(keys, 2):
            x = np.array([r[a] for r in rows]); y = np.array([r[b] for r in rows])
            rho, pv = spearmanr(x, y)
            res[f"{a}~{b}"] = {"rho": round(float(rho), 3), "p": float(pv), "n": len(rows)}
        return res

    keys = ["ent", f"pr{L}", f"unif{L}", "n_gen"]
    report["r0a"] = {
        "clean": _corr([s for s in summ if not s["is_loop"]], keys),
        "loop_preonset": _corr([s for s in summ if s["is_loop"]], keys),
        "pooled": _corr(summ, keys),
        "secondary_layers": {str(l): _corr(summ, ["ent", f"pr{l}", f"unif{l}"])
                             for l in (15, 16)},
    }
    prim = {**report["r0a"]["clean"], **report["r0a"]["loop_preonset"]}
    ladder_pairs = [v for k, v in prim.items() if "n_gen" not in k]
    report["P_R0_1_dissociation"] = {
        "max_abs_rho_primary": max(abs(v["rho"]) for v in ladder_pairs),
        "all_redundant": all(abs(v["rho"]) >= 0.9 for v in ladder_pairs),
        "verdict": ("LADDER REDUNDANT — programme collapses to one level"
                    if all(abs(v["rho"]) >= 0.9 for v in ladder_pairs)
                    else "PASS — levels dissociate (gate open)"),
    }

    # ── R0.b loop entropy profile ─────────────────────────────────────────────
    loops = [p for p in pairs if p["is_loop"] and p["onset_tok_gen"] > 0]
    cleans = [p for p in pairs if not p["is_loop"]]
    med_frac = float(np.median([p["onset_tok_gen"] / p["n_gen_tokens"] for p in loops]))

    def _tail_mean(p, onset) -> float | None:
        m = p["centers"] >= onset
        return float(np.nanmean(p["ent_win"][m])) if m.sum() >= 2 else None

    in_loop = [v for p in loops
               if (v := _tail_mean(p, int(p["onset_tok_gen"]))) is not None and np.isfinite(v)]
    in_clean = [v for p in cleans
                if (v := _tail_mean(p, int(med_frac * p["n_gen_tokens"]))) is not None
                and np.isfinite(v)]
    mw = mannwhitneyu(in_loop, in_clean, alternative="less")
    report["P_R0_2_jam"] = {
        "median_onset_frac": med_frac,
        "in_loop_ent_mean": float(np.mean(in_loop)), "n_loop": len(in_loop),
        "matched_clean_ent_mean": float(np.mean(in_clean)), "n_clean": len(in_clean),
        "mw_p_one_sided_less": float(mw.pvalue),
        "verdict": "CONFIRMED (in-loop entropy lower)" if mw.pvalue < 0.01 else "NOT confirmed",
    }

    def _delta(p, onset) -> float | None:
        c = p["centers"]
        pre_m = (c >= onset - PRE) & (c < onset)
        base_m = (c >= BASE_LO) & (c < BASE_HI)
        if pre_m.sum() >= 2 and base_m.sum() >= 2 and onset - PRE > BASE_HI:
            return float(np.nanmean(p["ent_win"][pre_m]) - np.nanmean(p["ent_win"][base_m]))
        return None

    d_loop = [v for p in loops
              if (v := _delta(p, int(p["onset_tok_gen"]))) is not None and np.isfinite(v)]
    d_clean = [v for p in cleans
               if (v := _delta(p, int(med_frac * p["n_gen_tokens"]))) is not None
               and np.isfinite(v)]
    mw2 = mannwhitneyu(d_loop, d_clean, alternative="two-sided")
    report["P_R0_3_preonset"] = {
        "delta_loop_mean": float(np.mean(d_loop)), "n_loop": len(d_loop),
        "delta_clean_mean": float(np.mean(d_clean)), "n_clean": len(d_clean),
        "mw_p_two_sided": float(mw2.pvalue),
        # within-loop Wilcoxon = the E9.0b companion statistic (clean-side n is small
        # because short clean chains fail the onset−PRE>BASE_HI window condition)
        "wilcoxon_p_loop_within": (float(wilcoxon(d_loop).pvalue)
                                   if len(d_loop) >= 10 else None),
        "verdict": ("STATE-FIRST (no loop-specific pre-onset E-1 differential)"
                    if mw2.pvalue >= 0.05 else
                    ("THERMOSTAT-FAILURE direction (loop pre-onset decline exceeds clean)"
                     if np.mean(d_loop) < np.mean(d_clean) else
                     "UNEXPECTED (loop pre-onset E-1 ELEVATED vs clean)")),
    }

    # aligned curves (loop chains, offsets rel. onset) for ent + pr + unif
    bins = np.arange(-2048, 1025, 128)
    curves = {k: [[] for _ in range(len(bins) - 1)] for k in ("ent", "pr", "unif")}
    for p in loops:
        rel = p["centers"] - int(p["onset_tok_gen"])
        which = np.digitize(rel, bins) - 1
        series = {"ent": p["ent_win"], "pr": p[f"pr_{L}"], "unif": p[f"unif_{L}"]}
        for k, v in series.items():
            for b in range(len(bins) - 1):
                m = which == b
                if m.any():
                    curves[k][b].append(float(np.nanmean(v[m])))
    report["r0b_aligned_curve"] = {
        "bin_lo": bins[:-1].tolist(),
        **{k: [round(float(np.mean(c)), 3) if c else None for c in curves[k]]
           for k in curves},
        "n": [len(c) for c in curves["ent"]],
    }

    # ── R0.c linkage to E-3 (T06 vanilla) ─────────────────────────────────────
    t06_path = out / "t06_ent.json"
    if t06_path.exists():
        recs = json.loads(t06_path.read_text())
        n_t06_nan = sum(1 for r in recs if not np.isfinite(r["mean_ent"]))
        recs = [r for r in recs if np.isfinite(r["mean_ent"])]
        report["r0c_n_nan_excluded"] = n_t06_nan
        rows = [r for r in json.loads(Path(args.t06_results).read_text())
                if r["method"] == "vanilla"]
        text_by_id = {r["task_id"]: r["chain"] for r in rows}
        by_task: dict[str, list] = {}
        for r in recs:
            by_task.setdefault(r["base_task_id"], []).append(r)
        table = []
        for task, rs in sorted(by_task.items()):
            if len(rs) < 2:
                continue
            grams = [_ngrams(text_by_id[r["task_id"]]) for r in rs]
            jac = [len(a & b) / max(len(a | b), 1) for a, b in combinations(grams, 2)]
            table.append({"task": task, "n_samples": len(rs),
                          "diversity": 1.0 - float(np.mean(jac)),
                          "mean_ent": float(np.mean([r["mean_ent"] for r in rs])),
                          "mean_pr": float(np.mean([r["mean_pr"] for r in rs])),
                          "mean_unif": float(np.mean([r["mean_unif"] for r in rs]))})
        report["r0c_n_tasks"] = len(table)
        div = np.array([t["diversity"] for t in table])
        res = {}
        for k in ("mean_ent", "mean_pr", "mean_unif"):
            rho, pv = spearmanr(np.array([t[k] for t in table]), div)
            res[k] = {"rho": round(float(rho), 3), "p": float(pv)}
        e = res["mean_ent"]
        report["P_R0_4_ent_diversity"] = {
            **e,
            "verdict": ("CONFIRMED (mid-range coupling)" if e["p"] < 0.05 and 0.2 <= e["rho"] < 0.9
                        else "REDUNDANT (rho>=0.9)" if e["rho"] >= 0.9
                        else "DISSOCIATED / not significant"),
        }
        report["P_R0_5_pr_diversity_exploratory"] = res["mean_pr"]
        report["r0c_correlations"] = res
        report["r0c_table"] = table
    else:
        report["r0c_n_tasks"] = 0
        logger.warning("analyse: no t06_ent.json — R0.c skipped")

    (out / "report.json").write_text(json.dumps(report, indent=1))
    _write_md(out, report)
    logger.info(f"analyse: wrote {out}/report.json + REPORT.md")


def _write_md(out: Path, r: dict) -> None:
    L = r["layer"]
    md = ["# R0 — entropy-ladder bookkeeping (creativity–entropy rung 0)\n",
          f"Chains: {r['n_chains']} ({r['n_loop']} loop; {r['n_excluded']} excluded on "
          f"grid/gen-token mismatch or missing shard). Prereg: R0_ENTROPY_LADDER_PREREG.md\n",
          "## R0.a cross-level Spearman (primary regions)\n",
          "| pair | clean ρ (p) | loop pre-onset ρ (p) | pooled ρ (p) |",
          "|---|--:|--:|--:|"]
    for key in (f"ent~pr{L}", f"ent~unif{L}", f"pr{L}~unif{L}",
                "ent~n_gen", f"pr{L}~n_gen", f"unif{L}~n_gen"):
        cells = []
        for reg in ("clean", "loop_preonset", "pooled"):
            v = r["r0a"][reg].get(key)
            cells.append(f"{v['rho']:+.3f} ({v['p']:.2g})" if v else "—")
        md.append(f"| {key} | " + " | ".join(cells) + " |")
    g = r["P_R0_1_dissociation"]
    md += [f"\n**P-R0.1 gate:** {g['verdict']} — max |ρ| among ladder pairs "
           f"{g['max_abs_rho_primary']:.3f} (bar 0.9)\n",
           "## R0.b loop entropy profile\n"]
    j = r["P_R0_2_jam"]
    md += [f"**P-R0.2 (jam):** in-loop E-1 {j['in_loop_ent_mean']:.3f} nats "
           f"(n={j['n_loop']}) vs matched clean {j['matched_clean_ent_mean']:.3f} "
           f"(n={j['n_clean']}), one-sided MW p={j['mw_p_one_sided_less']:.2g} → "
           f"**{j['verdict']}**\n"]
    q = r["P_R0_3_preonset"]
    md += [f"**P-R0.3 (pre-onset):** Δ(pre−base) loop {q['delta_loop_mean']:+.3f} "
           f"(n={q['n_loop']}) vs clean {q['delta_clean_mean']:+.3f} (n={q['n_clean']}), "
           f"two-sided MW p={q['mw_p_two_sided']:.2g} → **{q['verdict']}**\n"]
    if r.get("r0c_n_tasks"):
        md += ["## R0.c linkage to E-3 (T06 vanilla; re-scoring caveat applies)\n",
               "| metric | ρ vs diversity | p |", "|---|--:|--:|"]
        for k, v in r["r0c_correlations"].items():
            md.append(f"| {k} | {v['rho']:+.3f} | {v['p']:.2g} |")
        md += [f"\n**P-R0.4:** {r['P_R0_4_ent_diversity']['verdict']} "
               f"(n={r['r0c_n_tasks']} tasks)\n"]
    (out / "REPORT.md").write_text("\n".join(md) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    sample = stage_sample(args)
    if args.stage in ("all", "extract-base"):
        stage_extract_base(args, sample)
    if args.stage in ("all", "extract-t06"):
        stage_extract_t06(args)
    if args.stage in ("all", "analyse") and not args.smoke:
        stage_analyse(args, sample)


if __name__ == "__main__":
    main()
