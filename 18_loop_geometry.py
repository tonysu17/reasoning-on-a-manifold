#!/usr/bin/env python3
"""
E9.0 — Loop-geometry gate (COLLAPSE_AND_ENTROPY.md §5, E9.0a + E9.0b).

Three checkpointed stages over the generation corpus (default
data/chains_R1-1.5B.json — the 1000-chain corpus; ~half are cap-hit, so loops
are abundant without any new generation):

  detect   (CPU)  Label every chain: periodic loop tail (onset/period) via
                  src.loop_geometry.detect_loop_tail + 4-gram repetition.
                  → <out>/loop_labels.json
  extract  (GPU)  One forward pass per selected chain; capture residual-stream
                  states at the steering layers (default 15 16 17, the
                  ActivationCache/steered-hook convention); save per-chain npz
                  shards: windowed PR + token-uniformity series over the
                  GENERATED region only, plus sampled in-loop / out-of-loop
                  token vectors (f16) for the probe.
                  → <out>/shards/<task_id>.npz
  analyse  (CPU)  E9.0a: chain-grouped logistic loop-probe per layer → unit
                  loop direction; cosine table vs the E1-pooled steering
                  vectors (each behaviour compared at ITS OWN steering layer)
                  with the analytic random-direction null.
                  E9.0b: state-collapse precedence — PR/uniformity in the
                  pre-onset window vs an early-chain baseline, against the
                  matched-position contrast in clean chains.
                  → <out>/report.json + <out>/REPORT.md + <out>/directions.npz

Decision gate (pre-registered in COLLAPSE_AND_ENTROPY.md §5): |cos| ≳ 0.3
between a steering vector and the loop direction ⇒ H-B (vector contamination)
is live and E9.1 must carry a cleaned-vector arm.

Usage:
  python3 18_loop_geometry.py --stage all            # detect → extract → analyse
  python3 18_loop_geometry.py --stage detect         # CPU-only, runs anywhere
  python3 18_loop_geometry.py --stage analyse        # after shards exist
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("loop_geometry")

BEHAVIOURS = ["backtracking", "uncertainty-estimation", "example-testing",
              "adding-knowledge"]
ARMS = ["single", "manifold_k3", "manifold_k5"]


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "detect", "extract", "analyse"])
    ap.add_argument("--chains", default="data/chains_R1-1.5B.json")
    ap.add_argument("--out", default="results/loop_geometry/R1-1.5B")
    ap.add_argument("--vectors", default="results/steering_vectors/R1-1.5B__E1_pooled",
                    help="E1-pooled steering-vector dir (metadata.json + *.npy)")
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--layers", type=int, nargs="+", default=[15, 16, 17])
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--tokens-per-class", type=int, default=192,
                    help="max sampled token vectors per class per chain")
    ap.add_argument("--guard-words", type=int, default=200,
                    help="words before the loop onset excluded from out-of-loop")
    ap.add_argument("--clean-rep-max", type=float, default=0.30)
    ap.add_argument("--max-chains", type=int, default=0,
                    help="cap TOTAL chains extracted (0 = all loop+clean chains)")
    ap.add_argument("--max-seq-tokens", type=int, default=10240)
    ap.add_argument("--pre-window", type=int, default=512,
                    help="E9.0b: pre-onset window length (generated tokens)")
    ap.add_argument("--base-lo", type=int, default=128)
    ap.add_argument("--base-hi", type=int, default=640,
                    help="E9.0b: early-chain baseline window [lo, hi)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="detect+extract on the first 12 chains only")
    return ap.parse_args()


# ── stage: detect ─────────────────────────────────────────────────────────────

def stage_detect(args) -> list[dict]:
    from src.loop_geometry import loop_labels_for_chain

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels_path = out / "loop_labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
        logger.info(f"detect: reusing {labels_path} ({len(labels)} chains)")
        return labels

    chains = json.loads(Path(args.chains).read_text())
    if args.smoke:
        chains = chains[:12]
    labels = []
    for rec in chains:
        lab = loop_labels_for_chain(rec["chain"], clean_rep_max=args.clean_rep_max)
        labels.append({
            "task_id": rec["task_id"],
            "category": rec.get("category", "unknown"),
            "n_tokens": rec.get("n_tokens"),
            "prompt_chars": len(rec.get("prompt", "")),
            **lab,
        })
    labels_path.write_text(json.dumps(labels, indent=1))

    n = len(labels)
    by = {c: sum(1 for l in labels if l["class"] == c)
          for c in ("loop", "clean", "ambiguous")}
    loops = [l for l in labels if l["class"] == "loop"]
    logger.info(f"detect: {n} chains → loop {by['loop']} / clean {by['clean']} / "
                f"ambiguous {by['ambiguous']} (excluded)")
    if loops:
        periods = np.array([l["period"] for l in loops])
        onset_frac = np.array([l["onset_word"] / max(l["n_words"], 1) for l in loops])
        logger.info(f"detect: loop periods median {np.median(periods):.0f} words "
                    f"(P10 {np.percentile(periods, 10):.0f} / P90 {np.percentile(periods, 90):.0f}); "
                    f"onset at median {np.median(onset_frac):.0%} of chain")
    return labels


# ── stage: extract ────────────────────────────────────────────────────────────

def _select_chains(labels: list[dict], max_chains: int) -> list[dict]:
    """All loop + all clean chains (ambiguous excluded), optional total cap."""
    keep = [l for l in labels if l["class"] in ("loop", "clean")]
    if max_chains and len(keep) > max_chains:
        # preserve the loop:clean mix under the cap, deterministically
        loops = [l for l in keep if l["class"] == "loop"]
        cleans = [l for l in keep if l["class"] == "clean"]
        n_loop = min(len(loops), max(1, max_chains // 2))
        n_clean = max_chains - n_loop
        keep = loops[:n_loop] + cleans[:n_clean]
    return keep


def stage_extract(args, labels: list[dict]) -> None:
    import torch
    from src.chain_gen import load_model
    from src.hooks import ActivationCache
    from src.loop_geometry import (select_class_token_indices, windowed_state_metrics,
                                   word_spans)

    out = Path(args.out)
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)

    chains = {r["task_id"]: r for r in json.loads(Path(args.chains).read_text())}
    todo = [l for l in _select_chains(labels, args.max_chains)
            if not (shards / f"{l['task_id']}.npz").exists()]
    if args.smoke:
        todo = todo[:12]
    logger.info(f"extract: {len(todo)} chains to run "
                f"(shards dir already has {len(list(shards.glob('*.npz')))})")
    if not todo:
        return

    model, tokenizer = load_model(args.model_id, dtype="float16")
    model.eval()
    device = next(model.parameters()).device
    rng = np.random.default_rng(args.seed)

    from tqdm import tqdm
    n_fail = 0
    for lab in tqdm(todo, desc="extract"):
        try:
            rec = chains[lab["task_id"]]
            prompt = rec["prompt"]
            full_text = rec.get("full_text") or (prompt + rec["chain"])
            gen_start_char = len(prompt)

            enc = tokenizer(full_text, return_offsets_mapping=True,
                            return_tensors="pt", add_special_tokens=False,
                            truncation=True, max_length=args.max_seq_tokens)
            offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
            enc = {k: v.to(device) for k, v in enc.items()}

            # use_cache=False: a plain forward pass over an 8k-token cap-hit
            # chain otherwise allocates a full KV cache that is never reused and,
            # under variable sequence lengths, fragments GPU memory so per-chain
            # latency climbs without bound (56s->171s observed). The cache is
            # useless here — we read residual-stream hooks, not generate.
            with ActivationCache(model, layers=args.layers) as cache:
                with torch.no_grad():
                    model(**enc, use_cache=False)
                cache_np = {L: cache[L][0].float().numpy() for L in args.layers}

            starts = np.asarray([s for s, _ in offsets], dtype=np.int64)
            ends = np.asarray([e for _, e in offsets], dtype=np.int64)
            gen_tok = np.nonzero((starts >= gen_start_char) & (ends > starts))[0]
            if gen_tok.size < args.window:
                raise ValueError(f"only {gen_tok.size} generated tokens")
            gen_lo = int(gen_tok[0])

            # loop onset / guard in absolute char space
            if lab["class"] == "loop":
                onset_abs = gen_start_char + lab["onset_char"]
                spans = word_spans(rec["chain"])
                g_word = max(lab["onset_word"] - args.guard_words, 0)
                guard_abs = gen_start_char + spans[g_word][0]
                onset_tok = int(np.searchsorted(starts, onset_abs))
            else:
                onset_abs, guard_abs, onset_tok = -1, -1, -1

            sel = select_class_token_indices(offsets, gen_start_char, onset_abs,
                                             guard_abs, args.tokens_per_class, rng)

            payload: dict[str, np.ndarray] = {}
            for L in args.layers:
                H = cache_np[L]                           # (seq, hidden), on CPU
                m = windowed_state_metrics(H[gen_lo:], window=args.window,
                                           stride=args.stride)
                payload[f"pr_{L}"] = m["pr"]
                payload[f"unif_{L}"] = m["unif"]
                if L == args.layers[0]:
                    payload["centers"] = m["centers"]     # gen-relative token idx
                payload[f"Xin_{L}"] = H[sel["in"]].astype(np.float16)
                payload[f"Xout_{L}"] = H[sel["out"]].astype(np.float16)

            np.savez_compressed(
                shards / f"{lab['task_id']}.npz",
                **payload,
                onset_tok_gen=np.int64(onset_tok - gen_lo if onset_tok >= 0 else -1),
                n_gen_tokens=np.int64(gen_tok.size),
                is_loop=np.int64(lab["class"] == "loop"),
            )
        except Exception as e:                            # fail-soft per chain
            n_fail += 1
            logger.warning(f"extract FAILED {lab['task_id']}: {e}")
        finally:
            if torch.cuda.is_available():                 # de-fragment before next chain
                torch.cuda.empty_cache()
    logger.info(f"extract: done ({n_fail} failures)")


# ── stage: analyse ────────────────────────────────────────────────────────────

def _load_shards(out: Path) -> list[dict]:
    rows = []
    for p in sorted((out / "shards").glob("*.npz")):
        with np.load(p) as z:
            rows.append({"task_id": p.stem, **{k: z[k] for k in z.files}})
    return rows


def _probe_layer(shards: list[dict], L: int, seed: int) -> dict:
    """Chain-grouped loop probe at layer L → metrics + unit directions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import mannwhitneyu

    X, y, grp = [], [], []
    for s in shards:
        for cls, lab in ((f"Xin_{L}", 1), (f"Xout_{L}", 0)):
            V = s[cls]
            if V.shape[0]:
                X.append(V.astype(np.float32))
                y.append(np.full(V.shape[0], lab))
                grp.append(np.full(V.shape[0], s["task_id"], dtype=object))
    X = np.concatenate(X); y = np.concatenate(y); grp = np.concatenate(grp)

    def _fit(Xtr, ytr):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                 random_state=seed).fit(sc.transform(Xtr), ytr)
        return sc, clf

    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups=grp):
        sc, clf = _fit(X[tr], y[tr])
        oof[te] = clf.decision_function(sc.transform(X[te]))
    token_auc = float(roc_auc_score(y, oof))

    # chain-level: top-quartile mean of a chain's OOF scores ("does it contain a loop")
    chain_score, chain_is_loop = {}, {}
    for s in shards:
        m = grp == s["task_id"]
        sc_ = np.sort(oof[m])
        if sc_.size == 0:
            continue
        chain_score[s["task_id"]] = float(sc_[-max(1, sc_.size // 4):].mean())
        chain_is_loop[s["task_id"]] = bool(s["is_loop"])
    lo = [v for k, v in chain_score.items() if chain_is_loop[k]]
    cl = [v for k, v in chain_score.items() if not chain_is_loop[k]]
    u = mannwhitneyu(lo, cl, alternative="greater")
    chain_auc = float(u.statistic / (len(lo) * len(cl)))

    # final directions on ALL data (raw-space normal = w / sigma, then unit)
    sc, clf = _fit(X, y)
    w = clf.coef_[0] / np.where(sc.scale_ == 0, 1.0, sc.scale_)
    probe_dir = (w / np.linalg.norm(w)).astype(np.float32)
    mu_diff = X[y == 1].mean(0) - X[y == 0].mean(0)
    meandiff_dir = (mu_diff / np.linalg.norm(mu_diff)).astype(np.float32)

    return {
        "n_tokens": int(len(y)), "n_in": int(y.sum()),
        "n_chains": len(chain_score), "n_loop_chains": len(lo),
        "token_auc_grouped_oof": token_auc,
        "chain_auc": chain_auc, "chain_mw_p": float(u.pvalue),
        "cos_probe_vs_meandiff": float(probe_dir @ meandiff_dir),
        "probe_dir": probe_dir, "meandiff_dir": meandiff_dir,
    }


def _precedence(shards: list[dict], L: int, pre: int, base_lo: int, base_hi: int) -> dict:
    """E9.0b: Δ(pre-onset − baseline) for loop chains vs matched-position clean."""
    from scipy.stats import wilcoxon, mannwhitneyu

    onset_fracs = [s["onset_tok_gen"] / s["n_gen_tokens"]
                   for s in shards if s["is_loop"] and s["onset_tok_gen"] > 0]
    med_frac = float(np.median(onset_fracs)) if onset_fracs else 0.5

    def _delta(s, onset):
        c = s["centers"]
        for key in ("pr", "unif"):
            v = s[f"{key}_{L}"]
            pre_m = (c >= onset - pre) & (c < onset)
            base_m = (c >= base_lo) & (c < base_hi)
            if pre_m.sum() >= 2 and base_m.sum() >= 2 and onset - pre > base_hi:
                yield key, float(np.nanmean(v[pre_m]) - np.nanmean(v[base_m]))

    d = {"pr": {"loop": [], "clean": []}, "unif": {"loop": [], "clean": []}}
    for s in shards:
        if s["is_loop"] and s["onset_tok_gen"] > 0:
            onset, grp_ = int(s["onset_tok_gen"]), "loop"
        else:
            onset, grp_ = int(med_frac * s["n_gen_tokens"]), "clean"
        for key, val in _delta(s, onset):
            d[key][grp_].append(val)

    outres = {"median_onset_frac": med_frac}
    for key in ("pr", "unif"):
        lo_, cl_ = d[key]["loop"], d[key]["clean"]
        r = {"n_loop": len(lo_), "n_clean": len(cl_),
             "delta_loop_mean": float(np.mean(lo_)) if lo_ else None,
             "delta_clean_mean": float(np.mean(cl_)) if cl_ else None}
        if len(lo_) >= 10:
            r["wilcoxon_p_loop"] = float(wilcoxon(lo_).pvalue)
        if len(lo_) >= 10 and len(cl_) >= 10:
            alt = "less" if key == "pr" else "greater"   # PR drops, uniformity rises
            r["mw_p_loop_vs_clean"] = float(
                mannwhitneyu(lo_, cl_, alternative=alt).pvalue)
        outres[key] = r

    # aligned average curve (loop chains, offset relative to onset)
    bins = np.arange(-2048, 1025, 128)
    curve = {k: [[] for _ in range(len(bins) - 1)] for k in ("pr", "unif")}
    for s in shards:
        if not (s["is_loop"] and s["onset_tok_gen"] > 0):
            continue
        rel = s["centers"] - int(s["onset_tok_gen"])
        which = np.digitize(rel, bins) - 1
        for k in ("pr", "unif"):
            v = s[f"{k}_{L}"]
            for b in range(len(bins) - 1):
                m = which == b
                if m.any():
                    curve[k][b].append(float(np.nanmean(v[m])))
    outres["aligned_curve"] = {
        "bin_lo": bins[:-1].tolist(),
        **{k: [float(np.mean(c)) if c else None for c in curve[k]] for k in curve},
        "n": [len(c) for c in curve["pr"]],
    }
    return outres


def stage_analyse(args, labels: list[dict]) -> None:
    from src.loop_geometry import cosine_report

    out = Path(args.out)
    shards = _load_shards(out)
    if not shards:
        raise SystemExit("analyse: no shards — run --stage extract first")
    logger.info(f"analyse: {len(shards)} shards "
                f"({sum(bool(s['is_loop']) for s in shards)} loop)")

    vec_dir = Path(args.vectors)
    vmeta = json.loads((vec_dir / "metadata.json").read_text())
    beh_layers = vmeta["_provenance"]["layers"]

    report: dict = {"config": {k: v for k, v in vars(args).items()},
                    "n_shards": len(shards)}
    directions: dict[str, np.ndarray] = {}

    report["probe"] = {}
    for L in args.layers:
        res = _probe_layer(shards, L, args.seed)
        directions[f"probe_L{L}"] = res.pop("probe_dir")
        directions[f"meandiff_L{L}"] = res.pop("meandiff_dir")
        report["probe"][str(L)] = res
        logger.info(f"probe L{L}: token AUC {res['token_auc_grouped_oof']:.3f} | "
                    f"chain AUC {res['chain_auc']:.3f} (p={res['chain_mw_p']:.2g})")

    # E9.0a cosine table — each behaviour at its own steering layer
    report["cosines"] = {}
    for beh in BEHAVIOURS:
        L = int(beh_layers[beh])
        if L not in args.layers:
            logger.warning(f"{beh}: steering layer {L} not extracted — skipped")
            continue
        row = {}
        for arm in ARMS:
            v = np.load(vec_dir / f"{beh}_{arm}.npy").astype(np.float32)
            row[arm] = {
                "probe": cosine_report(directions[f"probe_L{L}"], v),
                "meandiff": cosine_report(directions[f"meandiff_L{L}"], v),
            }
        report["cosines"][beh] = {"layer": L, **row}

    # E9.0b precedence per layer
    report["precedence"] = {
        str(L): _precedence(shards, L, args.pre_window, args.base_lo, args.base_hi)
        for L in args.layers
    }

    np.savez(out / "directions.npz", **directions)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    _write_md(out, report)
    logger.info(f"analyse: wrote {out}/report.json + REPORT.md + directions.npz")


def _write_md(out: Path, r: dict) -> None:
    md = ["# E9.0 loop-geometry report\n",
          f"Shards: {r['n_shards']}\n",
          "## Probe (E9.0a)\n",
          "| layer | token AUC (grouped OOF) | chain AUC | chain p | probe↔meandiff cos |",
          "|--:|--:|--:|--:|--:|"]
    for L, p in r["probe"].items():
        md.append(f"| {L} | {p['token_auc_grouped_oof']:.3f} | {p['chain_auc']:.3f} "
                  f"| {p['chain_mw_p']:.2g} | {p['cos_probe_vs_meandiff']:.3f} |")
    md += ["\n## Loop-direction vs steering vectors (H-B gate: |cos| ≳ 0.3)\n",
           "| behaviour | layer | arm | cos (probe) | z | cos (meandiff) |",
           "|---|--:|---|--:|--:|--:|"]
    for beh, row in r["cosines"].items():
        for arm in ARMS:
            c = row[arm]
            md.append(f"| {beh} | {row['layer']} | {arm} | {c['probe']['cos']:+.3f} "
                      f"| {c['probe']['z_vs_random']:.1f} | {c['meandiff']['cos']:+.3f} |")
    md += ["\n## State-collapse precedence (E9.0b)\n",
           "| layer | metric | Δ loop (pre−base) | Δ clean | Wilcoxon p | MW p (loop vs clean) |",
           "|--:|---|--:|--:|--:|--:|"]
    for L, pr in r["precedence"].items():
        for key in ("pr", "unif"):
            q = pr[key]
            md.append(f"| {L} | {key} | {q.get('delta_loop_mean') if q.get('delta_loop_mean') is None else format(q['delta_loop_mean'], '.3f')} "
                      f"| {q.get('delta_clean_mean') if q.get('delta_clean_mean') is None else format(q['delta_clean_mean'], '.3f')} "
                      f"| {q.get('wilcoxon_p_loop', '—')} | {q.get('mw_p_loop_vs_clean', '—')} |")
    (out / "REPORT.md").write_text("\n".join(md) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    labels = stage_detect(args)
    if args.stage in ("all", "extract"):
        stage_extract(args, labels)
    if args.stage in ("all", "analyse"):
        stage_analyse(args, labels)


if __name__ == "__main__":
    main()
