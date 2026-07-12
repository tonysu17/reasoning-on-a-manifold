#!/usr/bin/env python3
"""24_belief_lens.py — A1: belief-state trajectory via the logit lens (PG §12).

The model's OWN evolving next-token belief, read per reasoning step with the
logit lens (unembed the residual through the final norm), turned into a
LABEL-FREE value/confidence trajectory. Two things this buys, both pre-registered
in PREDICTIVE_GEOMETRY.md §12 (A1):

  1. The strongest baseline the trained value head (R3) must beat: if a free
     belief-trajectory predicts COLLAPSE as well as a trained sequence model,
     "learned value" is not earned.
  2. The M3 order-sensitive POSITIVE the pilot lacked — belief *trajectory*
     features (slope / end-rise) are order-bearing by construction, so their
     step-shuffle null should drop while the order-free *occupancy* features
     (mean / min entropy) do not.

Primary outcome = COLLAPSE (loop onset, src/loop_geometry.detect_loop_tail), with
the sealed R3 leak-guard: for a collapse-bound chain, only steps STRICTLY BEFORE
the detected loop onset enter the features (loop-region steps never leak in).
Correctness is a secondary transfer endpoint on the full trajectory.

SPARSE ARM ($0, local, this script): the lens is applied to the saved
mean-pooled behaviour-span vectors — a *smeared* belief, "suggestive not
confirmatory" (M4). DENSE ARM (confirmatory, owed): per-token lens at the real
read token, folded into the R4 extraction pod. TUNED LENS (Belrose et al., a
learned per-layer affine map) is the faithful upgrade (A1b) — left as a hook.

  python3 24_belief_lens.py --layers 14,17,27           # real run (loads model)
  python3 24_belief_lens.py --smoke 120                 # random lens, validates plumbing (no model)

Output: results/predict/R1-1.5B/belief/belief_lens_{tag}.json + .md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import backup_existing, provenance  # noqa: E402
from src.predict.collapse import collapse_labels, truncate_preonset  # noqa: E402
from src.predict.trajectory_dataset import StepDataset, build_step_datasets  # noqa: E402
from src.predict.evaluation import (  # noqa: E402
    align_labels, grouped_auc, length_features, raw_step_features,
    raw_curvature_features,
)
from src.predict.nulls_predict import step_shuffle_null  # noqa: E402

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def log(msg: str) -> None:
    print(f"[A1 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# collapse_labels + truncate_preonset now live in src/predict/collapse.py (shared
# with 26_value_head.py; single source of truth — no copy-paste drift).


# ── the logit lens ──────────────────────────────────────────────────────────

def load_lens():
    """Return (norm, lm_head, device, dtype) — the two submodules the lens needs."""
    import torch
    from transformers import AutoModelForCausalLM

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    torch.set_grad_enabled(False)
    log(f"loading {MODEL_ID} on {device}/{dtype} for the lens (norm + lm_head)")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    return model.model.norm, model.lm_head, device, dtype


def lens_entropy_top1(X: np.ndarray, bundle, chunk: int = 256):
    """Per-row next-token belief: (entropy in nats, top-1 prob). Chunked over rows."""
    import torch

    norm, lm_head, device, dtype = bundle
    n = len(X)
    H = np.empty(n, dtype=np.float32)
    P1 = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        xb = torch.tensor(X[i:i + chunk], device=device, dtype=dtype)
        logits = lm_head(norm(xb)).float()          # (b, V) — logit lens at this layer
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        H[i:i + chunk] = (-(p * logp).sum(-1)).cpu().numpy()
        P1[i:i + chunk] = p.max(-1).values.cpu().numpy()
    return H, P1


def build_lens_lookup(datasets, bundle, chunk: int) -> dict:
    """{(chain_id, orig_span_idx): (entropy, top1)} over every retained step."""
    index, rows = [], []
    for ds in datasets:
        for t in range(ds.T):
            index.append((ds.chain_id, int(ds.orig_indices[t])))
            rows.append(ds.X[t])
    if not rows:
        return {}
    H, P1 = lens_entropy_top1(np.asarray(rows, dtype=np.float32), bundle, chunk)
    return {index[i]: (float(H[i]), float(P1[i])) for i in range(len(index))}


def random_lens_lookup(datasets, seed: int = 42) -> dict:
    """Smoke lookup: random belief stats (validates plumbing; AUCs should be ~0.5)."""
    rng = np.random.default_rng(seed)
    out = {}
    for ds in datasets:
        for t in range(ds.T):
            out[(ds.chain_id, int(ds.orig_indices[t]))] = (
                float(rng.uniform(0.0, 6.0)), float(rng.uniform(0.0, 1.0)))
    return out


def belief_datasets(datasets, lookup: dict) -> list[StepDataset]:
    """Re-express each chain as a (T, 2) belief trajectory: columns [entropy, top1]."""
    out = []
    for ds in datasets:
        stats = np.asarray(
            [lookup[(ds.chain_id, int(ds.orig_indices[t]))] for t in range(ds.T)],
            dtype=np.float32,
        ).reshape(-1, 2)
        out.append(StepDataset(
            chain_id=ds.chain_id, X=stats, orig_indices=ds.orig_indices,
            behaviours=ds.behaviours, truncated=ds.truncated, correct=ds.correct,
            difficulty=ds.difficulty, category=ds.category,
        ))
    return out


# ── belief feature sets (order-bearing vs order-free) ───────────────────────

def _slope(y: np.ndarray) -> float:
    T = y.shape[0]
    if T < 2:
        return 0.0
    t = np.linspace(0.0, 1.0, T)
    try:
        return float(np.polyfit(t, y, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def belief_trajectory_features(bel_ds):
    """ORDER-BEARING lens features (the M3 target): slope, end-rise over the chain."""
    names = ["H_slope", "H_end", "H_end_minus_start", "H_delta_last3", "p1_slope"]
    cids, rows = [], []
    for ds in bel_ds:
        H = ds.X[:, 0]
        p1 = ds.X[:, 1]
        T = ds.T
        rows.append([
            _slope(H),
            float(H[-1]),
            float(H[-1] - H[0]),
            float(H[-1] - H[max(0, T - 4)]),
            _slope(p1),
        ])
        cids.append(ds.chain_id)
    return (np.asarray(rows, dtype=float).reshape(-1, len(names)),
            np.asarray(cids, dtype=object), names)


def belief_occupancy_features(bel_ds):
    """ORDER-FREE lens features: mean/spread of belief over the chain (shuffle-invariant)."""
    names = ["H_mean", "H_std", "H_min", "H_max", "p1_mean"]
    cids, rows = [], []
    for ds in bel_ds:
        H = ds.X[:, 0]
        p1 = ds.X[:, 1]
        rows.append([float(H.mean()), float(H.std()), float(H.min()),
                     float(H.max()), float(p1.mean())])
        cids.append(ds.chain_id)
    return (np.asarray(rows, dtype=float).reshape(-1, len(names)),
            np.asarray(cids, dtype=object), names)


def belief_all_features(bel_ds):
    """Trajectory + occupancy concatenated — the headline belief feature set."""
    tf, cids, tn = belief_trajectory_features(bel_ds)
    of, _, on = belief_occupancy_features(bel_ds)
    return np.concatenate([tf, of], axis=1), cids, tn + on


# ── scoring ──────────────────────────────────────────────────────────────────

def _auc(feat_tuple, label_map) -> dict:
    feats, cids, _ = feat_tuple
    return grouped_auc(feats, align_labels(cids, label_map), cids)


def _concat(ft_a, ft_b):
    """Column-concatenate two (feats, cids, names) tuples built from the SAME,
    equally-ordered datasets — used for the incremental-over-length test."""
    fa, ca, na = ft_a
    fb, cb, nb = ft_b
    if not np.array_equal(ca, cb):
        raise ValueError("feature tuples are not row-aligned")
    return np.concatenate([fa, fb], axis=1), ca, na + nb


def score(emb_ds, bel_ds, label_map: dict, args, do_shuffle: bool) -> dict:
    """AUROC table (belief sets vs baselines) + the M3 step-shuffle null.

    CF-8 note: for the COLLAPSE endpoint, chain length is a dominant confound
    (loops run to the token cap ⇒ `length_gap_trunc` alone separates loop/clean
    almost perfectly). So the operative test is not "belief beats chance" but
    "belief beats length" and — the real bar — "length+belief beats length"
    (incremental). `belief_trajectory` (order-bearing, length-neutral) is the
    scientifically load-bearing set; `belief_occupancy` min/max partly encode
    step-count, so read it against the length baseline.
    """
    emb = [d for d in emb_ds if d.chain_id in label_map]
    bel = [d for d in bel_ds if d.chain_id in label_map]
    n = len(bel)
    n_pos = sum(1 for d in bel if label_map.get(d.chain_id))
    res = {"n": n, "n_pos": n_pos, "n_neg": n - n_pos}
    if n < 10 or n_pos < 3 or (n - n_pos) < 3:
        res["status"] = "insufficient (need >=10 chains, >=3/class)"
        return res

    lf = length_features(emb)
    res["auc"] = {
        "belief_all": _auc(belief_all_features(bel), label_map),
        "belief_trajectory": _auc(belief_trajectory_features(bel), label_map),
        "belief_occupancy": _auc(belief_occupancy_features(bel), label_map),
        "length_gap_trunc": _auc(lf, label_map),
        "persistence_step": _auc(raw_step_features(emb), label_map),
        "rung0_curvature": _auc(raw_curvature_features(emb), label_map),
        # incremental controls (CF-8): does belief ADD over length?
        "length_PLUS_belief_traj": _auc(_concat(lf, belief_trajectory_features(bel)), label_map),
        "length_PLUS_belief_all": _auc(_concat(lf, belief_all_features(bel)), label_map),
    }

    if do_shuffle:
        def stat(dsets):
            feats, cids, _ = belief_trajectory_features(dsets)
            return grouped_auc(feats, align_labels(cids, label_map), cids)["auc_oof"]
        ss = step_shuffle_null(bel, stat, n_resamples=args.shuffle_resamples, seed=42)
        res["m3_step_shuffle_trajectory"] = {
            "real": ss.real_value, "null_mean": ss.null_mean,
            "null_p97_5": ss.null_p97_5, "p_value": ss.p_value, "n": ss.n_resamples,
        }
    return res


def evaluate_layer(chains, chains_by_id, activations_dir, layer, collapse_map,
                   onset_map, correctness_map, bundle, args) -> dict:
    emb_full = build_step_datasets(chains, activations_dir, layer)
    emb_pre = truncate_preonset(emb_full, chains_by_id, onset_map)
    lookup = (random_lens_lookup(emb_full) if bundle is None
              else build_lens_lookup(emb_full, bundle, args.chunk))
    bel_full = belief_datasets(emb_full, lookup)
    bel_pre = belief_datasets(emb_pre, lookup)

    out = {"layer": layer,
           "n_chains_full": len(emb_full),
           "n_chains_preonset": len(emb_pre),
           "collapse_primary": score(emb_pre, bel_pre, collapse_map, args, do_shuffle=True)}
    if correctness_map:
        out["correctness_secondary"] = score(
            emb_full, bel_full, correctness_map, args, do_shuffle=True)
    return out


# ── correctness labels (secondary endpoint) ─────────────────────────────────

def correctness_label_map(path: str, min_confidence: str | None) -> dict:
    from src.predict.labels import load_correctness_labels
    labels = load_correctness_labels(path)
    order = {"low": 0, "medium": 1, "high": 2}
    floor = order.get(min_confidence, -1)
    out = {}
    for cid, rec in labels.items():
        if rec.get("correct") is None:
            continue
        if min_confidence and order.get(str(rec.get("confidence")).lower(), -1) < floor:
            continue
        out[cid] = bool(rec["correct"])
    return out


# ── reporting ────────────────────────────────────────────────────────────────

def _auc_rows(auc: dict) -> list[str]:
    lines = ["| feature set | AUC (oof) | fold mean±std | stable | n_pos/n |",
             "|---|---|---|---|---|"]
    for name, a in auc.items():
        lines.append(f"| {name} | {a['auc_oof']:.3f} | "
                     f"{a['auc_fold_mean']:.3f}±{a['auc_fold_std']:.3f} | "
                     f"{a['stable']} | {a['n_pos']}/{a['n']} |")
    return lines


def _block_md(title: str, blk: dict) -> list[str]:
    lines = [f"### {title}"]
    if "auc" not in blk:
        return lines + [f"- {blk.get('status')}", ""]
    lines.append(f"- chains: {blk['n']} (pos {blk['n_pos']} / neg {blk['n_neg']})")
    lines += _auc_rows(blk["auc"])
    if "m3_step_shuffle_trajectory" in blk:
        m = blk["m3_step_shuffle_trajectory"]
        lines.append("")
        lines.append(f"- **M3 order test** (trajectory features): real {m['real']:.3f} "
                     f"vs step-shuffle null {m['null_mean']:.3f} (p97.5 {m['null_p97_5']:.3f}), "
                     f"p={m['p_value']:.3f} — order-bearing if real ≫ null.")
    lines.append("")
    return lines


def write_md(path: Path, results: list[dict], counts: dict, smoke: bool) -> None:
    lines = ["# A1 — belief-state trajectory (logit lens, sparse arm)", ""]
    if smoke:
        lines.append("**SMOKE — random lens stats; every AUC should sit near 0.5.**\n")
    lines.append(f"Collapse classes over corpus: loop {counts.get('loop',0)} / "
                 f"clean {counts.get('clean',0)} / ambiguous {counts.get('ambiguous',0)} "
                 f"(ambiguous dropped).\n")
    lines.append("Sparse arm: lens applied to mean-pooled span vectors (smeared belief; "
                 "M4 — suggestive, dense per-token arm confirmatory). Entropy in nats.\n")
    for r in results:
        lines.append(f"## layer {r['layer']}  "
                     f"(chains: {r['n_chains_preonset']} pre-onset / {r['n_chains_full']} full)")
        lines += _block_md("Collapse (primary, leak-guarded)", r["collapse_primary"])
        if "correctness_secondary" in r:
            lines += _block_md("Correctness (secondary, full trajectory)",
                               r["correctness_secondary"])
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", default="data/activations/R1-1.5B")
    ap.add_argument("--chains", default="data/annotated_R1-1.5B.json")
    ap.add_argument("--labels", default="data/correctness_R1-1.5B_pilot.json")
    ap.add_argument("--layers", default="14,17,27")
    ap.add_argument("--min-confidence", choices=["low", "medium", "high"], default=None)
    ap.add_argument("--shuffle-resamples", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=256, help="lens matmul batch (rows)")
    ap.add_argument("--no-model", action="store_true",
                    help="skip the lens load; use random belief stats (plumbing only)")
    ap.add_argument("--smoke", type=int, default=0,
                    help="N: cap corpus to first N chains + random lens (fast plumbing check)")
    ap.add_argument("--out", default="results/predict/R1-1.5B/belief")
    args = ap.parse_args()

    chains = json.load(open(args.chains))
    smoke = bool(args.smoke)
    if smoke:
        chains = chains[: args.smoke]
    chains_by_id = {(c.get("task_id") or c.get("id") or ""): c for c in chains}

    collapse_map, onset_map, counts = collapse_labels(chains)
    log(f"collapse classes: {counts} ({len(collapse_map)} labelled loop/clean)")

    correctness_map = {} if smoke else correctness_label_map(args.labels, args.min_confidence)
    if correctness_map:
        log(f"{len(correctness_map)} correctness labels "
            f"({sum(correctness_map.values())} correct)")

    bundle = None if (smoke or args.no_model) else load_lens()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for layer in [int(x) for x in args.layers.split(",")]:
        log(f"=== layer {layer} ===")
        r = evaluate_layer(chains, chains_by_id, args.activations, layer,
                           collapse_map, onset_map, correctness_map, bundle, args)
        results.append(r)
        cp = r["collapse_primary"]
        if "auc" in cp:
            for name, a in cp["auc"].items():
                log(f"  collapse {name:18s} AUC={a['auc_oof']:.3f} "
                    f"(pos {a['n_pos']}/{a['n']})")
            if "m3_step_shuffle_trajectory" in cp:
                m = cp["m3_step_shuffle_trajectory"]
                log(f"  M3 trajectory real={m['real']:.3f} shuffle-null={m['null_mean']:.3f} "
                    f"p={m['p_value']:.3f}")
        else:
            log(f"  collapse: {cp.get('status')}")

    tag = "smoke" if smoke else "sparse"
    out_json = out_dir / f"belief_lens_{tag}.json"
    backup_existing(out_json)
    payload = {"results": results, "collapse_class_counts": counts, "smoke": smoke,
               "provenance": provenance(args, inputs=[args.chains] +
                                        ([] if smoke else [args.labels]))}
    json.dump(payload, open(out_json, "w"), indent=2, default=str)
    write_md(out_dir / f"belief_lens_{tag}.md", results, counts, smoke)
    log(f"wrote {out_json} and belief_lens_{tag}.md")


if __name__ == "__main__":
    main()
