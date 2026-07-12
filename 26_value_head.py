#!/usr/bin/env python3
"""26_value_head.py — R3: the trained value head vs the free belief lens (PG §11.3).

The decisive value-track test. A1 (24_belief_lens) showed the model's OWN logit-lens
confidence already predicts CORRECTNESS at AUROC 0.71 (occupancy, order-free), and that
COLLAPSE is trivially length-dominated (0.986). R3 asks whether a TRAINED sequence model
over the pooled reasoning-step embeddings earns its keep:

  (1) VALUE — does the trained V beat the free belief floor (A1: 0.71 correctness) and
      the length/persistence/curvature baselines? If V ≤ best baseline ⇒ value track
      CLOSED (R3-K): occupancy is all there is.
  (2) ORDER (M3) — does temporal context help? Two order-blind counterparts: the SAME
      GRU trained on step-SHUFFLED sequences (capacity-matched), and a mean/std-pooled
      logistic (architecture-blind). If real ≈ shuffled ≈ pooled, reasoning-value is
      state-occupancy even with a model that COULD use order — the strongest version of
      the pilot's occupancy-not-dynamics verdict. If real ≫ shuffled, R3 is the first
      dynamics-value result (pairs with A5's causal-fidelity signal).

Primary outcome = COLLAPSE (leak-guarded pre-onset, shared `src/predict/collapse.py`);
correctness = secondary. Chain-grouped 5-fold OOF, inner-val early stop, class-weighted
BCE. Local CPU (tiny GRU, d=64) — does not touch MPS, so it runs alongside a sweep.

  python3 26_value_head.py --layers 14,17,27
  python3 26_value_head.py --smoke 300         # random labels, validates plumbing (AUC~0.5)

Output: results/predict/R1-1.5B/value/value_head_{tag}.json + .md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import backup_existing, provenance  # noqa: E402
from src.predict.collapse import collapse_labels, truncate_preonset  # noqa: E402
from src.predict.trajectory_dataset import build_step_datasets  # noqa: E402
from src.predict.evaluation import (  # noqa: E402
    align_labels, grouped_auc, length_features, raw_step_features, raw_curvature_features,
)

BELIEF_FLOOR = {"correctness": 0.71, "collapse": 0.986}  # A1 free-lens / length bars to beat


def log(msg: str) -> None:
    print(f"[R3 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── the value head ───────────────────────────────────────────────────────────

def _make_gru(d_in: int, d_hidden: int, dropout: float):
    import torch.nn as nn

    class StepGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(d_in, d_hidden)
            self.gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(d_hidden, 1)

        def forward(self, x, lengths):
            import torch
            from torch.nn.utils.rnn import pack_padded_sequence
            h = torch.relu(self.proj(x))
            packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            _, hn = self.gru(packed)
            return self.head(self.drop(hn[-1])).squeeze(-1)

    return StepGRU()


def _prep(datasets, label_map, max_steps: int):
    """→ (seqs: list[(T,d)], y: (n,), groups: (n,)) over labelled chains; last max_steps kept."""
    seqs, ys, groups = [], [], []
    for ds in datasets:
        if ds.chain_id not in label_map:
            continue
        X = np.asarray(ds.X, dtype=np.float32)
        if X.shape[0] > max_steps:
            X = X[-max_steps:]                      # recent history most relevant
        seqs.append(X)
        ys.append(1.0 if label_map[ds.chain_id] else 0.0)
        groups.append(ds.chain_id)
    return seqs, np.asarray(ys, dtype=np.float32), np.asarray(groups, dtype=object)


def _batches(seqs, y, idx, bs, device, rng, shuffle_steps=False):
    import torch
    order = rng.permutation(idx)
    for i in range(0, len(order), bs):
        b = order[i:i + bs]
        arrs = []
        for j in b:
            s = seqs[j]
            if shuffle_steps and s.shape[0] > 1:
                s = s[rng.permutation(s.shape[0])]
            arrs.append(s)
        lens = np.array([a.shape[0] for a in arrs])
        maxT = int(lens.max())
        X = torch.zeros(len(arrs), maxT, arrs[0].shape[1])
        for r, a in enumerate(arrs):
            X[r, :a.shape[0]] = torch.from_numpy(a)
        yield (X.to(device), torch.from_numpy(lens), torch.from_numpy(y[b]).to(device))


def _eval_logits(model, seqs, idx, device):
    import torch
    model.eval()
    out = np.full(len(seqs), np.nan)
    with torch.no_grad():
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            lens = np.array([seqs[j].shape[0] for j in b])
            maxT = int(lens.max())
            X = torch.zeros(len(b), maxT, seqs[b[0]].shape[1])
            for r, j in enumerate(b):
                X[r, :seqs[j].shape[0]] = torch.from_numpy(seqs[j])
            logits = model(X.to(device), torch.from_numpy(lens))
            out[b] = logits.cpu().numpy()
    return out


def _train(seqs, y, tr, va, device, d_hidden, dropout, epochs, patience, lr, seed,
           shuffle_steps):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _make_gru(seqs[0].shape[1], d_hidden, dropout).to(device)
    n_pos = float(y[tr].sum())
    n_neg = float(len(tr) - n_pos)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / max(n_pos, 1.0)], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_auc, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for X, lens, yb in _batches(seqs, y, tr, 32, device, rng, shuffle_steps):
            opt.zero_grad()
            crit(model(X, lens), yb).backward()
            opt.step()
        va_logits = _eval_logits(model, seqs, va, device)
        try:
            va_auc = roc_auc_score(y[va], va_logits[va])
        except ValueError:
            va_auc = 0.5
        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def oof_gru_auc(seqs, y, groups, *, device="cpu", d_hidden=64, dropout=0.1, epochs=60,
                patience=8, lr=1e-3, n_splits=5, seed=0, shuffle_steps=False) -> float:
    """Chain-grouped OOF AUROC of the GRU value head (optionally on shuffled steps)."""
    if len(seqs) < 10 or np.unique(y).size < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=min(n_splits, np.unique(groups).size))
    oof = np.full(len(seqs), np.nan)
    Xdummy = np.zeros((len(seqs), 1))
    for fold, (tr, te) in enumerate(gkf.split(Xdummy, groups=groups)):
        # inner val split by group for early stopping
        gtr = np.unique(groups[tr])
        rng = np.random.default_rng(seed + fold)
        va_groups = set(rng.choice(gtr, size=max(1, int(0.15 * len(gtr))), replace=False))
        va = np.array([i for i in tr if groups[i] in va_groups])
        tr_in = np.array([i for i in tr if groups[i] not in va_groups])
        if va.size == 0 or tr_in.size == 0 or np.unique(y[tr_in]).size < 2:
            tr_in, va = tr, tr
        # standardize on training steps only
        scaler = StandardScaler().fit(np.concatenate([seqs[i] for i in tr_in], axis=0))
        sc = [scaler.transform(s).astype(np.float32) for s in seqs]
        model = _train(sc, y, tr_in, va, device, d_hidden, dropout, epochs, patience,
                       lr, seed + fold, shuffle_steps)
        oof[te] = _eval_logits(model, sc, te, device)[te]
    valid = ~np.isnan(oof)
    if np.unique(y[valid]).size < 2:
        return float("nan")
    return float(roc_auc_score(y[valid], oof[valid]))


# ── order-blind + baseline references ────────────────────────────────────────

def pooled_logistic_auc(seqs, y, groups) -> dict:
    """Mean+std pooled sequence → chain-grouped logistic (architecture order-blind)."""
    feats = np.array([np.concatenate([s.mean(0), s.std(0)]) for s in seqs], dtype=float)
    labels = np.where(np.isfinite(y), y, np.nan)
    return grouped_auc(feats, labels, groups)


def _auc(feat_tuple, label_map) -> dict:
    feats, cids, _ = feat_tuple
    return grouped_auc(feats, align_labels(cids, label_map), cids)


# ── endpoint scoring ─────────────────────────────────────────────────────────

def score(datasets, label_map, endpoint, args) -> dict:
    dsub = [d for d in datasets if d.chain_id in label_map]
    n = len(dsub)
    n_pos = sum(1 for d in dsub if label_map.get(d.chain_id))
    res = {"n": n, "n_pos": n_pos, "n_neg": n - n_pos}
    if n < 20 or n_pos < 5 or (n - n_pos) < 5:
        res["status"] = "insufficient (need >=20 chains, >=5/class)"
        return res

    seqs, y, groups = _prep(dsub, label_map, args.max_steps)
    v_real = oof_gru_auc(seqs, y, groups, device=args.device, epochs=args.epochs, seed=0)
    v_shuf = oof_gru_auc(seqs, y, groups, device=args.device, epochs=args.epochs, seed=0,
                         shuffle_steps=True)

    res["auc"] = {
        "value_gru": {"auc_oof": v_real},
        "value_gru_stepshuffled": {"auc_oof": v_shuf},           # M3: order destroyed, capacity-matched
        "pooled_logistic": pooled_logistic_auc(seqs, y, groups),  # M3: order-blind architecture
        "length_gap_trunc": _auc(length_features(dsub), label_map),
        "persistence_step": _auc(raw_step_features(dsub), label_map),
        "rung0_curvature": _auc(raw_curvature_features(dsub), label_map),
    }
    # verdict (R3-P1 / R3-K)
    baselines = [res["auc"][k]["auc_oof"] for k in
                 ("pooled_logistic", "length_gap_trunc", "persistence_step", "rung0_curvature")]
    best_base = float(np.nanmax(baselines + [BELIEF_FLOOR.get(endpoint, 0.5)]))
    res["best_baseline_incl_belief_floor"] = best_base
    res["value_beats_baselines_by"] = float(v_real - best_base)
    res["order_gain_vs_shuffled"] = float(v_real - v_shuf)
    res["verdict"] = (
        "value_earns_keep" if (v_real - best_base) >= 0.05 and v_real >= 0.65
        else "value_track_closed (V <= best free baseline)"
    )
    res["m3_order"] = ("order_bearing" if (v_real - v_shuf) >= 0.05
                       else "occupancy (order adds <0.05 even with a sequence model)")
    return res


# ── reporting ────────────────────────────────────────────────────────────────

def write_md(path: Path, results, counts, smoke):
    L = ["# R3 — trained value head vs the free belief lens", ""]
    if smoke:
        L.append("**SMOKE — random labels; every AUROC ~0.5.**\n")
    L.append(f"Bars to beat: correctness = A1 belief_occupancy **0.71**; "
             f"collapse = length **0.986** (both from 24_belief_lens). "
             f"Collapse classes: loop {counts.get('loop',0)} / clean {counts.get('clean',0)} "
             f"/ ambiguous {counts.get('ambiguous',0)} (dropped).\n")
    for r in results:
        L.append(f"## layer {r['layer']}")
        for ep in ("collapse", "correctness"):
            blk = r.get(ep)
            if not blk:
                continue
            L.append(f"### {ep}")
            if "auc" not in blk:
                L.append(f"- {blk.get('status')}\n")
                continue
            L.append(f"- chains {blk['n']} (pos {blk['n_pos']} / neg {blk['n_neg']})")
            L.append("| model | AUROC (oof) |")
            L.append("|---|---|")
            for name, a in blk["auc"].items():
                L.append(f"| {name} | {a['auc_oof']:.3f} |")
            L += ["",
                  f"- value beats best free baseline (incl. belief floor {blk['best_baseline_incl_belief_floor']:.3f}) "
                  f"by **{blk['value_beats_baselines_by']:+.3f}** → **{blk['verdict']}**",
                  f"- order gain vs step-shuffled GRU: **{blk['order_gain_vs_shuffled']:+.3f}** → {blk['m3_order']}",
                  ""]
    path.write_text("\n".join(L))


def _correctness_map(path, min_conf):
    from src.predict.labels import load_correctness_labels
    labels = load_correctness_labels(path)
    order = {"low": 0, "medium": 1, "high": 2}
    floor = order.get(min_conf, -1)
    out = {}
    for cid, rec in labels.items():
        if rec.get("correct") is None:
            continue
        if min_conf and order.get(str(rec.get("confidence")).lower(), -1) < floor:
            continue
        out[cid] = bool(rec["correct"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", default="data/activations/R1-1.5B")
    ap.add_argument("--chains", default="data/annotated_R1-1.5B.json")
    ap.add_argument("--labels", default="data/correctness_R1-1.5B_pilot.json")
    ap.add_argument("--layers", default="14,17,27")
    ap.add_argument("--max-steps", type=int, default=64, help="cap sequence length (keep last k)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cpu", help="cpu (default; leaves MPS free) or mps")
    ap.add_argument("--min-confidence", choices=["low", "medium", "high"], default=None)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default="results/predict/R1-1.5B/value")
    args = ap.parse_args()

    chains = json.load(open(args.chains))
    smoke = bool(args.smoke)
    if smoke:
        chains = chains[: args.smoke]
    chains_by_id = {(c.get("task_id") or c.get("id") or ""): c for c in chains}

    collapse_map, onset_map, counts = collapse_labels(chains)
    if smoke:
        rng = random.Random(42)
        collapse_map = {cid: bool(rng.getrandbits(1)) for cid in collapse_map}
        correctness_map = {c.get("task_id"): bool(rng.getrandbits(1)) for c in chains[:120]}
    else:
        correctness_map = _correctness_map(args.labels, args.min_confidence)
    log(f"collapse {counts}; correctness labels {len(correctness_map)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for layer in [int(x) for x in args.layers.split(",")]:
        log(f"=== layer {layer} ===")
        emb_full = build_step_datasets(chains, args.activations, layer)
        emb_pre = truncate_preonset(emb_full, chains_by_id, onset_map)
        r = {"layer": layer,
             "collapse": score(emb_pre, collapse_map, "collapse", args),
             "correctness": score(emb_full, correctness_map, "correctness", args)}
        results.append(r)
        for ep in ("collapse", "correctness"):
            b = r[ep]
            if "auc" in b:
                log(f"  {ep}: V={b['auc']['value_gru']['auc_oof']:.3f} "
                    f"shuf={b['auc']['value_gru_stepshuffled']['auc_oof']:.3f} "
                    f"pooled={b['auc']['pooled_logistic']['auc_oof']:.3f} "
                    f"| {b['verdict']} / {b['m3_order']}")
            else:
                log(f"  {ep}: {b.get('status')}")

    tag = "smoke" if smoke else "sparse"
    out_json = out_dir / f"value_head_{tag}.json"
    backup_existing(out_json)
    json.dump({"results": results, "collapse_class_counts": counts, "smoke": smoke,
               "belief_floor": BELIEF_FLOOR,
               "provenance": provenance(args, inputs=[args.chains] +
                                        ([] if smoke else [args.labels]))},
              open(out_json, "w"), indent=2, default=str)
    write_md(out_dir / f"value_head_{tag}.md", results, counts, smoke)
    log(f"wrote {out_json}")


if __name__ == "__main__":
    main()
