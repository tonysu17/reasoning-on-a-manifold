#!/usr/bin/env python3
"""28_forward_patch_dense.py — A5 DENSE arm: onset-token forward-map fidelity (PG §12).

The sparse arm (25_forward_map_patch.py) patched a MEAN-POOLED span vector at a single
onset token — coarse (all KLs ~15 nats; `real_pooled` was not even a valid ceiling). This
dense arm fixes both: it extracts the residual at each step's ONSET TOKEN (a single real
state, not a span average), trains the forward map on those, and patches the predicted
onset-token state back at that same token. Now `real_onset` (the true state patched at its
own position) is a VALID CEILING — it must give ≈0 KL — so the recovery metric is
meaningful and the absolute fidelity is readable.

Candidates at the onset token of step t+1 (all norm-matched to ‖real_onset‖ except raw pred):
    real_onset (true state, ≈0 KL ceiling) ≤ pred_normed < {persist (prev onset), random}
Load-bearing contrast (from the sparse-arm position control, which sent pred-vs-random to
chance off-onset): **pred_normed vs random** = the onset-specific direction test.

Two passes over the chain subset (extraction, then patch), so the forward map is trained on
ALL subset chains before any patching (chain-grouped OOF honesty). GPU (RTX 4090) job.

  python3 28_forward_patch_dense.py --layer 17 --n-chains 300 --max-steps 10 --max-len 4096
  python3 28_forward_patch_dense.py --smoke 6      # local sanity: real_onset KL must be ~0

Output: results/predict/R1-1.5B/forward_patch/dense_L{layer}.json + .md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import backup_existing, provenance  # noqa: E402
from src.text_offsets import locate_annotation_offsets  # noqa: E402
from src.cbs.trajectory import PHASE_4_BEHAVIOURS  # noqa: E402
from src.predict.predictor import RidgeConfig, oof_residuals  # noqa: E402

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CANDIDATES = ("real_onset", "pred_normed", "pred", "persist", "random")


def log(msg: str) -> None:
    print(f"[A5-dense {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def onset_token(offsets, abs_char: int) -> int | None:
    for idx, (_s, e) in enumerate(offsets):
        if e > abs_char:
            return idx
    return None


def _softmax_logprob(logits):
    import torch
    return torch.log_softmax(logits.float(), dim=-1)


def kl_patched_vs_clean(logp_clean, logp_patched) -> float:
    p = logp_patched.exp()
    return float((p * (logp_patched - logp_clean)).sum().item())


@contextmanager
def patched_residual_batched(model, layer_idx, position, donors):
    """Patch block-`layer_idx` output at `position` with per-batch-element donors (B,d)."""
    block = model.model.layers[layer_idx]

    def _hook(_m, _in, output):
        is_tuple = isinstance(output, tuple)
        hidden = (output[0] if is_tuple else output).clone()
        hidden[:, position, :] = donors.to(hidden.dtype).to(hidden.device)
        return ((hidden,) + tuple(output[1:])) if is_tuple else hidden

    h = block.register_forward_hook(_hook)
    try:
        yield
    finally:
        h.remove()


# ── pass 1: extract onset-token residuals ────────────────────────────────────

def extract_onset_trajectory(model, tokenizer, chain, layer, max_len):
    """Residual at block-`layer` output at each Phase-4 step's ONSET token."""
    import torch
    full = chain["prompt"] + chain["chain"]
    plen = len(chain["prompt"])
    ann = chain.get("annotations", []) or []
    offs = locate_annotation_offsets(chain["chain"], [a.get("text", "") for a in ann])
    enc = tokenizer(full, return_tensors="pt", return_offsets_mapping=True,
                    truncation=True, max_length=max_len)
    toffs = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(model.device)
    seq = ids.shape[1]
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states[layer + 1][0]  # (seq,d) block-L out
    X, oi, pos = [], [], []
    for i, a in enumerate(ann):
        if a.get("label") not in PHASE_4_BEHAVIOURS:
            continue
        char = offs[i]
        if char is None:
            continue
        p = onset_token(toffs, plen + char)
        if p is None or p >= seq or p == 0:
            continue
        X.append(hs[p].float().cpu().numpy())
        oi.append(i)
        pos.append(int(p))
    if len(X) < 2:
        return None
    return {"chain_id": chain.get("task_id") or chain.get("id") or "",
            "X": np.asarray(X, dtype=np.float32), "orig_indices": oi, "onset_pos": pos}


# ── build dense candidates from the onset-token forward map ──────────────────

def build_dense_sites(trajs, rng) -> dict:
    Xh, Xn, groups, gaps, posn, patchpos = [], [], [], [], [], []
    for tr in trajs:
        T = len(tr["X"])
        for t in range(T - 1):
            Xh.append(tr["X"][t]); Xn.append(tr["X"][t + 1]); groups.append(tr["chain_id"])
            gaps.append(tr["orig_indices"][t + 1] - tr["orig_indices"][t]); posn.append(t)
            patchpos.append(tr["onset_pos"][t + 1])
    if not Xh:
        return {}
    pairs = {"X_hist": np.asarray(Xh, np.float32), "X_next": np.asarray(Xn, np.float32),
             "groups": np.asarray(groups, object), "gaps": np.asarray(gaps, int),
             "pos": np.asarray(posn, int)}
    res = oof_residuals(pairs, RidgeConfig(target="delta"))
    by_chain: dict = {}
    for i in range(len(Xh)):
        z = np.asarray(res.pred[i], np.float32)
        xn = pairs["X_next"][i]
        nreal = float(np.linalg.norm(xn))
        pn = z * (nreal / max(float(np.linalg.norm(z)), 1e-8))
        rnd = rng.standard_normal(z.shape).astype(np.float32)
        rnd *= nreal / max(float(np.linalg.norm(rnd)), 1e-8)
        by_chain.setdefault(str(groups[i]), []).append((int(patchpos[i]), {
            "real_onset": xn, "pred_normed": pn, "pred": z,
            "persist": pairs["X_hist"][i], "random": rnd}))
    return by_chain


# ── pass 2: patch + KL ───────────────────────────────────────────────────────

def patch_chain(model, tokenizer, chain, csites, layer, max_len):
    import torch
    full = chain["prompt"] + chain["chain"]
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=max_len)
    seq = enc["input_ids"].shape[1]
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    B = len(CANDIDATES)
    batched = {k: v.repeat(B, 1) for k, v in inputs.items()}
    with torch.no_grad():
        clean_hs = model.model(**inputs).last_hidden_state[0]
    rows = []
    for p, cands in csites:
        if p <= 0 or p >= seq:
            continue
        logp_clean = _softmax_logprob(model.lm_head(clean_hs[p]))
        donors = torch.tensor(np.stack([cands[c] for c in CANDIDATES]))
        with patched_residual_batched(model, layer, p, donors):
            with torch.no_grad():
                hs_p = model.model(**batched).last_hidden_state[:, p, :]
        logits_p = model.lm_head(hs_p)
        rec = {"pos": p, "kl": {}}
        for bi, c in enumerate(CANDIDATES):
            rec["kl"][c] = kl_patched_vs_clean(logp_clean, _softmax_logprob(logits_p[bi]))
        rows.append(rec)
    return rows


# ── aggregation ──────────────────────────────────────────────────────────────

def _paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {"frac_a_lt_b": float(np.mean(a < b)), "median_b_minus_a": float(np.median(b - a)),
           "n": len(a), "wilcoxon_p": float("nan")}
    try:
        from scipy.stats import wilcoxon
        if len(a) >= 6:
            out["wilcoxon_p"] = float(wilcoxon(a, b, alternative="two-sided").pvalue)
    except Exception:  # noqa: BLE001
        pass
    return out


def aggregate(rows) -> dict:
    if not rows:
        return {"n": 0, "status": "no sites"}
    kl = {c: np.array([r["kl"][c] for r in rows], float) for c in CANDIDATES}
    out = {"n": len(rows),
           "mean_kl": {c: float(kl[c].mean()) for c in CANDIDATES},
           "median_kl": {c: float(np.median(kl[c])) for c in CANDIDATES},
           "prednormed_vs_persist": _paired(kl["pred_normed"], kl["persist"]),
           "prednormed_vs_random": _paired(kl["pred_normed"], kl["random"]),
           "real_vs_prednormed": _paired(kl["real_onset"], kl["pred_normed"])}
    denom = kl["persist"] - kl["real_onset"]
    good = np.abs(denom) > 1e-9
    if good.any():
        out["recovery_fraction_median"] = float(
            np.median((kl["persist"][good] - kl["pred_normed"][good]) / denom[good]))
    out["real_onset_is_valid_ceiling"] = bool(out["mean_kl"]["real_onset"] < 0.5)  # ≈0 sanity
    return out


def write_md(path: Path, layer, agg, counts):
    L = [f"# A5 DENSE — onset-token forward-map fidelity (layer {layer})", "",
         f"Patch = block-{layer} output at each step's ONSET token (single real state, not a "
         f"span average). KL(patched‖clean), nats. {counts.get('n_sites','?')} sites over "
         f"{counts.get('n_chains','?')} chains.", ""]
    if agg.get("status"):
        L.append(agg["status"]); path.write_text("\n".join(L)); return
    L += ["| candidate | mean KL | median KL |", "|---|---|---|"]
    for c in CANDIDATES:
        L.append(f"| {c} | {agg['mean_kl'][c]:.3f} | {agg['median_kl'][c]:.3f} |")
    pv, pr = agg["prednormed_vs_persist"], agg["prednormed_vs_random"]
    L += ["",
          f"- **real_onset a valid ceiling? {agg['real_onset_is_valid_ceiling']}** "
          f"(mean KL {agg['mean_kl']['real_onset']:.3f} — must be ≈0)",
          f"- **pred_normed vs random** (onset-specific direction test): pred lower in "
          f"{pr['frac_a_lt_b']*100:.0f}% (p={pr['wilcoxon_p']:.3g}, n={pr['n']})",
          f"- pred_normed vs persist: pred lower in {pv['frac_a_lt_b']*100:.0f}% "
          f"(p={pv['wilcoxon_p']:.3g})",
          f"- recovery vs ceiling (median): {agg.get('recovery_fraction_median', float('nan')):.3f} "
          f"(1 = pred as faithful as the true state; 0 = = persistence)", ""]
    path.write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chains", default="data/annotated_R1-1.5B.json")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--n-chains", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", type=int, default=0, help="N chains, local sanity (real_onset≈0)")
    ap.add_argument("--out", default="results/predict/R1-1.5B/forward_patch")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32
    log(f"device={device} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()

    chains = json.load(open(args.chains))
    n_want = args.smoke or args.n_chains
    # subset: chains within the token cap, in corpus order
    subset = [c for c in chains if int(c.get("n_tokens", 0)) <= args.max_len][:n_want]
    log(f"pass 1: extracting onset trajectories over {len(subset)} chains (≤{args.max_len} tok)")
    trajs = []
    for c in subset:
        tr = extract_onset_trajectory(model, tok, c, args.layer, args.max_len)
        if tr is not None:
            tr["onset_pos"] = tr["onset_pos"][: args.max_steps + 1]
            tr["orig_indices"] = tr["orig_indices"][: args.max_steps + 1]
            tr["X"] = tr["X"][: args.max_steps + 1]
            trajs.append(tr)
    log(f"  {len(trajs)} usable trajectories")

    rng = np.random.default_rng(args.seed)
    by_chain = build_dense_sites(trajs, rng)
    log(f"pass 2: patching {sum(len(v) for v in by_chain.values())} sites")
    all_rows, n_chains = [], 0
    subset_by_id = {(c.get("task_id") or c.get("id") or ""): c for c in subset}
    for cid, csites in by_chain.items():
        chain = subset_by_id.get(cid)
        if chain is None:
            continue
        all_rows.extend(patch_chain(model, tok, chain, csites[: args.max_steps], args.layer, args.max_len))
        n_chains += 1

    agg = aggregate(all_rows)
    counts = {"n_sites": len(all_rows), "n_chains": n_chains}
    if "mean_kl" in agg:
        log("mean KL: " + ", ".join(f"{c}={agg['mean_kl'][c]:.3f}" for c in CANDIDATES))
        log(f"real_onset ceiling ok: {agg['real_onset_is_valid_ceiling']} "
            f"(KL {agg['mean_kl']['real_onset']:.3f}); pred<random "
            f"{agg['prednormed_vs_random']['frac_a_lt_b']*100:.0f}% "
            f"(p={agg['prednormed_vs_random']['wilcoxon_p']:.3g})")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else f"dense_L{args.layer}"
    out_json = out_dir / f"{tag}.json"
    backup_existing(out_json)
    json.dump({"layer": args.layer, "aggregate": agg, "counts": counts,
               "provenance": provenance(args, inputs=[args.chains])},
              open(out_json, "w"), indent=2, default=str)
    write_md(out_dir / f"{tag}.md", args.layer, agg, counts)
    log(f"wrote {out_json}")


if __name__ == "__main__":
    main()
