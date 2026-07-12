#!/usr/bin/env python3
"""25_forward_map_patch.py — A5: causal-fidelity test of the forward map (PG §12).

Is the learned forward predictor's ẑ_{t+1} the model's OWN update, or an external
curve-fit? The pilot showed forward-prediction skill and detector value dissociate
(overfit worsens f, AUC holds), so the dynamics track needs this before R7. We
ground the forward map the way E10.1 grounds the steering frame: **patch the
predicted next-state into the running model and measure how little it perturbs the
model's own next-token distribution** — an interchange intervention, NOT a steering
method (no task-accuracy objective, no dose knob).

At the onset token p of step t+1 (block-L residual = the `layer{L}.npy` space, i.e.
the OUTPUT of model.model.layers[L] — confirmed via src/hooks.ActivationCache), we
replace the residual with a candidate pooled vector and read the shift in the
next-token distribution vs unpatched:
    KL( D_patched(·|p) || D_unpatched(·|p) )      (lower = more faithful)
Candidates, with the pre-registered ordering (A5-P):
    real_pooled (ds.X[t+1]) ≤ pred_normed < {persist (x_t), random}, ALL at the real
    magnitude ‖x_{t+1}‖ so the contrast is DIRECTION not norm (the ridge ẑ is
    norm-shrunk ~0.85×, reported raw as `pred`). `real_pooled` is the POOLING CEILING —
    the best any pooled-space vector can do; operative test = KL(pred_normed) <
    KL(persist) and the recovery (KL_persist − KL_prednormed)/(KL_persist − KL_real).

SPARSE ARM ($ tiny, local, this script): predictor + candidates live in the
mean-pooled span space and are patched at the single onset token — a smeared
injection (M4, suggestive). DENSE ARM (confirmatory, owed): per-token predictor
patched per-token, folded into the R4 extraction pod.

Prior art (PG §12.1): Patchscopes (2401.06102) patches REAL states, never a learned
predictor's output; ASM (NeurIPS-25, no arXiv) ADDS a delta (control, not
validation); SSP (2604.18464) is offline-only — A5 closes their causal gap. Report
the persistence baseline prominently (the occupancy-not-order pilot makes "beats
persistence under patching" the bar).

  python3 25_forward_map_patch.py --layer 17 --n-chains 24 --max-steps 6   # real (loads model)
  python3 25_forward_map_patch.py --dry-run                                # pipeline only, no model

Output: results/predict/R1-1.5B/forward_patch/forward_patch_L{layer}.json + .md
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
from src.activation_extraction import _sentence_to_token_positions  # noqa: E402
# A5 patches with a LOCAL per-batch-element variant (patched_residual_batched below),
# not src.activation_patching.patched_residual (which broadcasts ONE donor to all rows).
from src.predict.trajectory_dataset import build_step_datasets, make_supervised_pairs  # noqa: E402
from src.predict.predictor import RidgeConfig, oof_residuals  # noqa: E402

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CANDIDATES = ("real_pooled", "pred_normed", "pred", "persist", "random")


def log(msg: str) -> None:
    print(f"[A5 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── patch sites: one per (chain, step-transition t→t+1) with all candidates ──

def build_patch_sites(chains, activations_dir, layer, rng, predictor="ridge") -> list[dict]:
    """Train the OOF forward predictor on ALL chains, then emit one site per pair.

    Each site carries the four pooled-space candidate vectors and the original
    annotation-span index of step t+1 (whose ONSET token is the patch position).
    The predictor is out-of-fold (chain-grouped) so ẑ for a chain never used that
    chain in training — the honesty the pilot's integrity lesson demands.
    """
    datasets = build_step_datasets(chains, activations_dir, layer)
    pairs = make_supervised_pairs(datasets)          # no max_gap: row order == (ds, t) loop
    if pairs["X_hist"].shape[0] == 0:
        return []
    if predictor == "jepa":
        from src.predict.jepa import JEPAConfig, oof_residuals_jepa
        res = oof_residuals_jepa(pairs, JEPAConfig(target="delta"))
    else:
        res = oof_residuals(pairs, RidgeConfig(target="delta"))

    sites: list[dict] = []
    i = 0
    for ds in datasets:
        for t in range(ds.T - 1):
            assert pairs["groups"][i] == ds.chain_id, "pair/dataset misalignment"
            z_hat = np.asarray(res.pred[i], dtype=np.float32)
            x_t = np.asarray(pairs["X_hist"][i], dtype=np.float32)
            x_next = np.asarray(ds.X[t + 1], dtype=np.float32)   # real pooled step t+1
            n_real = float(np.linalg.norm(x_next))
            # pred_normed: PREDICTED DIRECTION at the real magnitude — isolates
            # direction quality from the ridge's norm shrinkage (‖ẑ‖≈0.85‖x‖).
            pred_normed = z_hat * (n_real / max(float(np.linalg.norm(z_hat)), 1e-8))
            rand = rng.standard_normal(z_hat.shape).astype(np.float32)
            rand *= (n_real / max(float(np.linalg.norm(rand)), 1e-8))   # matched to REAL norm
            sites.append({
                "chain_id": ds.chain_id, "t": int(t),
                "orig_next": int(ds.orig_indices[t + 1]),
                "cands": {"real_pooled": x_next, "pred_normed": pred_normed,
                          "pred": z_hat, "persist": x_t, "random": rand},
            })
            i += 1
    return sites


def onset_token(offsets, abs_char: int) -> int | None:
    """First token whose end exceeds abs_char — the extraction onset rule."""
    for idx, (_s, e) in enumerate(offsets):
        if e > abs_char:
            return idx
    return None


# ── KL of the next-token distribution ────────────────────────────────────────

def _softmax_logprob(logits):
    import torch
    return torch.log_softmax(logits.float(), dim=-1)


def kl_patched_vs_clean(logp_clean, logp_patched) -> float:
    """KL(P_patched || P_clean) at one position, in nats."""
    p = logp_patched.exp()
    return float((p * (logp_patched - logp_clean)).sum().item())


@contextmanager
def patched_residual_batched(model, layer_idx, position, donors):
    """Patch (block-`layer_idx` output, `position`) with a PER-BATCH-ELEMENT donor:
    hidden[b, position, :] = donors[b]. Lets one forward (batch = #candidates) score
    every candidate for a site at once — the option-2 speedup. `donors`: (B, d)."""
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


# ── main per-chain loop ──────────────────────────────────────────────────────

def run_chain(model, tokenizer, chain, sites, layer, max_len, dry_run,
              position_control=False, rng=None):
    """Resolve onset tokens for this chain's sites; if not dry_run, patch + KL."""
    import torch

    full_text = chain["prompt"] + chain["chain"]
    prompt_len = len(chain["prompt"])
    ann = chain.get("annotations", []) or []
    char_offsets = locate_annotation_offsets(chain["chain"], [a.get("text", "") for a in ann])

    enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True,
                    truncation=True, max_length=max_len)
    offsets = enc.pop("offset_mapping")[0].tolist()
    seq_len = enc["input_ids"].shape[1]

    resolved = []
    gen_start = next((j for j, (s0, _e0) in enumerate(offsets) if s0 >= prompt_len), 1)
    for s in sites:
        oi = s["orig_next"]
        char = char_offsets[oi] if 0 <= oi < len(char_offsets) else None
        if char is None:
            continue
        p = onset_token(offsets, prompt_len + char)
        if p is None or p >= seq_len or p == 0:
            continue
        if position_control and rng is not None and seq_len - gen_start > 5:
            # CONTROL: inject the step-t+1 prediction at a WRONG (random, non-onset)
            # generated token. If pred_normed still beats persistence here, the effect
            # is a generic vector property, not onset-specific step dynamics.
            for _ in range(8):
                pc = int(rng.integers(gen_start, seq_len - 1))
                if abs(pc - p) > 3:
                    p = pc
                    break
        resolved.append((s, p))

    stats = {"n_sites": len(sites), "n_resolved": len(resolved)}
    if dry_run or not resolved:
        if resolved:
            norms = {c: float(np.mean([np.linalg.norm(s["cands"][c]) for s, _ in resolved]))
                     for c in CANDIDATES}
            stats["cand_norms"] = norms
        return stats, []

    inputs = {k: v.to(model.device) for k, v in enc.items()}
    B = len(CANDIDATES)
    batched_inputs = {k: v.repeat(B, 1) for k, v in inputs.items()}
    # Route through model.model (base) + lm_head at the patched position only, so we
    # never materialise the full (B, seq, vocab) logits tensor. last_hidden_state is
    # post-final-norm, so lm_head(·) == logits.
    clean_hs = model.model(**inputs).last_hidden_state[0]          # (seq, d)

    rows = []
    for s, p in resolved:
        logp_clean = _softmax_logprob(model.lm_head(clean_hs[p]))
        donors = torch.tensor(np.stack([s["cands"][c] for c in CANDIDATES]))   # (B, d)
        with patched_residual_batched(model, layer, p, donors):
            hs_p = model.model(**batched_inputs).last_hidden_state[:, p, :]     # (B, d)
        logits_p = model.lm_head(hs_p)                                          # (B, V)
        rec = {"chain_id": s["chain_id"], "t": s["t"], "pos": p, "kl": {}}
        for bi, c in enumerate(CANDIDATES):
            rec["kl"][c] = kl_patched_vs_clean(logp_clean, _softmax_logprob(logits_p[bi]))
        rows.append(rec)
    return stats, rows


# ── aggregation ──────────────────────────────────────────────────────────────

def _paired(a, b):
    """Wilcoxon signed-rank p (two-sided) + fraction a<b + median difference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    frac = float(np.mean(a < b))
    med = float(np.median(b - a))
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(a, b, alternative="two-sided").pvalue) if len(a) >= 6 else float("nan")
    except Exception:  # noqa: BLE001
        p = float("nan")
    return {"frac_a_lt_b": frac, "median_b_minus_a": med, "wilcoxon_p": p, "n": len(a)}


def aggregate(rows) -> dict:
    if not rows:
        return {"n": 0, "status": "no resolved patch sites"}
    kl = {c: np.array([r["kl"][c] for r in rows], float) for c in CANDIDATES}
    out = {"n": len(rows),
           "mean_kl": {c: float(kl[c].mean()) for c in CANDIDATES},
           "median_kl": {c: float(np.median(kl[c])) for c in CANDIDATES}}
    # primary contrasts
    # PRIMARY (norm-matched): predicted DIRECTION vs do-nothing and vs chance.
    out["prednormed_vs_persist"] = _paired(kl["pred_normed"], kl["persist"])
    out["prednormed_vs_random"] = _paired(kl["pred_normed"], kl["random"])
    out["real_vs_prednormed"] = _paired(kl["real_pooled"], kl["pred_normed"])  # pooling ceiling
    out["predraw_vs_persist"] = _paired(kl["pred"], kl["persist"])   # raw ridge output (norm-shrunk)
    # recovery vs pooling ceiling: 1 = pred_normed as faithful as the real pooled state, 0 = persist
    denom = kl["persist"] - kl["real_pooled"]
    good = np.abs(denom) > 1e-9
    if good.any():
        rec = (kl["persist"][good] - kl["pred_normed"][good]) / denom[good]
        out["recovery_fraction_median"] = float(np.median(rec))
    m = out["mean_kl"]
    out["A5P_ordering_holds"] = bool(m["real_pooled"] <= m["pred_normed"] < m["persist"]
                                     and m["pred_normed"] < m["random"])
    return out


def write_md(path: Path, layer: int, agg: dict, dry: dict, args) -> None:
    L = [f"# A5 — forward-map causal fidelity (layer {layer}, sparse arm)", ""]
    L.append(f"Patch site = onset token of step t+1, at the OUTPUT of block {layer} "
             f"(`layer{layer}.npy` space). KL(patched‖clean) of the next-token dist, nats.\n")
    L.append(f"- resolved patch sites: {dry.get('n_resolved_total','?')} / "
             f"{dry.get('n_sites_total','?')} attempted, over {dry.get('n_chains','?')} chains\n")
    if agg.get("status"):
        L.append(f"**{agg['status']}**")
        path.write_text("\n".join(L)); return
    L.append("| candidate | mean KL | median KL |")
    L.append("|---|---|---|")
    for c in CANDIDATES:
        L.append(f"| {c} | {agg['mean_kl'][c]:.4f} | {agg['median_kl'][c]:.4f} |")
    pv = agg["prednormed_vs_persist"]
    L += ["",
          f"- **pred_normed vs persist** (want KL lower): pred_normed lower in "
          f"{pv['frac_a_lt_b']*100:.0f}% of sites, median Δ(persist−pred)={pv['median_b_minus_a']:.4f}, "
          f"Wilcoxon p={pv['wilcoxon_p']:.3g} (n={pv['n']})",
          f"- pred_normed vs random (norm-matched DIRECTION test): pred_normed lower in "
          f"{agg['prednormed_vs_random']['frac_a_lt_b']*100:.0f}% "
          f"(p={agg['prednormed_vs_random']['wilcoxon_p']:.3g})",
          f"- pooling ceiling (real ≤ pred_normed): real lower in "
          f"{agg['real_vs_prednormed']['frac_a_lt_b']*100:.0f}%",
          f"- raw ridge ẑ (norm-shrunk) vs persist: pred lower in "
          f"{agg['predraw_vs_persist']['frac_a_lt_b']*100:.0f}%",
          f"- recovery vs pooling ceiling (median): "
          f"{agg.get('recovery_fraction_median', float('nan')):.3f} "
          f"(1.0 = pred_normed as faithful as the real pooled state; 0 = no better than persistence)",
          f"- **A5-P ordering real≤pred_normed<persist,random holds: {agg['A5P_ordering_holds']}**",
          ""]
    path.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", default="data/activations/R1-1.5B")
    ap.add_argument("--chains", default="data/annotated_R1-1.5B.json")
    ap.add_argument("--layer", type=int, default=17, help="extraction layer = block-output site")
    ap.add_argument("--n-chains", type=int, default=24, help="subset of chains to patch")
    ap.add_argument("--max-steps", type=int, default=6, help="patch sites per chain (first k pairs)")
    ap.add_argument("--max-len", type=int, default=1024, help="token cap per chain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predictor", choices=["ridge", "jepa"], default="ridge",
                    help="forward map under test: linear ridge (Rung-1) or JEPA MLP (Rung-2)")
    ap.add_argument("--position-control", action="store_true",
                    help="patch at a random NON-onset generated token (is the effect onset-specific?)")
    ap.add_argument("--dry-run", action="store_true", help="resolve sites only; skip the model")
    ap.add_argument("--out", default="results/predict/R1-1.5B/forward_patch")
    args = ap.parse_args()

    chains = json.load(open(args.chains))
    chains_by_id = {(c.get("task_id") or c.get("id") or ""): c for c in chains}
    rng = np.random.default_rng(args.seed)

    log(f"building OOF forward predictor @ layer {args.layer} over {len(chains)} chains")
    sites = build_patch_sites(chains, args.activations, args.layer, rng, predictor=args.predictor)
    by_chain: dict = {}
    for s in sites:
        by_chain.setdefault(s["chain_id"], []).append(s)
    # subset: chains with sites and within the token cap, in corpus order
    subset = []
    for c in chains:
        cid = c.get("task_id") or c.get("id") or ""
        if cid in by_chain and int(c.get("n_tokens", 0)) <= args.max_len:
            subset.append(c)
        if len(subset) >= args.n_chains:
            break
    log(f"{len(sites)} total sites; patching {len(subset)} chains "
        f"(≤{args.max_steps} sites each)")

    model = tokenizer = None
    if not args.dry_run:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_grad_enabled(False)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        log(f"loading model on {device}/{dtype}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    all_rows, n_sites_total, n_resolved_total = [], 0, 0
    for c in subset:
        cid = c.get("task_id") or c.get("id") or ""
        csites = by_chain[cid][: args.max_steps]
        stats, rows = run_chain(model, tokenizer, c, csites, args.layer, args.max_len,
                                args.dry_run, position_control=args.position_control, rng=rng)
        n_sites_total += stats["n_sites"]
        n_resolved_total += stats["n_resolved"]
        all_rows.extend(rows)
        if args.dry_run:
            log(f"  {cid}: resolved {stats['n_resolved']}/{stats['n_sites']} "
                f"{stats.get('cand_norms', '')}")
        else:
            log(f"  {cid}: {len(rows)} sites patched")

    agg = aggregate(all_rows)
    dry = {"n_sites_total": n_sites_total, "n_resolved_total": n_resolved_total,
           "n_chains": len(subset)}
    if not args.dry_run and "mean_kl" in agg:
        log(f"mean KL: " + ", ".join(f"{c}={agg['mean_kl'][c]:.4f}" for c in CANDIDATES))
        log(f"pred_normed<persist in {agg['prednormed_vs_persist']['frac_a_lt_b']*100:.0f}% "
            f"(p={agg['prednormed_vs_persist']['wilcoxon_p']:.3g}); "
            f"A5-P ordering holds: {agg['A5P_ordering_holds']}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = (("_" + args.predictor) if args.predictor != "ridge" else "") + \
              ("_poscontrol" if args.position_control else "")
    tag = "dryrun" if args.dry_run else f"L{args.layer}{variant}"
    out_json = out_dir / f"forward_patch_{tag}.json"
    backup_existing(out_json)
    json.dump({"layer": args.layer, "aggregate": agg, "site_counts": dry,
               "dry_run": args.dry_run,
               "provenance": provenance(args, inputs=[args.chains])},
              open(out_json, "w"), indent=2, default=str)
    if not args.dry_run:
        write_md(out_dir / f"forward_patch_{tag}.md", args.layer, agg, dry, args)
    log(f"wrote {out_json}")


if __name__ == "__main__":
    main()
