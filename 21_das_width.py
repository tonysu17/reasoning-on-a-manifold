#!/usr/bin/env python
"""E10.2/E10.3 — the causal width of a behaviour's subspace (featurizer rung three).

Originally executed for backtracking at hs[17] (E10.2); Amendment 4 parameterizes
it over behaviours. For non-backtracking behaviours --layer is REQUIRED and must
be the grounded layer selected from the main run's controls.json by the sealed
rule (e10_pick_grounded_layer.py); if no layer grounds, the width probe is
skipped (that IS the sealed outcome, not a failure to run).

Learns orthonormal k-frames U in R^{d x k} for k in {1,2,4,8,16,32} by the same
windowed interchange objective as E10.1 (swap = h_b + U U^T (h_s - h_b)), at the
grounded site, and asks where the transfer-vs-width curve saturates.

Pre-registered readings (E10_DAS_PREREG.md Amendment 3):
  - saturation at k=1  -> the causal object is one-dimensional; the E8 manifold
    null is explained at its root;
  - rising to k ~ 6-8  -> matches the correlation-dimension estimate; the
    thesis's low-dimensionality claim is certified causally;
  - rising far beyond  -> the behaviour is causally wide; the 1-D story was a
    floor, and both correlational estimators under-counted.
Grounding travels with every width: k-dim coordinate class-separation AUC
(logistic, 5-fold, chain-agnostic pairs) + base-donor state-dependence; a width
whose extra transfer is ungrounded is optimizer carving, not structure.

Bonus probe: adding-knowledge REMOVAL (1-D, same recipe, donors = add-know
onsets) — the swap is symmetric, so the behaviour that failed amplification in
E8 can still be tested for causal removability.

Reuses 20_das_backtracking.py helpers via importlib. GPU: ~3 h on a 4090.
Output: results/das/R1-1.5B/width/
"""

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("das", ROOT / "20_das_backtracking.py")
das = importlib.util.module_from_spec(spec)
spec.loader.exec_module(das)

DAS_ROOT = ROOT / "results" / "das" / "R1-1.5B"
OUT = DAS_ROOT / "width"                       # resolved per behaviour in main()
PAIRS_MAIN = DAS_ROOT / "main" / "pairs.json"  # resolved per behaviour in main()


def log(msg):
    print(f"[e10.2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Frame(torch.nn.Module):
    """Orthonormal (d, k) frame via torch's orthogonal parametrization."""

    def __init__(self, d, k, device, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        lin = torch.nn.Linear(k, d, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(d, k, generator=g))
        self.lin = torch.nn.utils.parametrizations.orthogonal(lin)
        self.to(device)

    def U(self):
        return self.lin.weight  # (d, k), orthonormal columns


def swap_states_frame(hb, hs, U):
    proj = (hs - hb) @ U @ U.T
    return hb + proj, hs - proj


def interchange_loss_frame(model, batch, layer, U, W):
    hb_new, hs_new = swap_states_frame(batch["hb"], batch["hs"], U)
    lp_b = das.window_logprob(model, batch["induce_ids"], batch["induce_attn"], layer, hb_new, W)
    lp_s = das.window_logprob(model, batch["remove_ids"], batch["remove_attn"], layer, hs_new, W)
    return -(lp_b.mean() + lp_s.mean())


def train_frame(model, tok, pairs, layer, k, cfg, device, shuffle_pairs=False):
    if shuffle_pairs:
        perm = list(range(len(pairs)))
        random.Random(cfg.seed + 1).shuffle(perm)
        pairs = [{**pairs[i], "source_ids": pairs[perm[i]]["source_ids"],
                  "source_cont": pairs[perm[i]]["source_cont"]} for i in range(len(pairs))]
    batches = das.make_batches(pairs, tok, model, layer, device, cfg.bs, cfg.window)
    frame = Frame(model.config.hidden_size, k, device, seed=cfg.seed)
    opt = torch.optim.Adam(frame.parameters(), lr=cfg.lr)
    curve = []
    for ep in range(cfg.epochs):
        random.Random(cfg.seed + ep).shuffle(batches)
        tot = 0.0
        for batch in batches:
            opt.zero_grad()
            loss = interchange_loss_frame(model, batch, layer, frame.U(), cfg.window)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        curve.append(tot / len(batches))
        if ep % max(1, cfg.epochs // 6) == 0 or ep == cfg.epochs - 1:
            log(f"  k={k}{' [shuf]' if shuffle_pairs else ''} epoch {ep+1}/{cfg.epochs} "
                f"loss {curve[-1]:.4f}")
    return frame.U().detach(), curve


@torch.no_grad()
def eval_frame(model, tok, pairs, layer, U, cfg, device, positional=False):
    W = cfg.window
    batches = das.make_batches(pairs, tok, model, layer, device, cfg.bs, W)
    off = -2 if positional else 0
    ind, rem, n = 0.0, 0.0, 0
    for batch in batches:
        hb_new, hs_new = swap_states_frame(batch["hb"], batch["hs"], U)
        lp_ic = das.window_logprob(model, batch["induce_ids"], batch["induce_attn"], layer, None, W)
        lp_i = das.window_logprob(model, batch["induce_ids"], batch["induce_attn"], layer,
                                  hb_new, W, pos_offset=off)
        lp_rc = das.window_logprob(model, batch["remove_ids"], batch["remove_attn"], layer, None, W)
        lp_r = das.window_logprob(model, batch["remove_ids"], batch["remove_attn"], layer,
                                  hs_new, W, pos_offset=off)
        ind += float((lp_i - lp_ic).sum()); rem += float((lp_r - lp_rc).sum())
        n += batch["hb"].shape[0]
    return {"n": n, "induce_dlp": ind / n, "remove_dlp": rem / n, "sym_dlp": (ind + rem) / (2 * n)}


@torch.no_grad()
def grounding(model, tok, pairs, layer, U, cfg, device):
    """k-dim coordinate class-separation AUC + base-donor state-dependence."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    batches = das.make_batches(pairs, tok, model, layer, device, cfg.bs, cfg.window)
    X, y = [], []
    for b in batches:
        X.append((b["hb"] @ U).cpu().numpy()); y += [0] * b["hb"].shape[0]
        X.append((b["hs"] @ U).cpu().numpy()); y += [1] * b["hs"].shape[0]
    X = np.concatenate(X); y = np.array(y)
    prob = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=5,
                             method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, prob))
    null_pairs = [{**p, "source_ids": pairs[(i + 1) % len(pairs)]["base_ids"],
                   "source_cont": p["source_cont"]} for i, p in enumerate(pairs)]
    real = eval_frame(model, tok, pairs, layer, U, cfg, device)
    null = eval_frame(model, tok, null_pairs, layer, U, cfg, device)
    return {"coord_auc": auc, "real_induce": real["induce_dlp"],
            "base_donor_null_induce": null["induce_dlp"],
            "state_dependence": real["induce_dlp"] - null["induce_dlp"]}


def main():
    global OUT, PAIRS_MAIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour", default="backtracking", choices=sorted(das.BEHAVIOURS),
                    help="source behaviour (Amendment 4); resolves pairs + output dirs")
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--layer", type=int, default=None,
                    help="hidden_states index. Default 17 for backtracking (the executed "
                         "E10.2 site); REQUIRED for other behaviours — pass the grounded "
                         "layer picked from the main run's controls.json")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=320)
    ap.add_argument("--min-ctx", type=int, default=16)
    ap.add_argument("--max-chains", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--skip-ak", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    cfg = ap.parse_args()
    if cfg.layer is None:
        if cfg.behaviour == "backtracking":
            cfg.layer = 17
        else:
            ap.error(f"--layer is required for {cfg.behaviour}: pass the grounded layer "
                     "(e10_pick_grounded_layer.py on the main run's controls.json)")
    das.SOURCE_LABEL = cfg.behaviour
    short = das.BEHAVIOURS[cfg.behaviour]["short"]
    if cfg.behaviour != "backtracking":  # bt keeps the executed E10.2 dirs
        OUT = DAS_ROOT / f"{short}_width"
        PAIRS_MAIN = DAS_ROOT / f"{short}_main" / "pairs.json"
    if cfg.smoke:
        cfg.widths, cfg.epochs, cfg.bs, cfg.n_pairs, cfg.ctx, cfg.max_chains = \
            [1, 2], 2, 4, 8, 96, 60
    OUT.mkdir(parents=True, exist_ok=True)

    device, dtype = das.pick_device()
    torch.set_grad_enabled(True)
    log(f"behaviour={cfg.behaviour} device={device} widths={cfg.widths} "
        f"layer=hs[{cfg.layer}] smoke={cfg.smoke} out={OUT.name}")
    tok, model = das.load_model(device, dtype)

    # reuse the behaviour's E10.1/E10.3 main pairs verbatim when available
    if PAIRS_MAIN.exists() and not cfg.smoke:
        pairs = json.load(open(PAIRS_MAIN))[:cfg.n_pairs]
        log(f"reusing main pairs from {PAIRS_MAIN.parent.name} ({len(pairs)})")
    else:
        pairs = das.build_pairs(tok, cfg)

    report = {"experiment": f"E10.2 causal width — {cfg.behaviour}",
              "date": time.strftime("%Y-%m-%d"),
              "device": device, "layer_hs": cfg.layer, "n_pairs": len(pairs),
              "config": vars(cfg), "widths": {}}
    for k in cfg.widths:
        log(f"=== width k={k} ===")
        U, curve = train_frame(model, tok, pairs, cfg.layer, k, cfg, device)
        U_s, _ = train_frame(model, tok, pairs, cfg.layer, k, cfg, device, shuffle_pairs=True)
        np.save(OUT / f"frame_k{k}.npy", U.cpu().numpy())
        cell = {"train_curve": curve,
                "learned": eval_frame(model, tok, pairs, cfg.layer, U, cfg, device),
                "learned_positional": eval_frame(model, tok, pairs, cfg.layer, U, cfg,
                                                 device, positional=True),
                "shuffled": eval_frame(model, tok, pairs, cfg.layer, U_s, cfg, device),
                "grounding": grounding(model, tok, pairs, cfg.layer, U, cfg, device)}
        G = torch.linalg.qr(torch.randn(model.config.hidden_size, k, device=device))[0]
        cell["random_frame"] = eval_frame(model, tok, pairs, cfg.layer, G, cfg, device)
        report["widths"][str(k)] = cell
        g = cell["grounding"]
        log(f"  k={k}: sym={cell['learned']['sym_dlp']:+.4f} shuf={cell['shuffled']['sym_dlp']:+.4f} "
            f"rand={cell['random_frame']['sym_dlp']:+.4f} AUC={g['coord_auc']:.3f} "
            f"state-dep={g['state_dependence']:+.4f}")

    # --- adding-knowledge removal probe (1-D)
    if not cfg.skip_ak:
        log("=== adding-knowledge removal probe (k=1) ===")
        das.SOURCE_LABEL = "adding-knowledge"
        ak_pairs = das.build_pairs(tok, cfg)
        d_ak, _ = das.train_das(model, tok, ak_pairs, cfg.layer, cfg, device)
        np.save(OUT / f"dir_ak_L{cfg.layer}.npy", d_ak.cpu().numpy())
        ak = {"learned": das.eval_direction(model, tok, ak_pairs, cfg.layer, d_ak, cfg, device),
              "controls": das.controls_stage(model, tok, ak_pairs, cfg.layer, d_ak, cfg, device)}
        rand = torch.randn(model.config.hidden_size, device=device)
        ak["random"] = das.eval_direction(model, tok, ak_pairs, cfg.layer, rand / rand.norm(),
                                          cfg, device)
        report["adding_knowledge_removal"] = ak
        log(f"  ak removal: remove_dlp={ak['learned']['remove_dlp']:+.4f} "
            f"(random {ak['random']['remove_dlp']:+.4f}); coord_AUC="
            f"{ak['controls']['coord_auc_source_vs_base']:.3f}")

    json.dump(report, open(OUT / "report.json", "w"), indent=2)
    lines = [f"# {report['experiment']} — REPORT", "",
             f"Date: {report['date']} · {report['n_pairs']} pairs · site hs[{cfg.layer}]", "",
             "| k | sym Δlogprob | shuffled | random frame | coord AUC | state-dep |",
             "|---|---|---|---|---|---|"]
    for k, c in report["widths"].items():
        g = c["grounding"]
        lines.append(f"| {k} | {c['learned']['sym_dlp']:+.4f} | {c['shuffled']['sym_dlp']:+.4f} | "
                     f"{c['random_frame']['sym_dlp']:+.4f} | {g['coord_auc']:.3f} | "
                     f"{g['state_dependence']:+.4f} |")
    if "adding_knowledge_removal" in report:
        ak = report["adding_knowledge_removal"]
        lines += ["", f"Adding-knowledge removal (1-D): remove Δlogprob "
                      f"{ak['learned']['remove_dlp']:+.4f} vs random "
                      f"{ak['random']['remove_dlp']:+.4f}; coord AUC "
                      f"{ak['controls']['coord_auc_source_vs_base']:.3f}."]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    log(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
