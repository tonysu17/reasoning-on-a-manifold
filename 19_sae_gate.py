#!/usr/bin/env python
"""E10.0 — SAE dictionary gate (featurizer programme, rung F0).

Pre-registered BEFORE any SAE-arm experiment (2026-07-05). Decides whether any
PUBLIC pretrained SAE is usable as the sparse featurizer/unfeaturizer pair for a
residual-stream comparator arm, or whether a self-trained residual SAE is required.

Candidates (web-verified 2026-07-05; see RESULTS_LEDGER §B4):
  1. DGurgurov/DeepSeek-R1-Distill-Qwen-1.5B-sae — the ONLY public residual-stream
     SAE for this model. UNDOCUMENTED and self-contradictory: model card says
     blocks.19.hook_resid_pre, folder is named blocks.1.hook_resid_pre, cfg.json
     says blocks.19.hook_resid_post. SAELens 5.5.2, standard ReLU, d_sae 24576
     (x16), l1=5, 498M tokens of LMSYS-chat, ctx 1024,
     normalize_activations='expected_average_only_in'. The gate therefore first
     RESOLVES THE SITE EMPIRICALLY: FVU is probed at all three claimed sites x
     {raw, norm-folded} conventions on a 5-chain pre-pass, and the best cell is
     evaluated in full.
  2. EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k — layers.19.mlp (MLP out),
     TopK, FineWeb-Edu. WRONG SITE for our comparator; evaluated on its own site
     as a documented-healthy REFERENCE point only.

Gate metrics on OUR distribution (R1-1.5B CoT chains, token level, special
token 151643 excluded):
  - FVU = sum ||x - x_hat||^2 / sum ||x - mean(x)||^2
  - L0  = mean active latents
  - CE splice: teacher-forced CE with the site activation replaced by its SAE
    reconstruction, vs clean CE and zero-ablation CE;
    ce_recovered = (CE_zero - CE_recon) / (CE_zero - CE_clean).

Pre-registered verdict thresholds (residual candidate, best site/convention):
  PASS  : FVU <= 0.15 and ce_recovered >= 0.85  -> usable as-is (single layer)
  AMBER : FVU <= 0.40 and ce_recovered >= 0.60  -> usable only with caveats;
          self-trained SAE still preferred
  FAIL  : otherwise -> self-train residual SAEs on the CoT distribution at the
          causal-candidate layers (Resa-style recipe, residual hookpoint)

Local MPS run. No API credits. Output: results/sae_gate/R1-1.5B/e10_0_gate.json
+ REPORT.md
"""

import json
import math
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "sae_gate" / "R1-1.5B"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
RESID_SAE_REPO = "DGurgurov/DeepSeek-R1-Distill-Qwen-1.5B-sae"
MLP_SAE_REPO = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"
MLP_LAYER = 19
# candidate sites for the resid SAE, as hidden_states index (== input to layers[idx]):
#   blocks.1.hook_resid_pre  -> 1   (folder name)
#   blocks.19.hook_resid_pre -> 19  (model card)
#   blocks.19.hook_resid_post-> 20  (cfg.json)
RESID_SITES = {"blocks.1.resid_pre": 1, "blocks.19.resid_pre": 19, "blocks.19.resid_post": 20}
N_CHAINS = 60
N_PREPASS = 5
MAX_TOKENS = 1024  # SAE trained at ctx 1024
EXCLUDE_TOKEN = 151643
SEED = 0
GATE = {"pass_fvu": 0.15, "pass_ce": 0.85, "amber_fvu": 0.40, "amber_ce": 0.60}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32


def log(msg):
    print(f"[e10.0 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class GenericSAE:
    """Hand-loaded SAE (SAELens standard-ReLU or EleutherAI TopK); float32 math."""

    def __init__(self, weights, cfg, fmt):
        self.fmt, self.cfg = fmt, cfg
        if fmt == "saelens":
            self.W_enc = weights["W_enc"].float()           # (d_in, d_sae)
            self.b_enc = weights["b_enc"].float()
            self.W_dec = weights["W_dec"].float()           # (d_sae, d_in)
            self.b_dec = weights["b_dec"].float()
            self.apply_b_dec = bool(cfg.get("apply_b_dec_to_input", True))
            self.topk = None
        elif fmt == "eleuther":
            self.W_enc = weights["encoder.weight"].float().T  # stored (d_sae, d_in)
            self.b_enc = weights["encoder.bias"].float()
            self.W_dec = weights["W_dec"].float()
            self.b_dec = weights["b_dec"].float()
            self.apply_b_dec = True
            self.topk = int(cfg["k"])
        else:
            raise ValueError(fmt)
        self.d_in = self.W_enc.shape[0]

    def to(self, device):
        for n in ("W_enc", "b_enc", "W_dec", "b_dec"):
            setattr(self, n, getattr(self, n).to(device))
        return self

    @torch.no_grad()
    def forward(self, x, normalize=False):
        """x: (n, d_in) float32 -> (recon, l0). normalize=True applies the SAELens
        'expected_average_only_in' convention: scale x so E||x|| = sqrt(d_in),
        decode, then unscale (in case the saved weights were NOT norm-folded)."""
        if normalize:
            scale = math.sqrt(self.d_in) / x.norm(dim=-1).mean().clamp_min(1e-6)
            x_s = x * scale
        else:
            scale, x_s = 1.0, x
        x_in = x_s - self.b_dec if self.apply_b_dec else x_s
        pre = x_in @ self.W_enc + self.b_enc
        if self.topk is not None:
            vals, idx = torch.topk(pre, self.topk, dim=-1)
            acts = torch.zeros_like(pre)
            acts.scatter_(-1, idx, torch.relu(vals))
        else:
            acts = torch.relu(pre)
        recon = (acts @ self.W_dec + self.b_dec) / scale
        return recon, (acts > 0).sum(dim=-1).float()


def load_sae(repo, fmt, allow=None):
    path = Path(snapshot_download(repo, allow_patterns=allow))
    st = sorted(path.rglob("*sae*.safetensors")) or sorted(path.rglob("*.safetensors"))
    st = [f for f in st if "sparsity" not in f.name]
    cfgf = sorted(path.rglob("cfg.json")) or sorted(path.rglob("config.json"))
    weights = load_file(str(st[0]))
    cfg = json.loads(cfgf[0].read_text()) if cfgf else {}
    log(f"loaded {repo} [{fmt}] keys={ {k: tuple(v.shape) for k, v in weights.items()} }")
    return GenericSAE(weights, cfg, fmt)


def fvu(x, recon):
    return float((x - recon).pow(2).sum() / (x - x.mean(0)).pow(2).sum())


@torch.no_grad()
def ce_loss(model, input_ids, splice=None):
    """Teacher-forced CE. splice: (site, fn); site = hidden_states index -> pre-hook
    on layers[site]; site = ('mlp', L) -> post-hook on layers[L].mlp. fn maps the
    float32 activation to its replacement."""
    handles = []
    if splice is not None:
        site, fn = splice
        if isinstance(site, tuple) and site[0] == "mlp":
            def post_hook(m, a, out):
                return fn(out.float()).to(out.dtype)
            handles.append(model.model.layers[site[1]].mlp.register_forward_hook(post_hook))
        else:
            def pre_hook(m, args, kwargs):
                h = args[0]
                return (fn(h.float()).to(h.dtype),) + args[1:], kwargs
            handles.append(model.model.layers[site].register_forward_pre_hook(pre_hook, with_kwargs=True))
    try:
        return float(model(input_ids, labels=input_ids).loss)
    finally:
        for h in handles:
            h.remove()


def main():
    torch.set_grad_enabled(False)  # inference only; without this, autograd graphs accumulate -> MPS OOM
    log(f"device={DEVICE} dtype={DTYPE}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    log("model loaded")

    sae_resid = load_sae(RESID_SAE_REPO, "saelens").to(DEVICE)
    try:
        sae_mlp = load_sae(MLP_SAE_REPO, "eleuther",
                           allow=[f"layers.{MLP_LAYER}.mlp/*"]).to(DEVICE)
    except Exception as e:
        log(f"WARNING: MLP reference SAE failed to load ({e}); continuing without it")
        sae_mlp = None

    chains = json.load(open(ROOT / "data" / "chains_R1-1.5B.json"))
    by_cat = defaultdict(list)
    for c in chains:
        by_cat[c["category"]].append(c)
    rng = random.Random(SEED)
    per_cat = max(1, N_CHAINS // len(by_cat))
    sample = [c for cat in sorted(by_cat)
              for c in rng.sample(by_cat[cat], min(per_cat, len(by_cat[cat])))]
    log(f"sampled {len(sample)} chains across {len(by_cat)} categories")

    def encode(chain):
        ids = tok(chain["full_text"], return_tensors="pt", truncation=True,
                  max_length=MAX_TOKENS).input_ids.to(DEVICE)
        keep = (ids[0] != EXCLUDE_TOKEN)
        return ids, keep

    # ---- pre-pass: resolve the candidate's true site + normalization convention
    log("pre-pass: probing candidate sites x conventions on "
        f"{N_PREPASS} chains: {list(RESID_SITES)} x [raw, norm]")
    probe = defaultdict(list)
    for chain in sample[:N_PREPASS]:
        ids, keep = encode(chain)
        hs = model(ids, output_hidden_states=True).hidden_states
        for name, idx in RESID_SITES.items():
            x = hs[idx][0][keep].float()
            for mode in (False, True):
                recon, _ = sae_resid.forward(x, normalize=mode)
                probe[(name, mode)].append(fvu(x, recon))
    probe_mean = {k: float(np.mean(v)) for k, v in probe.items()}
    (best_site_name, best_mode) = min(probe_mean, key=probe_mean.get)
    best_site = RESID_SITES[best_site_name]
    log("pre-pass FVU: " + "; ".join(
        f"{n}/{'norm' if m else 'raw'}={v:.3f}" for (n, m), v in sorted(probe_mean.items())))
    log(f"resolved site: {best_site_name} ({'norm' if best_mode else 'raw'}), "
        f"FVU={probe_mean[(best_site_name, best_mode)]:.3f}")

    resid_fn = lambda h: sae_resid.forward(h[0], normalize=best_mode)[0][None]
    zero_fn = lambda h: torch.zeros_like(h)

    # ---- main loop
    acc, per_chain = defaultdict(list), []
    for i, chain in enumerate(sample):
        ids, keep = encode(chain)
        mlp_cache = {}
        if sae_mlp is not None:
            hcap = model.model.layers[MLP_LAYER].mlp.register_forward_hook(
                lambda m, a, o: mlp_cache.__setitem__("out", o.detach().float()))
        out = model(ids, labels=ids, output_hidden_states=True)
        if sae_mlp is not None:
            hcap.remove()
        x = out.hidden_states[best_site][0][keep].float()
        row = {"task_id": chain["task_id"], "category": chain["category"],
               "n_tokens": int(ids.shape[1]), "ce_clean": float(out.loss)}
        del out

        recon, l0 = sae_resid.forward(x, normalize=best_mode)
        row["resid_fvu"] = fvu(x, recon)
        row["resid_l0"] = float(l0.mean())
        row["resid_cos"] = float(torch.nn.functional.cosine_similarity(x, recon, dim=-1).mean())
        row["resid_ce_recon"] = ce_loss(model, ids, (best_site, resid_fn))
        row["resid_ce_zero"] = ce_loss(model, ids, (best_site, zero_fn))

        if sae_mlp is not None and "out" in mlp_cache:
            m = mlp_cache["out"][0][keep]
            recon_m, l0_m = sae_mlp.forward(m)
            row["mlp_fvu"] = fvu(m, recon_m)
            row["mlp_l0"] = float(l0_m.mean())
            row["mlp_ce_recon"] = ce_loss(
                model, ids, (("mlp", MLP_LAYER), lambda h: sae_mlp.forward(h[0])[0][None]))
            row["mlp_ce_zero"] = ce_loss(model, ids, (("mlp", MLP_LAYER), zero_fn))

        per_chain.append(row)
        for k, v in row.items():
            if isinstance(v, float):
                acc[k].append(v)
        if DEVICE == "mps" and (i + 1) % 10 == 0:
            torch.mps.empty_cache()
        if (i + 1) % 5 == 0:
            log(f"{i+1}/{len(sample)} | resid FVU {np.mean(acc['resid_fvu']):.3f} "
                f"L0 {np.mean(acc['resid_l0']):.0f} | mlp FVU "
                f"{np.mean(acc['mlp_fvu']):.3f}" if acc.get("mlp_fvu") else
                f"{i+1}/{len(sample)} | resid FVU {np.mean(acc['resid_fvu']):.3f}")

    summary = {k: float(np.mean(v)) for k, v in acc.items()}

    def recovered(prefix):
        cz, cr, cc = (summary.get(f"{prefix}_ce_zero"), summary.get(f"{prefix}_ce_recon"),
                      summary.get("ce_clean"))
        if cz is None or cr is None or cz <= cc:
            return None
        return (cz - cr) / (cz - cc)

    summary["resid_ce_recovered"] = recovered("resid")
    if "mlp_fvu" in summary:
        summary["mlp_ce_recovered"] = recovered("mlp")

    fvu_v = summary["resid_fvu"]
    cer = summary["resid_ce_recovered"] if summary["resid_ce_recovered"] is not None else 0.0
    if fvu_v <= GATE["pass_fvu"] and cer >= GATE["pass_ce"]:
        verdict = "PASS"
    elif fvu_v <= GATE["amber_fvu"] and cer >= GATE["amber_ce"]:
        verdict = "AMBER"
    else:
        verdict = "FAIL"

    result = {"experiment": "E10.0 SAE dictionary gate", "date": time.strftime("%Y-%m-%d"),
              "model": MODEL_ID, "n_chains": len(sample), "max_tokens": MAX_TOKENS,
              "seed": SEED, "gate_thresholds": GATE,
              "candidate": RESID_SAE_REPO,
              "site_probe_fvu": {f"{n}|{'norm' if m else 'raw'}": v
                                 for (n, m), v in probe_mean.items()},
              "resolved_site": best_site_name,
              "resolved_normalization": "norm" if best_mode else "raw",
              "reference": MLP_SAE_REPO if sae_mlp else None,
              "summary": summary, "verdict": verdict, "per_chain": per_chain}
    (OUT_DIR / "e10_0_gate.json").write_text(json.dumps(result, indent=2))

    def fmt_v(v, spec=".4f"):
        return format(v, spec) if isinstance(v, (int, float)) else "n/a"

    lines = [
        "# E10.0 SAE dictionary gate — REPORT", "",
        f"Date: {result['date']} · {len(sample)} chains · cap {MAX_TOKENS} tok · device {DEVICE}",
        "",
        f"## VERDICT (residual candidate {RESID_SAE_REPO}): **{verdict}**", "",
        f"Site resolved empirically: **{best_site_name}** "
        f"({result['resolved_normalization']}); repo metadata was self-contradictory "
        "(card: blocks.19.resid_pre; folder: blocks.1.resid_pre; cfg: blocks.19.resid_post).",
        "",
        "Pre-pass site probe (mean FVU over "
        f"{N_PREPASS} chains): " + "; ".join(
            f"{n} {'norm' if m else 'raw'} = {v:.3f}" for (n, m), v in sorted(probe_mean.items())),
        "",
        "| metric | resid candidate (resolved site) | MLP reference (layers.19.mlp, own site) |",
        "|---|---|---|",
        f"| FVU | {fmt_v(summary['resid_fvu'])} | {fmt_v(summary.get('mlp_fvu'))} |",
        f"| L0 | {fmt_v(summary['resid_l0'], '.1f')} | {fmt_v(summary.get('mlp_l0'), '.1f')} |",
        f"| CE clean / recon / zero | {summary['ce_clean']:.3f} / {summary['resid_ce_recon']:.3f} / "
        f"{summary['resid_ce_zero']:.3f} | {summary['ce_clean']:.3f} / "
        f"{fmt_v(summary.get('mlp_ce_recon'), '.3f')} / {fmt_v(summary.get('mlp_ce_zero'), '.3f')} |",
        f"| CE recovered | {fmt_v(summary['resid_ce_recovered'], '.3f')} | "
        f"{fmt_v(summary.get('mlp_ce_recovered'), '.3f')} |",
        "",
        f"Gate (pre-registered): PASS iff FVU<={GATE['pass_fvu']} and CE-recovered>="
        f"{GATE['pass_ce']}; AMBER iff FVU<={GATE['amber_fvu']} and CE-recovered>="
        f"{GATE['amber_ce']}; else FAIL.",
        "",
        "FAIL/AMBER consequence: self-train residual-stream SAEs on the CoT distribution",
        "at the causal-candidate layers (Resa-style recipe, residual hookpoint) before any",
        "SAE comparator arm. The DAS arm is unaffected either way (no dictionary needed).",
    ]
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    log(f"DONE verdict={verdict} site={best_site_name} resid_FVU={fvu_v:.4f} "
        f"CE_recovered={cer:.3f} -> {OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
