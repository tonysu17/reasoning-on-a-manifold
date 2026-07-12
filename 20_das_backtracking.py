#!/usr/bin/env python
"""E10.1 — DAS-1D on backtracking (featurizer programme, rung F1).

Learns a ONE-dimensional orthogonal featurizer frame for backtracking by the
interchange-intervention criterion of distributed alignment search (Geiger et al.
2024; Wu et al. 2023), and compares it against the correlationally-built
difference-of-means direction (E1) and a random-rotation floor. This is the
causal upgrade of the fixed-featurizer steering of Phase 7: the frame is selected
because SWAPPING the coordinate it designates transfers the behaviour, not because
it correlates with annotations.

METHOD (pre-registered — see E10_DAS_PREREG.md):
  Counterfactual pairs, onset-anchored:
    source = the token position that PREDICTS a backtracking-span onset
             (the model is about to emit "Wait"/"Actually"/...);
    base   = the token position that predicts an ordinary forward-reasoning
             continuation (deduction / initializing span onset).
  For a learned unit direction d, the interchange swap at steering layer L
  (resid_pre[L] == hidden_states[L]) replaces the base's d-component with the
  source's and vice versa:
    h_b' = h_b + d dᵀ (h_s - h_b)     (induce backtracking in the base)
    h_s' = h_s + d dᵀ (h_b - h_s)     (remove it from the source)
  Objective = teacher-forced CE of the patched run toward the OTHER member's
  real onset token (the counterfactual label). Only d is learned; h_b, h_s are
  frozen constants from clean forwards. A 1-D component that flips the next-token
  behaviour is the causal claim.

  Controls: random-rotation floor (untrained d); shuffled-pair (train on permuted
  base/source — the Makelov illusion control); positional (swap at a misaligned
  position). Comparator: the E1 diff-of-means backtracking direction, evaluated
  on the identical interchange metric (P1: causal frame >= diff-of-means).

STAGES:
  pairs : build + tokenize counterfactual pairs         (CPU; run anywhere)
  train : DAS-1D training + shuffled-pair control        (GPU)
  eval  : interchange metrics for learned / diffmeans /   (GPU)
          random dirs, with positional control
  all   : pairs -> train -> eval

Layers: 17 (E1 attribution), 11 (de-confounded 07d mid-peak), 27 (read-out
proximity control). CUDA-first; --smoke runs a tiny end-to-end pass on MPS/CPU.

Cost: ~1-3 GPU-hours per layer on a 4090 (pairs cheap, train ~few hundred steps,
eval one pass). No API credits.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))
from src.text_offsets import locate_annotation_offsets  # noqa: E402

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ANNOT = ROOT / "data" / "annotated_R1-1.5B.json"
DIFFMEANS = ROOT / "results" / "steering_vectors" / "R1-1.5B__E1_pooled" / "backtracking_single.npy"
OUT_ROOT = ROOT / "results" / "das" / "R1-1.5B"

SOURCE_LABEL = "backtracking"
BASE_LABELS = ("deduction", "initializing")  # ordinary forward reasoning
HELDOUT_TASKS = ROOT / "results" / "steering_vectors" / "R1-1.5B__E1_pooled"  # eval split via metadata


def log(msg):
    print(f"[e10.1 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


# ----------------------------------------------------------------- pair building

def build_pairs(tok, cfg):
    """Construct onset-anchored counterfactual pairs.

    Each pair: {base_ids, base_target, source_ids, source_target}, where *_ids
    is the token context up to (and including) the prediction position, and
    *_target is the real onset token that position predicts. Contexts are
    right-truncated to cfg.ctx tokens (keep the last ctx before the onset).
    """
    data = json.load(open(ANNOT))
    rng = random.Random(cfg.seed)
    sources, bases = [], []  # (task_id, context_ids:list[int], target:int)

    W = cfg.window

    def onsets_for(rec):
        """Yield (label, onset_tok, ids) for every locatable span whose onset
        sits at token >= 1 (needs a prediction position before it)."""
        chain = rec["chain"]
        prompt_len = len(tok(rec["prompt"], add_special_tokens=False).input_ids)
        # tokenize full_text with offsets so we can map char->token in the chain
        enc = tok(rec["full_text"], add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc.input_ids, enc.offset_mapping
        chain_start_char = rec["full_text"].find(chain[:40])
        if chain_start_char < 0:
            return
        sents = [a["text"] for a in rec["annotations"]]
        labels = [a["label"] for a in rec["annotations"]]
        char_offsets = locate_annotation_offsets(chain, sents)
        for lab, coff in zip(labels, char_offsets):
            if coff is None:
                continue
            abs_char = chain_start_char + coff
            onset_tok = next((i for i, (s, e) in enumerate(offs) if e > abs_char), None)
            if onset_tok is None or onset_tok < prompt_len + 1:
                continue  # need a prediction position, and stay inside the chain
            yield lab, onset_tok, ids

    n_scanned = 0
    for rec in data:
        if not rec.get("annotation_complete"):
            continue
        n_scanned += 1
        for lab, onset_tok, ids in onsets_for(rec):
            ctx = ids[max(0, onset_tok - cfg.ctx):onset_tok]
            cont = ids[onset_tok:onset_tok + W]
            if len(ctx) < cfg.min_ctx or len(cont) < W:
                continue  # need a full continuation window (chain must not end inside it)
            item = (rec["task_id"], ctx, cont)
            if lab == SOURCE_LABEL:
                sources.append(item)
            elif lab in BASE_LABELS:
                bases.append(item)
        if cfg.max_chains and n_scanned >= cfg.max_chains:
            break

    rng.shuffle(sources)
    rng.shuffle(bases)
    n = min(len(sources), len(bases), cfg.n_pairs)
    pairs = []
    for i in range(n):
        s, b = sources[i], bases[i]
        pairs.append({"task_source": s[0], "source_ids": s[1], "source_cont": s[2],
                      "task_base": b[0], "base_ids": b[1], "base_cont": b[2]})
    log(f"pairs: scanned {n_scanned} chains -> {len(sources)} source / {len(bases)} base onsets "
        f"-> {len(pairs)} pairs (ctx<= {cfg.ctx}, window {W})")
    return pairs


# ----------------------------------------------------------------- model helpers

def load_model(device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"  # prediction position is always the last token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def batch_encode(tok, list_of_ids, device):
    maxlen = max(len(x) for x in list_of_ids)
    input_ids = torch.full((len(list_of_ids), maxlen), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(list_of_ids), maxlen), dtype=torch.long)
    for i, ids in enumerate(list_of_ids):
        input_ids[i, maxlen - len(ids):] = torch.tensor(ids)
        attn[i, maxlen - len(ids):] = 1
    return input_ids.to(device), attn.to(device)


@torch.no_grad()
def clean_resid(model, input_ids, attn, layer):
    """resid_pre[layer] at the last position (the prediction position) -> (B, D)."""
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
    return out.hidden_states[layer][:, -1, :].float()


def window_logprob(model, input_ids, attn, layer, h_new, W, pos_offset=0):
    """Teacher-forced mean logprob of the last-W continuation tokens, with
    resid_pre[layer] at the last CONTEXT position (index -W-1, shifted by
    pos_offset for the positional control) replaced by h_new (B, D). h_new=None
    runs clean. Gradients flow through h_new. Returns (B,) mean logprob."""
    handles = []
    if h_new is not None:
        p = -W - 1 + pos_offset

        def pre_hook(module, args, kwargs):
            hs = args[0].clone()
            hs[:, p, :] = h_new.to(hs.dtype)
            return (hs,) + args[1:], kwargs

        handles.append(model.model.layers[layer].register_forward_pre_hook(
            pre_hook, with_kwargs=True))
    try:
        out = model(input_ids=input_ids, attention_mask=attn)
    finally:
        for h in handles:
            h.remove()
    logits = out.logits[:, -W - 1:-1, :]                       # predict the window
    labels = input_ids[:, -W:]
    lp = F.log_softmax(logits.float(), -1).gather(2, labels[:, :, None]).squeeze(2)
    return lp.mean(dim=1)                                       # (B,)


# ----------------------------------------------------------------- DAS training

class Direction(torch.nn.Module):
    """A learnable unit direction in R^d (the 1-D orthogonal frame)."""

    def __init__(self, d, init=None, device="cpu"):
        super().__init__()
        v = torch.randn(d) if init is None else torch.as_tensor(init, dtype=torch.float32)
        self.raw = torch.nn.Parameter(v.to(device))

    def unit(self):
        return self.raw / self.raw.norm().clamp_min(1e-8)


def swap_states(hb, hs, d):
    """1-D interchange along unit d: base gets source's d-component and vice versa."""
    proj = ((hs - hb) @ d).unsqueeze(1) * d.unsqueeze(0)   # (B, D)
    return hb + proj, hs - proj


def interchange_loss(model, batch, layer, unit, W):
    """Symmetric windowed interchange loss: the patched base must produce the
    SOURCE's real W-token continuation; the patched source must produce the
    BASE's. A direction that merely promotes generic onset tokens cannot match
    the donor's specific continuation — that is what defeats the single-token
    degeneracy minival-1 exposed."""
    hb_new, hs_new = swap_states(batch["hb"], batch["hs"], unit)
    lp_b = window_logprob(model, batch["induce_ids"], batch["induce_attn"], layer, hb_new, W)
    lp_s = window_logprob(model, batch["remove_ids"], batch["remove_attn"], layer, hs_new, W)
    return -(lp_b.mean() + lp_s.mean())


def make_batches(pairs, tok, model, layer, device, bs, W):
    """Precompute frozen hb/hs (from ctx-only forwards) and the padded
    counterfactual inputs: induce = base_ctx + source_cont, remove = source_ctx
    + base_cont. All constants; only the direction is learned."""
    batches = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        b_ids, b_attn = batch_encode(tok, [p["base_ids"] for p in chunk], device)
        s_ids, s_attn = batch_encode(tok, [p["source_ids"] for p in chunk], device)
        hb = clean_resid(model, b_ids, b_attn, layer)
        hs = clean_resid(model, s_ids, s_attn, layer)
        ind_ids, ind_attn = batch_encode(
            tok, [p["base_ids"] + p["source_cont"] for p in chunk], device)
        rem_ids, rem_attn = batch_encode(
            tok, [p["source_ids"] + p["base_cont"] for p in chunk], device)
        batches.append({
            "induce_ids": ind_ids, "induce_attn": ind_attn,
            "remove_ids": rem_ids, "remove_attn": rem_attn,
            "hb": hb, "hs": hs,
        })
    return batches


def train_das(model, tok, pairs, layer, cfg, device, init=None, shuffle_pairs=False):
    d = model.config.hidden_size
    if shuffle_pairs:  # Makelov illusion control: break the base<->source correspondence
        perm = list(range(len(pairs)))
        random.Random(cfg.seed + 1).shuffle(perm)
        pairs = [{**pairs[i], "source_ids": pairs[perm[i]]["source_ids"],
                  "source_cont": pairs[perm[i]]["source_cont"]} for i in range(len(pairs))]
    batches = make_batches(pairs, tok, model, layer, device, cfg.bs, cfg.window)
    direction = Direction(d, init=init, device=device)
    opt = torch.optim.Adam(direction.parameters(), lr=cfg.lr)
    curve = []
    for ep in range(cfg.epochs):
        random.Random(cfg.seed + ep).shuffle(batches)
        tot = 0.0
        for batch in batches:
            opt.zero_grad()
            loss = interchange_loss(model, batch, layer, direction.unit(), cfg.window)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        curve.append(tot / len(batches))
        if ep % max(1, cfg.epochs // 8) == 0 or ep == cfg.epochs - 1:
            log(f"  L{layer}{' [shuf]' if shuffle_pairs else ''} epoch {ep+1}/{cfg.epochs} "
                f"loss {curve[-1]:.4f}")
    return direction.unit().detach(), curve


# ----------------------------------------------------------------- interchange eval

@torch.no_grad()
def eval_direction(model, tok, pairs, layer, unit, cfg, device, positional=False,
                   per_pair=False):
    """Windowed interchange transfer metrics for a fixed unit direction.
      induce_dlp = mean logP(source cont | base patched) - (| base clean)
      remove_dlp = mean logP(base cont | source patched) - (| source clean)
    positional=True injects the same swap two tokens early (misaligned control).
    per_pair=True additionally returns the per-pair deltas (for bootstrap CIs)."""
    W = cfg.window
    batches = make_batches(pairs, tok, model, layer, device, cfg.bs, W)
    off = -2 if positional else 0
    ind, rem, n = 0.0, 0.0, 0
    pp_ind, pp_rem = [], []
    for batch in batches:
        hb_new, hs_new = swap_states(batch["hb"], batch["hs"], unit)
        lp_ind_clean = window_logprob(model, batch["induce_ids"], batch["induce_attn"],
                                      layer, None, W)
        lp_ind = window_logprob(model, batch["induce_ids"], batch["induce_attn"],
                                layer, hb_new, W, pos_offset=off)
        lp_rem_clean = window_logprob(model, batch["remove_ids"], batch["remove_attn"],
                                      layer, None, W)
        lp_rem = window_logprob(model, batch["remove_ids"], batch["remove_attn"],
                                layer, hs_new, W, pos_offset=off)
        d_ind = (lp_ind - lp_ind_clean)
        d_rem = (lp_rem - lp_rem_clean)
        ind += float(d_ind.sum()); rem += float(d_rem.sum())
        if per_pair:
            pp_ind += [float(v) for v in d_ind]
            pp_rem += [float(v) for v in d_rem]
        n += batch["hb"].shape[0]
    out = {"n": n, "induce_dlp": ind / n, "remove_dlp": rem / n,
           "sym_dlp": (ind + rem) / (2 * n)}
    if per_pair:
        out["per_pair_induce"] = pp_ind
        out["per_pair_remove"] = pp_rem
    return out


def _boot_ci(vals, n_boot=10000, seed=0):
    """Percentile bootstrap 95% CI of the mean."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=np.float64)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# ----------------------------------------------------------------- driver

@torch.no_grad()
def controls_stage(model, tok, pairs, layer, unit, cfg, device):
    """Eval-time controls for a BINARY behaviour variable (prereg Amendment 2).

    Within-class donor shuffling is NOT a null for a binary variable (any
    behaviour-positive donor is a valid interchange), so the illusion test is:
    (a) base-donor null: swap in another BASE state instead of a source state;
        a direction that reads behaviour state collapses to ~0, a constant-bias
        direction keeps firing;
    (b) coordinate separation: AUC of h·d for source vs base prediction states;
        a causal frame whose coordinate does not covary with the behaviour is
        suspect regardless of its transfer number.
    """
    W = cfg.window
    # (b) class separation along d
    from sklearn.metrics import roc_auc_score
    batches = make_batches(pairs, tok, model, layer, device, cfg.bs, W)
    xs, ys = [], []
    for b in batches:
        xs += [float(v) for v in (b["hb"] @ unit)]; ys += [0] * b["hb"].shape[0]
        xs += [float(v) for v in (b["hs"] @ unit)]; ys += [1] * b["hs"].shape[0]
    auc = float(roc_auc_score(ys, xs))

    # (a) base-donor null: rotate the base pool by one so donor != own base
    null_pairs = [{**p,
                   "source_ids": pairs[(i + 1) % len(pairs)]["base_ids"],
                   "source_cont": p["source_cont"]}   # label stays the REAL source cont
                  for i, p in enumerate(pairs)]
    real = eval_direction(model, tok, pairs, layer, unit, cfg, device)
    null = eval_direction(model, tok, null_pairs, layer, unit, cfg, device)
    return {"coord_auc_source_vs_base": auc,
            "real_donor": real, "base_donor_null": null,
            "state_dependence": real["induce_dlp"] - null["induce_dlp"]}


@torch.no_grad()
def cis_stage(model, tok, pairs, cfg, device, OUT, diffmeans):
    """E10.1 follow-up (pre-registered): per-pair bootstrap CIs for the
    interchange table, PLUS the diff-of-means SITE-CHECK. Site convention
    discovered post-run: the pipeline's 'layer L' vectors are built at block-L
    OUTPUT = hidden_states[L+1], so the E1 'bt17' diff-means vector's true site
    is hs[18]; E10.1 evaluated it at hs[17] (one block off). Decision rule
    (sealed): if sym_dlp(dm @ hs18) > 2x sym_dlp(dm @ hs17), the 13x headline
    ratio is revised to the hs18 value and both are reported."""
    arms = {}
    lp = OUT / "dir_learned_L17.npy"
    if lp.exists():
        learned = torch.tensor(np.load(lp), dtype=torch.float32, device=device)
        arms["learned@hs17"] = (learned, 17)
        arms["learned_positional@hs17"] = (learned, 17)  # handled below
    wp = OUT / "dir_warm_L17.npy"
    if wp.exists():
        arms["warm@hs17"] = (torch.tensor(np.load(wp), dtype=torch.float32,
                                          device=device), 17)
    if diffmeans is not None:
        arms["diff_of_means@hs17"] = (diffmeans, 17)
        arms["diff_of_means@hs18_site_check"] = (diffmeans, 18)
    rand = torch.randn(model.config.hidden_size, device=device)
    arms["random@hs17"] = (rand / rand.norm(), 17)

    out = {}
    for name, (unit, site) in arms.items():
        res = eval_direction(model, tok, pairs, site, unit, cfg, device,
                             positional=("positional" in name), per_pair=True)
        sym = [(a + b) / 2 for a, b in zip(res["per_pair_induce"], res["per_pair_remove"])]
        res["sym_ci95"] = _boot_ci(sym)
        out[name] = res
        log(f"  {name}: sym={res['sym_dlp']:+.4f} CI95=[{res['sym_ci95'][0]:+.4f}, "
            f"{res['sym_ci95'][1]:+.4f}]")
    # paired difference: learned vs dm at each dm site
    if "learned@hs17" in out and "diff_of_means@hs17" in out:
        for dm_name in ("diff_of_means@hs17", "diff_of_means@hs18_site_check"):
            if dm_name not in out:
                continue
            l, d = out["learned@hs17"], out[dm_name]
            diff = [ (a + b) / 2 - (c + e) / 2 for a, b, c, e in
                     zip(l["per_pair_induce"], l["per_pair_remove"],
                         d["per_pair_induce"], d["per_pair_remove"]) ]
            out[f"paired_learned_minus_{dm_name}"] = {
                "mean": float(np.mean(diff)), "ci95": _boot_ci(diff)}
    json.dump(out, open(OUT / "cis.json", "w"), indent=2)
    log(f"cis -> {OUT}/cis.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pairs", "train", "eval", "controls", "cis", "all"],
                    default="all")
    ap.add_argument("--layers", type=int, nargs="+", default=[17, 11, 27])
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--ctx", type=int, default=320)
    ap.add_argument("--min-ctx", type=int, default=16)
    ap.add_argument("--max-chains", type=int, default=0)  # 0 = all
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=12,
                    help="teacher-forced continuation window W (W=1 reproduces the "
                         "degenerate single-token objective minival-1 exposed)")
    ap.add_argument("--no-warm", action="store_true", help="skip the warm-start arm")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    cfg = ap.parse_args()
    if cfg.smoke:
        cfg.n_pairs, cfg.ctx, cfg.bs, cfg.epochs, cfg.max_chains, cfg.layers = 8, 96, 4, 3, 60, [17]
    OUT = OUT_ROOT / (cfg.tag or ("smoke" if cfg.smoke else "main"))
    OUT.mkdir(parents=True, exist_ok=True)

    device, dtype = pick_device()
    torch.set_grad_enabled(True)
    log(f"device={device} dtype={dtype} stage={cfg.stage} layers={cfg.layers} "
        f"n_pairs={cfg.n_pairs} smoke={cfg.smoke}")

    from transformers import AutoTokenizer
    tok_only = AutoTokenizer.from_pretrained(MODEL_ID)

    # --- pairs
    pairs_path = OUT / "pairs.json"
    if cfg.stage in ("pairs", "all") or not pairs_path.exists():
        pairs = build_pairs(tok_only, cfg)
        json.dump(pairs, open(pairs_path, "w"))
        log(f"wrote {len(pairs)} pairs -> {pairs_path}")
        if cfg.stage == "pairs":
            return
    else:
        pairs = json.load(open(pairs_path))[:cfg.n_pairs]

    tok, model = load_model(device, dtype)

    diffmeans = None
    if DIFFMEANS.exists():
        v = np.load(DIFFMEANS)
        diffmeans = torch.tensor(v / (np.linalg.norm(v) + 1e-8), dtype=torch.float32, device=device)

    if cfg.stage == "cis":
        cis_stage(model, tok, pairs, cfg, device, OUT, diffmeans)
        return

    if cfg.stage == "controls":
        out = {}
        for layer in cfg.layers:
            dir_path = OUT / f"dir_learned_L{layer}.npy"
            if not dir_path.exists():
                log(f"no trained direction at {dir_path}; skipping L{layer}")
                continue
            unit = torch.tensor(np.load(dir_path), dtype=torch.float32, device=device)
            cell = {"learned": controls_stage(model, tok, pairs, layer, unit, cfg, device)}
            if diffmeans is not None:
                cell["diff_of_means"] = controls_stage(model, tok, pairs, layer, diffmeans,
                                                       cfg, device)
            out[str(layer)] = cell
            lc = cell["learned"]
            log(f"L{layer} learned: coord_AUC={lc['coord_auc_source_vs_base']:.3f} "
                f"induce real={lc['real_donor']['induce_dlp']:+.4f} vs base-donor "
                f"null={lc['base_donor_null']['induce_dlp']:+.4f} "
                f"(state-dependence {lc['state_dependence']:+.4f})")
        json.dump(out, open(OUT / "controls.json", "w"), indent=2)
        log(f"controls -> {OUT}/controls.json")
        return

    report = {"experiment": "E10.1 DAS-1D backtracking", "date": time.strftime("%Y-%m-%d"),
              "device": device, "n_pairs": len(pairs), "config": vars(cfg), "layers": {}}

    dm_init = diffmeans.cpu().numpy() if diffmeans is not None else None
    for layer in cfg.layers:
        log(f"=== layer {layer} ===")
        # --- train: cold (random init, the honest 'found from scratch' arm) +
        #     warm (init from diff-of-means: does the causal criterion move it further?)
        learned, curve = train_das(model, tok, pairs, layer, cfg, device)
        warm, curve_warm = ((None, None) if (dm_init is None or cfg.no_warm)
                            else train_das(model, tok, pairs, layer, cfg, device, init=dm_init))
        shuf, curve_shuf = train_das(model, tok, pairs, layer, cfg, device, shuffle_pairs=True)
        np.save(OUT / f"dir_learned_L{layer}.npy", learned.cpu().numpy())
        if warm is not None:
            np.save(OUT / f"dir_warm_L{layer}.npy", warm.cpu().numpy())
        rand = torch.randn(model.config.hidden_size, device=device)
        rand = rand / rand.norm()

        # --- eval: learned / warm / diffmeans / random / shuffled, + positional control
        cell = {"train_curve": curve, "warm_train_curve": curve_warm, "shuf_train_curve": curve_shuf,
                "cos_learned_diffmeans": (float(learned @ diffmeans) if diffmeans is not None else None),
                "cos_warm_diffmeans": (float(warm @ diffmeans) if warm is not None else None)}
        cell["learned"] = eval_direction(model, tok, pairs, layer, learned, cfg, device)
        cell["learned_positional"] = eval_direction(model, tok, pairs, layer, learned, cfg, device,
                                                     positional=True)
        if warm is not None:
            cell["warm"] = eval_direction(model, tok, pairs, layer, warm, cfg, device)
        cell["shuffled_pair"] = eval_direction(model, tok, pairs, layer, shuf, cfg, device)
        cell["random_rotation"] = eval_direction(model, tok, pairs, layer, rand, cfg, device)
        if diffmeans is not None:
            cell["diff_of_means"] = eval_direction(model, tok, pairs, layer, diffmeans, cfg, device)
        # pair-specificity: what the true correspondence adds over the shuffled one
        cell["pair_specificity_gap"] = (cell["learned"]["sym_dlp"]
                                        - cell["shuffled_pair"]["sym_dlp"])
        report["layers"][str(layer)] = cell

        def dlp(name):
            return cell.get(name, {}).get("sym_dlp", float("nan"))
        log(f"  L{layer} sym Δlogprob: learned {dlp('learned'):+.3f} | warm {dlp('warm'):+.3f} | "
            f"diffmeans {dlp('diff_of_means'):+.3f} | random {dlp('random_rotation'):+.3f} | "
            f"shuffled {dlp('shuffled_pair'):+.3f} | positional {dlp('learned_positional'):+.3f} | "
            f"pair-gap {cell['pair_specificity_gap']:+.3f}")

    json.dump(report, open(OUT / "report.json", "w"), indent=2)
    _write_report_md(report, OUT)
    log(f"DONE -> {OUT}/report.json + REPORT.md")


def _write_report_md(report, OUT):
    W = report["config"].get("window")
    L = ["# E10.1 DAS-1D backtracking — REPORT", "",
         f"Date: {report['date']} · device {report['device']} · {report['n_pairs']} pairs · "
         f"window W={W}", "",
         "Windowed interchange transfer: does swapping the 1-D component make the base",
         "context produce the SOURCE's real W-token continuation (induce) and the source",
         "context produce the BASE's (remove)? Metric = mean per-token Δlogprob, patched",
         "vs clean. The W-token window defeats the generic-onset-token degeneracy the",
         "single-token objective admitted (minival-1): a 'Wait-booster' direction can",
         "raise the first token but not the donor's specific continuation.", "",
         "| layer | dir | sym Δlogprob | induce | remove |",
         "|---|---|---|---|---|"]
    for lyr, c in report["layers"].items():
        for name in ("learned", "warm", "diff_of_means", "random_rotation",
                     "shuffled_pair", "learned_positional"):
            if name in c:
                m = c[name]
                L.append(f"| {lyr} | {name} | {m['sym_dlp']:+.3f} | "
                         f"{m['induce_dlp']:+.3f} | {m['remove_dlp']:+.3f} |")
        L.append(f"| {lyr} | pair_specificity_gap (learned − shuffled) | "
                 f"{c['pair_specificity_gap']:+.3f} | | |")
        L.append(f"| {lyr} | cos(learned,diffmeans) | {c.get('cos_learned_diffmeans')} | "
                 f"cos(warm,diffmeans)={c.get('cos_warm_diffmeans')} | |")
    L += ["", "**P1 (sealed):** learned (causal) direction sym Δlogprob >= diff_of_means.",
          "**Controls that must be near zero:** random_rotation and learned_positional.",
          "**Pair-specificity:** the learned direction must beat shuffled_pair by a clear",
          "margin (the gap row) — a direction that survives pair shuffling is a generic",
          "behaviour-token promoter, not a pair-aligned causal frame (Makelov illusion).",
          "", "Follow-on (P2, separate stage): free-generation collapse rate, swap vs projective",
          "ablation at matched on-target effect — requires generation, not run here."]
    (OUT / "REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
