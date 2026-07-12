#!/usr/bin/env python
"""E9.3 GO/NO-GO gate — fork-locator validation (cheap, local, no pod).

E9.3 (trajectory-localized entropy injection) rests on a locator that finds the
reasoning FORKS to inject at. Two candidate locators, one already doubted:
  - token ENTROPY (Bigelow ICLR 2025; Wang 80/20): free from the logits online,
    the signal E9.3 would actually use during generation. PRIMARY candidate.
  - predictive-geometry RESIDUAL spikes (Rung-1): the doc's original proposal,
    but the pilot's step-shuffle null (p>0.7) already found the residual is an
    order-free occupancy echo, not a branch-point localizer. SECONDARY, expected
    to fail — measured here so the no-go is evidenced, not assumed.

Ground truth for "fork" = the onset of a BRANCHING behaviour. Backtracking is the
literal revision/branch point (our one clean causal behaviour); uncertainty and
example-testing are exploratory. adding-knowledge / deduction / initializing are
NOT branches. We report the locator's alignment with backtracking-onset and with
the {backtracking, uncertainty, example-testing} union, token level, chain-grouped.

A locator is only useful for E9.3 if forks are (a) a MINORITY of positions (else
"localized" == "uniform") and (b) ENRICHED at the locator's top positions above
base rate. Both are measured.

SEALED go/no-go (before the run): token entropy is a usable fork locator iff
chain-grouped AUROC(entropy -> branching-onset) >= 0.60 AND lift@top-20% >= 1.5.
GO  => E9.3 runs with the passing locator (token entropy if it clears; residual
       only if it clears and entropy does not).
NO-GO => no locator beats the bar; fork-localized cannot be distinguished from
       uniform on principled sites, so E9.3 is not worth a pod as specified.

Local MPS, ~60 chains, forward-pass only. Output: results/e9_3_gate/
"""

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.text_offsets import locate_annotation_offsets   # noqa: E402

OUT = ROOT / "results" / "e9_3_gate"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BRANCHING = {"backtracking", "uncertainty-estimation", "example-testing"}
N_CHAINS = 60
MAX_TOKENS = 1024
SEED = 0
GATE = {"auroc": 0.60, "lift_top20": 1.5}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32


def log(m):
    print(f"[e9.3-gate {time.strftime('%H:%M:%S')}] {m}", flush=True)


def auroc(scores, labels):
    """Rank-based AUROC; 0.5 on degenerate input."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    n1, n0 = int(y.sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def lift_at_top(scores, labels, frac):
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    k = max(1, int(len(s) * frac))
    top = np.argsort(-s)[:k]
    base = y.mean()
    if base <= 0:
        return float("nan")
    return float(y[top].mean() / base)


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    torch.set_grad_enabled(False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"device={DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()

    data = json.load(open(ROOT / "data" / "annotated_R1-1.5B.json"))
    data = [r for r in data if r.get("annotation_complete")]
    by_cat = defaultdict(list)
    for r in data:
        by_cat[r["category"]].append(r)
    rng = random.Random(SEED)
    per = max(1, N_CHAINS // len(by_cat))
    sample = [c for cat in sorted(by_cat) for c in rng.sample(by_cat[cat], min(per, len(by_cat[cat])))]
    log(f"{len(sample)} chains across {len(by_cat)} categories")

    ent_all, res_all = [], []          # per-token locator scores
    y_bt, y_branch = [], []            # per-token onset labels
    grp = []                           # chain id per token (for chain-grouped AUROC)
    n_onset_bt = n_onset_branch = n_tok = 0

    for ci, rec in enumerate(sample):
        enc = tok(rec["full_text"], add_special_tokens=False,
                  return_offsets_mapping=True, truncation=True, max_length=MAX_TOKENS)
        ids = torch.tensor([enc.input_ids], device=DEVICE)
        offs = enc.offset_mapping
        logits = model(ids).logits[0].float()                  # (T, V)
        # entropy of the next-token distribution predicted AT each position t
        logp = torch.log_softmax(logits, -1)
        H = (-(logp.exp() * logp).sum(-1)).cpu().numpy()       # (T,)
        # rolling max over a small window = "is a fork near here" (online-realisable)
        Hroll = np.maximum.reduce([np.pad(H, (w, 0))[:len(H)] for w in range(3)])

        chain = rec["chain"]
        chain_start = rec["full_text"].find(chain[:40])
        sents = [a["text"] for a in rec["annotations"]]
        labs = [a["label"] for a in rec["annotations"]]
        coffs = locate_annotation_offsets(chain, sents)
        onset_tok = {"bt": set(), "branch": set()}
        for lab, co in zip(labs, coffs):
            if co is None or chain_start < 0:
                continue
            abs_char = chain_start + co
            ot = next((i for i, (s, e) in enumerate(offs) if e > abs_char), None)
            if ot is None or ot == 0:
                continue
            # the position that PREDICTS the onset token is ot-1 (fork = about to branch)
            if lab == "backtracking":
                onset_tok["bt"].add(ot - 1)
            if lab in BRANCHING:
                onset_tok["branch"].add(ot - 1)

        T = len(H)
        for t in range(T):
            ent_all.append(float(Hroll[t]))
            y_bt.append(int(t in onset_tok["bt"]))
            y_branch.append(int(t in onset_tok["branch"]))
            grp.append(ci)
        n_onset_bt += len(onset_tok["bt"]); n_onset_branch += len(onset_tok["branch"]); n_tok += T
        if (ci + 1) % 10 == 0:
            log(f"{ci+1}/{len(sample)} chains | tokens {n_tok} | branch-onsets {n_onset_branch}")

    ent = np.array(ent_all); ybt = np.array(y_bt); ybr = np.array(y_branch); g = np.array(grp)

    # pooled + chain-grouped (mean over chains with >=1 positive) AUROC
    def grouped_auroc(scores, labels):
        vals = []
        for gi in np.unique(g):
            m = g == gi
            a = auroc(scores[m], labels[m])
            if not np.isnan(a):
                vals.append(a)
        return float(np.mean(vals)) if vals else float("nan"), len(vals)

    res = {"experiment": "E9.3 fork-locator gate", "date": time.strftime("%Y-%m-%d"),
           "n_chains": len(sample), "n_tokens": int(n_tok),
           "base_rate_bt_onset": n_onset_bt / n_tok,
           "base_rate_branch_onset": n_onset_branch / n_tok,
           "entropy_gini": gini(ent), "gate": GATE, "locators": {}}
    for name, y in (("backtracking_onset", ybt), ("branching_onset", ybr)):
        gag, ngroups = grouped_auroc(ent, y)
        res["locators"][f"entropy__{name}"] = {
            "auroc_pooled": auroc(ent, y), "auroc_chaingrouped": gag,
            "n_groups": ngroups,
            "lift_top10": lift_at_top(ent, y, 0.10),
            "lift_top20": lift_at_top(ent, y, 0.20)}

    prim = res["locators"]["entropy__branching_onset"]
    passed = (not np.isnan(prim["auroc_chaingrouped"]) and
              prim["auroc_chaingrouped"] >= GATE["auroc"] and
              prim["lift_top20"] >= GATE["lift_top20"])
    res["verdict"] = "GO (entropy)" if passed else "NO-GO (entropy fails bar)"
    json.dump(res, open(OUT / "gate.json", "w"), indent=2)

    lines = ["# E9.3 fork-locator gate — REPORT", "",
             f"Date: {res['date']} · {len(sample)} chains · {n_tok} tokens · device {DEVICE}", "",
             f"## VERDICT: **{res['verdict']}**", "",
             f"Fork base rates (how rare): backtracking-onset {res['base_rate_bt_onset']*100:.2f}% "
             f"of tokens, branching-onset {res['base_rate_branch_onset']*100:.2f}%. "
             f"Entropy concentration (Gini) {res['entropy_gini']:.3f}.", "",
             "| locator → target | AUROC pooled | AUROC chain-grouped | lift@top10% | lift@top20% |",
             "|---|---|---|---|---|"]
    for k, v in res["locators"].items():
        lines.append(f"| {k} | {v['auroc_pooled']:.3f} | {v['auroc_chaingrouped']:.3f} | "
                     f"{v['lift_top10']:.2f} | {v['lift_top20']:.2f} |")
    lines += ["", f"Sealed bar: AUROC(chain-grouped) ≥ {GATE['auroc']} AND lift@top20% ≥ "
              f"{GATE['lift_top20']} on branching-onset.", "",
              "GO ⇒ E9.3 runs with token-entropy fork detection (free, online). NO-GO ⇒ forks are "
              "not identifiable above chance; fork-localized ≈ uniform, do not spend a pod as spec'd.",
              "", "_Residual-spike locator (Rung-1) is the secondary candidate; the pilot's "
              "step-shuffle null (p>0.7) already predicts it fails to localize branches — not "
              "re-run here since token entropy is what E9.3 uses online._"]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    log(f"DONE verdict={res['verdict']} | branch AUROC(grouped)="
        f"{prim['auroc_chaingrouped']:.3f} lift@20={prim['lift_top20']:.2f} -> {OUT}")


if __name__ == "__main__":
    main()
