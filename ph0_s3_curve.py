#!/usr/bin/env python3
"""ph0 s3 — full-sequence contraction curve + SV-profile domain check (pod stage).

Freeze: results/prereg/PHASE0_TRANSPORT_FREEZE_2026-08-02.md §1.1 + §2 s3.
pt13b's local run FAILED Gate B2 (stored Xout_* are class-selected non-contiguous token
subsamples, not sequence excerpts) — this stage does the job on full sequences.

Modes:
  --extract --arm {r1,deepscaler}   (pod, GPU) teacher-force the 200-task matched set with the
      R1 tokenizer (byte-identical rule; matched-ids gate already in results/r1_compression/
      gate_tokenizer.json), cache L17+L16 gen-token states, save per task: top-256 singular
      values, windowed-PR mean (W=128/S=64), and (r1 only) the k=5 c-grid injection deltas.
  --analyse                          (anywhere, CPU) gates + family check + mapping:
      Gate S3-A: paired median dPR (deepscaler − r1) within 20% rel. of report.json −0.0167
                 (fp16/re-extraction tolerance; exact equality not expected).
      Family check: median SV-ratio profile sigma_ds(i)/sigma_r1(i); family-consistent iff a
                 flat-head(i<=5)/flat-tail step profile fits with RMS residual <= 0.05.
      Mapping: observed deepscaler (and star1, family-unverified) medians on the full-seq curve.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "results/safety_posttrain/ph0_s3"
C_GRID = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]
K = 5
LAYERS = [17, 16]
N_SV = 256


def stage_extract(arm: str, limit: int | None) -> None:
    import torch  # noqa: F401  (pod env)
    from src.chain_gen import load_model
    from src.hooks import ActivationCache
    from src.loop_geometry import windowed_state_metrics

    r0 = importlib.import_module("29_r0_entropy_ladder")
    r1c = importlib.import_module("30_r1_compression")

    sample = json.loads((ROOT / "results/r0_entropy_ladder/R1-1.5B/sample.json").read_text())
    tids = sample["loop"] + sample["clean"]
    if limit:
        tids = tids[:limit]
    chains = {r["task_id"]: r for r in
              json.loads((ROOT / "data/chains_R1-1.5B.json").read_text())}

    from transformers import AutoTokenizer
    ref_tok = AutoTokenizer.from_pretrained(r1c.MODEL_IDS["r1"])  # byte-identical rule (C1)
    model, _ = load_model(r1c.MODEL_IDS[arm], dtype="float16")
    model.eval()
    device = next(model.parameters()).device

    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict = {}
    n_fail = 0
    for i, tid in enumerate(tids):
        try:
            rec = chains[tid]
            full_text = rec.get("full_text") or (rec["prompt"] + rec["chain"])
            enc, gen_tok = r0._tokenize(ref_tok, full_text, len(rec["prompt"]), 8192, device)
            if gen_tok.size < 128:
                raise ValueError("too short")
            gen_lo = int(gen_tok[0])
            import torch
            with ActivationCache(model, layers=LAYERS) as cache, torch.no_grad():
                model(**enc)
                H = {L: cache[L][0].float().numpy() for L in LAYERS}
            for L in LAYERS:
                X = H[L][gen_lo:].astype(np.float64)
                mu = X.mean(axis=0, keepdims=True)
                U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
                base = float(np.nanmean(
                    windowed_state_metrics(X, window=128, stride=64)["pr"]))
                payload[f"sv_{L}_{tid}"] = s[:N_SV].astype(np.float32)
                payload[f"prwin_{L}_{tid}"] = np.float64(base)
                if arm == "r1":
                    dpr = []
                    for c in C_GRID:
                        s2 = s.copy()
                        s2[K:] *= c
                        Xr = (U * s2) @ Vt + mu
                        pr = float(np.nanmean(
                            windowed_state_metrics(Xr, window=128, stride=64)["pr"]))
                        dpr.append(pr - base)
                    payload[f"dpr_{L}_{tid}"] = np.array(dpr)
                    payload[f"vrem_{L}_{tid}"] = np.array(
                        [1.0 - float((np.where(np.arange(len(s)) < K, s, s * c) ** 2).sum()
                                     / (s ** 2).sum()) for c in C_GRID])
        except Exception as e:  # fail-soft, counted (house pattern)
            n_fail += 1
            print(f"FAIL {tid}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"[{arm}] {i+1}/{len(tids)} ({n_fail} fail)", flush=True)
    np.savez_compressed(OUT / f"{arm}_fullseq.npz", **payload)
    print(f"extract[{arm}] done: {len(tids)} tasks, {n_fail} failures")


def stage_analyse() -> None:
    report = json.loads((ROOT / "results/r1_compression/report.json").read_text())
    z_r1 = np.load(OUT / "r1_fullseq.npz")
    z_ds = np.load(OUT / "deepscaler_fullseq.npz")
    res: dict = {"date": "2026-08-02+", "freeze": "PHASE0_TRANSPORT_FREEZE §1.1/§2-s3"}

    tids = sorted({k.split("_", 2)[2] for k in z_r1.files if k.startswith("prwin_17_")}
                  & {k.split("_", 2)[2] for k in z_ds.files if k.startswith("prwin_17_")})
    d = [float(z_ds[f"prwin_17_{t}"]) - float(z_r1[f"prwin_17_{t}"]) for t in tids]
    med = float(np.median(d))
    want = report["rep_paired_vs_r1"]["deepscaler"]["pr"]["median_delta"]
    res["gate_s3a"] = {"recomputed_median": med, "report_median": want, "n": len(d),
                       "pass": bool(abs(med - want) <= 0.2 * abs(want))}

    # family check: median SV-ratio profile, step-fit head(i<K) / tail(i>=K)
    n = min(min(len(z_ds[f"sv_17_{t}"]), len(z_r1[f"sv_17_{t}"])) for t in tids)
    prof = np.median(np.stack(
        [z_ds[f"sv_17_{t}"][:n] / np.maximum(z_r1[f"sv_17_{t}"][:n], 1e-12) for t in tids]), 0)
    head, tail = float(np.median(prof[:K])), float(np.median(prof[K:]))
    fit = np.where(np.arange(n) < K, head, tail)
    rms = float(np.sqrt(np.mean((prof - fit) ** 2)))
    res["family_check"] = {"head_ratio": head, "tail_ratio": tail, "rms_residual": rms,
                           "pass": bool(rms <= 0.05), "profile_first32": prof[:32].tolist()}

    # full-seq curve (r1) + mapping
    curve = {}
    for L in LAYERS:
        ts = [k.split("_", 2)[2] for k in z_r1.files if k.startswith(f"dpr_{L}_")]
        dpr = np.stack([z_r1[f"dpr_{L}_{t}"] for t in ts])
        vre = np.stack([z_r1[f"vrem_{L}_{t}"] for t in ts])
        curve[L] = {"c": C_GRID, "median_dpr": np.median(dpr, 0).tolist(),
                    "mean_var_removed": np.mean(vre, 0).tolist(), "n_tasks": len(ts)}
    res["curve_fullseq"] = curve
    xs = np.array(curve[17]["median_dpr"])[::-1]
    ys = np.array(curve[17]["mean_var_removed"])[::-1]
    res["mapping"] = {}
    for arm in ("deepscaler", "star1"):
        obs = report["rep_paired_vs_r1"][arm]["pr"]["median_delta"]
        ok = bool(xs.min() <= obs <= xs.max())
        res["mapping"][arm] = {
            "observed_median_dpr": obs,
            "var_removed_interp": float(np.interp(obs, xs, ys)) if ok else None,
            "family_verified": bool(arm == "deepscaler" and res["family_check"]["pass"]),
        }
    (OUT / "s3_analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("gate_s3a", "family_check", "mapping")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--arm", choices=["r1", "deepscaler"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.extract:
        stage_extract(a.arm, a.limit)
    if a.analyse:
        stage_analyse()
